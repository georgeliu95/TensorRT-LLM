# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contract, parity, and CUDA-graph coverage for the phase2-only NVFP4 quantize op.

``trtllm::fp4_quantize_phase2`` is opt-in: unlike ``fp4_quantize_fused`` it never
scans for amax, never spins on a retirement counter, and never runs the
grid-wide barrier between phases -- it only consumes a caller-supplied
``{amax, global_scale}`` pair and runs the same phase-2 quantize step that
``fp4_quantize_fused`` runs internally. Parity against ``fp4_quantize_fused``
(fed that call's own returned ``amaxScale``) is therefore the correctness
oracle: both must produce bit-identical packed FP4 and scale factors.
"""

from __future__ import annotations

import pytest
import torch
from torch._subclasses.fake_tensor import FakeTensorMode

import tensorrt_llm  # noqa: F401  (registers the trtllm:: op library)
import tensorrt_llm.quantization.utils.fp4_utils as fp4_utils
from tensorrt_llm._utils import get_sm_version

_SM100_AVAILABLE = torch.cuda.is_available() and get_sm_version() in (100, 103)


def test_fp4_quantize_phase2_requires_a_cuda_tensor() -> None:
    """The op is CUDA-only; a CPU tensor must fail loudly, not silently no-op."""
    input_cpu = torch.randn(8, 64, dtype=torch.bfloat16)
    amax_scale_cpu = torch.tensor([1.0, 2688.0], dtype=torch.float32)

    with pytest.raises(RuntimeError):
        torch.ops.trtllm.fp4_quantize_phase2(input_cpu, amax_scale_cpu, 16, True,
                                             0)


@pytest.mark.parametrize("is_sf_swizzled_layout", [True, False])
def test_fp4_quantize_phase2_fake_shapes(is_sf_swizzled_layout: bool) -> None:
    """The fake/meta registration must report the same shapes fp4_quantize_ex does."""
    rows, cols, sf_vec_size = 37, 64, 16

    with FakeTensorMode():
        fake_input = torch.empty(rows, cols, dtype=torch.bfloat16, device="cuda")
        fake_amax_scale = torch.empty(2, dtype=torch.float32, device="cuda")
        fp4, sf = torch.ops.trtllm.fp4_quantize_phase2(
            fake_input, fake_amax_scale, sf_vec_size, is_sf_swizzled_layout, 0)

    expected_fp4_shape, expected_sf_size = fp4_utils.get_fp4_shape(
        [rows, cols], sf_vec_size, is_sf_swizzled_layout)
    assert list(fp4.shape) == expected_fp4_shape
    assert fp4.dtype == torch.uint8
    assert sf.shape == (expected_sf_size, )
    assert sf.dtype == torch.uint8


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
@pytest.mark.parametrize(
    ("scale_rule", "quant_range"),
    [
        (0, 448.0 * 6.0),  # standard NVFP4
        (1, 1536.0),  # adaptive MSE 4/6
    ],
)
def test_fp4_quantize_phase2_matches_fp4_quantize_fused(
    scale_rule: int,
    quant_range: float,
) -> None:
    """Bit-exact parity with fp4_quantize_fused fed that call's own amaxScale.

    This op takes no MoE routing mask, so every row is compared -- there is
    no padding region to exclude. The row count (197) is not a multiple of
    128 to exercise the SWIZZLED-layout row-padding path, and per-row
    magnitude varies so a row-indexing bug in the phase2-only launch would
    show up as a mismatch rather than being hidden by uniform data.
    """
    torch.manual_seed(0)
    rows, cols = 197, 512
    row_scale = torch.linspace(0.1, 8.0, rows, device="cuda").unsqueeze(1)
    activation = (torch.randn(rows, cols, device="cuda") *
                 row_scale).to(torch.bfloat16)

    fp4_ref, sf_ref, amax_scale = torch.ops.trtllm.fp4_quantize_fused(
        activation, 16, False, True, scale_rule, quant_range, 1e-12, 0, 0,
        None, None, 0)

    fp4_phase2, sf_phase2 = torch.ops.trtllm.fp4_quantize_phase2(
        activation, amax_scale, 16, True, scale_rule)

    torch.testing.assert_close(fp4_phase2, fp4_ref, rtol=0, atol=0)
    torch.testing.assert_close(sf_phase2, sf_ref, rtol=0, atol=0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
def test_fp4_quantize_phase2_rejects_unsupported_scale_rule() -> None:
    """Only rule 0 (standard) and rule 1 (MSE 4/6) are wired for this op."""
    activation = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    amax_scale = torch.tensor([1.0, 2688.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="adaptive MSE"):
        torch.ops.trtllm.fp4_quantize_phase2(activation, amax_scale, 16, True,
                                             2)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
def test_fp4_quantize_phase2_rejects_unsupported_sf_vec_size() -> None:
    activation = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    amax_scale = torch.tensor([1.0, 2688.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="sfVecSize"):
        torch.ops.trtllm.fp4_quantize_phase2(activation, amax_scale, 32, True,
                                             0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
def test_fp4_quantize_phase2_rejects_wrong_amax_scale_shape() -> None:
    activation = torch.randn(8, 64, dtype=torch.bfloat16, device="cuda")
    amax_scale = torch.tensor([1.0, 2688.0, 0.0],
                              dtype=torch.float32,
                              device="cuda")

    with pytest.raises(RuntimeError, match="amaxScale"):
        torch.ops.trtllm.fp4_quantize_phase2(activation, amax_scale, 16, True,
                                             0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
def test_fp4_quantize_phase2_rejects_unsupported_dtype() -> None:
    activation = torch.randn(8, 64, dtype=torch.float32, device="cuda")
    amax_scale = torch.tensor([1.0, 2688.0], dtype=torch.float32, device="cuda")

    with pytest.raises(RuntimeError, match="fp16/bf16"):
        torch.ops.trtllm.fp4_quantize_phase2(activation, amax_scale, 16, True,
                                             0)


@pytest.mark.skipif(not _SM100_AVAILABLE,
                    reason="fp4_quantize_phase2 requires SM100/SM103.")
def test_fp4_quantize_phase2_replays_in_cuda_graph() -> None:
    """Replay must consume both the replay-time input and a replay-time-updated scale.

    No amax scan, retirement counter, or grid-wide barrier means this is a
    single, non-persistent kernel launch. Unlike the fused prologue kernels'
    graph tests, the second graph input -- the precomputed amaxScale tensor
    -- is also mutated in place before replay, to demonstrate the
    precomputed-scale contract this op is built around.
    """
    rows, cols = 128, 256
    activation = torch.full((rows, cols), 2.0, dtype=torch.bfloat16,
                            device="cuda")
    amax_scale = torch.tensor([2.0, 2688.0 / 2.0],
                              dtype=torch.float32,
                              device="cuda")

    def quantize():
        return torch.ops.trtllm.fp4_quantize_phase2(activation, amax_scale,
                                                     16, True, 0)

    quantize()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_fp4, captured_sf = quantize()

    # Replay-time input and replay-time scale both change; the captured
    # output buffers must reflect the new values, not the ones seen at
    # capture time.
    activation.fill_(4.0)
    amax_scale.copy_(torch.tensor([4.0, 2688.0 / 4.0], device="cuda"))
    graph.replay()
    torch.cuda.synchronize()

    fp4_expected, sf_expected = torch.ops.trtllm.fp4_quantize_phase2(
        activation, amax_scale, 16, True, 0)
    torch.testing.assert_close(captured_fp4, fp4_expected, rtol=0, atol=0)
    torch.testing.assert_close(captured_sf, sf_expected, rtol=0, atol=0)
