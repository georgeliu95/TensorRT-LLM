# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for the NVFP4 MoE SVDQuant math contract."""

from __future__ import annotations

import pytest
import torch

from tensorrt_llm._torch.modules.fused_moe import svdquant_helpers as svdh


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in svdh.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def test_load_config_defaults_disabled(clean_env: pytest.MonkeyPatch) -> None:
    # Given no SVDQuant environment variables, when configuration is parsed.
    config = svdh.load_config()

    # Then the feature is off and its documented defaults remain observable.
    assert config == svdh.SvdquantConfig(
        enabled=False,
        rank=64,
        us_dtype=torch.bfloat16,
        device="cuda",
        fc13=False,
        fc2=False,
    )


@pytest.mark.parametrize(
    ("fc13", "fc2", "expected"),
    [(None, None, (True, True)), ("0", None, (False, True)),
     (None, "0", (True, False)), ("0", "0", (False, False))],
)
def test_stage_flags_follow_master(
    clean_env: pytest.MonkeyPatch,
    fc13: str | None,
    fc2: str | None,
    expected: tuple[bool, bool],
) -> None:
    # Given the master switch and optional per-stage overrides.
    clean_env.setenv(svdh.ENV_ENABLED, "1")
    if fc13 is not None:
        clean_env.setenv(svdh.ENV_FC13, fc13)
    if fc2 is not None:
        clean_env.setenv(svdh.ENV_FC2, fc2)

    # When parsed, then each stage follows the master unless overridden.
    config = svdh.load_config()
    assert (config.fc13, config.fc2) == expected
    assert config.any_stage is any(expected)


@pytest.mark.parametrize(
    ("name", "value", "message"),
    [(svdh.ENV_RANK, "0", svdh.ENV_RANK),
     (svdh.ENV_RANK, "rank", svdh.ENV_RANK),
     (svdh.ENV_US_DTYPE, "int8", svdh.ENV_US_DTYPE),
     (svdh.ENV_DEVICE, "tpu", svdh.ENV_DEVICE),
     (svdh.ENV_FC13, "sometimes", svdh.ENV_FC13)],
)
def test_invalid_config_fails_closed(
    clean_env: pytest.MonkeyPatch,
    name: str,
    value: str,
    message: str,
) -> None:
    # Given SVDQuant enabled with one malformed option.
    clean_env.setenv(svdh.ENV_ENABLED, "1")
    clean_env.setenv(name, value)

    # When parsed, then the boundary rejects it with a stable typed error.
    with pytest.raises(svdh.SvdquantConfigError, match=message):
        svdh.load_config()


@pytest.mark.parametrize("source_dtype", [torch.bfloat16, torch.float16])
def test_decompose_preserves_source_dtype_and_uses_fp32_svd(
    monkeypatch: pytest.MonkeyPatch,
    source_dtype: torch.dtype,
) -> None:
    # Given a supported dense source and an observer around torch.linalg.svd.
    torch.manual_seed(17)
    weight = torch.randn(24, 32, dtype=torch.float32).to(source_dtype)
    observed_dtypes: list[torch.dtype] = []
    original_svd = torch.linalg.svd

    def observing_svd(
        tensor: torch.Tensor,
        *,
        full_matrices: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        observed_dtypes.append(tensor.dtype)
        return original_svd(tensor, full_matrices=full_matrices)

    monkeypatch.setattr(torch.linalg, "svd", observing_svd)

    # When decomposed on CPU.
    us, vh, residual = svdh.decompose_per_tensor(
        weight, rank=8, us_dtype=torch.bfloat16, device="cpu")

    # Then SVD is FP32 while storage and the residual follow their contracts.
    assert observed_dtypes == [torch.float32]
    assert (us.dtype, vh.dtype) == (torch.bfloat16, torch.bfloat16)
    assert residual.dtype is source_dtype
    reconstructed = svdh.reconstruct_lowrank(us, vh).to(source_dtype) + residual
    torch.testing.assert_close(reconstructed, weight, rtol=2e-2, atol=2e-2)


def test_batched_decomposition_and_runtime_formula_match_dense_reference() -> None:
    # Given two expert matrices and token inputs.
    torch.manual_seed(23)
    weights = torch.randn(2, 20, 28, dtype=torch.float32)
    inputs = torch.randn(7, 28, dtype=torch.float32)

    # When decomposed and the runtime low-rank formula is evaluated.
    us, vh, residual = svdh.decompose_batched(
        weights, rank=12, us_dtype=torch.float32, device="cpu")
    result = svdh.lowrank_gemm(inputs, us[1], vh[1])

    # Then shapes are per expert and (x @ vh.T) @ us.T matches dense math.
    assert us.shape == (2, 20, 12)
    assert vh.shape == (2, 12, 28)
    assert residual.shape == weights.shape
    reference = inputs @ (us[1] @ vh[1]).T
    torch.testing.assert_close(result, reference, rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("projection", ["column", "row"])
def test_tp4_factor_ownership_reconstructs_each_local_dense_shard(
    projection: str,
) -> None:
    # Given one global dense projection decomposed before TP ownership is set.
    torch.manual_seed(29)
    weight = torch.randn(32, 48, dtype=torch.float32)
    us, vh, residual = svdh.decompose_per_tensor(
        weight, rank=16, us_dtype=torch.float32, device="cpu")

    # When each TP=4 rank takes its projection-local factors.
    for tp_rank in range(4):
        local_us, local_vh = svdh.shard_lowrank_factors(
            us, vh, tp_size=4, tp_rank=tp_rank, projection=projection)
        split_dim = 0 if projection == "column" else 1
        local_weight = torch.tensor_split(weight, 4, dim=split_dim)[tp_rank]
        local_residual = torch.tensor_split(residual, 4, dim=split_dim)[tp_rank]

        # Then the locally owned factors reconstruct that rank's dense shard.
        reconstructed = local_us @ local_vh + local_residual
        torch.testing.assert_close(reconstructed, local_weight,
                                   rtol=1e-4, atol=1e-4)


def test_storage_shapes_are_projection_local() -> None:
    # Given an expert-local projection shape, when storage shapes are derived.
    us_shape, vh_shape = svdh.lowrank_storage_shape(
        num_experts=6, out_features=40, in_features=96, rank=8)

    # Then US owns the local output and Vh owns the local input dimensions.
    assert us_shape == (6, 40, 8)
    assert vh_shape == (6, 8, 96)
