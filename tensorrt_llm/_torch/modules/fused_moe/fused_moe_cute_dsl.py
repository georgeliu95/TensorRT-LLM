# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
import os as _os
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from tensorrt_llm._utils import get_sm_version, is_sm_100f, nvtx_range_debug
from tensorrt_llm.models.modeling_utils import QuantAlgo

from ...autotuner import (AutoTuner, ConstraintSpec, DynamicTensorSpec,
                          OptimizationProfile, TunableRunner, TuningConfig)
from ...custom_ops.cute_dsl_custom_ops import (
    GroupedGemmInputsHelper,
    Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
    Sm100BlockScaledContiguousGroupedGemmRunner,
    Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner)
from ...distributed import allgather
from ...model_config import ModelConfig
from ...utils import (ActivationType, AuxStreamType, EventType,
                      Fp4QuantizedTensor,
                      get_last_power_of_2_num_tokens_buckets,
                      last_positive_power_of_2)
from . import svdquant_helpers
from .fused_moe_cutlass import CutlassFusedMoE
from .interface import AlltoallMethodType
from .quantization import (SVDQUANT_FC13_PACKED_ORDER, SVDQUANT_FC13_PACKED_VH,
                           SVDQUANT_FC13_SEPARATED_WEIGHT_LAYOUT,
                           MoEWeightLoadingMode, NVFP4CuteDslFusedMoEMethod)
from .routing import BaseMoeRoutingMethod

# ---------------------------------------------------------------------------
# Runtime NVFP4 activation quantization.  Existing deployments keep the legacy
# adaptive flags; the benchmark override selects standard or 4o6 explicitly.
# ---------------------------------------------------------------------------
_RUNTIME_ACTIVATION_ENV = "TRTLLM_NVFP4_RUNTIME_ACTIVATION"
_STANDARD_NVFP4_QUANT_RANGE = 448.0 * 6.0
_ADAPTIVE_QUANT_RANGE_FC2 = 1536.0


class RuntimeActivationQuantization(NamedTuple):
    """One runtime NVFP4 activation encoding rule."""

    scale_rule: int
    quant_range: float


_RUNTIME_STANDARD = RuntimeActivationQuantization(
    scale_rule=0, quant_range=_STANDARD_NVFP4_QUANT_RANGE)
_RUNTIME_4O6 = RuntimeActivationQuantization(
    scale_rule=1, quant_range=_ADAPTIVE_QUANT_RANGE_FC2)


@dataclass(frozen=True, slots=True)
class RuntimeActivationModeError(ValueError):
    """The runtime NVFP4 activation mode is unsupported."""

    value: str

    def __str__(self) -> str:
        return (f"{_RUNTIME_ACTIVATION_ENV} must be 'standard' or '4o6', "
                f"got {self.value!r}")


def _adaptive_flag(name: str) -> bool:
    """Return the explicit opt-in value for one adaptive 4/6 feature flag."""
    return _os.environ.get(name, "0").strip().lower() in (
        "1", "true", "yes", "on")


def _runtime_activation_quantization(
    legacy_adaptive_flag: str,
) -> RuntimeActivationQuantization | None:
    """Resolve runtime activation quantization with legacy 4o6 compatibility."""
    value = _os.environ.get(_RUNTIME_ACTIVATION_ENV)
    normalized = value.strip().lower() if value is not None else None
    if normalized == "standard":
        return _RUNTIME_STANDARD
    if normalized == "4o6":
        return _RUNTIME_4O6
    if normalized is None:
        return _RUNTIME_4O6 if _adaptive_flag(legacy_adaptive_flag) else None
    raise RuntimeActivationModeError(normalized)


def _require_runtime_quantization_ops(*extra_ops: str) -> None:
    required_ops = ("fp4_quantize_fused", *extra_ops)
    missing = [name for name in required_ops
               if not hasattr(torch.ops.trtllm, name)]
    if missing:
        raise RuntimeError(
            "Runtime NVFP4 activation operators are missing from this "
            "TensorRT-LLM "
            "build: " + ", ".join(missing))


def _dequant_nvfp4_cutedsl(x_fp4: torch.Tensor, x_sf: torch.Tensor,
                            global_scale: torch.Tensor,
                            scaling_vector_size: int = 16) -> torch.Tensor:
    """Dequantize native SWIZZLED NVFP4 scale factors without a Python remap."""
    return torch.ops.trtllm.dequant_nvfp4_swizzled_sf(
        x_fp4.view(torch.uint8), x_sf.view(torch.uint8), global_scale,
        scaling_vector_size)


def _runtime_alpha_scalar(scalar: torch.Tensor) -> torch.Tensor:
    """Shape a runtime alpha operand the way the grouped-GEMM ABI wants it.

    The grouped GEMMs take ``alpha_numerator``/``alpha_denominator`` as
    single-element float32 tensors and fold them into the per-expert alpha in
    their epilogue.  Reshaping is metadata-only, so this adds no kernel, no
    synchronization, and stays CUDA-graph capturable.
    """
    return scalar.reshape(1)


def _runtime_nvfp4_quantize(
    input_bf16: torch.Tensor,
    quantization: RuntimeActivationQuantization,
    scaling_vector_size: int = 16,
    swizzled: bool = True,
    tile_idx_to_mn_limit: Optional[torch.Tensor] = None,
    num_non_exiting_tiles: Optional[torch.Tensor] = None,
    tile_size: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize the current BF16 activation directly to runtime-scaled NVFP4."""
    _require_runtime_quantization_ops()
    return torch.ops.trtllm.fp4_quantize_fused(
        input_bf16.contiguous(),
        scaling_vector_size,
        False,  # sfUseUE8M0
        swizzled,
        quantization.scale_rule,
        quantization.quant_range,
        1e-12,
        0,  # testMaxActiveBlocks
        0,  # forceV2
        tile_idx_to_mn_limit,
        num_non_exiting_tiles,
        tile_size,
    )


def _runtime_nvfp4_quantize_phase2(
    input_bf16: torch.Tensor,
    amax_scale: torch.Tensor,
    quantization: RuntimeActivationQuantization,
    scaling_vector_size: int = 16,
    swizzled: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Encode BF16 after its producer already published runtime amax/scale."""
    _require_runtime_quantization_ops("fp4_quantize_phase2")
    return torch.ops.trtllm.fp4_quantize_phase2(
        input_bf16.contiguous(),
        amax_scale,
        scaling_vector_size,
        swizzled,
        quantization.scale_rule,
    )


def _runtime_swiglu_nvfp4_quantize(
    preactivation: torch.Tensor,
    tile_idx_to_mn_limit: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    quantization: RuntimeActivationQuantization,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Apply SwiGLU once, preserving its BF16 output for SVDQuant FC2.

    SwiGLU is evaluated inside phase 1 of the persistent adaptive quantizer, so
    the pre-activation is read once and no standalone activation kernel runs.
    Phase 1 still stores the BF16 activation -- the amax reduction and phase 2
    both consume that stored tensor, which is also what the FC2 low-rank
    correction needs -- and routing padding stays excluded from the reduction.
    """
    _require_runtime_quantization_ops("fp4_swiglu_quantize_fused")
    output, output_sf, amax, output_bf16 = (
        torch.ops.trtllm.fp4_swiglu_quantize_fused(
            preactivation.contiguous(),
            16,
            True,  # isSfSwizzledLayout
            quantization.scale_rule,
            quantization.quant_range,
            1e-12,
            0,  # testMaxActiveBlocks
            tile_idx_to_mn_limit,
            num_non_exiting_tiles,
            tile_size,
        ))
    return output, output_sf, amax, output_bf16


@dataclass
class NvFp4WeightView:
    """Bundles all NVFP4 weight tensors for MoE computation.

    Under the VA-based DWDP pipeline ``param.data`` is swapped to a
    composite [num_experts, ...] tensor before the kernel call, so every
    field is a single tensor — the bundle is just a convenient grouping
    that lets the runner forward a single object instead of six.
    """
    w3_w1_weight: torch.Tensor
    fc1_weight_scale: torch.Tensor
    fc1_global_scale: torch.Tensor
    w2_weight: torch.Tensor
    fc2_weight_scale: torch.Tensor
    fc2_global_scale: torch.Tensor
    expert_size_per_partition: int
    slot_start: int


@torch.compile(options={"max-autotune": True})
def swiglu_fused_moe(x):
    x, gate = x.chunk(2, dim=-1)
    return F.silu(gate) * x


def _deinterleave_linear_and_gate(
    x: torch.Tensor,
    group_size: int = 64,
    dim: int = -1,
) -> torch.Tensor:
    """Undo the CuteDSL FC13 interleave before low-rank SwiGLU math."""
    normalized_dim = dim % x.dim()
    sizes = x.size()
    if sizes[normalized_dim] % (2 * group_size) != 0:
        raise svdquant_helpers.SvdquantLoadError(
            f"Cannot deinterleave dimension {sizes[normalized_dim]} with "
            f"group size {group_size}.")
    prefix = sizes[:normalized_dim]
    suffix = sizes[normalized_dim + 1:]
    grouped = x.view(*prefix, sizes[normalized_dim] // (2 * group_size),
                     2, group_size, *suffix)
    return grouped.transpose(normalized_dim,
                             normalized_dim + 1).contiguous().view(*sizes)


def _deinterleave_linear_and_gate_cutedsl(
    x: torch.Tensor,
    group_size: int = 64,
) -> torch.Tensor:
    """Run the FC13 block permutation without TensorIterator staging."""
    if x.dim() != 2:
        raise svdquant_helpers.SvdquantLoadError(
            f"CuTeDSL FC13 deinterleave expects a 2-D tensor; got "
            f"{tuple(x.shape)}.")
    if x.shape[1] % (2 * group_size) != 0:
        raise svdquant_helpers.SvdquantLoadError(
            f"Cannot deinterleave dimension {x.shape[1]} with group size "
            f"{group_size}.")
    return torch.ops.trtllm.cute_dsl_bf16_deinterleave_blackwell(
        x, group_size)


def _svdquant_packed_fc13_vh(module: torch.nn.Module) -> torch.Tensor | None:
    """Return the finalized ``[E, 2r, H]`` FC13 Vh Parameter, if it exists.

    ``NVFP4FusedMoEMethod._finalize_svdquant_params`` replaces the two FC13 Vh
    Parameters with this one; before finalization it is absent and the caller
    falls back to the per-projection storage.  The Parameter itself is returned
    -- never a cached view -- so a later ``module._apply`` (``.to(device)``, a
    dtype cast) is picked up on the next forward instead of leaving a stale
    tensor behind.
    """
    return getattr(module, SVDQUANT_FC13_PACKED_VH, None)


def _svdquant_packed_fc13_vh_half(
    packed: torch.Tensor,
    projection: str,
    us: torch.Tensor,
) -> torch.Tensor:
    """Slice one projection's Vh out of the packed FC13 factor.

    The packed layout is ``SVDQUANT_FC13_PACKED_ORDER`` back to back along the
    rank dimension, so the half a projection owns is fixed by its position in
    that tuple.  Every relationship the slice depends on -- expert count, rank,
    the pack holding exactly the two halves -- is checked against ``us`` here,
    because a silently wrong offset would produce a plausible correction rather
    than an error.
    """
    if projection not in SVDQUANT_FC13_PACKED_ORDER:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant {projection} has no half in the packed FC13 Vh; the "
            f"pack holds {SVDQUANT_FC13_PACKED_ORDER}.")
    if packed.dim() != 3:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant packed FC13 Vh must be 3-D; got {tuple(packed.shape)}.")
    if not packed.is_contiguous():
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant packed FC13 Vh must be contiguous; a strided pack "
            "interleaves the two halves instead of separating them.")
    if us.dim() != 3:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant {projection} US must be 3-D; got {tuple(us.shape)}.")
    if packed.shape[0] != us.shape[0]:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant packed FC13 Vh expert count {packed.shape[0]} does not "
            f"match {projection} US {us.shape[0]}.")
    rank = us.shape[2]
    expected_rank = len(SVDQUANT_FC13_PACKED_ORDER) * rank
    if packed.shape[1] != expected_rank:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant packed FC13 Vh rank {packed.shape[1]} does not hold "
            f"{len(SVDQUANT_FC13_PACKED_ORDER)} ranks of {rank}.")
    index = SVDQUANT_FC13_PACKED_ORDER.index(projection)
    return packed[:, index * rank:(index + 1) * rank]


def _svdquant_factor_pair(
    module: torch.nn.Module,
    projection: str,
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Return one complete factor pair and reject partial runtime state.

    After finalization the FC13 projections have no ``{projection}_vh``
    Parameter of their own -- both live in one packed buffer -- so their Vh is
    resolved to the matching view of that buffer.  The view is rebuilt on every
    call rather than cached, which keeps it in step with the Parameter across
    device and dtype moves.  Before finalization nothing is packed and the two
    independent Parameters are returned unchanged.
    """
    us = getattr(module, f"{projection}_us", None)
    vh = getattr(module, f"{projection}_vh", None)
    if vh is None and us is not None:
        packed = _svdquant_packed_fc13_vh(module)
        if packed is not None and projection in SVDQUANT_FC13_PACKED_ORDER:
            vh = _svdquant_packed_fc13_vh_half(packed, projection, us)
    if us is None and vh is None:
        return None
    if us is None or vh is None:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant {projection} requires US and Vh together.")
    return us, vh


def _permuted_svdquant_fc13_input(
    source_bf16: torch.Tensor | None,
    tile_idx_to_mn_limit: torch.Tensor,
    permuted_idx_to_expanded_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    top_k: int,
) -> torch.Tensor:
    """Build the expert-major BF16 input without requantizing local activations."""
    if source_bf16 is None:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant FC13 requires the original BF16 activation; "
            "FP4 dequantization is not a valid substitute.")
    permuted_bf16, permuted_sf = torch.ops.trtllm.moe_permute(
        source_bf16,
        None,
        tile_idx_to_mn_limit,
        permuted_idx_to_expanded_idx,
        num_non_exiting_tiles,
        tile_size,
        top_k,
    )
    if permuted_sf is not None:
        raise svdquant_helpers.SvdquantLoadError(
            "BF16 SVDQuant FC13 permutation unexpectedly returned scale factors.")
    return permuted_bf16


def _validate_grouped_lowrank_operands(
    x_bf16: torch.Tensor,
    us: torch.Tensor,
    vh: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_size: int,
) -> None:
    """Reject operand shapes the grouped low-rank ABI cannot express.

    Every check is shape- or dtype-only so the validation never reads device
    memory and stays legal inside a CUDA graph capture.
    """
    if us.dim() != 3 or vh.dim() != 3:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant grouped low-rank expects 3-D per-expert factors; got "
            f"US {tuple(us.shape)} and Vh {tuple(vh.shape)}.")
    if us.shape[0] != vh.shape[0]:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant US expert count {us.shape[0]} does not match Vh "
            f"{vh.shape[0]}.")
    if us.shape[2] != vh.shape[1]:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant US rank {us.shape[2]} does not match Vh rank "
            f"{vh.shape[1]}.")
    if x_bf16.dim() != 2:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant grouped low-rank expects a 2-D activation; got "
            f"{tuple(x_bf16.shape)}.")
    if x_bf16.shape[1] < vh.shape[2]:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant activation width {x_bf16.shape[1]} is narrower than the "
            f"Vh contraction {vh.shape[2]}.")
    if tile_size <= 0 or x_bf16.shape[0] % tile_size != 0:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant permuted rows {x_bf16.shape[0]} are not a multiple of "
            f"tile size {tile_size}.")
    num_tiles = x_bf16.shape[0] // tile_size
    if tile_idx_to_expert_idx.numel() != num_tiles:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant tile map holds {tile_idx_to_expert_idx.numel()} entries "
            f"for {num_tiles} permuted tiles.")
    if tile_idx_to_expert_idx.dtype != torch.int32:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant tile map must be int32; got "
            f"{tile_idx_to_expert_idx.dtype}.")
    if (us.dtype != torch.bfloat16 or vh.dtype != torch.bfloat16
            or x_bf16.dtype != torch.bfloat16):
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant grouped low-rank needs BF16 operands; got activation "
            f"{x_bf16.dtype}, US {us.dtype}, Vh {vh.dtype}.")


def _supports_grouped_lowrank(x_bf16: torch.Tensor) -> bool:
    """Return whether the grouped CuteDSL low-rank path can run here.

    The grouped GEMM is a Blackwell CuteDSL kernel, so anything else -- CPU
    tensors in unit tests, or an older GPU -- has to use the reference loop.
    """
    return x_bf16.is_cuda and is_sm_100f()


def _svdquant_grouped_lowrank_cutedsl(
    x_bf16: torch.Tensor,
    us: torch.Tensor,
    vh: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_size: int,
    num_non_exiting_tiles: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Evaluate ``(x @ Vh.T) @ US.T`` over the ``moe_sort`` permuted layout.

    ``tile_idx_to_expert_idx`` supplies the batch ("fake L") coordinate of the
    weight operand for every ``tile_size`` row block, so a single grouped GEMM
    covers every expert.  Both stages ride on one fused custom op: two kernels
    still launch, but the host pays one dispatch per factor pair instead of
    two.  Both grids follow from ``x_bf16.shape`` alone, which keeps the path
    legal under CUDA graph capture.

    Only the leading ``vh.shape[2]`` columns of the activation are contracted;
    the op takes that slice internally as a view, so a wider activation costs
    nothing.  ``M == 0`` yields an empty ``[0, us.shape[1]]`` result.

    ``num_non_exiting_tiles`` is an optional device-side tile count; when given
    both stages skip the trailing padded tiles instead of computing them.
    """
    _validate_grouped_lowrank_operands(x_bf16, us, vh, tile_idx_to_expert_idx,
                                       tile_size)
    if num_non_exiting_tiles is None:
        return torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_blackwell(
            x_bf16, us, vh, tile_idx_to_expert_idx, tile_size)
    return torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_blackwell(
        x_bf16, us, vh, tile_idx_to_expert_idx, tile_size,
        num_non_exiting_tiles=num_non_exiting_tiles)


def _validate_grouped_lowrank_destination(
    out: torch.Tensor,
    x_bf16: torch.Tensor,
    us: torch.Tensor,
) -> None:
    """Reject destinations the accumulating grouped ABI cannot write.

    Only the relationships the caller controls are checked here; the row stride
    and pointer alignment TMA needs belong to the op.  Every check is shape- or
    dtype-only, so this never reads device memory and stays legal inside a CUDA
    graph capture.
    """
    if out.dim() != 2:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant low-rank destination must be 2-D; got "
            f"{tuple(out.shape)}.")
    if tuple(out.shape) != (x_bf16.shape[0], us.shape[1]):
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant low-rank destination {tuple(out.shape)} does not match "
            f"the correction shape {(x_bf16.shape[0], us.shape[1])}.")
    if out.dtype != torch.bfloat16:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant low-rank destination must be BF16; got {out.dtype}.")
    if out.stride(1) != 1:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant low-rank destination must be contiguous along its last "
            f"dimension; got strides {tuple(out.stride())}.")


def _svdquant_grouped_lowrank_accumulate_cutedsl(
    x_bf16: torch.Tensor,
    us: torch.Tensor,
    vh: torch.Tensor,
    out: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_size: int,
    num_non_exiting_tiles: Optional[torch.Tensor] = None,
) -> None:
    """Add ``(x @ Vh.T) @ US.T`` onto ``out`` over the permuted layout.

    The same two grouped GEMMs as :func:`_svdquant_grouped_lowrank_cutedsl`,
    except the second stage's epilogue accumulates onto ``out`` instead of
    writing a fresh ``[M, N]`` tensor the caller then has to add.  That removes
    both the temporary and the elementwise add, which is where the SVDQuant path
    was spending several times the cost of the GEMMs themselves.

    ``out`` only has to be contiguous along N -- its row stride may exceed N --
    so one half of a wider destination is a valid target and costs no copy.  It
    must already hold the value being added to, so whatever produces that value
    has to be enqueued first.  Tiles at or past ``num_non_exiting_tiles`` are
    skipped, which leaves their rows of ``out`` exactly as they were rather than
    folding in an uninitialized correction.
    """
    # CUDA production is validated again by the custom op immediately below;
    # avoid paying the same Python shape/layout walk twice on every forward.
    # CPU tests replace the op with a host stub, so retain the wrapper contract
    # there to keep fallback and error-path coverage meaningful.
    if not x_bf16.is_cuda:
        _validate_grouped_lowrank_operands(
            x_bf16, us, vh, tile_idx_to_expert_idx, tile_size)
        _validate_grouped_lowrank_destination(out, x_bf16, us)
    if num_non_exiting_tiles is None:
        torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_accumulate_blackwell(
            x_bf16, us, vh, out, tile_idx_to_expert_idx, tile_size)
        return
    torch.ops.trtllm.cute_dsl_bf16_grouped_lowrank_accumulate_blackwell(
        x_bf16, us, vh, out, tile_idx_to_expert_idx, tile_size,
        num_non_exiting_tiles=num_non_exiting_tiles)


def _validate_grouped_dual_lowrank_operands(
    x_bf16: torch.Tensor,
    us_lo: torch.Tensor,
    us_hi: torch.Tensor,
    vh_packed: torch.Tensor,
    out: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_size: int,
) -> None:
    """Reject operands the packed dual low-rank ABI cannot express.

    Each half still has to satisfy everything one factor pair does, so those
    rules are delegated to the single-pair validators on views of the pack.
    What is left is specific to the packing: the two US factors have to agree,
    the pack has to hold their two ranks back to back, and ``out`` has to be the
    two destinations side by side in the same order.  Shape- and dtype-only, so
    nothing here reads device memory or breaks CUDA graph capture; the pointer
    alignment TMA needs still belongs to the op.
    """
    if us_lo.dim() != 3 or us_hi.dim() != 3:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant dual low-rank expects 3-D per-expert US factors; got "
            f"{tuple(us_lo.shape)} and {tuple(us_hi.shape)}.")
    if tuple(us_lo.shape) != tuple(us_hi.shape):
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant dual low-rank US factors {tuple(us_lo.shape)} and "
            f"{tuple(us_hi.shape)} must share a shape.")
    if vh_packed.dim() != 3:
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant dual low-rank packed Vh must be 3-D; got "
            f"{tuple(vh_packed.shape)}.")
    if not vh_packed.is_contiguous():
        raise svdquant_helpers.SvdquantLoadError(
            "SVDQuant dual low-rank packed Vh must be contiguous; splitting "
            "its rank dimension only yields the two factors when it is dense.")
    rank = us_lo.shape[2]
    if vh_packed.shape[1] != 2 * rank:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant dual low-rank packed Vh rank {vh_packed.shape[1]} does "
            f"not hold two ranks of {rank} back to back.")
    _validate_grouped_lowrank_operands(x_bf16, us_lo, vh_packed[:, :rank],
                                       tile_idx_to_expert_idx, tile_size)
    _validate_grouped_lowrank_operands(x_bf16, us_hi, vh_packed[:, rank:],
                                       tile_idx_to_expert_idx, tile_size)
    out_features = us_lo.shape[1]
    if out.dim() != 2:
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant dual low-rank destination must be 2-D; got "
            f"{tuple(out.shape)}.")
    if tuple(out.shape) != (x_bf16.shape[0], 2 * out_features):
        raise svdquant_helpers.SvdquantLoadError(
            f"SVDQuant dual low-rank destination {tuple(out.shape)} does not "
            f"match the paired correction shape "
            f"{(x_bf16.shape[0], 2 * out_features)}.")
    # The halves are column slices of one row, so the destination they share
    # has to satisfy the single-pair destination rules on each of them.
    _validate_grouped_lowrank_destination(out[:, :out_features], x_bf16, us_lo)
    _validate_grouped_lowrank_destination(out[:, out_features:], x_bf16, us_hi)


def _svdquant_grouped_dual_lowrank_accumulate_cutedsl(
    x_bf16: torch.Tensor,
    us_lo: torch.Tensor,
    us_hi: torch.Tensor,
    vh_packed: torch.Tensor,
    out: torch.Tensor,
    tile_idx_to_expert_idx: torch.Tensor,
    tile_size: int,
    num_non_exiting_tiles: Optional[torch.Tensor] = None,
) -> None:
    """Add both packed low-rank corrections onto the two halves of ``out``.

    With ``N = us_lo.shape[1]`` and ``r`` the shared rank::

        out[:, :N] += (x @ vh_packed[:r].T) @ us_lo.T
        out[:, N:]  += (x @ vh_packed[r:].T) @ us_hi.T

    Calling :func:`_svdquant_grouped_lowrank_accumulate_cutedsl` once per half
    costs two host dispatches, four kernel launches and two ``[M, r]`` scratch
    buffers.  Packing the two Vh factors along their rank makes the two down
    projections one grouped GEMM of ``N = 2 * r`` -- one launch, and a wider,
    better-shaped one, since a lone rank of 64 sits at the kernel's minimum
    tile.  ``out`` must already hold the value being added to.
    """
    # As above, the CUDA custom op owns the canonical device-side contract.
    # Keep wrapper validation only for CPU stubs/fallback tests rather than
    # duplicating it in the latency-critical production path.
    if not x_bf16.is_cuda:
        _validate_grouped_dual_lowrank_operands(
            x_bf16, us_lo, us_hi, vh_packed, out,
            tile_idx_to_expert_idx, tile_size)
    if num_non_exiting_tiles is None:
        torch.ops.trtllm.cute_dsl_bf16_grouped_dual_lowrank_accumulate_blackwell(
            x_bf16, us_lo, us_hi, vh_packed, out, tile_idx_to_expert_idx,
            tile_size)
        return
    torch.ops.trtllm.cute_dsl_bf16_grouped_dual_lowrank_accumulate_blackwell(
        x_bf16, us_lo, us_hi, vh_packed, out, tile_idx_to_expert_idx,
        tile_size, num_non_exiting_tiles=num_non_exiting_tiles)


def cute_dsl_fp8_group_blockwise_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    offset_array: torch.Tensor,
) -> torch.Tensor:
    m, k = a.shape[0], a.shape[1]
    l, n, k = b.shape[0], b.shape[1], b.shape[2]
    num_group, w_n, w_k = b_sf.shape[0], b_sf.shape[1], b_sf.shape[2]

    # Note: view(int8) will cause error.
    a_tmp = a.as_strided((m, k, 1), (k, 1, m * k))
    b_tmp = b.permute(1, 2, 0)

    # Note: we have different output scale shape for fp8_quantize_1x128, so we need to handle it differently for sm100 and other archs.
    if is_sm_100f():
        input_scale_tmp = a_sf.permute(1, 0).as_strided((m, w_k, 1),
                                                        (1, m, m * w_k))
    else:
        m_padded = (m + 3) // 4 * 4
        input_scale_tmp = a_sf[0:m_padded * w_k]
        input_scale_tmp = input_scale_tmp.reshape(-1, m_padded)
        input_scale_tmp = input_scale_tmp[:w_k, :m].contiguous().permute(1, 0)
        input_scale_tmp = input_scale_tmp.as_strided((m, w_k, 1),
                                                     (1, m, m * w_k))

    weight_scale_tmp = b_sf.permute(1, 2, 0)

    def pad_and_multiply(scale, tensor):
        cm, ck, _ = scale.shape
        m, k, _ = tensor.shape
        IsGroupWise = False
        IsBlockWise = False
        if ck == math.ceil(k / 128):
            IsGroupWise = True
        if cm == math.ceil(m / 128):
            IsBlockWise = True
        if not IsBlockWise and not IsGroupWise:
            raise ValueError("Only support granularity = 128")

        k_idx = torch.arange(k, device=scale.device)
        if IsGroupWise:
            k_idx = k_idx // 128
        m_idx = torch.arange(m, device=scale.device)
        if IsBlockWise:
            m_idx = m_idx // 128
        expanded_scale = scale[m_idx[:, None], k_idx, :]

        result = expanded_scale * tensor

        return result

    updated_a = pad_and_multiply(input_scale_tmp, a_tmp.to(torch.float32))
    updated_b = pad_and_multiply(weight_scale_tmp, b_tmp.to(torch.float32))

    ref = torch.zeros((m, n), device="cuda", dtype=torch.float32)

    len_offset_array = offset_array.shape[0]
    for i in range(len_offset_array - 1):
        start = offset_array[i]
        end = offset_array[i + 1]
        # assert start <= end, f"Invalid group boundaries: start={start} > end={end}"
        ref[start:end, :] = torch.einsum("mk,nk->mn", updated_a[start:end, :,
                                                                0],
                                         updated_b[:, :, i])
    ref = ref.to(torch.bfloat16)
    return ref


def cute_dsl_nvfp4_grouped_gemm_ref(
    a: torch.Tensor,
    b: torch.Tensor,
    a_sf: torch.Tensor,
    b_sf: torch.Tensor,
    alpha: torch.Tensor,
    tile_idx_to_group_idx: torch.Tensor,
    num_non_exiting_tiles: torch.Tensor,
    tile_size: int,
    output_dtype: torch.dtype,
    scaling_vector_size: int = 16,
):
    assert a.dtype == torch.float4_e2m1fn_x2
    assert a.dim() == 2
    assert b.dtype == torch.float4_e2m1fn_x2
    assert b.dim() == 3
    assert a_sf.dtype == torch.uint8
    assert a_sf.dim() == 1
    assert b_sf.dtype == torch.uint8
    assert b_sf.dim() == 3
    assert alpha.dtype == torch.float32
    assert alpha.dim() == 1

    m, k = a.size(0), a.size(1) * 2
    l, n = b.size(0), b.size(1)
    scale_k = k // scaling_vector_size
    assert m % tile_size == 0
    assert k % (scaling_vector_size * 4) == 0
    assert b.size(2) * 2 == k
    assert a_sf.size(0) == m * scale_k
    assert b_sf.size(0) == l
    assert b_sf.size(1) == n
    assert b_sf.size(2) == scale_k
    assert alpha.size(0) == l

    num_tiles = m // tile_size
    assert tile_idx_to_group_idx.dtype == torch.int32
    assert tile_idx_to_group_idx.size() == (num_tiles, )
    assert num_non_exiting_tiles.dtype == torch.int32
    assert num_non_exiting_tiles.size() == (1, )

    num_tiles_per_expert = torch.bincount(
        tile_idx_to_group_idx[:num_non_exiting_tiles[0].item()], minlength=l)
    offsets = [0] + num_tiles_per_expert.cumsum(dim=0).tolist()

    ref = torch.empty(m, n, dtype=output_dtype, device="cuda")
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        if end <= start:
            continue
        a_sliced = a[start * tile_size:end * tile_size]
        a_sf_sliced = a_sf[start * tile_size * k // scaling_vector_size:end *
                           tile_size * k // scaling_vector_size]
        ref[start * tile_size:end * tile_size] = torch.ops.trtllm.nvfp4_gemm(
            a_sliced.view(torch.uint8), b[i].view(torch.uint8), a_sf_sliced,
            b_sf[i], alpha[i], output_dtype)

    return ref


class CuteDslFusedMoENvfp4InputsHelper(GroupedGemmInputsHelper):

    def __init__(self, num_experts: int, top_k: int, num_local_experts: int,
                 local_expert_offset: int):
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset

    def infer_shape_num_tokens(self, input_shapes: List[torch.Size]) -> int:
        return input_shapes[0][0]

    def inputs_pre_hook(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        x, token_selected_experts, *others = inputs
        num_tokens = token_selected_experts.size(0)
        num_tokens_per_expert = self.generate_num_tokens_per_expert(
            num_tokens, approx_max_load=True)

        new_token_selected_experts = []
        for i, curr_num_tokens in enumerate(num_tokens_per_expert,
                                            start=self.local_expert_offset):
            new_token_selected_experts.extend([i] * curr_num_tokens)
        new_token_selected_experts = new_token_selected_experts + [-1] * (
            num_tokens * self.top_k - len(new_token_selected_experts))
        new_token_selected_experts = torch.tensor(
            new_token_selected_experts,
            dtype=token_selected_experts.dtype,
            device=token_selected_experts.device)
        new_token_selected_experts = new_token_selected_experts.view(
            self.top_k, num_tokens).transpose(0, 1).contiguous()
        return x, new_token_selected_experts, *others


class CuteDslFusedMoENvfp4Runner(TunableRunner):
    tuning_config_cache = dict()

    def __init__(self,
                 forward_impl: Callable,
                 num_experts: int,
                 top_k: int,
                 num_local_experts: int,
                 local_expert_offset: int,
                 enable_finalize_fusion: bool = True,
                 enable_alltoall: bool = False,
                 output_dtype: torch.dtype = torch.bfloat16,
                 scaling_vector_size: int = 16):
        super().__init__()
        self.forward_impl = forward_impl
        self.num_experts = num_experts
        self.top_k = top_k
        self.num_local_experts = num_local_experts
        self.local_expert_offset = local_expert_offset
        self.enable_finalize_fusion = enable_finalize_fusion
        self.enable_alltoall = enable_alltoall

        assert output_dtype == torch.bfloat16
        self.output_dtype = output_dtype
        self.scaling_vector_size = scaling_vector_size

    def unique_id(self):
        return (
            self.num_experts,
            self.top_k,
            self.num_local_experts,
            self.local_expert_offset,
            self.enable_finalize_fusion,
            self.enable_alltoall,
            self.output_dtype,
            self.scaling_vector_size,
        )

    def get_valid_tactics(
        self,
        inputs: List[torch.Tensor],
        profile: OptimizationProfile,
        **kwargs,
    ) -> List[int]:
        return [128, 256]

    def get_tuning_config(self) -> TuningConfig:
        key = self.unique_id()
        if key not in self.__class__.tuning_config_cache:
            helper = CuteDslFusedMoENvfp4InputsHelper(self.num_experts,
                                                      self.top_k,
                                                      self.num_local_experts,
                                                      self.local_expert_offset)
            self.__class__.tuning_config_cache[key] = TuningConfig(
                dynamic_tensor_specs=(DynamicTensorSpec(
                    0, 0, get_last_power_of_2_num_tokens_buckets,
                    last_positive_power_of_2), ),
                constraint_specs=(ConstraintSpec(1, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(2, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(3, 0,
                                                 helper.infer_shape_num_tokens),
                                  ConstraintSpec(
                                      4, 0, helper.infer_shape_num_tokens)),
                inputs_pre_hook=helper.inputs_pre_hook,
                use_cold_l2_cache=True,
            )
        return self.__class__.tuning_config_cache[key]

    def forward(self, inputs: List[torch.Tensor],
                tactic: Optional[int]) -> torch.Tensor:
        if isinstance(tactic, int) and tactic > 0:
            tile_size = tactic
        else:
            tile_size = 128
        return self.forward_impl(*inputs,
                                 enable_alltoall=self.enable_alltoall,
                                 tile_size=tile_size)

    @AutoTuner.TacticsCapture.register_runner_tactic_comb_checker
    @staticmethod
    def runner_tactic_comb_checker(
            comb: List[Tuple[TunableRunner, Any]]) -> bool:
        tile_size = None
        for runner, tactic in comb:
            if isinstance(runner, CuteDslFusedMoENvfp4Runner):
                tile_size = tactic
        if tile_size is None:
            return True

        for runner, tactic in comb:
            if isinstance(
                    runner,
                (Sm100BlockScaledContiguousGroupedGemmRunner,
                 Sm100BlockScaledContiguousGroupedGemmFinalizeFusionRunner,
                 Sm100BlockScaledContiguousGroupedGemmSwigluFusionRunner,
                 Sm100BlockScaledContiguousGatherGroupedGemmActFusionRunner)):
                mma_tiler_mn, *_ = tactic
                if mma_tiler_mn[0] != tile_size:
                    return False
        return True


class CuteDslFusedMoE(CutlassFusedMoE):
    # CuteDSL dispatch/combine path exercises the ceil/floor partition
    # (NVLinkOneSided alltoall with kernel-level remainder handling), so this
    # backend is the only opt-in for non-divisible EP today.
    _supports_non_divisible_ep: bool = True
    """CuteDSL flow of fused mixture of experts (MoE) Layer.

    Args:
        num_experts (int): Number of experts in the MoE layer.
        top_k (int): Number of top experts to select for each input token.
        hidden_size (int): Size of the hidden state.
        intermediate_size (int): Size of the intermediate state.
        aux_stream_dict (Optional[Dict[AuxStreamType, torch.cuda.Stream]]): Auxiliary CUDA streams for overlapping.
        dtype (Optional[torch.dtype]): Data type for the weights.
        reduce_results (bool): Whether to reduce the results across devices.
        model_config (ModelConfig): Configuration object for the model.
    """

    @classmethod
    def can_implement(
        cls,
        quant_algo: Optional[QuantAlgo],
        dtype_activation: torch.dtype = torch.bfloat16,
        swiglu_gptoss_style: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """
        Check if CuteDslFusedMoE can implement the given quantization algorithm.

        CuteDslFusedMoE supports:
        - NVFP4: SM in {100, 103}

        Does NOT support unquantized mode. Output dtype is hardcoded to bfloat16.
        Does NOT support swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit).

        Args:
            quant_algo: The quantization algorithm to check (None for unquantized)
            dtype_activation: The activation input data type. Only bfloat16 is supported
                because output dtype is hardcoded to bfloat16 (input/output dtype must match).
            swiglu_gptoss_style: Whether swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit) is enabled.
                CuteDslFusedMoE does NOT support swiglu_gptoss_style.

        Returns:
            Tuple[bool, Optional[str]]: (can_implement, skip_reason)
        """
        from .interface import _warn_and_return

        sm_version = get_sm_version()

        # CuteDslFusedMoE requires at least SM90
        if sm_version < 90:
            return _warn_and_return(
                f"CuteDslFusedMoE requires SM >= 90, got SM{sm_version}")

        # Check dtype_activation: output is hardcoded to bfloat16, so input must also be bfloat16
        # to maintain input/output dtype consistency
        if dtype_activation != torch.bfloat16:
            return _warn_and_return(
                f"CuteDslFusedMoE only supports bfloat16 activation (output is hardcoded to bfloat16), "
                f"got {dtype_activation}")

        # CuteDslFusedMoE does NOT support unquantized mode
        if quant_algo is None:
            return _warn_and_return(
                "CuteDslFusedMoE does not support unquantized mode")

        # CuteDslFusedMoE does NOT support swiglu_gptoss_style
        if swiglu_gptoss_style:
            return _warn_and_return(
                "CuteDslFusedMoE does not support swiglu_gptoss_style (bias/swiglu with custom alpha/beta/limit)"
            )

        # NVFP4 - SM in {100, 103}
        if quant_algo == QuantAlgo.NVFP4:
            if sm_version not in {100, 103}:
                return _warn_and_return(
                    f"NVFP4 requires SM100 or SM103, got SM{sm_version}")
            return True, None

        return _warn_and_return(
            f"CuteDslFusedMoE does not support quant_algo={quant_algo}")

    def validate_configurable_moe(self, moe: torch.nn.Module) -> None:
        """Reject dynamic expert remapping after wrapper state is available."""
        super().validate_configurable_moe(moe)
        if not svdquant_helpers.load_config().any_stage:
            return
        if moe.enable_dwdp:
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant does not support ConfigurableMoE VA-DWDP.")
        if moe._using_load_balancer():
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant does not support online EPLB/shared expert remapping.")

    def __init__(
        self,
        *,
        routing_method: BaseMoeRoutingMethod,
        num_experts: int,
        hidden_size: int,
        intermediate_size: int,
        dtype: Optional[torch.dtype] = None,
        reduce_results: bool = False,
        model_config: ModelConfig = ModelConfig(),
        aux_stream_dict: Optional[Dict[AuxStreamType,
                                       torch.cuda.Stream]] = None,
        weight_loading_mode: MoEWeightLoadingMode = MoEWeightLoadingMode.
        VANILLA,
        apply_router_weight_on_input: bool = False,
        layer_idx: Optional[int] = None,
        init_load_balancer: bool = True,
        without_comm: bool = False,
        activation_type: ActivationType = ActivationType.Swiglu,
    ):
        super().__init__(
            routing_method=routing_method,
            num_experts=num_experts,
            hidden_size=hidden_size,
            intermediate_size=intermediate_size,
            dtype=dtype,
            reduce_results=reduce_results,
            model_config=model_config,
            aux_stream_dict=aux_stream_dict,
            weight_loading_mode=weight_loading_mode,
            apply_router_weight_on_input=apply_router_weight_on_input,
            layer_idx=layer_idx,
            init_load_balancer=init_load_balancer,
            without_comm=without_comm,
            activation_type=activation_type,
        )

        if self.aux_stream_dict is None:
            self.aux_stream_dict = aux_stream_dict if aux_stream_dict is not None else {}
        if AuxStreamType.MoeOutputMemset not in self.aux_stream_dict:
            self.aux_stream_dict[
                AuxStreamType.MoeOutputMemset] = torch.cuda.Stream()
        if self.event_dict is None:
            self.event_dict = {}
        for key in [EventType.Main, EventType.MoeOutputMemset]:
            if key not in self.event_dict:
                self.event_dict[key] = torch.cuda.Event()

    def _build_local_weight_view(self) -> NvFp4WeightView:
        """Build the weight view from this backend's per-layer weights."""
        return NvFp4WeightView(
            w3_w1_weight=self.w3_w1_weight,
            fc1_weight_scale=self.quant_scales.fc1_weight_block,
            fc1_global_scale=self.quant_scales.fc1_global,
            w2_weight=self.w2_weight,
            fc2_weight_scale=self.quant_scales.fc2_weight_block,
            fc2_global_scale=self.quant_scales.fc2_global,
            expert_size_per_partition=self.expert_size_per_partition,
            slot_start=self.slot_start,
        )

    def select_alltoall_method_type(self) -> AlltoallMethodType:
        return AlltoallMethodType.NotEnabled

    def _get_quant_method(self):
        if self.quant_config is not None and self.quant_config.layer_quant_mode.has_any_quant(
                exclude_kv_cache=True):
            if self.quant_config.layer_quant_mode.has_nvfp4():
                return NVFP4CuteDslFusedMoEMethod()
        return super()._get_quant_method()

    def supports_moe_output_in_alltoall_workspace(self):
        return self.has_nvfp4

    def quantize_input(self,
                       x: Union[torch.Tensor, Fp4QuantizedTensor],
                       post_quant_comm: bool = True,
                       return_bf16_input: bool = False):
        """Quantize inputs prior to post-communication (alltoall/allgather) or before MoE computation.

        Args:
            x: Input tensor to quantize
            post_quant_comm:
                If True, quantize for post-quant communication path.
                If False, quantize for non-communication path
            return_bf16_input:
                Also return the prepared contiguous BF16 activation used to
                produce the NVFP4 tensor. Required by SVDQuant FC13.

        Returns: ``(x, x_sf)`` or ``(x, x_sf, source_bf16)`` when
        ``return_bf16_input`` is true. ``x_sf`` is reshaped to 2D if needed.

        For quantization methods that produce scaling factors:
        - x_sf is reshaped from 1D to 2D: [num_elements] -> [batch_size, ceil_div(hidden_size, scaling_vector_size)]
        - The 2D shape is required for proper handling in alltoall/allgather operations
        - scaling_vector_size is typically the group size for block-wise quantization
        """
        x_sf = None
        runtime_global_scale = None
        source_bf16 = None
        if self.has_nvfp4:
            if isinstance(x, Fp4QuantizedTensor):
                if return_bf16_input:
                    raise svdquant_helpers.SvdquantLoadError(
                        "SVDQuant FC13 cannot preserve BF16 from a pre-quantized input.")
                assert not x.is_sf_swizzled, "Fp4QuantizedTensor should not be swizzled before communication"
                x_row = x.shape[0]
                x, x_sf = x.fp4_tensor, x.scaling_factor
            else:
                x_row = x.shape[0]

                runtime_fc13_quantization = _runtime_activation_quantization(
                    "TRTLLM_ADAPTIVE_FP4")
                if runtime_fc13_quantization is not None:
                    with nvtx_range_debug(
                            "[CUTEDSL][NVFP4] input_quantize.runtime"):
                        # Preserve the legacy feature-off path below. Explicit
                        # runtime modes share this scale and padding preparation.
                        if hasattr(
                                self, 'fc31_act_scale'
                        ) and self.fc31_act_scale is not None:
                            x = x * self.fc31_act_scale

                        pad_size = self.w3_w1_weight.shape[-1] * 2 - x.shape[-1]
                        if pad_size > 0:
                            x = torch.nn.functional.pad(x, (0, pad_size))

                        x_contig = x.contiguous()
                        source_bf16 = x_contig
                        x, x_sf, amax_buf = _runtime_nvfp4_quantize(
                            x_contig,
                            runtime_fc13_quantization,
                            scaling_vector_size=self.scaling_vector_size,
                            swizzled=False,
                        )
                        # ``amax_buf`` is ``[amax, quantRange / amax]``;
                        # the slice keeps the shape the GEMM ABI expects
                        # without copying or synchronizing.
                        runtime_global_scale = amax_buf[1:2]
                else:
                    with nvtx_range_debug(
                            "[CUTEDSL][NVFP4] input_quantize.native"):
                        source_bf16 = x.contiguous(
                        ) if return_bf16_input else x
                        x, x_sf = torch.ops.trtllm.fp4_quantize(
                            source_bf16, self.fc31_input_scale,
                            self.scaling_vector_size, False, False)
        elif self.has_deepseek_fp8_block_scales:
            # FP8 block scales doesn't support permutation of quantized inputs.
            # WAR: The quantization is in run_moe_fp8_block_scales.
            pass
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )

        # The checkpoint scale is only an algebraic loader placeholder here;
        # the measured activation encoding uses this invocation's global scale.
        self._runtime_fc13_global_scale = runtime_global_scale

        if x_sf is not None:
            x_sf = x_sf.view(x_row, -1)
        if return_bf16_input:
            return x, x_sf, source_bf16
        return x, x_sf

    def has_svdquant_fc13(self) -> bool:
        """Return whether this backend needs the original BF16 FC13 activation."""
        return _svdquant_factor_pair(self, "w1") is not None

    def run_moe_nvfp4(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        weight_view: Optional[NvFp4WeightView] = None,
        fc13_input_bf16: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """NVFP4 MoE computation.

        Uses the single-tensor ``run_moe_nvfp4_impl`` path. (The former
        multi-B DWDP path was removed once DWDP switched to VA: VA swaps
        ``param.data`` to a full [num_experts, ...] tensor, so the single-
        tensor kernel is sufficient.)

        Args:
            weight_view: Bundled weight tensors. Must not be None.
        """
        assert self.has_nvfp4
        assert weight_view is not None
        output_dtype = torch.bfloat16

        if moe_output is None:
            moe_output = torch.empty(
                (token_final_scales.size(0), self.hidden_size),
                dtype=output_dtype,
                device=x.device)
        else:
            assert moe_output.size() == (token_final_scales.size(0),
                                         self.hidden_size)
            assert moe_output.dtype == output_dtype

        effective_top_k = token_selected_experts.size(-1)

        forward_impl = self.run_moe_nvfp4_impl
        w1_factors = _svdquant_factor_pair(self, "w1")
        w3_factors = _svdquant_factor_pair(self, "w3")
        w2_factors = _svdquant_factor_pair(self, "w2")
        if (w1_factors is None) != (w3_factors is None):
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant FC13 requires complete w1 and w3 factor pairs.")
        has_svdquant = w1_factors is not None or w2_factors is not None
        if has_svdquant:
            factor_experts = (w1_factors[0].shape[0]
                              if w1_factors is not None else
                              w2_factors[0].shape[0])
            if (weight_view.w3_w1_weight.shape[0] != factor_experts
                    or weight_view.expert_size_per_partition != factor_experts):
                raise svdquant_helpers.SvdquantLoadError(
                    "SVDQuant factors cannot follow VA-DWDP or remapped "
                    "expert storage.")
        # Runtime FC2-input quantization materializes BF16 and FP4
        # outputs, whose allocations break the autotuner's offset tracking
        # ("Offset increment outside graph capture"). SVDQuant's Python
        # low-rank branch has the same constraint. Bypass autotuning and use a
        # fixed tile size whenever either is active.
        runtime_fc2_quantization = _runtime_activation_quantization(
            "TRTLLM_ADAPTIVE_FP4_FC2")
        if runtime_fc2_quantization is not None or has_svdquant:
            return self.run_moe_nvfp4_impl(
                x, token_selected_experts, token_final_scales, x_sf,
                moe_output, weight_view, fc13_input_bf16=fc13_input_bf16,
                enable_alltoall=enable_alltoall,
                tile_size=128)

        tuner = AutoTuner.get()
        runner = CuteDslFusedMoENvfp4Runner(
            forward_impl=forward_impl,
            num_experts=self.num_slots,
            top_k=effective_top_k,
            num_local_experts=weight_view.expert_size_per_partition,
            local_expert_offset=weight_view.slot_start,
            enable_finalize_fusion=self.use_fused_finalize,
            enable_alltoall=enable_alltoall,
        )

        inputs = [
            x,
            token_selected_experts,
            token_final_scales,
            x_sf,
            moe_output,
            weight_view,
        ]
        _, best_tactic = tuner.choose_one(
            "CuteDslFusedMoE::run_moe_nvfp4",
            [runner],
            runner.get_tuning_config(),
            inputs,
        )
        return runner(inputs, tactic=best_tactic)

    def _compute_svdquant_lr_permuted(
        self,
        x_bf16: torch.Tensor,
        us: torch.Tensor,
        vh: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        tile_size: int,
        slot_start: int,
        num_local_experts: int,
        num_non_exiting_tiles: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Evaluate one factor pair in the ``moe_sort`` permuted layout.

        On SM 100 this is two grouped CuteDSL GEMMs whose launch geometry
        depends only on the padded row count, so the call is graph-capturable
        and free of host readback.  Everywhere else it falls back to the
        per-tile reference loop, which is what the CPU unit tests exercise.
        """
        # ``moe_sort`` applies ``local_expert_offset`` while routing, but its
        # ``tile_idx_to_expert_idx`` output is a rank-local slot in
        # ``[0, num_local_experts)``.  Do not subtract ``slot_start`` again:
        # doing so skips every low-rank correction on nonzero EP ranks.
        del slot_start
        if _supports_grouped_lowrank(x_bf16):
            return _svdquant_grouped_lowrank_cutedsl(
                x_bf16, us, vh, tile_idx_to_expert_idx, tile_size,
                num_non_exiting_tiles)
        return self._compute_svdquant_lr_permuted_ref(
            x_bf16, us, vh, tile_idx_to_expert_idx, tile_idx_to_mn_limit,
            tile_size, num_local_experts)

    def _accumulate_svdquant_lr_permuted(
        self,
        x_bf16: torch.Tensor,
        us: torch.Tensor,
        vh: torch.Tensor,
        out: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        tile_size: int,
        slot_start: int,
        num_local_experts: int,
        num_non_exiting_tiles: Optional[torch.Tensor] = None,
    ) -> None:
        """Add one factor pair's correction onto ``out`` in place.

        This is the production form of :meth:`_compute_svdquant_lr_permuted`.
        On SM 100 the second grouped GEMM accumulates onto ``out`` from its own
        epilogue, so no ``[M, N]`` temporary is allocated and no elementwise add
        is dispatched; ``out`` may be a column slice of a wider tensor, which is
        what lets the two FC13 corrections land straight on the two halves of
        the pre-activation.  Everywhere else the per-tile reference loop runs
        and its result is added on the host side, which is what keeps the CPU
        unit tests meaningful.

        ``out`` must already hold the value being added to, so its producer has
        to be enqueued first.
        """
        if _supports_grouped_lowrank(x_bf16):
            _svdquant_grouped_lowrank_accumulate_cutedsl(
                x_bf16, us, vh, out, tile_idx_to_expert_idx, tile_size,
                num_non_exiting_tiles)
            return
        # Delegating keeps the rank-local slot rationale in one place; on this
        # branch it resolves to the reference loop.
        out += self._compute_svdquant_lr_permuted(
            x_bf16, us, vh, tile_idx_to_expert_idx, tile_idx_to_mn_limit,
            tile_size, slot_start, num_local_experts, num_non_exiting_tiles)

    def _accumulate_svdquant_fc13_lr_permuted(
        self,
        x_bf16: torch.Tensor,
        w3_factors: tuple[torch.Tensor, torch.Tensor],
        w1_factors: tuple[torch.Tensor, torch.Tensor],
        out: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        tile_size: int,
        slot_start: int,
        num_local_experts: int,
        num_non_exiting_tiles: Optional[torch.Tensor] = None,
    ) -> None:
        """Add both FC13 corrections onto the deinterleaved pre-activation.

        ``out`` is the whole ``[M, 2 * N]`` FC13 pre-activation in
        linear-then-gate order, which is exactly the order the packed Vh stores
        its halves in, so the linear (w3) correction lands on the low half and
        the gate (w1) correction on the high half.

        Where the packed factor and the Blackwell kernel are both available the
        pair goes out as one dispatch: the two down projections share a single
        grouped GEMM over the packed Vh, and each up projection accumulates
        straight onto its half from its own epilogue.  Everywhere else --
        CPU unit tests, pre-Blackwell devices, or a module whose factors were
        never packed -- the two corrections run as independent single-pair
        accumulations onto the two column slices, which is the behaviour this
        replaced and keeps the CPU tests meaningful.

        ``out`` must already hold the value being added to, so the main FC13
        GEMM and its deinterleave have to be enqueued first.
        """
        packed_vh = _svdquant_packed_fc13_vh(self)
        if packed_vh is not None and _supports_grouped_lowrank(x_bf16):
            _svdquant_grouped_dual_lowrank_accumulate_cutedsl(
                x_bf16, w3_factors[0], w1_factors[0], packed_vh, out,
                tile_idx_to_expert_idx, tile_size, num_non_exiting_tiles)
            return
        half_features = out.shape[1] // 2
        for factors, destination in (
            (w3_factors, out[:, :half_features]),
            (w1_factors, out[:, half_features:]),
        ):
            self._accumulate_svdquant_lr_permuted(
                x_bf16, factors[0], factors[1], destination,
                tile_idx_to_expert_idx, tile_idx_to_mn_limit, tile_size,
                slot_start, num_local_experts, num_non_exiting_tiles)

    @staticmethod
    def _compute_svdquant_lr_permuted_ref(
        x_bf16: torch.Tensor,
        us: torch.Tensor,
        vh: torch.Tensor,
        tile_idx_to_expert_idx: torch.Tensor,
        tile_idx_to_mn_limit: torch.Tensor,
        tile_size: int,
        num_local_experts: int,
    ) -> torch.Tensor:
        """Per-tile reference for platforms without the grouped CuteDSL GEMM.

        Reads the routing tensors on the host, so it synchronizes and cannot be
        captured in a CUDA graph.  The grouped path above is the production one.
        """
        result = torch.zeros(
            (x_bf16.shape[0], us.shape[1]),
            dtype=torch.bfloat16,
            device=x_bf16.device,
        )
        expert_ids = tile_idx_to_expert_idx.detach().to(
            device="cpu", dtype=torch.int64).tolist()
        limits = tile_idx_to_mn_limit.detach().to(
            device="cpu", dtype=torch.int64).tolist()
        for tile_idx, (expert_id, limit) in enumerate(
                zip(expert_ids, limits)):
            start = tile_idx * tile_size
            end = min(limit, start + tile_size, x_bf16.shape[0])
            local_slot = expert_id
            if end <= start or local_slot < 0 or local_slot >= num_local_experts:
                continue
            result[start:end] = svdquant_helpers.lowrank_gemm(
                x_bf16[start:end], us[local_slot], vh[local_slot])
        return result

    def run_moe_nvfp4_impl(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: torch.Tensor,
        moe_output: torch.Tensor,
        weight_view: NvFp4WeightView,
        fc13_input_bf16: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        tile_size: int = 128,
    ) -> torch.Tensor:
        """Non-DWDP NVFP4 MoE implementation using single-tensor ops."""
        output_dtype = torch.bfloat16
        effective_top_k = token_selected_experts.size(1)
        esp = weight_view.expert_size_per_partition
        slot_start = weight_view.slot_start
        w1_factors = _svdquant_factor_pair(self, "w1")
        w3_factors = _svdquant_factor_pair(self, "w3")
        w2_factors = _svdquant_factor_pair(self, "w2")
        if (w1_factors is None) != (w3_factors is None):
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant FC13 requires complete w1 and w3 factor pairs.")
        fc13_svdquant_active = w1_factors is not None
        fc2_svdquant_active = w2_factors is not None
        if fc13_svdquant_active and fc13_input_bf16 is None:
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant FC13 requires the original BF16 activation; "
                "FP4 dequantization is not a valid substitute.")
        if (fc13_svdquant_active
                and self.activation_type != int(ActivationType.Swiglu)):
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant FC13 supports only the SwiGLU activation.")
        use_fused_finalize = self.use_fused_finalize and not fc2_svdquant_active
        runtime_fc2_quantization = _runtime_activation_quantization(
            "TRTLLM_ADAPTIVE_FP4_FC2")
        runtime_fc2_amax = None
        fc2_input_bf16 = None

        with nvtx_range_debug("[CUTEDSL][NVFP4] moe_sort"):
            tile_idx_to_expert_idx, tile_idx_to_mn_limit, expanded_idx_to_permuted_idx, permuted_idx_to_expanded_idx, total_num_padded_tokens, num_non_exiting_tiles = torch.ops.trtllm.moe_sort(
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                num_experts=self.num_slots,
                top_k=effective_top_k,
                local_expert_offset=slot_start,
                local_num_experts=esp,
                tile_tokens_dim=tile_size,
            )

        if use_fused_finalize:
            self.event_dict[EventType.Main].record()
            moe_output.record_stream(
                self.aux_stream_dict[AuxStreamType.MoeOutputMemset])

        # Fused gather + GEMM + activation + quantize for FC1.
        # For gated (SwiGLU): weights are interleaved [up, gate], output is N/2.
        # For non-gated (Relu2): weights are plain, output is N.
        # --- FC13 runtime-global-scale alpha correction ---
        # The checkpoint's static per-expert alpha was baked with
        # ``fc31_input_scale``.  When the FC13 input was quantized at runtime
        # the encoding used a different global scale, so the corrected alpha is
        #   fc1_global_scale * fc31_input_scale / dynamic_global_scale
        # Both scalars travel with the GEMM and are folded in its epilogue, so
        # no standalone Div/Mul kernel runs here.  They stay ``None`` when the
        # feature is off, which reproduces the legacy alpha exactly.
        fc1_alpha = weight_view.fc1_global_scale
        fc1_alpha_numerator = None
        fc1_alpha_denominator = None
        runtime_fc13_global_scale = getattr(
            self, '_runtime_fc13_global_scale', None)
        if runtime_fc13_global_scale is not None:
            fc1_alpha_numerator = _runtime_alpha_scalar(self.fc31_input_scale)
            fc1_alpha_denominator = _runtime_alpha_scalar(
                runtime_fc13_global_scale)

        if fc13_svdquant_active:
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.permute"):
                x_permuted, x_sf_permuted = torch.ops.trtllm.moe_permute(
                    x.view(torch.float4_e2m1fn_x2),
                    x_sf.view(torch.uint8),
                    tile_idx_to_mn_limit,
                    permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles,
                    tile_size,
                    effective_top_k,
                )
                if x_sf_permuted is None:
                    raise svdquant_helpers.SvdquantLoadError(
                        "NVFP4 SVDQuant FC13 permutation did not return "
                        "scale factors.")
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.gemm"):
                fc13_preact = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                    input=x_permuted.view(torch.float4_e2m1fn_x2),
                    weight=weight_view.w3_w1_weight.view(
                        torch.float4_e2m1fn_x2),
                    input_scale=x_sf_permuted.view(torch.uint8),
                    weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
                    alpha=fc1_alpha,
                    alpha_numerator=fc1_alpha_numerator,
                    alpha_denominator=fc1_alpha_denominator,
                    tile_idx_to_group_idx=tile_idx_to_expert_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    num_experts=self.num_slots,
                    top_k=effective_top_k,
                    num_local_experts=esp,
                    local_expert_offset=slot_start,
                    tile_size=tile_size,
                    output_dtype=output_dtype,
                )
            if not getattr(self, SVDQUANT_FC13_SEPARATED_WEIGHT_LAYOUT, False):
                with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.deinterleave"):
                    fc13_preact = _deinterleave_linear_and_gate_cutedsl(
                        fc13_preact)
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.bf16_permute"):
                fc13_input_bf16 = _permuted_svdquant_fc13_input(
                    fc13_input_bf16,
                    tile_idx_to_mn_limit,
                    permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles,
                    tile_size,
                    effective_top_k,
                )
            # ``_deinterleave_linear_and_gate`` above put the linear half first
            # and the gate half second, and each low-rank output is exactly one
            # half wide, so the corrections land straight on the two halves.
            # Each one is accumulated inside its own second-stage epilogue, so
            # neither a [M, half] temporary nor an elementwise add is needed --
            # and with the packed Vh both go out as one dispatch.
            half_features = fc13_preact.shape[1] // 2
            assert w3_factors[0].shape[1] == half_features, (
                f"SVDQuant w3 low-rank width {w3_factors[0].shape[1]} must "
                f"match the FC13 half width {half_features}.")
            assert w1_factors[0].shape[1] == half_features, (
                f"SVDQuant w1 low-rank width {w1_factors[0].shape[1]} must "
                f"match the FC13 half width {half_features}.")
            # The gate half starts ``half_features`` elements into each row, and
            # TMA needs that offset 16B aligned; BF16 gives 8 elements per 16B.
            assert half_features % 8 == 0, (
                f"SVDQuant FC13 half width {half_features} must be a multiple "
                f"of 8 for the accumulating low-rank epilogue.")
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.svdq_lowrank"):
                self._accumulate_svdquant_fc13_lr_permuted(
                    fc13_input_bf16, w3_factors, w1_factors, fc13_preact,
                    tile_idx_to_expert_idx, tile_idx_to_mn_limit, tile_size,
                    slot_start, esp, num_non_exiting_tiles)
            with nvtx_range_debug(
                    "[CUTEDSL][NVFP4] fc13.activation_quantize"):
                if runtime_fc2_quantization is not None:
                    x, x_sf, runtime_fc2_amax, fc2_input_bf16 = (
                        _runtime_swiglu_nvfp4_quantize(
                            fc13_preact,
                            tile_idx_to_mn_limit,
                            num_non_exiting_tiles,
                            tile_size,
                            runtime_fc2_quantization,
                        ))
                else:
                    x, x_sf = torch.ops.trtllm.moe_swiglu_nvfp4_quantize(
                        fc13_preact,
                        self.fc2_input_scale,
                        tile_idx_to_mn_limit,
                        num_non_exiting_tiles,
                        tile_size,
                    )
        else:
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc13.fused"):
                fc13_inputs = dict(
                    input=x.view(torch.float4_e2m1fn_x2),
                    weight=weight_view.w3_w1_weight.view(
                        torch.float4_e2m1fn_x2),
                    input_scale=x_sf.view(torch.uint8),
                    weight_scale=weight_view.fc1_weight_scale.view(torch.uint8),
                    alpha=fc1_alpha,
                    alpha_numerator=fc1_alpha_numerator,
                    alpha_denominator=fc1_alpha_denominator,
                    tile_idx_to_group_idx=tile_idx_to_expert_idx,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    global_sf=self.fc2_input_scale,
                    num_experts=self.num_slots,
                    top_k=effective_top_k,
                    num_local_experts=esp,
                    local_expert_offset=slot_start,
                    tile_size=tile_size,
                    activation_type=self.activation_type,
                )
                if runtime_fc2_quantization is not None:
                    fc2_input_bf16, runtime_fc2_amax = (
                        torch.ops.trtllm
                        .cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_amax_blackwell(
                            **fc13_inputs,
                            quant_range=runtime_fc2_quantization.quant_range,
                        ))
                    with nvtx_range_debug(
                            "[CUTEDSL][NVFP4] fc13.phase2_quantize"):
                        x, x_sf = _runtime_nvfp4_quantize_phase2(
                            fc2_input_bf16,
                            runtime_fc2_amax,
                            runtime_fc2_quantization,
                        )
                else:
                    x, x_sf = (
                        torch.ops.trtllm
                        .cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell(
                            **fc13_inputs))

        # Native NVFP4 still needs BF16 for the optional FC2 low-rank branch.
        # Runtime activation quantization already preserved the exact FC13
        # SwiGLU output above.
        if fc2_svdquant_active and fc2_input_bf16 is None:
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc2.dequantize"):
                fc2_input_bf16 = _dequant_nvfp4_cutedsl(
                    x, x_sf, self.fc2_input_scale)

        # Correct FC2 alpha for the dynamic global scale selected while FC13
        # produced its SwiGLU activation.  No standard-NVFP4 decode is
        # involved, and the correction rides along with the GEMM rather than
        # running as a standalone Div/Mul pair.
        fc2_alpha = weight_view.fc2_global_scale
        fc2_alpha_numerator = None
        fc2_alpha_denominator = None
        if runtime_fc2_quantization is not None:
            assert runtime_fc2_amax is not None
            fc2_alpha_numerator = _runtime_alpha_scalar(self.fc2_input_scale)
            fc2_alpha_denominator = _runtime_alpha_scalar(runtime_fc2_amax[1:2])

        # The FC2 correction is applied after the FC2 GEMM below, not here: it
        # accumulates onto that GEMM's own output, so the base value has to be
        # in place first.  Both are stream-ordered on the same stream and never
        # overlapped, so the reordering costs nothing.  ``use_fused_finalize``
        # is already forced off whenever FC2 SVDQuant is active, so the fused
        # finalize branch never has a correction to apply.
        assert not (use_fused_finalize and fc2_svdquant_active), (
            "SVDQuant FC2 cannot run with the fused finalize epilogue.")

        if use_fused_finalize:
            with torch.cuda.stream(
                    self.aux_stream_dict[AuxStreamType.MoeOutputMemset]):
                self.event_dict[EventType.Main].wait()
                with nvtx_range_debug(
                        "[CUTEDSL][NVFP4] output_memset"):
                    torch.ops.trtllm.moe_output_memset_inplace(
                        input=moe_output,
                        tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                        expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                        permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                        num_non_exiting_tiles=num_non_exiting_tiles,
                        tile_tokens_dim=tile_size,
                        top_k=effective_top_k,
                        ep_size=self.mapping.moe_ep_size,
                        enable_alltoall=enable_alltoall,
                    )
                self.event_dict[EventType.MoeOutputMemset].record()
            self.event_dict[EventType.MoeOutputMemset].wait()

            with nvtx_range_debug("[CUTEDSL][NVFP4] fc2.gemm_finalize"):
                torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell(
                    input=x.view(torch.float4_e2m1fn_x2),
                    weight=weight_view.w2_weight.view(
                        torch.float4_e2m1fn_x2),
                    input_scale=x_sf.view(torch.uint8),
                    weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
                    alpha=fc2_alpha,
                    alpha_numerator=fc2_alpha_numerator,
                    alpha_denominator=fc2_alpha_denominator,
                    output=moe_output,
                    tile_idx_to_group_idx=tile_idx_to_expert_idx,
                    tile_idx_to_mn_limit=tile_idx_to_mn_limit,
                    permuted_idx_to_expanded_idx=permuted_idx_to_expanded_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    token_final_scales=token_final_scales,
                    num_experts=self.num_slots,
                    top_k=effective_top_k,
                    num_local_experts=esp,
                    local_expert_offset=slot_start,
                    tile_size=tile_size,
                    output_dtype=output_dtype,
                )
        else:
            with nvtx_range_debug("[CUTEDSL][NVFP4] fc2.gemm"):
                x = torch.ops.trtllm.cute_dsl_nvfp4_grouped_gemm_blackwell(
                    input=x.view(torch.float4_e2m1fn_x2),
                    weight=weight_view.w2_weight.view(
                        torch.float4_e2m1fn_x2),
                    input_scale=x_sf.view(torch.uint8),
                    weight_scale=weight_view.fc2_weight_scale.view(torch.uint8),
                    alpha=fc2_alpha,
                    alpha_numerator=fc2_alpha_numerator,
                    alpha_denominator=fc2_alpha_denominator,
                    tile_idx_to_group_idx=tile_idx_to_expert_idx,
                    num_non_exiting_tiles=num_non_exiting_tiles,
                    num_experts=self.num_slots,
                    top_k=effective_top_k,
                    num_local_experts=esp,
                    local_expert_offset=slot_start,
                    tile_size=tile_size,
                    output_dtype=output_dtype,
                )
            if fc2_svdquant_active:
                # Accumulates onto the GEMM output in place, so no [M, hidden]
                # correction buffer and no out-of-place add are allocated.
                with nvtx_range_debug(
                        "[CUTEDSL][NVFP4] fc2.svdq_lowrank"):
                    self._accumulate_svdquant_lr_permuted(
                        fc2_input_bf16, w2_factors[0], w2_factors[1], x,
                        tile_idx_to_expert_idx, tile_idx_to_mn_limit, tile_size,
                        slot_start, esp, num_non_exiting_tiles)
            with nvtx_range_debug("[CUTEDSL][NVFP4] unpermute"):
                torch.ops.trtllm.moe_unpermute_inplace(
                    permuted_input=x,
                    output=moe_output,
                    expanded_idx_to_permuted_idx=expanded_idx_to_permuted_idx,
                    topk_scales=token_final_scales,
                )
        return moe_output

    def run_moe_fp8_block_scales(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
    ) -> torch.Tensor:
        assert self.has_deepseek_fp8_block_scales
        assert x_sf is None
        assert self.activation_type == ActivationType.Swiglu, (
            "FP8 block-scales MoE path hardcodes SwiGLU (see swiglu_fused_moe "
            f"below); got activation_type={ActivationType(self.activation_type).name}"
        )
        weight_dtype = self.w3_w1_weight.dtype

        (
            permuted_row_to_unpermuted_row,
            permuted_token_selected_experts,
            x,
            expert_first_token_offset,
            permuted_token_final_scales,
            unpermuted_row_to_permuted_row,
        ) = torch.ops.trtllm.moe_permute_op(
            x,
            token_selected_experts,
            token_final_scales,
            None,  # w3_w1_weight.view(weight_dtype),
            None,  # w2_weight.view(weight_dtype),
            None,  # quant_scales,
            input_sf=None,
            num_experts_on_rank=self.expert_size_per_partition,
            tp_size=self.tp_size,
            tp_rank=self.tp_rank,
            ep_size=self.ep_size,
            ep_rank=self.ep_rank,
            cluster_size=self.cluster_size,
            cluster_rank=self.cluster_rank,
            min_latency_mode=False,
            use_fp8_block_scaling=True,
        )
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w3_w1_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[0],
            offset_array=expert_first_token_offset,
        )
        x = swiglu_fused_moe(x)
        x, x_sf = torch.ops.trtllm.fp8_quantize_1x128(x)
        x = cute_dsl_fp8_group_blockwise_gemm_ref(
            a=x,
            b=self.w2_weight.view(weight_dtype),
            a_sf=x_sf,
            b_sf=self.quant_scales[1],
            offset_array=expert_first_token_offset,
        )
        top_k = self.routing_method.top_k
        if token_selected_experts is not None:
            top_k = token_selected_experts.shape[-1]

        x = torch.ops.trtllm.moe_finalize_scale_op(
            x,
            None,  # biases
            token_final_scales,
            unpermuted_row_to_permuted_row,
            permuted_row_to_unpermuted_row,
            token_selected_experts,
            expert_first_token_offset,
            enable_alltoall,
            token_final_scales.size(0),  # num_rows
            self.hidden_size,  # (possibly padded) hidden_size
            self.unpadded_hidden_size,  # original hidden size
            top_k,
            self.expert_size_per_partition,  # num_experts_per_node
            self.tp_size,
            self.tp_rank,
            self.ep_size,
            self.ep_rank,
        )
        return x

    def run_moe(
        self,
        x: torch.Tensor,
        token_selected_experts: torch.Tensor,
        token_final_scales: Optional[torch.Tensor],
        x_sf: Optional[torch.Tensor] = None,
        moe_output: Optional[torch.Tensor] = None,
        enable_alltoall: bool = False,
        fc13_input_bf16: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Run MoE computation with CuteDSL backend.

        This method encapsulates the core MoE computation logic, handling different
        quantization schemes (fp8_block_scales and nvfp4).

        Args:
            # Standard MoE interface parameters:
            x: Input hidden states (may be pre-quantized)
            token_selected_experts: Expert IDs [num_tokens, top_k]. If EPLB is enabled,
                                    this represents expert slots [num_tokens, top_k] instead.
            token_final_scales: Final scaling factors for each token
            x_sf: Input scale factors (optional, for certain quantization schemes)
            moe_output: Pre-allocated MoE output buffer (optional, for NVLINK one-sided backend).
            enable_alltoall: Whether alltoall communication is enabled.

        Returns:
            final_hidden_states tensor.
        """
        # Execute MoE computation
        if self.has_nvfp4:
            weight_view = self._build_local_weight_view()
            result = self.run_moe_nvfp4(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                moe_output=moe_output,
                enable_alltoall=enable_alltoall,
                weight_view=weight_view,
                fc13_input_bf16=fc13_input_bf16,
            )
        elif self.has_deepseek_fp8_block_scales:
            result = self.run_moe_fp8_block_scales(
                x=x,
                token_selected_experts=token_selected_experts,
                token_final_scales=token_final_scales,
                x_sf=x_sf,
                enable_alltoall=enable_alltoall)
        else:
            raise ValueError(
                f"{self.__class__.__name__} doesn't support quantization mode {self.quant_config.quant_mode}."
            )
        return result

    def forward_chunk(
            self,
            x: Union[torch.Tensor, Fp4QuantizedTensor],
            router_logits: torch.Tensor,
            output_dtype: Optional[torch.dtype] = None,
            all_rank_num_tokens: Optional[List[int]] = None,
            use_dp_padding: Optional[bool] = None,
            repeating_info: tuple = (True, True),
    ) -> torch.Tensor:
        # Currently, the default path is that ConfigurableMoE calls CuteDslFusedMoE.run_moe.
        # This forward_chunk method is a reference implementation of the legacy path.
        if (self.has_svdquant_fc13() and self.use_dp
                and self.parallel_size > 1):
            raise svdquant_helpers.SvdquantLoadError(
                "SVDQuant FC13 with data parallel size > 1 is unsupported: "
                "the exact BF16 activation must be all-gathered together with "
                "its runtime NVFP4 encoding.")

        # Apply routing
        token_selected_experts, token_final_scales = self.routing_method.apply(
            router_logits)
        assert token_selected_experts.shape[
            1] == self.routing_method.experts_per_token
        assert token_selected_experts.shape == token_final_scales.shape
        assert token_selected_experts.shape[0] == router_logits.shape[0]
        assert token_final_scales.dtype == torch.float32
        assert token_selected_experts.dtype == torch.int32

        fc13_input_bf16 = None
        if (self.has_svdquant_fc13() and isinstance(x, torch.Tensor)
                and x.dtype == torch.bfloat16):
            x, x_sf, fc13_input_bf16 = self.quantize_input(
                x, return_bf16_input=True)
        else:
            x, x_sf = self.quantize_input(x)

        if self.use_dp and self.parallel_size > 1:
            x, x_sf, token_selected_experts, token_final_scales = allgather(
                [x, x_sf, token_selected_experts, token_final_scales],
                self.mapping,
                dim=0,
                sizes=None if use_dp_padding else all_rank_num_tokens)

        x = self.run_moe(x=x,
                         token_selected_experts=token_selected_experts,
                         token_final_scales=token_final_scales,
                         x_sf=x_sf,
                         fc13_input_bf16=fc13_input_bf16,
                         enable_alltoall=False)
        return x

    def load_weights(self,
                     weights: List[Dict],
                     allow_partial_loading: bool = False):
        super().load_weights(weights,
                             allow_partial_loading=allow_partial_loading)
