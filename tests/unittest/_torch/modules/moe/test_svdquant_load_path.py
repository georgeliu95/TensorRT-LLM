# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Load-lifecycle coverage for the NVFP4 MoE SVDQuant fallback."""

from __future__ import annotations

from collections.abc import Mapping

import pytest
import torch

from tensorrt_llm._torch.models import modeling_kimi_k25 as kimi_k25
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    ConsumableWeightsDict,
    WeightsDictWithMetadata,
)
from tensorrt_llm._torch.modules.fused_moe import svdquant_helpers as svdh
from tensorrt_llm._torch.modules.fused_moe.fused_moe_cute_dsl import (
    CuteDslFusedMoE,
    _deinterleave_linear_and_gate,
)
from tensorrt_llm._torch.modules.fused_moe.interface import MoEWeightLoadingMode
from tensorrt_llm._torch.modules.fused_moe.quantization import (
    NVFP4FusedMoEMethod,
    interleave_linear_and_gate,
)
from tensorrt_llm._torch.utils import ActivationType


class _ConcreteNVFP4Method(NVFP4FusedMoEMethod):
    """Concrete test seam for SVDQuant helpers on the abstract base method."""

    def load_expert_w3_w1_weight_scale_nvfp4(
        self,
        module: torch.nn.Module,
        w1_weight_scale: torch.Tensor,
        w3_weight_scale: torch.Tensor,
        dst_w3_w1_weight_scale: torch.Tensor,
    ) -> None:
        del module, w1_weight_scale, w3_weight_scale, dst_w3_w1_weight_scale

    def load_expert_w2_weight_scale_nvfp4(
        self,
        module: torch.nn.Module,
        w2_weight_scale: torch.Tensor,
        dst_w2_weight_scale: torch.Tensor,
    ) -> None:
        del module, w2_weight_scale, dst_w2_weight_scale


class _SharedWeights:
    def need_load_shared_weights(self) -> bool:
        return True


@pytest.fixture
def enabled_env(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    for name in svdh.ENV_NAMES:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(svdh.ENV_ENABLED, "1")
    monkeypatch.setenv(svdh.ENV_RANK, "4")
    monkeypatch.setenv(svdh.ENV_DEVICE, "cpu")
    return monkeypatch


def _make_module(*, tp_size: int = 1, tp_rank: int = 0) -> torch.nn.Module:
    module = torch.nn.Module()
    module.expert_size_per_partition = 2
    module.intermediate_size = 24
    module.intermediate_size_per_partition = 24 // tp_size
    module.hidden_size = 32
    module.tp_size = tp_size
    module.tp_rank = tp_rank
    module.initial_local_expert_ids = [3, 7]
    module.layer_load_balancer = None
    module.activation_type = int(ActivationType.Swiglu)
    module.is_gated_activation = True
    return module


def _make_dense_weights() -> dict[str, torch.Tensor]:
    torch.manual_seed(31)
    weights: dict[str, torch.Tensor] = {}
    for expert_id in (3, 7):
        weights[f"{expert_id}.w1.weight"] = torch.randn(
            24, 32, dtype=torch.bfloat16)
        weights[f"{expert_id}.w3.weight"] = torch.randn(
            24, 32, dtype=torch.bfloat16)
        weights[f"{expert_id}.w2.weight"] = torch.randn(
            32, 24, dtype=torch.bfloat16)
    return weights


def _persisted_metadata() -> dict[str, object]:
    return {
        "already_4o6_nvfp4": True,
        "producer": "llm_4o6.finalize_parallel_4o6_nvfp4",
        "svdquant_artifact": True,
        "svdquant_format": "int4-derived-offline-v1",
        "svdquant_rank": 4,
        "svdquant_factor_dtype": "bfloat16",
        "svdquant_stages": ("fc13", "fc2"),
        "svdquant_source_format": "int4-compressed-tensors",
        "svdquant_reference": "dequantized-native-int4",
    }


def _make_persisted_weights() -> WeightsDictWithMetadata:
    torch.manual_seed(37)
    weights = WeightsDictWithMetadata(metadata=_persisted_metadata())
    for expert_id in (3, 7):
        for projection, shape in (
            ("w1", (24, 32)),
            ("w3", (24, 32)),
            ("w2", (32, 24)),
        ):
            base = f"{expert_id}.{projection}"
            weights[f"{base}.weight"] = torch.zeros(shape, dtype=torch.uint8)
            weights[f"{base}.weight_scale"] = torch.ones(1)
            weights[f"{base}.weight_scale_2"] = torch.ones(())
            weights[f"{base}.input_scale"] = torch.ones(())
            weights[f"{base}.svdquant_us"] = torch.randn(
                shape[0], 4, dtype=torch.bfloat16)
            weights[f"{base}.svdquant_vh"] = torch.randn(
                4, shape[1], dtype=torch.bfloat16)
    return weights


def _snapshot(weights: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.clone() for name, tensor in weights.items()}


def test_kimi_language_model_filter_preserves_svdquant_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Given a Kimi checkpoint packet carrying the persisted SVDQuant contract.
    monkeypatch.setattr(kimi_k25, "DISAGG", True)
    captured: dict[str, ConsumableWeightsDict] = {}

    class _LlmRecorder:

        def load_weights(self, weights: ConsumableWeightsDict) -> None:
            captured["weights"] = weights

    class _KimiWrapper:
        _LANG_PREFIX = "language_model."
        mm_encoder = None
        llm = _LlmRecorder()

    weights = WeightsDictWithMetadata(metadata=_persisted_metadata())
    factor = torch.ones(24, 4, dtype=torch.bfloat16)
    weights["language_model.model.layers.1.mlp.experts.3.w1.svdquant_us"] = factor

    # When the Kimi VLM wrapper strips the language-model prefix and wraps the
    # packet for consumable loading.
    kimi_k25.KimiK25ForConditionalGeneration.load_weights(_KimiWrapper(),
                                                           weights)

    # Then both the factor and its checkpoint-level contract reach DeepSeek's
    # MoE loader together.
    forwarded = captured["weights"]
    assert isinstance(forwarded, ConsumableWeightsDict)
    assert forwarded.metadata == _persisted_metadata()
    assert forwarded[
        "model.layers.1.mlp.experts.3.w1.svdquant_us"] is factor


def test_dense_load_prepares_residual_then_tp_local_factors(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given rank two of a TP=4 MoE and dense BF16 expert weights.
    method = _ConcreteNVFP4Method()
    module = _make_module(tp_size=4, tp_rank=2)
    method._maybe_create_svdquant_weights(module)
    weights = _make_dense_weights()
    original = _snapshot(weights)

    # When the load substitution runs before the regular NVFP4 loader.
    adapted = method._maybe_prepare_svdquant_weights(
        module, weights, MoEWeightLoadingMode.VANILLA)

    # Then source tensors stay untouched and pending factors use TP-local shapes.
    for name, tensor in original.items():
        torch.testing.assert_close(weights[name], tensor)
    pending = module._svdquant_pending
    assert pending[(3, "w1")][0].shape == (6, 4)
    assert pending[(3, "w1")][1].shape == (4, 32)
    assert pending[(3, "w2")][0].shape == (32, 4)
    assert pending[(3, "w2")][1].shape == (4, 6)

    # And each low-rank term plus the corresponding residual shard reconstructs
    # the local dense shard that rc19's normal loader will consume.
    for projection, mode in (("w1", "column"), ("w2", "row")):
        source = original[f"3.{projection}.weight"]
        residual = adapted[f"3.{projection}.weight"]
        split_dim = 0 if mode == "column" else 1
        source_shard = torch.tensor_split(source, 4, dim=split_dim)[2]
        residual_shard = torch.tensor_split(residual, 4, dim=split_dim)[2]
        us, vh = pending[(3, projection)]
        reconstructed = (us.float() @ vh.float()).to(source_shard.dtype)
        reconstructed = reconstructed + residual_shard
        torch.testing.assert_close(reconstructed, source_shard,
                                   rtol=2e-2, atol=2e-2)


def test_persisted_load_stages_tp_local_factors_without_redecomposition(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given rank two of a TP=4 MoE and an exported residual-plus-factor packet.
    method = _ConcreteNVFP4Method()
    module = _make_module(tp_size=4, tp_rank=2)
    method._maybe_create_svdquant_weights(module)
    weights = _make_persisted_weights()
    original_w1_us = weights["3.w1.svdquant_us"].clone()
    original_w2_vh = weights["3.w2.svdquant_vh"].clone()

    # When preparation consumes the persisted artifact contract.
    adapted = method._maybe_prepare_svdquant_weights(
        module, weights, MoEWeightLoadingMode.VANILLA)

    # Then the regular loader receives the original residual packet unchanged,
    # while factors are staged with the projection's TP ownership.
    assert adapted is weights
    pending = module._svdquant_pending
    torch.testing.assert_close(pending[(3, "w1")][0], original_w1_us[12:18])
    torch.testing.assert_close(pending[(3, "w1")][1],
                               weights["3.w1.svdquant_vh"])
    torch.testing.assert_close(pending[(3, "w2")][0],
                               weights["3.w2.svdquant_us"])
    torch.testing.assert_close(pending[(3, "w2")][1], original_w2_vh[:, 12:18])

    method._finalize_svdquant_params(module)
    assert module._svdquant_loaded is True
    torch.testing.assert_close(module.w1_us[0], original_w1_us[12:18])
    torch.testing.assert_close(module.w2_vh[0], original_w2_vh[:, 12:18])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda weights: weights.metadata.__setitem__("svdquant_rank", 8),
         "conflict"),
        (lambda weights: weights.pop("7.w3.svdquant_vh"), "complete"),
        (lambda weights: weights.__setitem__(
            "3.w2.svdquant_us", weights["3.w2.svdquant_us"].float()),
         "BF16"),
        (lambda weights: weights.__setitem__(
            "3.w1.svdquant_vh",
            torch.zeros(4, 31, dtype=torch.bfloat16)), "shape"),
    ],
)
def test_invalid_persisted_inputs_fail_before_pending_state(
    enabled_env: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    # Given one invalid persisted artifact and pristine module state.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    method._maybe_create_svdquant_weights(module)
    weights = _make_persisted_weights()
    mutate(weights)

    # When prepared, then the complete packet is rejected before staging.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        method._maybe_prepare_svdquant_weights(
            module, weights, MoEWeightLoadingMode.VANILLA)
    assert not hasattr(module, "_svdquant_pending")


def test_persisted_factor_keys_without_metadata_fail_closed(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given factor-bearing weights whose checkpoint contract was lost.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    method._maybe_create_svdquant_weights(module)
    weights = _make_persisted_weights()
    weights.metadata.clear()

    # When prepared, then they cannot silently fall through to dense handling.
    with pytest.raises(svdh.SvdquantLoadError, match="metadata"):
        method._maybe_prepare_svdquant_weights(
            module, weights, MoEWeightLoadingMode.VANILLA)
    assert not hasattr(module, "_svdquant_pending")


def test_persisted_artifact_requires_runtime_enablement(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given a recognized artifact but disabled SVDQuant runtime flags.
    enabled_env.setenv(svdh.ENV_ENABLED, "0")
    method = _ConcreteNVFP4Method()
    module = _make_module()

    # When prepared, then it fails instead of loading residual-only weights.
    with pytest.raises(svdh.SvdquantLoadError, match="flags"):
        method._maybe_prepare_svdquant_weights(
            module, _make_persisted_weights(),
            MoEWeightLoadingMode.VANILLA)
    assert not hasattr(module, "_svdquant_pending")


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda weights: weights.__setitem__(
            "3.w1.weight_scale", torch.ones(1)), "dense BF16/FP16"),
        (lambda weights: weights.pop("7.w3.weight"), "complete"),
        (lambda weights: weights.__setitem__(
            "3.w2.weight", weights["3.w2.weight"].float()), "BF16/FP16"),
        (lambda weights: weights.__setitem__(
            "3.w1.weight", torch.zeros(23, 32, dtype=torch.bfloat16)),
         "shape"),
    ],
)
def test_invalid_dense_inputs_fail_before_module_or_source_mutation(
    enabled_env: pytest.MonkeyPatch,
    mutate,
    message: str,
) -> None:
    # Given one invalid dense load packet and pristine module state.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    method._maybe_create_svdquant_weights(module)
    weights = _make_dense_weights()
    mutate(weights)
    before = _snapshot(weights)

    # When prepared, then validation fails before pending state or source edits.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        method._maybe_prepare_svdquant_weights(
            module, weights, MoEWeightLoadingMode.VANILLA)
    assert not hasattr(module, "_svdquant_pending")
    for name, tensor in before.items():
        torch.testing.assert_close(weights[name], tensor)


@pytest.mark.parametrize(
    ("mode", "partial", "shared", "loaded", "message"),
    [
        (MoEWeightLoadingMode.FUSED_GATE_UP_PROJ, False, False, False,
         "VANILLA"),
        (MoEWeightLoadingMode.VANILLA, True, False, False, "partial"),
        (MoEWeightLoadingMode.VANILLA, False, True, False, "EPLB"),
        (MoEWeightLoadingMode.VANILLA, False, False, True, "reload"),
    ],
)
def test_unsupported_lifecycle_fails_closed(
    enabled_env: pytest.MonkeyPatch,
    mode: MoEWeightLoadingMode,
    partial: bool,
    shared: bool,
    loaded: bool,
    message: str,
) -> None:
    # Given an unsupported lifecycle combination.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    method._maybe_create_svdquant_weights(module)
    if shared:
        module.layer_load_balancer = _SharedWeights()
    if loaded:
        module._svdquant_loaded = True

    # When preparation starts, then it fails before creating pending state.
    with pytest.raises(svdh.SvdquantLoadError, match=message):
        method._maybe_prepare_svdquant_weights(
            module, _make_dense_weights(), mode,
            allow_partial_loading=partial)
    assert not hasattr(module, "_svdquant_pending")


def test_finalize_copies_all_pairs_and_marks_load_complete(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given a complete prepared packet and registered factor parameters.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    method._maybe_create_svdquant_weights(module)
    method._maybe_prepare_svdquant_weights(
        module, _make_dense_weights(), MoEWeightLoadingMode.VANILLA)
    pending = {key: (us.clone(), vh.clone())
               for key, (us, vh) in module._svdquant_pending.items()}

    # When rc19 finalization copies the factors.
    method._finalize_svdquant_params(module)

    # Then every local slot is populated and transient state is cleared.
    assert module._svdquant_loaded is True
    assert not hasattr(module, "_svdquant_pending")
    for local_slot, expert_id in enumerate((3, 7)):
        for projection in ("w1", "w3", "w2"):
            expected_us, expected_vh = pending[(expert_id, projection)]
            torch.testing.assert_close(
                getattr(module, f"{projection}_us")[local_slot], expected_us)
            torch.testing.assert_close(
                getattr(module, f"{projection}_vh")[local_slot], expected_vh)


def test_fc13_rejects_non_swiglu_before_parameter_registration(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given FC13 enabled on a non-SwiGLU MoE.
    method = _ConcreteNVFP4Method()
    module = _make_module()
    module.activation_type = int(ActivationType.Relu2)
    module.is_gated_activation = False

    # When factor storage is created, then no partial parameter set is left.
    with pytest.raises(svdh.SvdquantLoadError, match="SwiGLU"):
        method._maybe_create_svdquant_weights(module)
    assert not any(name.endswith(("_us", "_vh"))
                   for name, _ in module.named_parameters())


def test_fc2_only_allows_non_swiglu(
    enabled_env: pytest.MonkeyPatch,
) -> None:
    # Given only FC2 correction enabled for a generic activation.
    enabled_env.setenv(svdh.ENV_FC13, "0")
    method = _ConcreteNVFP4Method()
    module = _make_module()
    module.activation_type = int(ActivationType.Relu2)
    module.is_gated_activation = False

    # When factor storage is created, then only the generic FC2 pair exists.
    method._maybe_create_svdquant_weights(module)
    assert hasattr(module, "w2_us") and hasattr(module, "w2_vh")
    assert not hasattr(module, "w1_us") and not hasattr(module, "w3_us")


def test_fc13_deinterleave_restores_residual_gemm_channel_order() -> None:
    # Given gate/up channels in the unfused order expected by SwiGLU math.
    source = torch.arange(3 * 256, dtype=torch.float32).view(3, 256)
    interleaved = interleave_linear_and_gate(source, group_size=64, dim=-1)

    # When the residual-only FC13 fallback reverses the kernel layout.
    result = _deinterleave_linear_and_gate(interleaved, group_size=64)

    # Then w3/w1 low-rank corrections can be added to their original channels.
    torch.testing.assert_close(result, source)


def test_svdquant_permuted_math_uses_rank_local_expert_slots() -> None:
    # Given moe_sort-style local expert slots on an EP rank whose global
    # expert range starts at 96.
    backend = CuteDslFusedMoE.__new__(CuteDslFusedMoE)
    x = torch.tensor([[2.0, 3.0], [5.0, 7.0]], dtype=torch.bfloat16)
    us = torch.tensor([[[1.0], [2.0]], [[3.0], [4.0]]],
                      dtype=torch.bfloat16)
    vh = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]],
                      dtype=torch.bfloat16)
    tile_idx_to_expert_idx = torch.tensor([0, 1], dtype=torch.int32)
    tile_idx_to_mn_limit = torch.tensor([1, 2], dtype=torch.int32)

    # When the low-rank correction is evaluated with a nonzero global offset.
    result = backend._compute_svdquant_lr_permuted(
        x,
        us,
        vh,
        tile_idx_to_expert_idx,
        tile_idx_to_mn_limit,
        tile_size=1,
        slot_start=96,
        num_local_experts=2,
    )

    # Then local slots 0 and 1 both contribute; the global offset must not be
    # subtracted from moe_sort's already-local IDs.
    expected = torch.tensor([[2.0, 4.0], [21.0, 28.0]],
                            dtype=torch.bfloat16)
    torch.testing.assert_close(result, expected)
