import json
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    ConsumableWeightsDict, LazySafetensorsWeightsDict
from tensorrt_llm._torch.models.checkpoints.hf.weight_mapper import \
    HfWeightMapper
from tensorrt_llm.mapping import Mapping


class MyError(Exception):
    pass


@pytest.mark.parametrize(
    "dir_name, safetensor_filenames, expected_safetensor_filenames, use_consolidated",
    [
        (
            "foo",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["model-00001-of-00002.safetensors", "model-000002-of-00002.safetensors"],
            False,
        ),
        # If use_consolidated specified explicitly.
        (
            "foo",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["consolidated.safetensors"],
            True,
        ),
        (
            "foo",
            [
                *(f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)),
                "foo-consolidated.safetensors",
            ],
            [f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)],
            False,
        ),
        # If there is only a consolidated safetensor, that one should still be used.
        (
            "foo",
            ["consolidated.safetensors"],
            ["consolidated.safetensors"],
            False,
        ),
        # If the directory contains "consolidated" in its name, but its contents are sharded tensors.
        (
            "consolidated-model",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["model-00001-of-00002.safetensors", "model-000002-of-00002.safetensors"],
            False,
        ),
    ],
)
def test_load_weights_ignores_consolidated_ckpt_when_sharded_ckpt_exists(
    tmp_path,
    dir_name: str,
    safetensor_filenames: list[str],
    expected_safetensor_filenames: list[str],
    use_consolidated: bool,
):
    checkpoint_dir = tmp_path / dir_name
    checkpoint_dir.mkdir()
    for filename in safetensor_filenames:
        (checkpoint_dir / filename).touch()
    expected_safetensor_filenames = set(
        str(checkpoint_dir / filename) for filename in expected_safetensor_filenames
    )

    loader = HfWeightLoader()
    with (
        mock.patch.object(
            loader, "_load_weights_in_parallel", side_effect=MyError
        ) as load_weights_in_parallel,
        mock.patch.object(loader, "prefetch_files") as prefetch_files,
        pytest.raises(MyError),
    ):
        loader.load_weights(
            checkpoint_dir=str(checkpoint_dir), mapping=Mapping(), use_consolidated=use_consolidated
        )

    prefetch_files.assert_called_once()
    prefetched_files = prefetch_files.call_args[0][0]
    assert set(prefetched_files) == expected_safetensor_filenames

    load_weights_in_parallel.assert_called_once()
    loaded_weight_files = load_weights_in_parallel.call_args[0][0]
    assert set(loaded_weight_files) == expected_safetensor_filenames


@pytest.mark.parametrize(
    "producer_name",
    [
        "llm_4o6.convert_ckpt_to_4o6_nvfp4",
        "llm_4o6.finalize_parallel_4o6_nvfp4",
    ],
)
def test_load_weights_uses_lazy_safetensors_for_4o6_export(
    tmp_path, producer_name
):
    checkpoint_dir = tmp_path / "exported-4o6"
    checkpoint_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    shard_path = checkpoint_dir / shard_name
    tensor_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    unrelated_key = "model.layers.1.mlp.experts.0.gate_proj.weight"
    save_file(
        {
            tensor_key: torch.arange(8, dtype=torch.int32),
            f"{tensor_key}_scale": torch.ones(1, dtype=torch.float32),
            f"{tensor_key}_scale_2": torch.ones((), dtype=torch.float32),
            "model.layers.0.mlp.experts.0.gate_proj.input_scale": torch.ones(
                (), dtype=torch.float32
            ),
            unrelated_key: torch.arange(4, dtype=torch.int32),
        },
        shard_path,
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": shard_path.stat().st_size},
                "weight_map": {
                    tensor_key: shard_name,
                    f"{tensor_key}_scale": shard_name,
                    f"{tensor_key}_scale_2": shard_name,
                    "model.layers.0.mlp.experts.0.gate_proj.input_scale": shard_name,
                    unrelated_key: shard_name,
                },
            }
        ),
        encoding="utf-8",
    )
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {"name": producer_name, "version": "0.1"},
                "quantization": {
                    "quant_algo": "NVFP4",
                    "group_size": 16,
                    "exclude_modules": [],
                },
            }
        ),
        encoding="utf-8",
    )

    loader = HfWeightLoader()
    with (
        mock.patch("safetensors.torch.load_file") as load_file,
        mock.patch.object(
            loader, "_get_local_available_host_memory", return_value=0
        ),
    ):
        weights = loader.load_weights(str(checkpoint_dir), Mapping())

    load_file.assert_not_called()
    assert isinstance(weights, LazySafetensorsWeightsDict)
    assert weights.metadata == {
        "already_4o6_nvfp4": True,
        "producer": producer_name,
    }

    with mock.patch("safetensors.safe_open") as safe_open:
        module_weights = weights.filter(
            "model.layers.0.mlp.experts.0.gate_proj"
        )
        safe_open.assert_not_called()

    assert isinstance(module_weights, LazySafetensorsWeightsDict)
    assert module_weights.metadata == weights.metadata
    assert set(module_weights) == {
        "weight",
        "weight_scale",
        "weight_scale_2",
        "input_scale",
    }
    assert torch.equal(
        module_weights["weight"], torch.arange(8, dtype=torch.int32)
    )
    assert unrelated_key in weights

    assert weights.mark_consumed(
        "model.layers.0.mlp.experts.0.gate_proj"
    ) == 4
    assert tensor_key not in weights
    assert unrelated_key in weights

    with mock.patch("safetensors.safe_open") as safe_open:
        renamed_weights = weights.rename_by_regex(
            {r"(.*)gate_proj(.*)": r"\1w1\2"}
        )
        safe_open.assert_not_called()

    assert isinstance(renamed_weights, LazySafetensorsWeightsDict)
    assert renamed_weights.metadata == weights.metadata
    assert "model.layers.1.mlp.experts.0.w1.weight" in renamed_weights
    assert torch.equal(
        renamed_weights["model.layers.1.mlp.experts.0.w1.weight"],
        torch.arange(4, dtype=torch.int32),
    )

    mapper_renamed_weights = HfWeightMapper().rename_by_params_map(
        {r"(.*)gate_proj(.*)": r"\1w1\2"}, weights
    )
    assert isinstance(mapper_renamed_weights, LazySafetensorsWeightsDict)
    assert mapper_renamed_weights.metadata == weights.metadata
    assert "model.layers.1.mlp.experts.0.w1.weight" in mapper_renamed_weights


def test_4o6_loader_exposes_exact_persisted_svdquant_contract(tmp_path):
    # Given a 4o6 export carrying the supported offline INT4-derived contract.
    checkpoint_dir = tmp_path / "exported-4o6-svdquant"
    checkpoint_dir.mkdir()
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {
                    "name": "llm_4o6.finalize_parallel_4o6_nvfp4",
                    "version": "0.2",
                },
                "quantization": {"quant_algo": "NVFP4"},
                "svdquant": {
                    "format": "int4-derived-offline-v1",
                    "rank": 64,
                    "factor_dtype": "bfloat16",
                    "source_format": "int4-compressed-tensors",
                    "stages": ["fc13", "fc2"],
                    "reference": "dequantized-native-int4",
                },
            }
        ),
        encoding="utf-8",
    )

    # When checkpoint metadata is resolved at the loader boundary.
    metadata = HfWeightLoader._get_4o6_exported_checkpoint_metadata(
        str(checkpoint_dir))

    # Then downstream MoE remapping receives an immutable, explicit contract.
    assert metadata == {
        "already_4o6_nvfp4": True,
        "producer": "llm_4o6.finalize_parallel_4o6_nvfp4",
        "svdquant_artifact": True,
        "svdquant_format": "int4-derived-offline-v1",
        "svdquant_rank": 64,
        "svdquant_factor_dtype": "bfloat16",
        "svdquant_stages": ("fc13", "fc2"),
        "svdquant_source_format": "int4-compressed-tensors",
        "svdquant_reference": "dequantized-native-int4",
    }


@pytest.mark.parametrize(
    "mutate",
    [
        lambda metadata: metadata.__setitem__("rank", 0),
        lambda metadata: metadata.__setitem__("factor_dtype", "float16"),
        lambda metadata: metadata.__setitem__("stages", ["fc13"]),
        lambda metadata: metadata.__setitem__("extra", True),
    ],
)
def test_4o6_loader_rejects_malformed_svdquant_contract(tmp_path, mutate):
    # Given recognized 4o6 metadata with one unsupported SVDQuant field.
    checkpoint_dir = tmp_path / "invalid-4o6-svdquant"
    checkpoint_dir.mkdir()
    svdquant = {
        "format": "int4-derived-offline-v1",
        "rank": 64,
        "factor_dtype": "bfloat16",
        "source_format": "int4-compressed-tensors",
        "stages": ["fc13", "fc2"],
        "reference": "dequantized-native-int4",
    }
    mutate(svdquant)
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {
                    "name": "llm_4o6.finalize_parallel_4o6_nvfp4"
                },
                "quantization": {"quant_algo": "NVFP4"},
                "svdquant": svdquant,
            }
        ),
        encoding="utf-8",
    )

    # When the loader inspects it, then it fails before any tensor is read.
    with pytest.raises(ValueError, match="SVDQuant artifact metadata"):
        HfWeightLoader._get_4o6_exported_checkpoint_metadata(
            str(checkpoint_dir))


def test_load_weights_uses_eager_safetensors_for_small_4o6_export(tmp_path):
    checkpoint_dir = tmp_path / "exported-4o6-eager"
    checkpoint_dir.mkdir()
    shard_name = "model-00001-of-00001.safetensors"
    tensor_key = "model.layers.0.mlp.experts.0.gate_proj.weight"
    save_file(
        {tensor_key: torch.arange(8, dtype=torch.int32)},
        checkpoint_dir / shard_name,
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": {tensor_key: shard_name}}),
        encoding="utf-8",
    )
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps(
            {
                "producer": {
                    "name": "llm_4o6.finalize_parallel_4o6_nvfp4"
                },
                "quantization": {"quant_algo": "NVFP4"},
            }
        ),
        encoding="utf-8",
    )

    loader = HfWeightLoader()
    with mock.patch.object(loader, "prefetch_files") as prefetch_files:
        weights = loader.load_weights(str(checkpoint_dir), Mapping())

    prefetch_files.assert_called_once()
    assert isinstance(weights, ConsumableWeightsDict)
    assert not isinstance(weights, LazySafetensorsWeightsDict)
    assert weights.metadata == {
        "already_4o6_nvfp4": True,
        "producer": "llm_4o6.finalize_parallel_4o6_nvfp4",
    }
    renamed = HfWeightMapper().rename_by_params_map(
        {r"(.*)gate_proj(.*)": r"\1w1\2"}, weights
    )
    assert isinstance(renamed, ConsumableWeightsDict)
    assert renamed.metadata == weights.metadata
    assert "model.layers.0.mlp.experts.0.w1.weight" in renamed
    filtered = HfWeightMapper().filter_weights(
        "model.layers.0.mlp.experts.0.gate_proj", weights
    )
    assert filtered.metadata == weights.metadata
    assert torch.equal(filtered["weight"], torch.arange(8, dtype=torch.int32))
