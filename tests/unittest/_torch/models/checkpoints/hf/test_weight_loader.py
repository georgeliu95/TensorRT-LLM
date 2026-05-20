import json
from unittest import mock

import pytest
import torch
from safetensors.torch import save_file

from tensorrt_llm._torch.models.checkpoints import HfWeightLoader
from tensorrt_llm._torch.models.checkpoints.base_weight_loader import \
    LazySafetensorsWeightsDict
from tensorrt_llm.mapping import Mapping


class MyError(Exception):
    pass


@pytest.mark.parametrize(
    "dir_name, safetensor_filenames, expected_safetensor_filenames",
    [
        (
            "foo",
            [
                "model-00001-of-00002.safetensors",
                "model-000002-of-00002.safetensors",
                "consolidated.safetensors",
            ],
            ["model-00001-of-00002.safetensors", "model-000002-of-00002.safetensors"],
        ),
        (
            "foo",
            [
                *(f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)),
                "foo-consolidated.safetensors",
            ],
            [f"model-0000{i}-of-00010.safetensors" for i in range(1, 11)],
        ),
        # If there is only a consolidated safetensor, that one should still be used.
        (
            "foo",
            ["consolidated.safetensors"],
            ["consolidated.safetensors"],
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
        ),
    ],
)
def test_load_weights_ignores_consolidated_ckpt_when_sharded_ckpt_exists(
    tmp_path,
    dir_name: str,
    safetensor_filenames: list[str],
    expected_safetensor_filenames: list[str],
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
        loader.load_weights(checkpoint_dir=str(checkpoint_dir), mapping=Mapping())

    prefetch_files.assert_called_once()
    prefetched_files = prefetch_files.call_args[0][0]
    assert set(prefetched_files) == expected_safetensor_filenames

    load_weights_in_parallel.assert_called_once()
    loaded_weight_files = load_weights_in_parallel.call_args[0][0]
    assert set(loaded_weight_files) == expected_safetensor_filenames


def test_load_weights_uses_lazy_safetensors_for_4o6_export(tmp_path):
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
            "model.layers.0.mlp.experts.0.gate_proj.input_scale":
            torch.ones((), dtype=torch.float32),
            unrelated_key: torch.arange(4, dtype=torch.int32),
        },
        shard_path,
    )
    (checkpoint_dir / "model.safetensors.index.json").write_text(
        json.dumps({
            "metadata": {
                "total_size": shard_path.stat().st_size
            },
            "weight_map": {
                tensor_key: shard_name,
                f"{tensor_key}_scale": shard_name,
                f"{tensor_key}_scale_2": shard_name,
                "model.layers.0.mlp.experts.0.gate_proj.input_scale":
                shard_name,
                unrelated_key: shard_name,
            },
        }),
        encoding="utf-8",
    )
    (checkpoint_dir / "hf_quant_config.json").write_text(
        json.dumps({
            "producer": {
                "name": "llm_4o6.convert_ckpt_to_4o6_nvfp4",
                "version": "0.1",
            },
            "quantization": {
                "quant_algo": "NVFP4",
                "group_size": 16,
                "exclude_modules": [],
            },
        }),
        encoding="utf-8",
    )

    loader = HfWeightLoader()
    with mock.patch("safetensors.torch.load_file") as load_file:
        weights = loader.load_weights(str(checkpoint_dir), Mapping())

    load_file.assert_not_called()
    assert isinstance(weights, LazySafetensorsWeightsDict)

    module_weights = weights.filter(
        "model.layers.0.mlp.experts.0.gate_proj")
    assert set(module_weights) == {
        "weight", "weight_scale", "weight_scale_2", "input_scale"
    }
    assert torch.equal(module_weights["weight"], torch.arange(8,
                                                             dtype=torch.int32))
    assert unrelated_key in weights

    assert weights.mark_consumed(
        "model.layers.0.mlp.experts.0.gate_proj") == 4
    assert tensor_key not in weights
    assert unrelated_key in weights
