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

import subprocess
import sys
from pathlib import Path
from typing import Final

import torch
from safetensors import safe_open

from tensorrt_llm.commands.convert_4o6_ckpt import (
    ShardWriter,
    build_svdquant_metadata,
)

REPO_ROOT: Final = Path(__file__).resolve().parents[3]
CONVERTER_MODULE: Final = (
    REPO_ROOT / "tensorrt_llm" / "commands" / "convert_4o6_ckpt.py"
)


def test_convert_4o6_ckpt_module_exposes_cli_help() -> None:
    # Given an invocation of the installable converter module.
    command = [sys.executable, str(CONVERTER_MODULE), "--help"]

    # When argparse renders the command help without loading model weights.
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
    )

    # Then the CLI exits successfully and advertises the Kimi INT4 input mode.
    assert result.returncode == 0, result.stderr
    assert "--source-format" in result.stdout
    assert "--svdquant-rank" in result.stdout


def test_svdquant_factor_serialization_and_metadata_round_trip(tmp_path) -> None:
    # Given one BF16 low-rank factor pair and the offline artifact contract.
    writer = ShardWriter(tmp_path, max_shard_bytes=1024 * 1024)
    us = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    vh = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)

    # When the converter's normal shard writer persists and indexes the pair.
    writer.add("model.layers.1.mlp.experts.0.gate_proj.svdquant_us", us)
    writer.add("model.layers.1.mlp.experts.0.gate_proj.svdquant_vh", vh)
    writer.finalize()

    # Then values, dtype, and immutable INT4-derived metadata round-trip.
    shard = tmp_path / "model-00001.safetensors"
    with safe_open(shard, framework="pt", device="cpu") as handle:
        torch.testing.assert_close(
            handle.get_tensor(
                "model.layers.1.mlp.experts.0.gate_proj.svdquant_us"), us)
        torch.testing.assert_close(
            handle.get_tensor(
                "model.layers.1.mlp.experts.0.gate_proj.svdquant_vh"), vh)
    assert build_svdquant_metadata(64, "bfloat16") == {
        "format": "int4-derived-offline-v1",
        "rank": 64,
        "factor_dtype": "bfloat16",
        "source_format": "int4-compressed-tensors",
        "stages": ["fc13", "fc2"],
        "reference": "dequantized-native-int4",
    }
