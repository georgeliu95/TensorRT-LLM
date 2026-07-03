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
