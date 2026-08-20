# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exact-process provenance for the NVFP4 overhead benchmark."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


_REQUIRED_OPERATOR_ARGUMENTS = {
    "trtllm::fp4_quantize_fused": (
        "scaleRule",
        "tileIdxToMnLimit",
        "numNonExitingTiles",
        "tileSize",
    ),
    "trtllm::fp4_swiglu_quantize_fused": (
        "scaleRule",
        "tileIdxToMnLimit",
        "numNonExitingTiles",
        "tileSize",
    ),
    "trtllm::cute_dsl_nvfp4_grouped_gemm_blackwell": (
        "alpha_numerator",
        "alpha_denominator",
    ),
    "trtllm::cute_dsl_nvfp4_grouped_gemm_finalize_inplace_blackwell": (
        "alpha_numerator",
        "alpha_denominator",
    ),
    "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_blackwell": (
        "alpha_numerator",
        "alpha_denominator",
    ),
    "trtllm::cute_dsl_nvfp4_gather_grouped_gemm_act_fusion_bf16_blackwell": (
        "alpha_numerator",
        "alpha_denominator",
    ),
}


@dataclass(frozen=True, slots=True)
class SourceIdentity:
    """Workspace and runtime identities for one required Python source file."""

    logical_name: str
    workspace_path: str
    runtime_path: str
    sha256: str


@dataclass(frozen=True, slots=True)
class Nvfp4ProvenanceError(RuntimeError):
    """The benchmark process does not match its declared workspace source."""

    logical_name: str
    workspace_path: Path
    runtime_path: Path
    workspace_sha256: str
    runtime_sha256: str

    def __str__(self) -> str:
        return (
            f"{self.logical_name} runtime source does not match the workspace: "
            f"{self.runtime_path} ({self.runtime_sha256}) != "
            f"{self.workspace_path} ({self.workspace_sha256})"
        )


@dataclass(frozen=True, slots=True)
class OperatorSchemaError(RuntimeError):
    """A required custom operator or ABI argument is absent."""

    operator: str
    missing_argument: str

    def __str__(self) -> str:
        return (
            f"{self.operator} loaded schema is missing required argument "
            f"{self.missing_argument}"
        )


@dataclass(frozen=True, slots=True)
class BuildIdentityError(RuntimeError):
    """The running source or custom-op library differs from its build manifest."""

    component: str
    actual: str
    expected: str

    def __str__(self) -> str:
        return (
            f"loaded {self.component} identity {self.actual} does not match "
            f"build manifest {self.expected}"
        )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_source_manifest(
    workspace_root: Path,
    relative_paths: Iterable[str],
) -> dict[str, Any]:
    """Hash build-relevant source files and their canonical aggregate."""
    files = {
        relative_path: _sha256(workspace_root / relative_path)
        for relative_path in sorted(set(relative_paths))
    }
    payload = json.dumps(
        files,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {"files": files, "sha256": hashlib.sha256(payload).hexdigest()}


def verify_source_identity(
    logical_name: str,
    workspace_path: Path,
    runtime_path: Path,
) -> SourceIdentity:
    """Return one source identity or reject stale installed Python code."""
    workspace_sha256 = _sha256(workspace_path)
    runtime_sha256 = _sha256(runtime_path)
    if runtime_sha256 != workspace_sha256:
        raise Nvfp4ProvenanceError(
            logical_name=logical_name,
            workspace_path=workspace_path,
            runtime_path=runtime_path,
            workspace_sha256=workspace_sha256,
            runtime_sha256=runtime_sha256,
        )
    return SourceIdentity(
        logical_name=logical_name,
        workspace_path=str(workspace_path.resolve()),
        runtime_path=str(runtime_path.resolve()),
        sha256=workspace_sha256,
    )


def verify_operator_schemas(schemas: Mapping[str, str]) -> None:
    """Reject a loaded custom-op library whose NVFP4 ABI is stale."""
    for operator, required_arguments in _REQUIRED_OPERATOR_ARGUMENTS.items():
        schema = schemas.get(operator, "")
        for argument in required_arguments:
            if argument not in schema:
                raise OperatorSchemaError(operator, argument)


def verify_build_identity(
    build_manifest: Mapping[str, Any],
    *,
    source_manifest_sha256: str,
    loaded_library_sha256: str,
) -> None:
    """Reject a process whose source or custom-op binary differs from its build."""
    expected_source = str(build_manifest.get("source_manifest_sha256", ""))
    if source_manifest_sha256 != expected_source:
        raise BuildIdentityError(
            "source manifest",
            source_manifest_sha256,
            expected_source,
        )
    library = build_manifest.get("custom_op_library")
    expected_library = str(library.get("sha256", "")) if isinstance(library, Mapping) else ""
    if loaded_library_sha256 != expected_library:
        raise BuildIdentityError(
            "custom-op library",
            loaded_library_sha256,
            expected_library,
        )
