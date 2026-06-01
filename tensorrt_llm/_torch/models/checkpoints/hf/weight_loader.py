# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
import glob
import json
import multiprocessing
import os
import re
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

import psutil
import safetensors
import torch
import tqdm
from mpi4py import MPI as _MPI

from tensorrt_llm._torch.models.checkpoints.base_weight_loader import (
    BaseWeightLoader, ConsumableWeightsDict, LazySafetensorsWeightsDict,
    four_o_six_load_timing_elapsed, four_o_six_load_timing_log,
    four_o_six_load_timing_now, four_o_six_load_timing_should_log,
    four_o_six_load_timing_tensor_nbytes)
from tensorrt_llm._torch.models.modeling_utils import (
    register_checkpoint_weight_loader, run_concurrently)
from tensorrt_llm._utils import (ENABLE_MULTI_DEVICE, local_mpi_barrier,
                                 local_mpi_comm, local_mpi_rank, local_mpi_size)
from tensorrt_llm.logger import logger
from tensorrt_llm.mapping import Mapping


@register_checkpoint_weight_loader("mistral")
@register_checkpoint_weight_loader("HF")
class HfWeightLoader(BaseWeightLoader):
    """
    Loads weights from SafeTensors/bin/pth files.
    """

    _FOUROVERSIX_CKPT_PRODUCERS = {
        "llm_4o6.convert_ckpt_to_4o6_nvfp4",
        "llm_4o6.finalize_parallel_4o6_nvfp4",
    }

    @staticmethod
    def _get_local_available_host_memory() -> int:
        """Determine the minimum available memory observed on the local node
        and distribute it to all local ranks

        Because psutil.virtual_memory().available is just a snapshot in time,
        it is possible for the local ranks to get different numbers due to
        timing differences. This can lead to disagreement among the local ranks
        as to whether prefetch should be enabled, which causes a deadlock,
        because the ranks that think prefetch is enabled will wait at a local
        mpi barrier indefinitely for the ranks that do not.
        """
        available_host_memory = psutil.virtual_memory().available
        if ENABLE_MULTI_DEVICE:
            return local_mpi_comm().allreduce(available_host_memory,
                                              op=_MPI.MIN)
        return available_host_memory

    @staticmethod
    def _env_flag(name: str, default: bool = False) -> bool:
        value = os.environ.get(name)
        if value is None:
            return default
        return value.strip().lower() in ("1", "true", "yes", "on")

    @staticmethod
    def _env_int(name: str, default: int) -> int:
        value = os.environ.get(name)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    @staticmethod
    def _get_lazy_prefetch_file_order(
            checkpoint_dir: str, weight_files: List[str]) -> Tuple[List[str], str]:
        order_mode = os.environ.get("TRTLLM_4O6_LAZY_PREFETCH_ORDER",
                                    "demand").strip().lower()
        if order_mode in ("", "glob", "none", "off"):
            return weight_files, "glob"

        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if not os.path.exists(index_path):
            return weight_files, "glob_no_index"

        try:
            with open(index_path, "r", encoding="utf-8") as f:
                weight_map = json.load(f)["weight_map"]
        except (OSError, KeyError, json.JSONDecodeError):
            return weight_files, "glob_index_error"

        file_set = {os.path.normpath(file_path) for file_path in weight_files}
        keys_by_file: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            file_path = os.path.normpath(os.path.join(checkpoint_dir, filename))
            if file_path in file_set:
                keys_by_file.setdefault(file_path, []).append(key)

        def priority(file_path: str) -> Tuple[int, int, str]:
            keys = keys_by_file.get(os.path.normpath(file_path), [])
            layer_priority: Optional[int] = None
            language_layerless = False
            has_embedding = False
            has_language = False
            for key in keys:
                if key.startswith("language_model."):
                    has_language = True
                if "language_model.model.embed_tokens." in key:
                    has_embedding = True
                match = re.search(r"(?:^|\.)model\.layers\.(\d+)\.", key)
                if match is not None:
                    layer = int(match.group(1))
                    layer_priority = (layer if layer_priority is None else min(
                        layer_priority, layer))
                elif key.startswith("language_model."):
                    language_layerless = True

            if layer_priority is None:
                if has_embedding:
                    layer_priority = -1
                elif language_layerless or has_language:
                    layer_priority = 10_000
                else:
                    layer_priority = 20_000

            basename = os.path.basename(file_path)
            if basename.startswith("source-model-"):
                category = 0
            elif keys:
                category = 1
            else:
                category = 9
            return layer_priority, category, basename

        referenced_files = [
            file_path for file_path in weight_files
            if os.path.normpath(file_path) in keys_by_file
        ]
        unreferenced_files = [
            file_path for file_path in weight_files
            if os.path.normpath(file_path) not in keys_by_file
        ]
        ordered_files = sorted(referenced_files, key=priority)
        if unreferenced_files:
            ordered_files.extend(sorted(unreferenced_files))
        return ordered_files, "demand"

    def load_weights(self, checkpoint_dir: str,
                     mapping: Mapping) -> dict[str, Any]:
        load_start = four_o_six_load_timing_now()
        weight_files = glob.glob(f"{checkpoint_dir}/*.safetensors")
        # Some model checkpoint directories contain not only the sharded safetensors, but one
        # consolidated tensor. In the presence of both, we favor the former, as there really is no need
        # to prefetch the (usually) ridiculously large consolidated tensor into memory in such a case.
        filtered_weight_files = [
            x for x in weight_files if "consolidated" not in os.path.split(x)[1]
        ]
        if len(filtered_weight_files) > 0:
            weight_files = filtered_weight_files
        if weight_files:
            stat_start = four_o_six_load_timing_now()
            prefetch_size = sum(os.path.getsize(file) for file in weight_files)
            four_o_six_metadata = self._get_4o6_exported_checkpoint_metadata(
                checkpoint_dir)
            # Prefetch the weight files to CPU memory if the size is less than 90% of the available memory.
            # This is a heuristic to avoid prefetching files that are too large and causing file cache thrashing.
            # If the layer number is overridden, it indicates that only a subset of layers are loaded.
            # Prefetching all layers is unnecessary.
            num_layers = int(os.environ.get("TLLM_OVERRIDE_LAYER_NUM", "0"))
            available_host_memory = self._get_local_available_host_memory()
            enable_prefetch = (prefetch_size < available_host_memory * 0.9
                               and num_layers == 0)
            four_o_six_load_timing_log(
                "hf_weight_loader_discovery",
                checkpoint_dir=checkpoint_dir,
                safetensors_files=len(weight_files),
                prefetch_size_gb=prefetch_size / (1024**3),
                available_host_memory_gb=available_host_memory / (1024**3),
                num_layers_override=num_layers,
                enable_prefetch=enable_prefetch,
                is_4o6_exported=four_o_six_metadata is not None,
                elapsed_sec=four_o_six_load_timing_elapsed(stat_start))

            force_lazy = False
            use_lazy_4o6 = False
            lazy_prefetch = False
            if four_o_six_metadata is not None:
                force_lazy = self._env_flag("TRTLLM_4O6_FORCE_LAZY", False)
                use_lazy_4o6 = force_lazy or not enable_prefetch
                lazy_prefetch = (use_lazy_4o6 and enable_prefetch
                                 and self._env_flag(
                                     "TRTLLM_4O6_LAZY_PREFETCH", True))

            if enable_prefetch and not use_lazy_4o6:
                prefetch_start = four_o_six_load_timing_now()
                logger.info(
                    f"Prefetching {prefetch_size / (1024**3):.2f}GB checkpoint files."
                )
                self.prefetch_files(weight_files)
                # Ensure that all local ranks have finished prefetching before loading weights
                local_mpi_barrier()
                four_o_six_load_timing_log(
                    "hf_weight_loader_prefetch",
                    files=len(weight_files),
                    size_gb=prefetch_size / (1024**3),
                    elapsed_sec=four_o_six_load_timing_elapsed(prefetch_start))

            if four_o_six_metadata is not None:
                index_path = os.path.join(checkpoint_dir,
                                          "model.safetensors.index.json")
                if use_lazy_4o6:
                    async_prefetch_files: List[str] = []
                    async_prefetch_workers = 0
                    if lazy_prefetch:
                        prefetch_files, prefetch_order = (
                            self._get_lazy_prefetch_file_order(
                                checkpoint_dir, weight_files))
                        async_prefetch_files = prefetch_files[
                            local_mpi_rank()::local_mpi_size()]
                        async_prefetch_workers = min(
                            max(
                                0,
                                self._env_int(
                                    "TRTLLM_4O6_LAZY_PREFETCH_WORKERS", 4)),
                            len(async_prefetch_files))
                        if async_prefetch_workers == 0:
                            async_prefetch_files = []
                    else:
                        prefetch_order = "disabled"
                    logger.info(
                        "Using lazy safetensors loading for exported 4o6 NVFP4 "
                        "checkpoint with asynchronous prefetch "
                        f"{'enabled' if async_prefetch_files else 'disabled'}."
                    )
                    four_o_six_load_timing_log(
                        "hf_weight_loader_lazy_prefetch",
                        enabled=bool(async_prefetch_files),
                        files=len(async_prefetch_files),
                        total_files=len(weight_files),
                        max_workers=async_prefetch_workers,
                        local_rank=local_mpi_rank(),
                        local_size=local_mpi_size(),
                        force_lazy=force_lazy,
                        enable_prefetch=enable_prefetch,
                        order=prefetch_order,
                        first_files=",".join(
                            os.path.basename(file_path)
                            for file_path in async_prefetch_files[:8]))
                    weights = LazySafetensorsWeightsDict.from_safetensors_files(
                        checkpoint_dir,
                        weight_files,
                        metadata=four_o_six_metadata,
                        async_prefetch_files=async_prefetch_files,
                        async_prefetch_workers=async_prefetch_workers)
                    four_o_six_load_timing_log(
                        "hf_weight_loader_done",
                        strategy=("lazy_async_prefetch"
                                  if async_prefetch_files else "lazy"),
                        elapsed_sec=four_o_six_load_timing_elapsed(load_start))
                    return weights
                if os.path.exists(index_path):
                    logger.info(
                        "Using index-aware eager parallel safetensors loading "
                        "for exported 4o6 NVFP4 checkpoint.")
                    weights = self._load_indexed_safetensors_in_parallel(
                        checkpoint_dir,
                        index_path,
                        metadata=four_o_six_metadata)
                    four_o_six_load_timing_log(
                        "hf_weight_loader_done",
                        strategy="indexed_eager",
                        elapsed_sec=four_o_six_load_timing_elapsed(load_start))
                    return weights
                logger.info(
                    "Using eager parallel safetensors loading for exported "
                    "4o6 NVFP4 checkpoint.")

            weights = self._load_weights_in_parallel(
                weight_files, self._load_safetensors_file,
                "Loading safetensors weights in parallel",
                metadata=four_o_six_metadata)
            four_o_six_load_timing_log(
                "hf_weight_loader_done",
                strategy="eager",
                elapsed_sec=four_o_six_load_timing_elapsed(load_start))
            return weights

        weight_files = glob.glob(f"{checkpoint_dir}/*.bin")
        if not weight_files:
            weight_files = glob.glob(f"{checkpoint_dir}/*.pth")

        if weight_files:
            return self._load_weights_in_parallel(
                weight_files, self._load_bin_or_path_file,
                "Loading bin weights in parallel")

        raise RuntimeError(f"No weight files found in {checkpoint_dir}.")

    @classmethod
    def _is_4o6_exported_checkpoint(cls, checkpoint_dir: str) -> bool:
        return cls._get_4o6_exported_checkpoint_metadata(
            checkpoint_dir) is not None

    @classmethod
    def _get_4o6_exported_checkpoint_metadata(
            cls, checkpoint_dir: str) -> dict[str, Any] | None:
        quant_config_path = os.path.join(checkpoint_dir, "hf_quant_config.json")
        if not os.path.exists(quant_config_path):
            return None

        try:
            with open(quant_config_path, "r", encoding="utf-8") as f:
                quant_config = json.load(f)
        except (OSError, json.JSONDecodeError):
            return None

        producer = quant_config.get("producer", {})
        producer_name = producer.get("name")
        if producer_name not in cls._FOUROVERSIX_CKPT_PRODUCERS:
            return None
        return {
            "already_4o6_nvfp4": True,
            "producer": producer_name,
        }

    def _load_weights_in_parallel(
            self,
            weight_files: List[str],
            load_func,
            description: str,
            metadata: Optional[Dict[str, Any]] = None) -> ConsumableWeightsDict:
        """
        Load weight files in parallel using the specified loading function.

        Args:
            weight_files: List of weight file paths
            load_func: Function to load individual weight files
            description: Description for the progress bar

        Returns:
            ConsumableWeightsDict containing all loaded weights
        """
        start = four_o_six_load_timing_now()
        weights = {}
        pbar = tqdm.tqdm(total=len(weight_files), desc=description)

        # Note that the function is called with a tuple of arguments, hence we need to wrap the arguments in a tuple via [(w,) for w in weight_files]
        # specifically the comma right after the w is important to make it a tuple.
        run_concurrently(load_func, [(w, ) for w in weight_files],
                         reduce_func=weights.update,
                         pbar=pbar)

        four_o_six_load_timing_log(
            "hf_parallel_load_done",
            description=description,
            files=len(weight_files),
            tensors=len(weights),
            elapsed_sec=four_o_six_load_timing_elapsed(start))
        return ConsumableWeightsDict(weights, metadata=metadata)

    def _load_indexed_safetensors_in_parallel(
            self,
            checkpoint_dir: str,
            index_path: str,
            metadata: Optional[Dict[str, Any]] = None) -> ConsumableWeightsDict:
        start = four_o_six_load_timing_now()
        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f)["weight_map"]

        keys_by_file: Dict[str, List[str]] = {}
        for key, filename in weight_map.items():
            keys_by_file.setdefault(os.path.join(checkpoint_dir, filename),
                                    []).append(key)

        weights = {}
        pbar = tqdm.tqdm(total=len(keys_by_file),
                         desc="Loading indexed safetensors weights in parallel")
        args = [(file, sorted(keys)) for file, keys in sorted(
            keys_by_file.items())]
        run_concurrently(self._load_safetensors_file_keys,
                         args,
                         reduce_func=weights.update,
                         pbar=pbar)
        four_o_six_load_timing_log(
            "hf_indexed_parallel_load_done",
            files=len(keys_by_file),
            tensors=len(weights),
            elapsed_sec=four_o_six_load_timing_elapsed(start))
        return ConsumableWeightsDict(weights, metadata=metadata)

    @staticmethod
    def _load_safetensors_file(file):
        start = four_o_six_load_timing_now()
        logger.info(f"Start to load safetensor file {file}")
        weights = safetensors.torch.load_file(file)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_FILE_THRESHOLD_SEC",
                0.25):
            nbytes = sum(
                four_o_six_load_timing_tensor_nbytes(value)
                for value in weights.values())
            four_o_six_load_timing_log("hf_load_file",
                                       file=os.path.basename(file),
                                       tensors=len(weights),
                                       nbytes=nbytes,
                                       elapsed_sec=elapsed_sec)
        return weights

    @staticmethod
    def _load_safetensors_file_keys(file, keys):
        start = four_o_six_load_timing_now()
        logger.info(f"Start to load {len(keys)} tensors from safetensor file {file}")
        with safetensors.safe_open(file, framework="pt", device="cpu") as f:
            weights = {key: f.get_tensor(key) for key in keys}
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_FILE_THRESHOLD_SEC",
                0.25):
            nbytes = sum(
                four_o_six_load_timing_tensor_nbytes(value)
                for value in weights.values())
            four_o_six_load_timing_log("hf_load_file_keys",
                                       file=os.path.basename(file),
                                       tensors=len(weights),
                                       nbytes=nbytes,
                                       elapsed_sec=elapsed_sec)
        return weights

    @staticmethod
    def _load_bin_or_path_file(file):
        try:
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=True)
        except Exception:
            logger.warning(
                f"Failed to load {file} with mmap=True, fallback to mmap=False")
            part_weights = torch.load(file,
                                      weights_only=True,
                                      map_location='cpu',
                                      mmap=False)
        finally:
            return part_weights

    def _prefetch_one_file(self, file_name):
        if os.path.exists(file_name):
            logger.info(f"Prefetching {file_name} to memory...")
            with open(file_name, 'rb') as f:
                f.read()
            logger.info(f"Finished prefetching {file_name}.")

    def prefetch_files(self, file_names: List[str]):
        """
        Prefetch safetensors files to memory so that the weight loading will be much faster.
        When multiple ranks run in parallel, each rank will prefetch some files.
        """
        # Find out the files to prefetch for the current rank.
        # Each rank loads files with indices local_rank, local_rank + local_mpi_size, local_rank + 2*local_mpi_size, etc.
        local_file_names = file_names[local_mpi_rank()::local_mpi_size()]
        if len(local_file_names) == 0:
            return

        max_workers = min(multiprocessing.cpu_count() * 2, 16,
                          len(local_file_names))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            list(executor.map(self._prefetch_one_file, local_file_names))
