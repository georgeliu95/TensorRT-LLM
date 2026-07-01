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

import json
import os
import re
import sys
import threading
import time
from abc import ABC, abstractmethod
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from tensorrt_llm.mapping import Mapping

_LazyWeightRecord = Tuple[str, str]


class _LazyPrefetchState:
    _READ_CHUNK_BYTES = 64 * 1024 * 1024

    def __init__(self, files: List[str], max_workers: int):
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._executor: Optional[ThreadPoolExecutor] = None
        self._futures: Dict[str, Future] = {}
        self._skip_wait_logged: set[str] = set()
        if files and max_workers > 0:
            self._executor = ThreadPoolExecutor(max_workers=max_workers)
            for file_path in files:
                self._futures[file_path] = self._executor.submit(
                    self._prefetch_one_file, file_path, self._cancel)
            four_o_six_load_timing_log(
                "lazy_async_prefetch_start",
                files=len(files),
                max_workers=max_workers)

    @staticmethod
    def _prefetch_one_file(file_path: str,
                           cancel_event: threading.Event) -> int:
        start = four_o_six_load_timing_now()
        total = 0
        status = "ok"
        error = ""
        try:
            with open(file_path, "rb") as f:
                while not cancel_event.is_set():
                    data = f.read(_LazyPrefetchState._READ_CHUNK_BYTES)
                    if not data:
                        break
                    total += len(data)
            if cancel_event.is_set():
                status = "cancelled"
        except Exception as exc:
            status = "error"
            error = repr(exc)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_PREFETCH_THRESHOLD_SEC",
                0.25) or status != "ok":
            four_o_six_load_timing_log("lazy_async_prefetch_file",
                                       file=os.path.basename(file_path),
                                       status=status,
                                       nbytes=total,
                                       error=error,
                                       elapsed_sec=elapsed_sec)
        return total

    def wait_for_file(self, file_path: str) -> float:
        with self._lock:
            future = self._futures.get(file_path)
        if future is None:
            return 0.0
        start = four_o_six_load_timing_now()
        try:
            future.result()
        except Exception as exc:
            four_o_six_load_timing_log("lazy_async_prefetch_wait_error",
                                       file=os.path.basename(file_path),
                                       error=repr(exc))
        return four_o_six_load_timing_elapsed(start)

    def file_done(self, file_path: str) -> bool:
        with self._lock:
            future = self._futures.get(file_path)
        return future is None or future.done()

    def should_log_skip_wait(self, file_path: str) -> bool:
        with self._lock:
            future = self._futures.get(file_path)
            if future is None or future.done() or file_path in self._skip_wait_logged:
                return False
            self._skip_wait_logged.add(file_path)
            return True

    def shutdown(self) -> None:
        with self._lock:
            executor = self._executor
            self._executor = None
            futures = list(self._futures.values())
        self._cancel.set()
        cancelled = 0
        done = 0
        for future in futures:
            if future.done():
                done += 1
            elif future.cancel():
                cancelled += 1
        if executor is not None:
            executor.shutdown(wait=False, cancel_futures=True)
            four_o_six_load_timing_log("lazy_async_prefetch_shutdown",
                                       futures=len(futures),
                                       done=done,
                                       cancelled=cancelled)


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def four_o_six_load_timing_enabled() -> bool:
    return _env_flag("TRTLLM_4O6_LOAD_TIMING", False)


def four_o_six_load_timing_now() -> float:
    return time.perf_counter()


def four_o_six_load_timing_elapsed(start: float) -> float:
    return time.perf_counter() - start


def _load_timing_rank() -> str:
    for name in (
            "OMPI_COMM_WORLD_RANK",
            "PMI_RANK",
            "SLURM_PROCID",
            "RANK",
            "LOCAL_RANK",
    ):
        value = os.environ.get(name)
        if value is not None:
            return value
    return "0"


def _load_timing_rank_enabled(rank: str) -> bool:
    ranks = os.environ.get("TRTLLM_4O6_LOAD_TIMING_RANKS", "0").strip()
    if ranks.lower() in ("all", "*"):
        return True
    return rank in {item.strip() for item in ranks.split(",") if item.strip()}


def _load_timing_format(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    text = str(value).replace("\n", "\\n")
    if " " in text:
        return json.dumps(text)
    return text


def four_o_six_load_timing_log(event: str, **fields: Any) -> None:
    if not four_o_six_load_timing_enabled():
        return
    rank = _load_timing_rank()
    if not _load_timing_rank_enabled(rank):
        return
    payload = {
        "event": event,
        "rank": rank,
        "pid": os.getpid(),
        "time": f"{time.time():.6f}",
    }
    payload.update(fields)
    message = "TRTLLM_4O6_LOAD_TIMING " + " ".join(
        f"{key}={_load_timing_format(value)}"
        for key, value in payload.items())
    print(message, file=sys.stderr, flush=True)


def four_o_six_load_timing_threshold(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def four_o_six_load_timing_should_log(elapsed_sec: float,
                                      threshold_env: str,
                                      default_threshold_sec: float) -> bool:
    if not four_o_six_load_timing_enabled():
        return False
    threshold_sec = four_o_six_load_timing_threshold(threshold_env,
                                                    default_threshold_sec)
    return elapsed_sec >= threshold_sec


def four_o_six_load_timing_tensor_nbytes(value: Any) -> int:
    if not hasattr(value, "numel") or not hasattr(value, "element_size"):
        return 0
    try:
        return int(value.numel() * value.element_size())
    except (RuntimeError, TypeError, ValueError):
        return 0


class WeightsDictWithMetadata(dict):
    """Plain dict carrying checkpoint-level metadata through key remapping."""

    def __init__(self,
                 *args,
                 metadata: Optional[Dict[str, Any]] = None,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.metadata = dict(metadata or {})


class ConsumableWeightsDict:
    """
    Wrapper around a weights dictionary that allows marking keys as consumed
    to free memory during model loading.

    This reduces peak memory usage by deleting weight tensors from the dictionary
    after they have been copied to the model, rather than keeping all weights
    in memory until loading completes.

    Thread-safe: uses a lock to protect concurrent access. Iteration methods
    (keys, values, items, __iter__) return snapshot copies to allow safe
    concurrent iteration while other threads may modify the dictionary.
    """

    def __init__(self,
                 weights: Dict[str, Any],
                 metadata: Optional[Dict[str, Any]] = None):
        self._weights = weights
        self._lock = threading.Lock()
        self.metadata = dict(metadata or {})

    def __getitem__(self, key: str) -> Any:
        return self._weights[key]

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._weights[key] = value

    def __delitem__(self, key: str) -> None:
        with self._lock:
            del self._weights[key]

    def __contains__(self, key: str) -> bool:
        return key in self._weights

    def __len__(self) -> int:
        return len(self._weights)

    def __iter__(self) -> Iterator[str]:
        # Return iterator over a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return iter(list(self._weights.keys()))

    def keys(self):
        # Return a snapshot copy of keys to allow concurrent modification
        with self._lock:
            return list(self._weights.keys())

    def values(self):
        # Return a snapshot copy of values to allow concurrent modification
        with self._lock:
            return list(self._weights.values())

    def items(self) -> Iterator[Tuple[str, Any]]:
        # Return a snapshot copy of items to allow concurrent modification
        with self._lock:
            return list(self._weights.items())

    def get(self, key: str, default: Any = None) -> Any:
        return self._weights.get(key, default)

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            self._weights.update(other)

    def mark_consumed(self, prefix: str) -> int:
        """
        Delete all keys starting with the given prefix to free memory.

        Args:
            prefix: The prefix to match. Keys starting with "{prefix}." will be deleted.

        Returns:
            The number of keys deleted.

        Thread-safe: uses a lock to prevent concurrent modification issues.
        """
        with self._lock:
            keys_to_delete = [
                k for k in self._weights.keys() if k.startswith(prefix + ".")
            ]
            for key in keys_to_delete:
                del self._weights[key]
            return len(keys_to_delete)


class LazySafetensorsWeightsDict(ConsumableWeightsDict):
    """
    Lazy SafeTensors-backed weights dictionary.

    TensorRT-LLM's normal HF loader materializes every tensor in every shard
    before model loading starts. That is too expensive for exported 4o6 NVFP4
    checkpoints where the safetensors index can be hundreds of GB. This wrapper
    keeps only the index resident and reads tensors by key or by module prefix.
    """

    requires_serial_weight_loading = True

    def __init__(self,
                 weight_map: Dict[str, Union[str, _LazyWeightRecord]],
                 metadata: Optional[Dict[str, Any]] = None,
                 async_prefetch_files: Optional[List[str]] = None,
                 async_prefetch_workers: int = 0,
                 prefetch_state: Optional[_LazyPrefetchState] = None):
        super().__init__({}, metadata=metadata)
        self._weight_map: Dict[str, _LazyWeightRecord] = {
            key: (value, key) if isinstance(value, str) else value
            for key, value in weight_map.items()
        }
        self._prefetch_state = prefetch_state
        if self._prefetch_state is None and async_prefetch_files:
            self._prefetch_state = _LazyPrefetchState(async_prefetch_files,
                                                      async_prefetch_workers)
        self._lock = threading.RLock()
        self._timing_load_count = 0
        self._timing_load_sec = 0.0
        self._timing_load_bytes = 0
        self._timing_filter_count = 0
        self._timing_filter_sec = 0.0
        self._timing_last_report_count = 0
        if four_o_six_load_timing_enabled():
            four_o_six_load_timing_log(
                "lazy_dict_init",
                pending_keys=len(self._weight_map),
                loaded_keys=len(self._weights),
                metadata_keys=",".join(sorted(self.metadata.keys())))

    @classmethod
    def from_safetensors_files(
            cls, checkpoint_dir: str,
            weight_files: List[str],
            metadata: Optional[Dict[str, Any]] = None,
            async_prefetch_files: Optional[List[str]] = None,
            async_prefetch_workers: int = 0
    ) -> "LazySafetensorsWeightsDict":
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            start = four_o_six_load_timing_now()
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = {
                key: os.path.join(checkpoint_dir, filename)
                for key, filename in index.get("weight_map", {}).items()
            }
            four_o_six_load_timing_log(
                "lazy_index_read",
                checkpoint_dir=checkpoint_dir,
                index_file=os.path.basename(index_path),
                tensor_keys=len(weight_map),
                elapsed_sec=four_o_six_load_timing_elapsed(start))
            return cls(weight_map,
                       metadata=metadata,
                       async_prefetch_files=async_prefetch_files,
                       async_prefetch_workers=async_prefetch_workers)

        from safetensors import safe_open

        start = four_o_six_load_timing_now()
        weight_map = {}
        for file_path in weight_files:
            file_start = four_o_six_load_timing_now()
            with safe_open(file_path, framework="pt", device="cpu") as f:
                keys = list(f.keys())
                for key in keys:
                    weight_map[key] = file_path
            elapsed_sec = four_o_six_load_timing_elapsed(file_start)
            if four_o_six_load_timing_should_log(
                    elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_FILE_THRESHOLD_SEC",
                    0.25):
                four_o_six_load_timing_log(
                    "lazy_scan_file",
                    file=os.path.basename(file_path),
                    tensor_keys=len(keys),
                    elapsed_sec=elapsed_sec)
        four_o_six_load_timing_log(
            "lazy_scan_files_done",
            files=len(weight_files),
            tensor_keys=len(weight_map),
            elapsed_sec=four_o_six_load_timing_elapsed(start))
        return cls(weight_map,
                   metadata=metadata,
                   async_prefetch_files=async_prefetch_files,
                   async_prefetch_workers=async_prefetch_workers)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._weights:
                return self._weights[key]
            file_path, tensor_key = self._weight_map[key]

        from safetensors import safe_open

        wait_sec = 0.0
        wait_for_prefetch = _env_flag(
            "TRTLLM_4O6_LAZY_PREFETCH_WAIT_FOR_TENSOR", False)
        if self._prefetch_state is not None and wait_for_prefetch:
            wait_sec = self._prefetch_state.wait_for_file(file_path)
            if four_o_six_load_timing_should_log(
                    wait_sec, "TRTLLM_4O6_LOAD_TIMING_PREFETCH_WAIT_THRESHOLD_SEC",
                    0.05):
                four_o_six_load_timing_log(
                    "lazy_async_prefetch_wait",
                    file=os.path.basename(file_path),
                    elapsed_sec=wait_sec)
        elif (self._prefetch_state is not None
              and self._prefetch_state.should_log_skip_wait(file_path)):
            four_o_six_load_timing_log(
                "lazy_async_prefetch_skip_wait",
                file=os.path.basename(file_path),
                reason="tensor_materialize")

        start = four_o_six_load_timing_now()
        with safe_open(file_path, framework="pt", device="cpu") as f:
            tensor = f.get_tensor(tensor_key)
        self._record_materialized_tensor(key, file_path, tensor_key, tensor,
                                         start)
        return tensor

    def _record_materialized_tensor(self, key: str, file_path: str,
                                    tensor_key: str, tensor: Any,
                                    start: float) -> None:
        if not four_o_six_load_timing_enabled():
            return
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        nbytes = four_o_six_load_timing_tensor_nbytes(tensor)
        with self._lock:
            self._timing_load_count += 1
            self._timing_load_sec += elapsed_sec
            self._timing_load_bytes += nbytes
            load_count = self._timing_load_count

        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_TENSOR_THRESHOLD_SEC",
                0.25):
            four_o_six_load_timing_log(
                "lazy_tensor_load",
                key=key,
                source_key=tensor_key,
                file=os.path.basename(file_path),
                nbytes=nbytes,
                elapsed_sec=elapsed_sec)

        report_every = int(
            os.environ.get("TRTLLM_4O6_LOAD_TIMING_REPORT_EVERY", "1000"))
        if report_every > 0 and load_count % report_every == 0:
            self.log_timing_summary("lazy_periodic_summary")

    def __setitem__(self, key: str, value: Any) -> None:
        with self._lock:
            self._weights[key] = value
            self._weight_map.pop(key, None)

    def __delitem__(self, key: str) -> None:
        with self._lock:
            deleted = False
            if key in self._weights:
                del self._weights[key]
                deleted = True
            if key in self._weight_map:
                del self._weight_map[key]
                deleted = True
            if not deleted:
                raise KeyError(key)

    def __contains__(self, key: str) -> bool:
        with self._lock:
            return key in self._weights or key in self._weight_map

    def __len__(self) -> int:
        with self._lock:
            return len(self._weights) + len(self._weight_map)

    def __iter__(self) -> Iterator[str]:
        return iter(self.keys())

    def keys(self):
        with self._lock:
            return list(self._weights.keys()) + list(self._weight_map.keys())

    def values(self):
        return [value for _, value in self.items()]

    def items(self) -> Iterator[Tuple[str, Any]]:
        from safetensors import safe_open

        start = four_o_six_load_timing_now()
        with self._lock:
            loaded_items = list(self._weights.items())
            weight_map = dict(self._weight_map)

        grouped_keys: Dict[str, List[Tuple[str, str]]] = {}
        for key, (file_path, tensor_key) in weight_map.items():
            grouped_keys.setdefault(file_path, []).append((key, tensor_key))

        if four_o_six_load_timing_enabled() and weight_map:
            four_o_six_load_timing_log(
                "lazy_items_materialize_all_start",
                pending_keys=len(weight_map),
                loaded_keys=len(loaded_items),
                files=len(grouped_keys))

        items = loaded_items
        for file_path, keys_and_tensor_keys in grouped_keys.items():
            file_start = four_o_six_load_timing_now()
            wait_sec = 0.0
            if self._prefetch_state is not None:
                wait_sec = self._prefetch_state.wait_for_file(file_path)
            file_bytes = 0
            file_items = []
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key, tensor_key in keys_and_tensor_keys:
                    tensor = f.get_tensor(tensor_key)
                    file_bytes += four_o_six_load_timing_tensor_nbytes(tensor)
                    file_items.append((key, tensor))
            items.extend(file_items)
            elapsed_sec = four_o_six_load_timing_elapsed(file_start)
            if four_o_six_load_timing_should_log(
                    elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_FILE_THRESHOLD_SEC",
                    0.25):
                four_o_six_load_timing_log(
                    "lazy_items_file",
                    file=os.path.basename(file_path),
                    tensor_keys=len(keys_and_tensor_keys),
                    nbytes=file_bytes,
                    prefetch_wait_sec=wait_sec,
                    elapsed_sec=elapsed_sec)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        four_o_six_load_timing_log(
            "lazy_items_materialize_all_done",
            pending_keys=len(weight_map),
            files=len(grouped_keys),
            elapsed_sec=elapsed_sec)
        return items

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            return default
        return self[key]

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            for key, value in other.items():
                self._weights[key] = value
                self._weight_map.pop(key, None)

    def shutdown_prefetch(self) -> None:
        if self._prefetch_state is not None:
            self._prefetch_state.shutdown()

    def rename_by_regex(
            self, pattern_mapping: Dict[str,
                                        str]) -> "LazySafetensorsWeightsDict":
        start = four_o_six_load_timing_now()
        renamed_weight_map = {}
        renamed_weights = {}
        with self._lock:
            for key, value in self._weights.items():
                renamed_weights[_rename_key_by_regex(key,
                                                      pattern_mapping)] = value
            for key, record in self._weight_map.items():
                renamed_weight_map[_rename_key_by_regex(
                    key, pattern_mapping)] = record

        renamed = LazySafetensorsWeightsDict(
            renamed_weight_map,
            metadata=self.metadata,
            prefetch_state=self._prefetch_state)
        renamed.update(renamed_weights)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_RENAME_THRESHOLD_SEC",
                0.05):
            four_o_six_load_timing_log(
                "lazy_rename_by_regex",
                patterns=len(pattern_mapping),
                pending_keys=len(renamed_weight_map),
                loaded_keys=len(renamed_weights),
                elapsed_sec=elapsed_sec)
        return renamed

    def filter(self, prefix: str) -> Dict[str, Any]:
        start = four_o_six_load_timing_now()
        prefix_with_separator = prefix + "."
        prefix_len = len(prefix_with_separator)
        with self._lock:
            filtered_weight_map = {
                key[prefix_len:]: record
                for key, record in self._weight_map.items()
                if key.startswith(prefix_with_separator)
            }
            filtered_weights = {
                key[prefix_len:]: value
                for key, value in self._weights.items()
                if key.startswith(prefix_with_separator)
            }

        filtered = LazySafetensorsWeightsDict(
            filtered_weight_map,
            metadata=self.metadata,
            prefetch_state=self._prefetch_state)
        filtered.update(filtered_weights)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_enabled():
            with self._lock:
                self._timing_filter_count += 1
                self._timing_filter_sec += elapsed_sec
                total_pending = len(self._weight_map)
                total_loaded = len(self._weights)
            if four_o_six_load_timing_should_log(
                    elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_FILTER_THRESHOLD_SEC",
                    0.05):
                four_o_six_load_timing_log(
                    "lazy_filter",
                    prefix=prefix,
                    matched_pending=len(filtered_weight_map),
                    matched_loaded=len(filtered_weights),
                    total_pending=total_pending,
                    total_loaded=total_loaded,
                    elapsed_sec=elapsed_sec)
        return filtered

    def mark_consumed(self, prefix: str) -> int:
        start = four_o_six_load_timing_now()
        prefix_with_separator = prefix + "."
        with self._lock:
            keys_to_delete = [
                key for key in self.keys()
                if key.startswith(prefix_with_separator)
            ]
            for key in keys_to_delete:
                self._weights.pop(key, None)
                self._weight_map.pop(key, None)
            deleted = len(keys_to_delete)
            remaining_pending = len(self._weight_map)
            remaining_loaded = len(self._weights)
        elapsed_sec = four_o_six_load_timing_elapsed(start)
        if four_o_six_load_timing_should_log(
                elapsed_sec, "TRTLLM_4O6_LOAD_TIMING_CONSUME_THRESHOLD_SEC",
                0.05):
            four_o_six_load_timing_log(
                "lazy_mark_consumed",
                prefix=prefix,
                deleted=deleted,
                remaining_pending=remaining_pending,
                remaining_loaded=remaining_loaded,
                elapsed_sec=elapsed_sec)
        return deleted

    def log_timing_summary(self, event: str) -> None:
        if not four_o_six_load_timing_enabled():
            return
        with self._lock:
            load_count = self._timing_load_count
            load_sec = self._timing_load_sec
            load_bytes = self._timing_load_bytes
            filter_count = self._timing_filter_count
            filter_sec = self._timing_filter_sec
            pending_keys = len(self._weight_map)
            loaded_keys = len(self._weights)
        mb = load_bytes / (1024**2)
        mb_per_sec = mb / load_sec if load_sec > 0 else 0.0
        four_o_six_load_timing_log(
            event,
            materialized_tensors=load_count,
            materialized_mb=mb,
            materialize_sec=load_sec,
            materialize_mb_per_sec=mb_per_sec,
            filter_calls=filter_count,
            filter_sec=filter_sec,
            pending_keys=pending_keys,
            loaded_keys=loaded_keys)


def _rename_key_by_regex(key: str, pattern_mapping: Dict[str, str]) -> str:
    for pattern, replacement in pattern_mapping.items():
        if re.match(pattern, key):
            return re.sub(pattern, replacement, key)
    return key


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(self, checkpoint_dir: str, mapping: Mapping,
                     **kwargs) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.
            **kwargs: Optional format-specific loader arguments.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
