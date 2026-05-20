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
import threading
from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator, List, Tuple, Union

from tensorrt_llm.mapping import Mapping


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

    def __init__(self, weights: Dict[str, Any]):
        self._weights = weights
        self._lock = threading.Lock()

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

    def __init__(self, weight_map: Dict[str, str]):
        super().__init__({})
        self._weight_map = dict(weight_map)
        self._lock = threading.RLock()

    @classmethod
    def from_safetensors_files(
            cls, checkpoint_dir: str,
            weight_files: List[str]) -> "LazySafetensorsWeightsDict":
        index_path = os.path.join(checkpoint_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path, "r", encoding="utf-8") as f:
                index = json.load(f)
            weight_map = {
                key: os.path.join(checkpoint_dir, filename)
                for key, filename in index.get("weight_map", {}).items()
            }
            return cls(weight_map)

        from safetensors import safe_open

        weight_map = {}
        for file_path in weight_files:
            with safe_open(file_path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    weight_map[key] = file_path
        return cls(weight_map)

    def __getitem__(self, key: str) -> Any:
        with self._lock:
            if key in self._weights:
                return self._weights[key]
            file_path = self._weight_map[key]

        from safetensors import safe_open

        with safe_open(file_path, framework="pt", device="cpu") as f:
            return f.get_tensor(key)

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
        return [self[key] for key in self.keys()]

    def items(self) -> Iterator[Tuple[str, Any]]:
        return [(key, self[key]) for key in self.keys()]

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self:
            return default
        return self[key]

    def update(self, other: Dict[str, Any]) -> None:
        with self._lock:
            for key, value in other.items():
                self._weights[key] = value
                self._weight_map.pop(key, None)

    def filter(self, prefix: str) -> Dict[str, Any]:
        prefix_with_separator = prefix + "."
        keys = [
            key for key in self.keys() if key.startswith(prefix_with_separator)
        ]
        return {
            key[len(prefix_with_separator):]: self[key]
            for key in keys
        }

    def mark_consumed(self, prefix: str) -> int:
        prefix_with_separator = prefix + "."
        with self._lock:
            keys_to_delete = [
                key for key in self.keys()
                if key.startswith(prefix_with_separator)
            ]
            for key in keys_to_delete:
                self._weights.pop(key, None)
                self._weight_map.pop(key, None)
            return len(keys_to_delete)


class BaseWeightLoader(ABC):

    @abstractmethod
    def load_weights(
            self, checkpoint_dir: str,
            mapping: Mapping) -> Union[Dict[str, Any], ConsumableWeightsDict]:
        """
        Loads weights from a checkpoint directory.

        Args:
            checkpoint_dir: A path to the checkpoint directory.
            mapping: A mapping object containing the distributed configuration.

        Returns:
            A dictionary (or ConsumableWeightsDict) where keys are tensor names
            and values are the tensors.
        """

    def cleanup(self) -> None:
        pass
