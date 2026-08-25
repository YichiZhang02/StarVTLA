#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from vtla.engine.utils.import_utils import require_package

require_package("datasets", extra="dataset")
require_package("av", extra="dataset")

from .factory import make_dataset, resolve_delta_timestamps
from .sampler import EpisodeAwareSampler, MixtureSampler

# Keep this package entrypoint light. Training only needs the dataset factory
# and sampler; heavier dataset tools should be imported from their modules.

__all__ = [
    "EpisodeAwareSampler",
    "MixtureSampler",
    "make_dataset",
    "resolve_delta_timestamps",
]
