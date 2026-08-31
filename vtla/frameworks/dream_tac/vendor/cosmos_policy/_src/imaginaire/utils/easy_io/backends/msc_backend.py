# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""MSC storage backend: uses NVIDIA multistorageclient when available.

If ``multistorageclient`` is missing or incompatible (e.g. no ``StorageClient``),
imports fall back to :class:`Boto3Backend` so training and local I/O still work.
"""

try:
    from multistorageclient import StorageClient as _StorageClient  # noqa: F401
    from multistorageclient import StorageClientConfig as _StorageClientConfig  # noqa: F401
    from multistorageclient.types import Range as _Range  # noqa: F401
except (ImportError, ModuleNotFoundError):
    from cosmos_policy._src.imaginaire.utils.easy_io.backends.boto3_backend import Boto3Backend as MSCBackend
else:
    from cosmos_policy._src.imaginaire.utils.easy_io.backends.msc_backend_impl import MSCBackend

__all__ = ["MSCBackend"]
