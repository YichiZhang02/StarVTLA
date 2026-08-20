# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

"""
tac_infra 部署层 (deployment)

自包含的硬件控制 + 数据采集 / 策略推理栈，移植自 lerobot_tactile_ws，
但与之完全独立：
- 硬件层 (robots / teleoperators / cameras / motors / sdk) 全部内置于本目录；
- 策略 / 数据集 / 处理管线复用本仓库的 `vtla` 包；
- 触觉传感器以 tactile_u16_fixed_v1 采集，并作为无损 FFV1 MKV video feature 保存。
"""
