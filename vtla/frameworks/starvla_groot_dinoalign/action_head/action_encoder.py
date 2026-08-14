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
#
# Vendored verbatim from
# starVLA/model/modules/action_model/flow_matching_head/action_encoder.py
# Only ``swish`` and ``SinusoidalPositionalEncoding`` are consumed by the GR00T
# head (the head defines its own ``ActionEncoder``), but the module is copied as
# a whole to keep it self-contained.

import torch
import torch.nn as nn


def swish(x):
    return x * torch.sigmoid(x)


class SinusoidalPositionalEncoding(nn.Module):
    """
    Produces a sinusoidal encoding of shape (B, T, w)
    given timesteps of shape (B, T).
    """

    def __init__(self, embedding_dim):
        super().__init__()
        self.embedding_dim = embedding_dim

    def forward(self, timesteps):
        # timesteps: shape (B, T)
        # We'll compute sin/cos frequencies across dim T
        timesteps = timesteps.float()  # ensure float

        B, T = timesteps.shape
        device = timesteps.device

        half_dim = self.embedding_dim // 2
        # typical log space frequencies for sinusoidal encoding
        exponent = -torch.arange(half_dim, dtype=torch.float, device=device) * (
            torch.log(torch.tensor(10000.0)) / half_dim
        )
        # Expand timesteps to (B, T, 1) then multiply
        freqs = timesteps.unsqueeze(-1) * exponent.exp()  # (B, T, half_dim)

        sin = torch.sin(freqs)
        cos = torch.cos(freqs)
        enc = torch.cat([sin, cos], dim=-1)  # (B, T, w)

        return enc


class ActionEncoder(nn.Module):
    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size

        # W1: R^{w x d}, W2: R^{w x 2w}, W3: R^{w x w}
        self.W1 = nn.Linear(action_dim, hidden_size)  # (d -> w)
        self.W2 = nn.Linear(2 * hidden_size, hidden_size)  # (2w -> w)
        self.W3 = nn.Linear(hidden_size, hidden_size)  # (w -> w)

        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            # shape (B,) => (B,T)
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.W1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.W2(x))
        x = self.W3(x)
        return x
