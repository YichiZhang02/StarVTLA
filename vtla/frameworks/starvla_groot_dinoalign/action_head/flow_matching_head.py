# Copyright 2025 NVIDIA Corp. and affiliates. All rights reserved.
# Modified by [Junqiu YU / Fudan University] in [2025].
# Action repeat is inspired by CogACT.
#
# Ported into vtla from
# starVLA/model/modules/action_model/GR00T_ActionHeader.py
#
# The only behavioural change versus the starVLA original is the constructor:
# instead of reading the OmegaConf tree ``full_config.framework.action_model.*``
# it takes a plain :class:`ActionHeadConfig` dataclass that the vtla policy builds
# from its own ``StarvlaGrootConfig``. The forward / predict_action math is
# unchanged.

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Beta

from .action_encoder import SinusoidalPositionalEncoding, swish  # noqa: F401  (kept for parity)
from .cross_attention_dit import DiT


@dataclass
class ActionHeadConfig:
    """Plain config consumed by :class:`FlowmatchingActionHead`.

    Mirrors the fields the starVLA head used to read from
    ``cfg.framework.action_model``. ``cross_attention_dim`` inside
    ``diffusion_model_cfg`` MUST be set to the VLM hidden size by the caller
    before constructing the head.
    """

    # DiT architecture preset: "DiT-B" | "DiT-L"
    action_model_type: str = "DiT-B"
    # Intermediate MLP width for state_encoder / action_decoder.
    hidden_size: int = 1024
    # Action / state dimensions.
    action_dim: int = 7
    state_dim: int = 7
    # Chunk length predicted by the head.
    action_horizon: int = 8
    # Denoising (Euler) steps at inference.
    num_inference_timesteps: int = 4
    # Learnable planning query tokens prepended before the action sequence.
    num_target_vision_tokens: int = 32
    # Positional embedding over the action sequence.
    add_pos_embed: bool = True
    max_seq_len: int = 1024
    # Flow-matching noise schedule (Beta distribution).
    noise_beta_alpha: float = 1.5
    noise_beta_beta: float = 1.0
    noise_s: float = 0.999
    num_timestep_buckets: int = 1000
    # DiT transformer sub-config. cross_attention_dim is aligned to VLM hidden at runtime.
    diffusion_model_cfg: dict = field(default_factory=dict)


# Base DiT shapes, identical to the starVLA original.
DiTConfig = {
    "DiT-B": {"input_embedding_dim": 768, "attention_head_dim": 64, "num_attention_heads": 12},
    "DiT-L": {"input_embedding_dim": 1536, "attention_head_dim": 48, "num_attention_heads": 32},
}


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        return self.layer2(F.relu(self.layer1(x)))


class ActionEncoder(nn.Module):
    """Per-timestep action encoder with sinusoidal time conditioning.

    (This is the head-local encoder used by the GR00T head; distinct from the
    embodiment-aware one in ``action_encoder.py``.)
    """

    def __init__(self, action_dim, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_dim = action_dim
        self.layer1 = nn.Linear(action_dim, hidden_size)
        self.layer2 = nn.Linear(2 * hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.pos_encoding = SinusoidalPositionalEncoding(hidden_size)

    def forward(self, actions, timesteps):
        """
        actions:   shape (B, T, action_dim)
        timesteps: shape (B,)  -- a single scalar per batch item
        returns:   shape (B, T, hidden_size)
        """
        B, T, _ = actions.shape

        if timesteps.dim() == 1 and timesteps.shape[0] == B:
            timesteps = timesteps.unsqueeze(1).expand(-1, T)
        else:
            raise ValueError("Expected `timesteps` to have shape (B,) so we can replicate across T.")

        a_emb = self.layer1(actions)
        tau_emb = self.pos_encoding(timesteps).to(dtype=a_emb.dtype)
        x = torch.cat([a_emb, tau_emb], dim=-1)
        x = swish(self.layer2(x))
        x = self.layer3(x)
        return x


class FlowmatchingActionHead(nn.Module):
    """GR00T flow-matching (DiT) action head.

    Cross-attends a noised action trajectory (+ optional state + learnable query
    tokens) over the VLM hidden states and regresses the flow-matching velocity.
    """

    def __init__(self, config: ActionHeadConfig):
        super().__init__()
        self.config = config

        action_model_cfg = DiTConfig[config.action_model_type]
        self.input_embedding_dim = action_model_cfg["input_embedding_dim"]

        diffusion_model_cfg = {**action_model_cfg, **dict(config.diffusion_model_cfg)}
        self.model = DiT(**diffusion_model_cfg)

        self.action_horizon = int(config.action_horizon)
        self.action_dim = config.action_dim
        self.num_inference_timesteps = config.num_inference_timesteps
        self.hidden_size = config.hidden_size

        self.state_encoder = (
            MLP(
                input_dim=config.state_dim,
                hidden_dim=self.hidden_size,
                output_dim=self.input_embedding_dim,
            )
            if config.state_dim
            else None
        )

        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            hidden_size=self.input_embedding_dim,
        )
        self.action_decoder = MLP(
            input_dim=self.model.config.output_dim,
            hidden_dim=self.hidden_size,
            output_dim=self.action_dim,
        )

        self.future_tokens = nn.Embedding(config.num_target_vision_tokens, self.input_embedding_dim)
        nn.init.normal_(self.future_tokens.weight, mean=0.0, std=0.02)

        if config.add_pos_embed:
            self.position_embedding = nn.Embedding(config.max_seq_len, self.input_embedding_dim)
            nn.init.normal_(self.position_embedding.weight, mean=0.0, std=0.02)

        self.beta_dist = Beta(config.noise_beta_alpha, config.noise_beta_beta)
        self.num_timestep_buckets = config.num_timestep_buckets

    def sample_time(self, batch_size, device, dtype):
        sample = self.beta_dist.sample([batch_size]).to(device, dtype=dtype).clamp(max=self.config.noise_s)
        return (self.config.noise_s - sample) / self.config.noise_s

    def forward(
        self,
        vl_embs: torch.Tensor,
        actions: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ):
        """
        vl_embs: shape (B, seq_length, feature_dim)
        actions: shape (B, action_horizon, action_dim)
        Returns the scalar flow-matching MSE loss.
        """
        device = vl_embs.device

        # Embed noised action trajectory.
        noise = torch.randn(actions.shape, device=actions.device, dtype=actions.dtype)
        t = self.sample_time(actions.shape[0], device=actions.device, dtype=actions.dtype)
        t = t[:, None, None]  # shape (B,1,1) for broadcast

        noisy_trajectory = (1 - t) * noise + t * actions
        velocity = actions - noise

        t_discretized = (t[:, 0, 0] * self.num_timestep_buckets).long()
        action_features = self.action_encoder(noisy_trajectory, t_discretized)

        state_features = self.state_encoder(state) if state is not None else None

        if self.config.add_pos_embed:
            pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
            pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
            action_features = action_features + pos_embs

        future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
        sa_embs = (
            torch.cat((state_features, future_tokens, action_features), dim=1)
            if state_features is not None
            else torch.cat((future_tokens, action_features), dim=1)
        )

        model_output = self.model(
            hidden_states=sa_embs,
            encoder_hidden_states=vl_embs,
            encoder_attention_mask=encoder_attention_mask,
            timestep=t_discretized,
            return_all_hidden_states=False,
        )
        pred = self.action_decoder(model_output)
        pred_actions = pred[:, -actions.shape[1] :]

        # Per-sample loss (B,) so the caller can choose mean / weighting.
        loss = ((pred_actions - velocity) ** 2).mean(dim=(1, 2))
        return loss

    @torch.no_grad()
    def predict_action(
        self,
        vl_embs: torch.Tensor,
        state: torch.Tensor = None,
        encoder_attention_mask=None,
    ) -> torch.Tensor:
        batch_size = vl_embs.shape[0]
        device = vl_embs.device
        actions = torch.randn(
            size=(batch_size, self.action_horizon, self.action_dim),
            dtype=vl_embs.dtype,
            device=device,
        )

        num_steps = self.num_inference_timesteps
        dt = 1.0 / num_steps

        state_features = self.state_encoder(state) if state is not None else None

        for t in range(num_steps):
            t_cont = t / float(num_steps)
            t_discretized = int(t_cont * self.num_timestep_buckets)

            timesteps_tensor = torch.full(size=(batch_size,), fill_value=t_discretized, device=device)
            action_features = self.action_encoder(actions, timesteps_tensor)
            if self.config.add_pos_embed:
                pos_ids = torch.arange(action_features.shape[1], dtype=torch.long, device=device)
                pos_embs = self.position_embedding(pos_ids).unsqueeze(0)
                action_features = action_features + pos_embs

            future_tokens = self.future_tokens.weight.unsqueeze(0).expand(vl_embs.shape[0], -1, -1)
            sa_embs = (
                torch.cat((state_features, future_tokens, action_features), dim=1)
                if state_features is not None
                else torch.cat((future_tokens, action_features), dim=1)
            )

            model_output = self.model(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                encoder_attention_mask=encoder_attention_mask,
                timestep=timesteps_tensor,
            )
            pred = self.action_decoder(model_output)
            pred_velocity = pred[:, -self.action_horizon :]
            actions = actions + dt * pred_velocity
        return actions

    @property
    def device(self):
        return next(iter(self.parameters())).device

    @property
    def dtype(self):
        return next(iter(self.parameters())).dtype
