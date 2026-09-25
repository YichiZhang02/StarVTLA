"""Trainable tactile predictor and identity-initialized action modulation."""

import json
from pathlib import Path

import torch
from torch import nn
from transformers import DynamicCache

from tacmind0.tactile.encoder import FrozenTactileEncoder
from tacmind0.tactile.lewm.module import MLP, ARPredictor
from tacmind0.tactile.lewm.multi_source_action import MultiSourceActionEmbedder

from .stream import SUPPORTED_CACHE_VERSIONS


class ActionBridge(nn.Module):
    def __init__(self, config):
        super().__init__()
        for name in ("q01", "q99", "mean", "std", "relative_mask"):
            value = torch.tensor(config[name], dtype=torch.float32)
            if value.shape != (8,) or not torch.isfinite(value).all():
                raise ValueError(f"Invalid action bridge {name}")
            self.register_buffer(name, value)
        self.normalization = config.get("normalization", "std_plus_epsilon")
        if self.normalization not in ("std", "std_plus_epsilon"):
            raise ValueError("Unsupported world action normalization")
        if (
            (self.std < 0).any()
            or (self.q99 < self.q01).any()
            or (self.normalization == "std" and (self.std <= 0).any())
        ):
            raise ValueError("Invalid action statistics")

    def physical(self, action, state):
        action = action[..., :8].float()
        value = (action + 1) / 2 * (self.q99 - self.q01 + 1e-6) + self.q01
        value = torch.where((self.q01 == 0) & (self.q99 == 0), 0.0, value)
        return value + state.float()[:, None, :] * self.relative_mask

    def normalize_physical(self, action):
        denominator = self.std if self.normalization == "std" else self.std + 1e-6
        return (action - self.mean) / denominator

    def forward(self, action, state):
        return self.normalize_physical(self.physical(action, state))


class TacFRS(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        if (
            config.get("version", 1) != 1
            or config.get("cache_version", "five-phase-sequence-v1")
            not in SUPPORTED_CACHE_VERSIONS
        ):
            raise ValueError("Unsupported Tac-FRS checkpoint/cache version")
        if config.get("vision_residual", False):
            raise ValueError("Tac-FRS only supports tactile/action world conditioning")
        spec = config["world_model"]

        def kwargs(name):
            return {
                k: v for k, v in spec[name].items() if k not in ("_target_", "norm_fn")
            }

        self.predictor = ARPredictor(**kwargs("predictor"))
        self.action_encoder = MultiSourceActionEmbedder(**kwargs("action_encoder"))
        self.projector = MLP(**kwargs("projector"), norm_fn=nn.BatchNorm1d)
        self.pred_proj = MLP(**kwargs("pred_proj"), norm_fn=nn.BatchNorm1d)
        self.encoder = FrozenTactileEncoder(config["backbone"])
        self.bridge = ActionBridge(config["action_bridge"])
        self.modulators = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(384, 384),
                    nn.SiLU(),
                    nn.Linear(384, 4 * config["hidden_size"]),
                )
                for _ in range(config["layers"])
            ]
        )
        for net in self.modulators:
            nn.init.zeros_(net[-1].weight)
            nn.init.zeros_(net[-1].bias)
        self.requires_grad_(False)
        self.predictor.requires_grad_(True)
        self.modulators.requires_grad_(True)
        self.train(False)

    def train(self, mode=True):
        # Deterministic rollout; gradients do not require dropout or BN train mode.
        super().train(False)
        return self

    def load_world(self, state):
        prefix = "jepa.encoder."
        self.encoder.encoder.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
            strict=True,
        )
        for name in ("predictor", "action_encoder", "projector", "pred_proj"):
            prefix = "jepa." + name + "."
            getattr(self, name).load_state_dict(
                {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)},
                strict=True,
            )

    def rollout(self, latent, actions):
        if latent.ndim != 2 or latent.shape[1] != 384:
            raise ValueError("Tac-FRS latent must be (B,384)")
        if (
            actions.ndim != 3
            or actions.shape[0] != latent.shape[0]
            or actions.shape[2] != 8
            or actions.shape[1] < 1
        ):
            raise ValueError(
                "Tac-FRS actions must be (B,H,8), aligned with latent batch"
            )
        if not torch.isfinite(latent).all() or not torch.isfinite(actions).all():
            raise ValueError("Tac-FRS latent and actions must be finite")
        horizon = actions.shape[1]
        context = latent[:, None]
        act_emb = self.action_encoder(actions.float())
        results = []
        window = min(
            self.config["world_model"]["history_size"],
            self.predictor.pos_embedding.shape[1],
        )
        for t in range(horizon):
            length = min(window, t + 1)
            predicted = self.predictor(
                context[:, -length:], act_emb[:, t + 1 - length : t + 1]
            )[:, -1]
            predicted = self.pred_proj(predicted)
            results.append(predicted)
            context = torch.cat((context, predicted[:, None]), dim=1)
        return torch.stack(results, dim=1)

    def generate(self, policy, inputs, latent, state, *, checkpoint_steps=False):
        batch = inputs["input_ids"].shape[0]
        if state.shape != (batch, 8) or latent.shape != (batch, 384):
            raise ValueError("Tac-FRS state/latent must match the policy batch")
        if not torch.isfinite(state).all() or not torch.isfinite(latent).all():
            raise ValueError("Tac-FRS state/latent must be finite")
        if policy.config.chunk_size != self.config["chunk_size"]:
            raise ValueError("Policy and Tac-FRS chunk size differ")
        if (
            policy.config.tactile_config.get("encoder_initialization")
            != self.config["encoder_provenance"]
        ):
            raise ValueError("Policy and Tac-FRS encoder provenance differ")
        if policy.config.use_suffix_graph:
            raise ValueError("Tac-FRS requires use_suffix_graph=false")
        if inputs.get("reference_action") is not None:
            raise ValueError(
                "Tac-FRS generates its own candidate; reference_action is not accepted"
            )
        with torch.no_grad():
            cache, hidden = policy._compute_prefix_cache(
                **{
                    k: inputs.get(k)
                    for k in (
                        "input_ids",
                        "attention_mask",
                        "pixel_values",
                        "token_type_ids",
                        "history_pixel_values",
                        "history_mask",
                        "tactile_pixel_values",
                    )
                },
                cache_cls=DynamicCache,
            )
            flow = dict(
                input_ids=inputs["input_ids"],
                kv_cache=cache,
                prefix_len=hidden.shape[1],
                diffusion_steps=inputs.get("diffusion_steps", 10),
                action_mask=inputs["action_mask"],
            )
            noise = torch.randn(
                len(state),
                policy.config.chunk_size,
                policy.config.action_dim,
                device=state.device,
                dtype=policy.model.action_in_proj.weight.dtype,
            )
            candidate = policy._integrate_action_flow(noise, **flow)
            reverse_noise = policy._integrate_action_flow(
                candidate, **flow, reverse=True
            )
            world_actions = self.bridge(candidate, state)
        prediction = self.rollout(latent.detach(), world_actions)
        modulations = torch.stack([net(prediction) for net in self.modulators])
        action = policy._integrate_action_flow(
            reverse_noise.detach(),
            **flow,
            tactile_modulations=modulations,
            checkpoint_steps=checkpoint_steps,
        )
        return action, prediction

    def save(self, path):
        path = Path(path)
        (path / "tac_frs.json").write_text(json.dumps(self.config, indent=2))
        torch.save(self.state_dict(), path / "tac_frs.pt")

    @classmethod
    def load(cls, path, device):
        path = Path(path)
        model = cls(json.loads((path / "tac_frs.json").read_text()))
        model.load_state_dict(
            torch.load(path / "tac_frs.pt", map_location="cpu", weights_only=True),
            strict=True,
        )
        return model.to(device).eval()


def masked_mse(prediction, target, mask):
    expanded = torch.broadcast_to(mask.bool(), prediction.shape)
    if not expanded.any():
        return prediction.sum() * 0
    return (prediction.float() - target.float()).square()[expanded].mean()
