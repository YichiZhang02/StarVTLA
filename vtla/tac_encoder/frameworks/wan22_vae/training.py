"""Wan2.2 tactile VAE post-training recipe."""

from __future__ import annotations

from pathlib import Path

from torch import Tensor, nn

from vtla.tac_encoder.eval import save_full_reconstruction_visualization

from ...common.training import (
    StepOutput,
    TrainingRecipe,
    require_fully_pretrained,
    split_state_dict,
)
from .model import Wan22VAEBackbone


class Wan22VAETrainingRecipe(TrainingRecipe):
    model_id = "wan22_vae"
    objective = "full_frame_vae_reconstruction"
    default_weight_decay = 0.0
    encoder_prefixes = Wan22VAEBackbone.checkpoint_prefixes

    def build_model(self, args) -> nn.Module:
        model = Wan22VAEBackbone(
            num_frames=args.num_frames,
            image_size=args.image_size,
            latent_dim=args.wan22_latent_dim,
            base_dim=args.wan22_base_dim,
            decoder_base_dim=args.wan22_decoder_base_dim,
            kl_weight=args.vae_kl_weight,
            pretrained_path=args.pretrained_path,
        )
        require_fully_pretrained(model, model.load_report)
        return model

    def step(self, model: nn.Module, images: Tensor, args) -> StepOutput:
        output = model(images, args.mask_ratio)
        return StepOutput(
            loss=output.loss,
            metrics={
                "loss": output.loss.detach(),
                "reconstruction_l1": output.reconstruction_loss.detach(),
                "kl": output.kl_loss.detach(),
            },
        )

    def encoder_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[0]

    def trainer_state_dict(self, model: nn.Module) -> dict[str, Tensor]:
        return split_state_dict(model, self.encoder_prefixes)[1]

    def save_visualization(
        self, model, dataset, indices, destination: Path, device, args, autocast_dtype
    ) -> None:
        save_full_reconstruction_visualization(
            model, dataset, indices, destination, device, autocast_dtype
        )


RECIPE = Wan22VAETrainingRecipe()
