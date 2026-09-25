"""Load local TacMind0 assets and run one real native inference on synthetic sensors."""

import argparse

import torch

from vtla.engine.configs import FeatureType, PolicyFeature
from vtla.frameworks.tacmind0.configuration_tacmind0 import TacMind0Config
from vtla.frameworks.tacmind0.modeling_tacmind0 import TacMind0Policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    keys = ["observation.images.cam_finger0", "observation.images.cam_finger1"]
    wrist = "observation.images.wrist"
    visual = PolicyFeature(FeatureType.VISUAL, (3, 42, 42))
    config = TacMind0Config(
        device=args.device, tactile_keys=keys, wrist_camera_keys=[wrist],
        top_camera_keys=[], state_mode="none", action_mode="absolute_joint",
        num_inference_steps=1,
        input_features={**{key: visual for key in keys}, wrist: visual},
        output_features={"action": PolicyFeature(FeatureType.ACTION, (8,))},
    )
    policy = TacMind0Policy(config).to(args.device).eval()
    batch = {key: torch.rand(1, 8, 3, 42, 42) for key in keys}
    batch[wrist] = torch.rand(1, 3, 42, 42)
    batch["task"] = ["pick up the object"]
    with torch.inference_mode():
        actions = policy.predict_action_chunk(batch)
    assert actions.shape == (1, 32, 8)
    assert torch.isfinite(actions).all()
    print(f"TacMind0 inference passed: shape={tuple(actions.shape)}, device={actions.device}")


if __name__ == "__main__":
    main()
