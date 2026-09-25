"""SO-101 dataset registration."""

from tacmind0.constants.robot import ROBOT_STATE_DESCS, RobotType
from tacmind0.dataset.register import register_dataset

register_dataset(
    {
        "pick_cube": {
            "jsonl_dir": "./data/so101_pick_cube/jsonl",
            "image_dir": "./data/so101_pick_cube/videos",
            "image_keys": ["images_1", "images_2"],
            "image_prompts": ["Head", "Left wrist"],
            "robot_type": RobotType.SO101,
            "state_desc": ROBOT_STATE_DESCS[RobotType.SO101],
        },
    },
    prefix="so101",
)
