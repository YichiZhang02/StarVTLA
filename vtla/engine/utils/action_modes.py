"""Canonical model action modes. All end-effector poses describe TCPs."""

ACTION_MODE_SPECS = {
    "absolute_joint": ("absolute", "joint"),
    "relative_joint": ("relative", "joint"),
    "absolute_rot6d": ("absolute", "rot6d"),
    "relative_rot6d": ("relative", "rot6d"),
}
ACTION_MODE_ALIASES = {"joint": "absolute_joint"}


def parse_action_mode(mode: str) -> tuple[str, str]:
    mode = ACTION_MODE_ALIASES.get(mode, mode)
    if mode not in ACTION_MODE_SPECS:
        raise ValueError(f"Unknown action_mode={mode!r}; expected one of {tuple(ACTION_MODE_SPECS)}. "
                         "EE actions use TCP rot6d; regenerate old datasets/checkpoints.")
    return ACTION_MODE_SPECS[mode]
