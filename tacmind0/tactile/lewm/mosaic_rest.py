"""The v6 sequence input uses no rest subtraction."""


def parse_mosaic_rest_sub(value):
    if (
        value is None
        or value is False
        or str(value).lower() in ("off", "none", "false", "0")
    ):
        return "off"
    raise ValueError("TacDream v6 fine-tuning supports mosaic_rest_sub=off only")
