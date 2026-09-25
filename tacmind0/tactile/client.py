"""Evaluation-side marker history and HTTP transport, independent of Isaac imports."""

import base64
from io import BytesIO

import numpy as np
import requests
from PIL import Image

from .preprocessing import MarkerHistory


def encode_png(image: np.ndarray) -> str:
    buffer = BytesIO()
    Image.fromarray(image).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


class TacDreamClient:
    """Feed observe() every control tick, then infer() whenever a new chunk is needed.

    A separate history is maintained for each lane. The caller applies its existing
    action chunk scheduler / temporal aggregation; this client does not alter actions.
    """

    def __init__(self, url: str, timeout: float = 120):
        self.url = url.rstrip("/") + "/v1/infer"
        self.timeout = timeout
        self.histories = {}

    def reset(self, lane=0):
        self.histories[lane] = MarkerHistory()

    def observe(self, left_rgb, right_rgb, *, frame_index: int, lane=0):
        if lane not in self.histories:
            self.reset(lane)
        self.histories[lane].append(left_rgb, right_rgb, frame_index=frame_index)

    def request_body(
        self, agentview_rgb, wrist_rgb, state, prompt: str, *, lane=0, sampling=None
    ):
        state = np.asarray(state, dtype=np.float32)
        if state.shape != (8,) or not np.isfinite(state).all():
            raise ValueError("TacDream state must contain eight finite values")
        if lane not in self.histories:
            raise ValueError(
                "Empty marker history; observe every control tick before inference"
            )
        left, right = self.histories[lane].selected()
        body = {
            "observation": {
                "images": {"1": encode_png(agentview_rgb), "2": encode_png(wrist_rgb)},
                "state": np.asarray(state).tolist(),
                "prompt": prompt,
                "tactile_history": {
                    "left": [encode_png(frame) for frame in left],
                    "right": [encode_png(frame) for frame in right],
                },
            }
        }
        if sampling is not None:
            body["sampling"] = dict(sampling)
            if sampling.get("reference_action") is not None:
                body["sampling"].setdefault("reference_action_space", "physical")
        return body

    def infer(
        self, agentview_rgb, wrist_rgb, state, prompt: str, *, lane=0, sampling=None
    ):
        body = self.request_body(
            agentview_rgb, wrist_rgb, state, prompt, lane=lane, sampling=sampling
        )
        response = requests.post(self.url, json=body, timeout=self.timeout)
        response.raise_for_status()
        actions = np.asarray(response.json()["actions"], dtype=np.float32)
        if (
            actions.ndim != 2
            or actions.shape[0] == 0
            or actions.shape[1] != 8
            or not np.isfinite(actions).all()
        ):
            raise ValueError("TacDream response must be finite (H,8) actions")
        return actions
