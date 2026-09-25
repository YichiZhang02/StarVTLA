"""Five-phase tactile KV cache with transactional, per-environment delivery."""

import copy
import hashlib
import json
import threading
from collections import OrderedDict

import torch

from tacmind0.tactile.preprocessing import MarkerHistory, prepare_frame_pair

CACHE_VERSION = "five-phase-sequence-v2"
SUPPORTED_CACHE_VERSIONS = {"five-phase-sequence-v1", CACHE_VERSION}


class PhaseHistory:
    def __init__(self):
        self.history = MarkerHistory()
        self.phases = {}
        self.latent = None

    def fork(self):
        new = copy.copy(self)
        new.history = copy.copy(self.history)
        new.history.frames = self.history.frames.copy()
        new.phases = self.phases.copy()
        return new

    @torch.no_grad()
    def append(self, model, left, right, index):
        self.history.append(left, right, frame_index=index)
        phase = index % 5
        encoder = model.encoder.encoder
        parameter = next(encoder.parameters())
        if phase not in self.phases:
            pixels = (
                self.history.tensor()
                .unsqueeze(0)
                .to(parameter.device, parameter.dtype)
            )
            cls, state = encoder.stream_init(pixels)
        else:
            version = getattr(model, "config", {}).get("cache_version", CACHE_VERSION)
            if version not in SUPPORTED_CACHE_VERSIONS:
                raise ValueError("Unsupported Tac-FRS cache version")
            current = (
                prepare_frame_pair(left, right)
                .unsqueeze(0)
                .to(parameter.device, parameter.dtype)
            )
            cls, state = encoder.stream_step(
                current,
                self.phases[phase],
                corrected_window=version == CACHE_VERSION,
            )
        self.phases[phase] = state
        self.latent = model.projector(cls.float())
        return self.latent


class StreamSessions:
    """Commit history and response together; retries never advance encoder state."""

    def __init__(self, max_environments=64, response_history=4):
        self.entries = {}
        self.lock = threading.RLock()
        self.max_environments = max_environments
        self.response_history = response_history

    def transact(self, body, model, decode, callback):
        stream = body.get("tactile_stream")
        if not isinstance(stream, dict):
            raise ValueError("tactile_stream is required for Tac-FRS")
        for name in ("session", "environment", "episode", "request_id"):
            if not isinstance(stream.get(name), str) or not stream[name]:
                raise ValueError(f"tactile_stream.{name} must be a nonempty string")
        end = stream.get("control_step")
        if isinstance(end, bool) or not isinstance(end, int) or end < 0:
            raise ValueError("control_step must be a nonnegative integer")
        reset = stream.get("reset", False)
        if not isinstance(reset, bool):
            raise ValueError("reset must be boolean")
        key = (stream["session"], stream["environment"])
        digest = hashlib.sha256(
            json.dumps(body, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()
        with self.lock:
            old = self.entries.get(key)
            if old is not None:
                match = old["responses"].get(stream["request_id"])
                if match is not None:
                    if match[0] != digest:
                        raise ValueError("Conflicting retry payload")
                    return copy.deepcopy(match[1])
                if not reset and stream["request_id"] in old["seen_requests"]:
                    raise ValueError("Expired request_id cannot be reused")
            if old is None:
                if len(self.entries) >= self.max_environments:
                    raise ValueError(
                        "Tac-FRS session capacity reached; release an idle session"
                    )
                if not reset:
                    raise ValueError("First request requires reset=true")
                pending = PhaseHistory()
                closed = set()
            elif reset:
                if (
                    stream["episode"] == old["episode"]
                    or stream["episode"] in old["closed"]
                ):
                    raise ValueError("Reset requires a new episode identifier")
                pending = PhaseHistory()
                closed = old["closed"] | {old["episode"]}
            else:
                if stream["episode"] != old["episode"]:
                    raise ValueError("Wrong episode; explicit reset required")
                pending = old["history"].fork()
                closed = old["closed"]
            frames = stream.get("frames")
            if not isinstance(frames, list) or not frames:
                raise ValueError("Continuous unacknowledged marker frames are required")
            if len(frames) != end - pending.history.frame_index:
                raise ValueError("Missing or duplicated control frames")
            for frame in frames:
                if not isinstance(frame, dict) or set(frame) != {
                    "index",
                    "left",
                    "right",
                }:
                    raise ValueError("Each stream frame requires index, left, right")
                left = decode(frame["left"], "stream.left", require_rgb=True)
                right = decode(frame["right"], "stream.right", require_rgb=True)
                pending.append(model, left, right, frame["index"])
            if pending.history.frame_index != end:
                raise ValueError("control_step does not match last frame")
            result = callback(pending)
            responses = (
                OrderedDict() if old is None or reset else old["responses"].copy()
            )
            responses[stream["request_id"]] = (digest, copy.deepcopy(result))
            while len(responses) > self.response_history:
                responses.popitem(last=False)
            self.entries[key] = dict(
                history=pending,
                episode=stream["episode"],
                closed=closed,
                responses=responses,
                seen_requests=(
                    {stream["request_id"]}
                    if old is None or reset
                    else old["seen_requests"] | {stream["request_id"]}
                ),
            )
            return result

    def release(self, session, environment):
        with self.lock:
            self.entries.pop((session, environment), None)
