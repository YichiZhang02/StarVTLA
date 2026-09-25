"""HTTP integration without mutable modulation hooks or shared episode state."""

import time

import numpy as np
import torch
from flask import jsonify, request
from werkzeug.exceptions import BadRequest


class TacFRSServingMixin:
    def _infer(self):
        with self.tac_frs_sessions.lock:
            return self._infer_locked()

    def _infer_locked(self):
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict):
                raise ValueError("request body must be an object")
            sampling = body.get("sampling", {})
            if sampling is None:
                sampling = {}
            if not isinstance(sampling, dict):
                raise ValueError("sampling must be an object")
            mode = sampling.get("mode", "frs")
            if mode not in ("frs", "tac_frs"):
                raise ValueError("sampling.mode must be frs or tac_frs")
            if mode == "frs":
                return super()._infer()
            if self.backend != "default":
                raise ValueError("Tac-FRS requires the native PyTorch backend")
            if not hasattr(self, "tac_frs_runtime"):
                raise ValueError("Checkpoint does not contain Tac-FRS weights")
            if sampling.get("reference_action") is not None:
                raise ValueError("Tac-FRS generates its own reference action")

            def predict(history):
                self._apply_v1_sampling(sampling)
                data = self._prepare_input(body)
                expected = history.history.tensor().unsqueeze(0)
                if not torch.equal(data["tactile_pixel_values"].cpu(), expected):
                    raise ValueError(
                        "tactile_history disagrees with streamed control frames"
                    )
                data["sampling_mode"] = "tac_frs"
                data["tac_frs_latent"] = history.latent
                start = time.monotonic()
                actions = self._predict(data)
                if not np.isfinite(actions).all():
                    raise ValueError("Tac-FRS produced nonfinite actions")
                return dict(
                    actions=actions.tolist(),
                    metadata=dict(
                        latency_ms=(time.monotonic() - start) * 1000,
                        acknowledged_step=history.history.frame_index,
                        request_id=body["tactile_stream"]["request_id"],
                        mode="tac_frs",
                    ),
                )

            return jsonify(
                self.tac_frs_sessions.transact(
                    body, self.tac_frs_runtime, self._decode_b64_image, predict
                )
            )
        except (BadRequest, ValueError, TypeError, KeyError) as exc:
            return jsonify(error=str(exc)), 400

    def _release_tac_frs(self):
        try:
            body = request.get_json(force=True)
            if not isinstance(body, dict) or any(
                not isinstance(body.get(k), str) or not body[k]
                for k in ("session", "environment")
            ):
                raise ValueError("session and environment are required")
            self.tac_frs_sessions.release(body["session"], body["environment"])
            return jsonify(released=True)
        except (BadRequest, ValueError, TypeError) as exc:
            return jsonify(error=str(exc)), 400
