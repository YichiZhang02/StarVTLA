"""Real-weight Tac-FRS reload, sensitivity, cache, and HTTP smoke verification."""

import argparse
import inspect
import json
import threading
import time
from pathlib import Path

import numpy as np
import requests
import torch
from flask import Flask
from transformers import AutoProcessor
from werkzeug.serving import make_server

from tacmind0.data.transforms import LoadImages
from tacmind0.tacdream import TacDreamInferenceConfig, strict_policy_load
from tacmind0.tactile.client import TacDreamClient
from tacmind0.tactile.preprocessing import MarkerHistory

from .client import TacFRSClient
from .model import TacFRS
from .stream import PhaseHistory
from .training import EpisodeFeatures, make_batch, make_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--jsonl-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    report = {}

    def record(name, value):
        report[name] = value
        args.report.write_text(json.dumps(report, indent=2))
        print(name, value, flush=True)

    path = args.checkpoint
    policy = (
        strict_policy_load(path, overrides={"use_suffix_graph": False}).cuda().eval()
    )
    policy.requires_grad_(False)
    model = TacFRS.load(path, "cuda")
    processor = AutoProcessor.from_pretrained(path, local_files_only=True)
    dataset, collator = make_data(policy, path, processor, args)
    features = EpisodeFeatures(model, str(args.image_dir))
    batch, states, latent, _, _, _ = make_batch(
        dataset, collator, [7], features, model, "cuda", policy.config.chunk_size
    )
    inputs = {k: v for k, v in batch.items() if k != "action"}
    inputs["diffusion_steps"] = 2

    def generate(module, current):
        torch.manual_seed(123)
        with torch.no_grad():
            return module.generate(policy, inputs, current, states)[0]

    torch.cuda.synchronize()
    start = time.perf_counter()
    expected = generate(model, latent)
    torch.cuda.synchronize()
    record("three_flow_plus_predictor_ms", (time.perf_counter() - start) * 1000)
    restored = TacFRS.load(path, "cuda")
    actual = generate(restored, latent)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    record("adapter_reload_max_abs", float((actual - expected).abs().max()))
    # Keep visual inputs, existing FiLM inputs, candidate noise and policy fixed;
    # change only the new marker-conditioned predictor path.
    history = PhaseHistory()
    rng = np.random.default_rng(123)
    for t in range(8):
        changed_latent = history.append(
            model,
            rng.integers(0, 256, (42, 42, 3), dtype=np.uint8),
            rng.integers(0, 256, (42, 42, 3), dtype=np.uint8),
            t,
        )
    changed = generate(model, changed_latent)
    diff = float((changed - expected).abs().max())
    record("new_tactile_path_action_max_abs", diff)
    assert diff > 0
    # Zero new modulation must equal unmodulated reverse -> forward, not the
    # first candidate (Euler forward/reverse need not be exact inverses).
    for net in restored.modulators:
        torch.nn.init.zeros_(net[-1].weight)
        torch.nn.init.zeros_(net[-1].bias)
    reference = generate(restored, latent)
    original = policy._integrate_action_flow

    def without_modulation(x, **kwargs):
        kwargs.pop("tactile_modulations", None)
        return original(x, **kwargs)

    policy._integrate_action_flow = without_modulation
    try:
        baseline = generate(restored, latent)
    finally:
        policy._integrate_action_flow = original
    torch.testing.assert_close(reference, baseline, rtol=0, atol=0)
    record("identity_modulation_max_abs", float((reference - baseline).abs().max()))
    # Also strict-reload the full policy, not only its new modules.
    policy2 = (
        strict_policy_load(path, overrides={"use_suffix_graph": False}).cuda().eval()
    )
    torch.manual_seed(123)
    with torch.no_grad():
        actual, _ = model.generate(policy2, inputs, latent, states)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    record("full_reload_max_abs", float((actual - expected).abs().max()))
    del policy2, restored
    # Quantify bounded-stream approximation on real observations, including eviction.
    rows = features.entries[dataset.id_to_jsonl[0]][0]
    loader = LoadImages(
        ["marker_left", "marker_right"], str(args.image_dir), require_rgb=True
    )
    phase, repeat, full = PhaseHistory(), PhaseHistory(), MarkerHistory()
    errors = []
    for t in range(min(46, len(rows))):
        pair = loader(dict(rows[t]))["images"]
        z = phase.append(model, *pair, t)
        z2 = repeat.append(model, *pair, t)
        torch.testing.assert_close(z, z2, rtol=0, atol=0)
        full.append(*pair, frame_index=t)
        with torch.no_grad():
            full_z = model.projector(model.encoder(full.tensor().unsqueeze(0).cuda()))
        errors.append(float((z - full_z).abs().max()))
    record("online_replay_max_abs", 0.0)
    record("approximate_cache_vs_full_max_abs", max(errors))
    record("cache_frames", len(errors))
    service = TacDreamInferenceConfig(diffusion_steps=2)
    service.default_robot_type = "Franka"
    service.default_state_desc = policy.config.tactile_config["state_desc"]
    service._initialize(
        policy,
        str(path),
        str(path / "norm_stats.json"),
        256,
        1024,
        use_absolute_action=policy.config.tactile_config["action_mode"] == "relative",
        add_state=True,
    )
    service.tac_frs_runtime = model
    app = Flask(__name__)
    app.add_url_rule("/v1/infer", "infer", service._infer, methods=["POST"])
    flow_calls = []
    original_flow = policy._integrate_action_flow

    def traced_flow(x, **kwargs):
        flow_calls.append(
            {
                "reverse": kwargs.get("reverse", False),
                "modulated": kwargs.get("tactile_modulations") is not None,
            }
        )
        return original_flow(x, **kwargs)

    policy._integrate_action_flow = traced_flow
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        url = f"http://127.0.0.1:{server.server_port}"
        client = TacFRSClient(url)
        default_client = TacDreamClient(url)
        for t in range(8):
            pair = [np.asarray(x) for x in loader(dict(rows[t]))["images"]]
            client.observe(*pair, frame_index=t)
            default_client.observe(*pair, frame_index=t)
        cameras = [
            np.asarray(x)
            for x in LoadImages(["images_1", "images_2"], str(args.image_dir))(
                dict(rows[7])
            )["images"]
        ]
        raw = rows[7]
        call = (*cameras, raw["state"], raw["prompt"])
        result = client.infer(*call, sampling={"seed": 123, "num_steps": 2})
        assert flow_calls == [
            {"reverse": False, "modulated": False},
            {"reverse": True, "modulated": False},
            {"reverse": False, "modulated": True},
        ]
        record("http_tac_frs_flow_calls", flow_calls.copy())
        record("http_tac_frs_shape", list(result.shape))
        assert result.shape == (policy.config.chunk_size, 8)
        entry = service.tac_frs_sessions.entries[(client.session, "0")]
        assert entry["history"].history.frame_index == 7
        # Retry the identical serialized payload and verify cached response.
        client2 = TacFRSClient(url, session="retry")
        for t in range(8):
            pair = [np.asarray(x) for x in loader(dict(rows[t]))["images"]]
            client2.observe(*pair, frame_index=t)
        body = client2.request_body(*call, sampling={"mode": "tac_frs", "seed": 123})
        body["tactile_stream"] = dict(
            session="retry",
            environment="0",
            episode="one",
            request_id="one",
            control_step=7,
            reset=True,
            frames=client2.streams[0]["frames"],
        )
        first = requests.post(url + "/v1/infer", json=body, timeout=120)
        retry = requests.post(url + "/v1/infer", json=body, timeout=120)
        assert first.status_code == retry.status_code == 200
        assert first.json() == retry.json()
        record("http_retry_identical", True)
        flow_calls.clear()
        plain = default_client.infer(*call, sampling={"seed": 123})
        assert np.isfinite(plain).all() and plain.shape == (policy.config.chunk_size, 8)
        assert flow_calls == [{"reverse": False, "modulated": False}]
        record("http_without_frs_flow_calls", flow_calls.copy())
        flow_calls.clear()
        frs = default_client.infer(
            *call, sampling={"reference_action": plain.tolist(), "seed": 123}
        )
        assert np.isfinite(frs).all() and frs.shape == plain.shape
        assert flow_calls == [
            {"reverse": True, "modulated": False},
            {"reverse": False, "modulated": False},
        ]
        record("http_frs_flow_calls", flow_calls.copy())
        record("default_frs_finite", True)
        bad = dict(body)
        bad.pop("tactile_stream")
        assert (
            requests.post(url + "/v1/infer", json=bad, timeout=120).status_code == 400
        )
        record("missing_stream_status", 400)
    finally:
        server.shutdown()
        thread.join()
        policy._integrate_action_flow = original_flow
    source = Path(inspect.getfile(TacFRS)).resolve()
    assert source.is_relative_to(Path(__file__).resolve().parents[2])
    record("project_module", str(source))
    record("cwd", str(Path.cwd()))
    record("peak_cuda_gib", torch.cuda.max_memory_allocated() / 2**30)


if __name__ == "__main__":
    main()
