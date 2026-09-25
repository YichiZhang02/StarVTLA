"""Real-weight, real-episode GPU smoke; writes only a new explicitly selected output."""

import argparse
import gc
import json
import shutil
import threading
from pathlib import Path

import numpy as np
import torch
from flask import Flask
from transformers import AutoProcessor
from werkzeug.serving import make_server

from tacmind0.data.transforms import LoadImages
from tacmind0.tacdream import (
    TacDreamDataConfig,
    TacDreamInferenceConfig,
    TacDreamModelConfig,
    strict_policy_load,
)
from tacmind0.tactile.client import TacDreamClient
from tacmind0.tactile.provenance import fingerprint


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl-dir", required=True)
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    report = {}

    def record(key, value):
        report[key] = value
        print(key, value, flush=True)
        (args.output_dir / "report.json").write_text(json.dumps(report, indent=2))

    torch.manual_seed(19)
    cfg = TacDreamModelConfig()
    model = cfg.build_model().cuda()
    processor = AutoProcessor.from_pretrained(
        cfg.model_name_or_path, local_files_only=True
    )
    data = TacDreamDataConfig(jsonl_dir=args.jsonl_dir, image_dir=args.image_dir)
    dataset, collator = data.build_dataset(processor, cfg.chunk_size)
    index = min(100, len(dataset) - 1)
    batch = {k: v.cuda() for k, v in collator([dataset[index]]).items()}
    batch["pixel_values"] = batch["pixel_values"].to(torch.bfloat16)
    batch["action"] = batch["action"].to(torch.bfloat16)
    record("sample_index", index)
    record("batch_shapes", {k: list(v.shape) for k, v in batch.items()})
    frozen_hash = fingerprint(model.tactile_encoder)
    record("encoder_before_sha256", frozen_hash)
    (args.output_dir / "extraction_report.json").write_text(
        json.dumps(model.tactile_load_report, indent=2)
    )
    record("encoder_loaded_keys", len(model.tactile_load_report["loaded_keys"]))

    def infer_input(batch):
        return {k: v for k, v in batch.items() if k != "action"}

    inputs = infer_input(batch)
    inputs["diffusion_steps"] = 2
    model.eval()
    torch.manual_seed(123)
    initial = model.inference_action(**inputs)
    contract = model.config.tactile_config
    model.config.tactile_config = None
    torch.manual_seed(123)
    baseline = model.inference_action(
        **{k: v for k, v in inputs.items() if k != "tactile_pixel_values"}
    )
    model.config.tactile_config = contract
    record("identity_policy_max_abs", float((initial - baseline).abs().max()))
    torch.testing.assert_close(initial, baseline, rtol=0, atol=0)
    del initial, baseline
    model.train()
    assert not any(m.training for m in model.tactile_encoder.modules())
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(parameters, lr=1e-4, foreach=False)
    groups = [p for group in optimizer.param_groups for p in group["params"]]
    assert len({id(p) for p in groups}) == len(parameters)
    assert not (
        {id(p) for p in groups} & {id(p) for p in model.tactile_encoder.parameters()}
    )
    record("trainable_parameters", sum(p.numel() for p in parameters))
    film_before = fingerprint(model.tactile_film)
    losses, grads = [], []
    for _step in range(2):
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss
        assert torch.isfinite(loss)
        loss.backward()
        grad = model.tactile_film.net[-1].weight.grad.float().norm()
        assert torch.isfinite(grad) and grad > 0
        torch.nn.utils.clip_grad_norm_(parameters, 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))
        grads.append(float(grad))
        record("training", {"losses": losses, "film_grad_norms": grads})
    assert fingerprint(model.tactile_encoder) == frozen_hash
    assert fingerprint(model.tactile_film) != film_before
    record("encoder_unchanged", True)
    record("film_updated", True)
    del optimizer, parameters, groups, loss, grad
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    model.eval()
    vision_outputs = []
    hook = model.tactile_film.register_forward_hook(
        lambda _module, _args, output: vision_outputs.append(output.detach().cpu())
    )
    torch.manual_seed(123)
    expected = model.inference_action(**inputs)
    torch.manual_seed(123)
    changed = model.inference_action(
        **{**inputs, "tactile_pixel_values": 1 - inputs["tactile_pixel_values"]}
    )
    hook.remove()
    assert len(vision_outputs) == 2
    vision_difference = float((vision_outputs[0] - vision_outputs[1]).abs().max())
    assert vision_difference > 0
    record("tactile_vision_max_abs", vision_difference)
    vision_outputs.clear()
    difference = float((expected - changed).abs().max())
    assert difference > 0
    record("tactile_action_max_abs", difference)
    expected = expected.cpu()
    save = args.output_dir / "checkpoint"
    model.save_pretrained(save, max_shard_size="5GB")
    processor.save_pretrained(save)
    shutil.copyfile(data.norm_file, save / "norm_stats.json")
    del model, changed
    gc.collect()
    torch.cuda.empty_cache()
    model = strict_policy_load(save).cuda().eval()
    model.set_attention_implementation(
        llm_attn_implementation="sdpa",
        vision_attn_implementation="sdpa",
        action_attn_implementation="sdpa",
        bf16=True,
    )
    torch.manual_seed(123)
    restored = model.inference_action(**inputs).cpu()
    torch.testing.assert_close(restored, expected, rtol=0, atol=0)
    record("checkpoint_reload_max_abs", float((restored - expected).abs().max()))
    # Real HTTP service endpoint, using the same native inference config as the CLI.
    service = TacDreamInferenceConfig(diffusion_steps=2)
    info = data._dataset_info()
    service.default_robot_type = info["robot_type"].value
    service.default_state_desc = info["state_desc"]
    service._initialize(
        model,
        str(save),
        str(save / "norm_stats.json"),
        256,
        1024,
        use_absolute_action=True,
        add_state=True,
    )
    app = Flask(__name__)
    app.add_url_rule("/v1/infer", "infer", service._infer, methods=["POST"])
    server = make_server("127.0.0.1", 0, app)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        client = TacDreamClient(f"http://127.0.0.1:{server.server_port}")
        rows = [
            json.loads(line)
            for line in Path(dataset.id_to_jsonl[0]).read_text().splitlines()
        ]
        loader = LoadImages(["marker_left", "marker_right"], args.image_dir)
        for t in range(index + 1):
            images = loader(rows[t])["images"]
            client.observe(*[np.asarray(img) for img in images], frame_index=t)
        raw = rows[index]
        vision = LoadImages(["images_1", "images_2"], args.image_dir)(raw)["images"]
        actions = client.infer(
            *[np.asarray(img) for img in vision],
            raw["state"],
            raw["prompt"],
            sampling={"seed": 123, "num_steps": 2},
        )
        steered = client.infer(
            *[np.asarray(img) for img in vision],
            raw["state"],
            raw["prompt"],
            sampling={
                "seed": 123,
                "num_steps": 2,
                "reference_action": actions.tolist(),
            },
        )
        record("http_action_shape", list(actions.shape))
        record("http_frs_finite", bool(np.isfinite(steered).all()))
        bad = client.request_body(
            *[np.asarray(img) for img in vision], raw["state"], raw["prompt"]
        )
        del bad["observation"]["tactile_history"]
        response = app.test_client().post("/v1/infer", json=bad)
        assert response.status_code == 400
        record("missing_tactile_status", response.status_code)
    finally:
        server.shutdown()
        thread.join()
    record("peak_cuda_gib", torch.cuda.max_memory_allocated() / 2**30)
    record("passed", True)


if __name__ == "__main__":
    main()
