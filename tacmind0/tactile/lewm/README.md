# Tac-LeWM

This is the Tac-LeWM implementation imported with TacMind-0. The source package name and checkpoint keys are retained. Model resources resolve within this StarVTLA checkout under playground/pretrained_models/tacmind0; no TacDream or external le-wm checkout is needed at runtime.

The bundled tac_lewm_v6_epoch1 directory contains the v6 epoch-1 starting weights and structure configs used by the fine-tuning command. The latest Gelsight marker-fixed, encoder-only fine-tune is under tac_lewm_finetuned/checkpoint-10000; TacMind-0 Stage 1 uses its strictly loaded encoder, and Tac-FRS Stage 2 uses its full world-model checkpoint. See vtla/frameworks/tacmind0/weights_manifest.json for source paths and SHA-256 checks.

For the input contract, training and serving commands, and isolated dependency environment, see vtla/frameworks/tacmind0/README.md.
