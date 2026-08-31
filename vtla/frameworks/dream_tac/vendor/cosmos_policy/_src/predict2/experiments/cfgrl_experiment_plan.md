## CFGRL-style regularized joint guidance (video+action) — experiment template

### Setting
- **Model**: joint diffusion/flow model over \(z=(video\_latent, action)\) using `pack_joint(...)`.
- **Optimality**: critic score \(Q_\psi(x,z)\) binned to \(o \in \{0..K-1\}\), with unconditional \(o=\varnothing\) realized by dropout.
- **Inference**: CFGRL guidance weight \(w\) swept at test time, without retraining.

### Metrics (report all)
- **Task success proxy**: per-task success detector / scripted metric / downstream policy success.
- **Action-video consistency**:
  - learned inverse dynamics error (predict action from generated video, MSE / NLL)
  - temporal alignment: cross-correlation peak at lag 0
- **Distribution shift / trust-region proxy**:
  - log-likelihood under unconditional branch (or a separate density model) vs \(w\)
  - feature-space distance to dataset (FID-like in latent space)
- **Physical plausibility**:
  - jerk/acceleration stats of action
  - optical flow smoothness / motion magnitude priors

### Sweeps
- **Guidance weight**: \(w \in \{0, 0.5, 1, 1.25, 1.5, 2, 3, 5\}\)
- **Schedule ablation**: constant w vs `guidance_schedule_linear(w0,w1)` (start low, end high)
- **Bins**: K=2 (binary) vs K=4 (quartiles) vs K=8 (octiles)

### Ablations (paper-friendly)
- **Optimality source**:
  - critic trained on dataset returns vs critic trained on consistency-only vs multi-objective critic
- **Joint vs factorized**:
  - joint z diffusion vs (action diffusion + video|action diffusion) factorized baseline
- **Token vs direct conditioning**:
  - optimality as `crossattn_emb` token (this PR) vs explicit net input (future work)
- **Dropout rate p_uncond**:
  - 0.05 / 0.1 / 0.2

### Failure modes & diagnostics (include in appendix)
- Divergence point: performance peaks then drops as w increases; correlate with KL proxy.
- Over-optimization: critic hacking; detect via OOD detectors / realism metrics.

