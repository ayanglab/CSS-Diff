# Checkpoints and reproducibility status

The code accepts existing checkpoints without rewriting their keys. Configure paths
with top-level YAML keys `sgp_weights`, `lsc_weights`, or the matching CLI options
(CLI takes precedence). Both required files are checked before model allocation.
No random fallback is used. Loading is strict and uses `weights_only=True` on
PyTorch 2.1.2 (the tested version).

| File | Architecture / format | Purpose |
| --- | --- | --- |
| `clip_1.pth` | Raw CLIP state_dict; embedding 512, resolution 120, 6 layers, width 256, patch 16 | SGP matching |
| `unet_weights_rotate.pth` | Raw MONAI 2D UNet state_dict; channels 16/32/64/128/256, strides 2/2/2/2, 2 residual units | LSC reconstruction |
| `<epoch>_net_G_A.pth` | Complete PSSGenerator state_dict (`model.*` keys) | LF to HF generator |
| `<epoch>_net_G_B.pth` | Complete PSSGenerator state_dict (`model.*` keys) | HF to LF generator |

G_A/G_B belong in `<checkpoints_dir>/<name>/`, loaded by `BaseModel.load_networks`.
Training weight initialization via `--continue_train` also needs D_A/D_B. This is
not exact training-state restoration: optimizer, scheduler and random states are
not saved. Set `--epoch_count` explicitly when continuing an experiment.

`PSSGenerator(ckpt_path=...)` is a different, internal interface: it loads only the
inner diffusion model (raw state_dict or the historical `model` container). It
must not be given a complete G_A/G_B file. CLI training uses BaseModel instead.
The `PSS.path` template field is not a checkpoint-loading option.

## Files located during the repair

Author storage contained the four files recorded in `checkpoint_validation.json`.
They all loaded strictly on CPU with the published architectures. These checks
establish architecture compatibility, not which files generated the paper tables.
Their originating run, data split, distribution permission and paper association
have not been established. No public download release is asserted here. Contact
the repository authors for these files; do not substitute unrelated weights.

A local SGP+LSC notebook and server code copies were located, but no independently
validated, portable pretraining pipeline is included in this repair.
