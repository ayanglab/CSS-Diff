# Implementation status (2026-11-04)

This engineering repair starts from public commit
`3958933d76bcaa95cd6cf703e9d93b06d1d5e785`. It does not certify reproduction of
published metrics. The templates are runnable configuration examples, not a
verified paper-training recipe.

## Completed

- Configurable SGP/LSC paths, complete file preflight, strict CPU-compatible loading.
- Single-run training by default, explicit `--sweep`, stable experiment names,
  disabled W&B without initialization/config access; shared `pss_train.py` entry.
- Deferred imports of unavailable comparison architectures so CSS-Diff can import.
- Removed the unavailable StructuredTransModel dependency from evaluation.
- Portable train/test templates and shared LSC options at inference.
- Explicit freezing/evaluation mode for pretrained auxiliaries, which the existing
  optimizer did not update. Generated images still receive gradients through LSC.
- Nonpersistent diffusion beta buffer: device migration without new checkpoint keys.
- Optional labeled trajectory output; invalid sampling sequences fail early.
- Dataset renaming previews by default, checks collisions before any rename.

## Paper/code correspondence and unresolved items

Source: [paper v1, Section III](https://arxiv.org/html/2510.13735v1#S3).

| Object | Located implementation / outstanding evidence |
| --- | --- |
| Cycle (Eq. 11) | `backward_G`: L1(real_A, rec_A) × lambda_A plus L1(real_B, rec_B) × lambda_B. |
| Path (Eq. 16) | No separate term found in public code, local backups or two server copies. The independent learned chain state and its reference, noise coupling and weights remain unidentified. Comparing the deterministic DDIM update with itself gives zero and is not a repair. |
| SGP | `set_input`: frozen features and batch Hungarian assignment; LF also uses the high-field encoder. Batch size 1 makes assignment trivial. Feature conditioning described in the paper is not independently implemented here. |
| LSC | `set_input`/`forward`: masking and fixed UNet; D_A receives the transformed real/fake branch. A corresponding G structural objective is not in the located total loss. |
| L1 / SSIM | Existing G losses compare generated images to selected targets. Their use in an unpaired setting needs experiment-specific justification. |

The original random input timestep and fixed reverse schedule are preserved for
checkpoint behavior; their mismatch remains unresolved. The trajectory interface
exposes states and timesteps but does not claim to implement Eq. 16. The legacy
inner-model eval-at-construction behavior is also retained; calling model.train()
sets generators to training mode, as in the existing training loop.

A saved evaluation configuration used 100 diffusion timesteps, whereas a recovered
training configuration used 50. The templates now agree at 50; use the actual run
configuration when reproducing historical inference. No output-equivalence or
paper-metric claim is made for this template correction or auxiliary eval mode.

Current evaluation requires both A and B. LF-only deployment remains unverified.
Full paper reproduction still requires the authoritative algorithm implementation,
run/config association, split, and metric evaluation. Do not relabel image cycle
loss as path loss or treat the located latest checkpoint as a confirmed paper model.

## Validation

- PyTorch 2.1.2+cu121 environment, CPU execution.
- Five automated tests: configuration precedence, missing-file preflight, strict
  loading and frozen-module input gradients, trajectory/checkpoint compatibility,
  and import without starting a sweep.
- Four real checkpoints strictly loaded; SHA-256 values recorded separately.
- Complete pss_lsc_sgp: two updates with real auxiliary weights and a reduced
  32-pixel diffusion network, followed by save/reload/evaluation, passed on CPU.
- CLI train.py: one synthetic-data epoch, disabled W&B, checkpoint save and paired
  validation passed. This smoke test is not a paper-metric experiment.
- CLI test.py loaded the saved networks and exported one synthetic evaluation
  sample successfully. No HF-free inference or clinical metrics were tested.
