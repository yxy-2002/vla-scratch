# BBox Co-Training Innovations (PaliGemma + PI Policy)

This document summarizes the main innovations used by:

- `official_cotraining-action_a-bbox_a.sh`

It also clarifies how this setup differs from `bbox_none` and `bbox_ab`.

## 1. What This Training Recipe Actually Runs

The script uses:

- `policy=pi-paligemma`
- `train_data=bbox_cotrain_baseline` (action stream only: `action_a`)
- `use_paligemma_tokens=true`

So this run is **not** "action-only with no bbox signal".
It is "action stream + bbox language supervision from the same samples".

## 2. Core Innovation: Joint Objective (Action + BBox Text)

The PI policy is trained with a joint loss:

- `flow_mse` for action denoising/flow matching
- `ce_loss` for language generation (including bbox coordinate tokens)

Combined objective:

- `loss = flow_mse + ce_loss_weight * ce_loss`

Why this matters:

- The same visual-linguistic prefix is optimized for both action prediction and bbox token generation.
- BBox supervision regularizes and enriches visual grounding used by the action head.

## 3. BBox Supervision in Native PaliGemma Coordinate Tokens

When `use_paligemma_tokens=true`:

- bbox targets are formatted as `<loc####>` coordinate tokens (quantized to `[0, 1023]`).
- tokenizer registers these location tokens explicitly.

This avoids relying only on free-form JSON text and aligns supervision with PaliGemma's coordinate-token style.

## 4. How BBox Visual Information Reaches the PaliGemma LM Head

Pipeline:

1. Dataset emits:
   - image tensor
   - generation prompt
   - generation answer (bbox labels + `<loc####>` coordinates)
2. PaliGemma processor builds one sequence:
   - `<image>... + prompt + suffix(answer)`
3. In bridge encoding:
   - image features are extracted by vision tower
   - image features are injected at `<image>` token positions via `masked_scatter`
4. Gemma decoder layers process fused multimodal sequence.
5. `lm_head` predicts target bbox tokens; CE loss backpropagates through:
   - `lm_head`
   - language model layers
   - vision tower

Net effect:

- bbox text loss directly trains the visual-language representation, not just a detached text branch.

## 5. PI-Specific Architectural Innovation (vs plain VLM finetune)

The policy is not only a VLM:

- Prefix encoder: VLM bridge (PaliGemma/Qwen/SmolVLM compatible abstraction)
- Suffix action expert: DiT-style flow-matching module

Key PI design points:

- Observation registers: learnable register tokens appended after prefix.
- Register-focused conditioning: action expert can consume only register tokens for compact, task-focused conditioning.
- Time-conditioned denoising: action prediction is modeled as flow matching from noisy action trajectories.

This is the main reason the model can use language/bbox grounding while still producing robot actions efficiently.

## 6. Performance/Systems Innovation

The repository adds custom forward paths to reduce overhead:

- custom/monkeypatched PaliGemma layer forward
- shape-stable image feature injection (`masked_scatter`)
- FSDP2 + gradient checkpointing + module-level LR groups

These changes target higher throughput and lower synchronization overhead during large-scale training.

## 7. Relation Between `bbox_none`, `bbox_a`, `bbox_ab`

Three scripts represent controlled variants:

- `bbox_none`: remove bbox prompt/answer from training samples
- `bbox_a`: keep bbox supervision in action stream (`action_a`) only
- `bbox_ab`: mix action stream (`action_a`) with additional bbox-only stream (`bbox_b`)

Interpretation:

- `bbox_none -> bbox_a` measures benefit of adding bbox supervision on the same action data.
- `bbox_a -> bbox_ab` measures benefit of adding extra bbox-only data source.

## 8. Practical Summary

For `official_cotraining-action_a-bbox_a.sh`, the key innovation is:

- a **shared multimodal prefix** jointly trained by:
  - action flow loss
  - PaliGemma-native bbox token generation loss

This gives stronger visual grounding than pure action training while keeping a policy architecture specialized for control.

## 9. Code Pointers

- Script:
  - `official_cotraining-action_a-bbox_a.sh`
- Joint loss:
  - `vla_scratch/policies/pi/policy.py`
- PaliGemma bridge + image-token fusion:
  - `vla_scratch/policies/modules/vlm_bridge/paligemma/bridge.py`
- PaliGemma processor + location token registration:
  - `vla_scratch/policies/modules/vlm_bridge/paligemma/processor.py`
- BBox-to-`<loc####>` formatting:
  - `vla_scratch/datasets/utils/paligemma_bbox_format.py`
- BBox cotrain dataset behavior:
  - `vla_scratch/datasets/bbox_cotrain/dataset.py`
- Train-data variants (`baseline`/`mix`):
  - `vla_scratch/configs/data/bbox_cotrain.py`
