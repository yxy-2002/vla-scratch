# BBox Co-Train Eval Example

## 🚀 Policy Serving
```bash
uv run scripts/serve_policy.py \
    checkpoint_path=/mnt/project_rlinf/yangxinye/vla-scratch/official-outputs/2026-02-09/23-59-25-state-0-action-10_norm_stats_bbox_mix-action_a-bbox_none/checkpoint_120 \
    data=bbox_cotrain_test \
    merge_policy_cfg=true
```
More training commands live in `examples/bbox_cotrain/scripts/`.

Pretrained checkpoints: [wandb runs](https://wandb.ai/elijahgalahad/vla-scratch/workspace?nw=77pbx7f06ur)

| Huggingface Id                                                                                           | Gradient Steps | Run Time   | Train Set Success | Test Set Success |
|----------------------------------------------------------------------------------------------------------|----------------|------------|---------------|--------------|
| [`elijahgalahad/checkpoint-action_a_bbox_none`](https://huggingface.co/elijahgalahad/checkpoint-action_a_bbox_none) | 15k            | 13h 39m 17s | 83.0%         | 26.5%        |
| [`elijahgalahad/checkpoint-action_a_bbox_a`](https://huggingface.co/elijahgalahad/checkpoint-action_a_bbox_a)       | 15k            | 13h 41m 45s | 94.5%         | 44.0%        |
| [`elijahgalahad/checkpoint-action_a_bbox_ab`](https://huggingface.co/elijahgalahad/checkpoint-action_a_bbox_ab)     | 15k            | 17h 21m 49s | 89.5%         | 59.5%        |

## 🤖 Simulation Environment

Set up simulation virtual environment (`examples/bbox_cotrain/.venv`):
```bash
git clone https://github.com/EGalahad/BlindVLA.git ../BlindVLA
export BLINDVLA_ROOT=$(pwd)/../BlindVLA

uv sync --project examples/bbox_cotrain  # installs pyzmq/msgpack/gym etc.
source examples/bbox_cotrain/.venv/bin/activate
uv pip install -e $BLINDVLA_ROOT/ManiSkill
uv pip install -e $BLINDVLA_ROOT/SimplerEnv
```

Run the simulation with policy client:
```bash
source examples/bbox_cotrain/.venv/bin/activate
python examples/bbox_cotrain/simulation.py \
    render=false sim_backend=cpu port=9000 obj_set=test video_path=outputs/evaluation/0212_maniskill_attention/action-a-bbox-ab/test/
```
