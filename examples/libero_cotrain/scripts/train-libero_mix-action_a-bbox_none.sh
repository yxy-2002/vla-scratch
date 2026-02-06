export HYDRA_FULL_ERROR=1

# Allow overrides like: DL_WORKERS=8 PREFETCH=4 NPROCS=2 bash train-libero_mix-action_a-bbox_none.sh
for arg in "$@"; do
    if [[ "$arg" == *=* ]]; then
        export "$arg"
    fi
done

DL_WORKERS=${DL_WORKERS:-4}
PREFETCH=${PREFETCH:-2}
NPROCS=${NPROCS:-8}

echo "train-libero_mix-action_a-bbox_none"
echo "  DL_WORKERS=$DL_WORKERS"
echo "  PREFETCH=$PREFETCH"
echo "  NPROCS=$NPROCS"
echo

torchrun --standalone --nnodes=1 --nproc_per_node=$NPROCS \
    scripts/train_policy.py \
    policy=pi-qwen \
    policy.state_history=0 \
    policy.action_horizon=10 \
    policy.transforms.0.max_length=500 \
    data=libero-mix-120-cotrain \
    batch_size=32 \
    train_data=libero_cotrain_baseline \
    train_data.datasets.action_a.data.remove_bbox=true \
    eval_data=libero_cotrain_eval \
    num_workers=$DL_WORKERS \
    prefetch_factor=$PREFETCH \
    lr.base=5e-5 \
    +lr.vlm_bridge=1e-5 \
    +lr.action_expert=5e-5 \
    epochs=121 \
    save_interval=40 \
    wandb.mode=online
