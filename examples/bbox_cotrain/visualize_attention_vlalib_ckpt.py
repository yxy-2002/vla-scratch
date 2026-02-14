#!/usr/bin/env python3
"""
Offline attention visualization for vla_lib checkpoints.

This script intentionally does NOT use online serving. It loads a checkpoint in
the same style as serve_policy_zmq.py, then extracts + visualizes attention
offline from dataset frames (similar workflow to visualize_attention.py).
"""

from __future__ import annotations

import argparse
import enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
import yaml
from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata

from vla_lib.models.vlas.openpi05 import attention_vis
from vla_lib.policies import policy_config
from vla_lib.utils.dist_utils import get_logger

logger = get_logger(__name__)


class EnvMode(enum.Enum):
    aloha = "aloha"
    droid = "droid"
    franka = "franka_demo"
    franka_3cam = "franka_3cam"
    libero = "libero"


KEY_MAPPINGS = {
    "libero": {
        "image": "observation/image",
        "wrist_image": "observation/wrist_image",
        "state": "observation/state",
        "task": "prompt",
    },
    "droid": {
        "exterior_image_1_left": "observation/exterior_image_1_left",
        "wrist_image_left": "observation/wrist_image_left",
        "joint_position": "observation/joint_position",
        "gripper_position": "observation/gripper_position",
        "task": "prompt",
    },
    "aloha": {
        "cam_high": "observation/images/cam_high",
        "cam_low": "observation/images/cam_low",
        "cam_left_wrist": "observation/images/cam_left_wrist",
        "cam_right_wrist": "observation/images/cam_right_wrist",
        "state": "observation/state",
        "task": "prompt",
    },
    "franka": {
        "observation.images.front_cam": "observation/images/front_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        "task": "prompt",
    },
    "franka_3cam": {
        "observation.images.left_cam": "observation/images/left_cam",
        "observation.images.right_cam": "observation/images/right_cam",
        "observation.images.wrist_cam": "observation/images/wrist_cam",
        "observation.state.tcp_pose": "observation/state/tcp_pose",
        "observation.state.gripper_pose": "observation/state/gripper_pose",
        "task": "prompt",
    },
}


def load_action_norm_skip_dims_from_config(checkpoint_dir: str) -> Optional[Dict[str, List[int]]]:
    checkpoint_path = Path(checkpoint_dir)
    for path in [checkpoint_path / "config.yaml", checkpoint_path.parent / "config.yaml"]:
        if not path.exists():
            continue
        try:
            with open(path) as f:
                training_config = yaml.safe_load(f)
            dataset_mixture = training_config.get("dataset_mixture", {})
            vla_config = dataset_mixture.get("vla", {})
            for ds in vla_config.get("datasets", []):
                skip_dims = ds.get("action_norm_skip_dims")
                if skip_dims is not None:
                    return skip_dims
        except Exception:
            continue
    return None


def load_raw_dataset(data_path: str, episode_id: Optional[int]) -> Tuple[LeRobotDataset, Dict[Any, str]]:
    path = Path(data_path).resolve()
    if path.exists():
        folder_name = path.name
        metadata = LeRobotDatasetMetadata(folder_name, root=path)
    else:
        metadata = LeRobotDatasetMetadata(data_path)

    episodes = [episode_id] if episode_id is not None else None
    if path.exists():
        dataset = LeRobotDataset(folder_name, root=path, download_videos=False, episodes=episodes)
    else:
        dataset = LeRobotDataset(data_path, download_videos=False, episodes=episodes)

    tasks: Dict[Any, str] = {}
    if path.exists():
        tasks_file = path / "meta" / "tasks.jsonl"
        if tasks_file.exists():
            with open(tasks_file, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    task_text = line.strip()
                    if task_text:
                        tasks[i] = task_text
    elif hasattr(metadata, "tasks") and metadata.tasks:
        tasks = metadata.tasks

    return dataset, tasks


def process_data_to_msg(data: Dict[str, Any], env_type: str, tasks: Dict[Any, str]) -> Dict[str, Any]:
    key_mapping = KEY_MAPPINGS.get(env_type)
    if key_mapping is None:
        raise ValueError(f"Unknown env_type: {env_type}")

    msg: Dict[str, Any] = {}
    for dataset_key, policy_key in key_mapping.items():
        if dataset_key == "task":
            if "task_index" in data:
                task_idx = data["task_index"]
                if hasattr(task_idx, "item"):
                    task_idx = int(task_idx.item())
                else:
                    task_idx = int(task_idx)
                msg[policy_key] = tasks.get(task_idx, "")
            elif "task" in data:
                task_val = data["task"]
                msg[policy_key] = task_val.item() if hasattr(task_val, "item") else str(task_val)
            else:
                msg[policy_key] = ""
        elif dataset_key in data:
            value = data[dataset_key]
            if isinstance(value, torch.Tensor):
                msg[policy_key] = value.detach().cpu().numpy()
            elif hasattr(value, "numpy"):
                msg[policy_key] = value.numpy()
            elif isinstance(value, str):
                msg[policy_key] = value
            else:
                msg[policy_key] = np.array(value)
    return msg


def to_chw_uint8(img: np.ndarray) -> np.ndarray:
    arr = np.asarray(img)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3 and arr.shape[-1] == 3:
        arr = np.transpose(arr, (2, 0, 1))
    if arr.dtype != np.uint8:
        arr_f = arr.astype(np.float32)
        if arr_f.min() >= 0.0 and arr_f.max() <= 1.0:
            arr = (arr_f * 255.0).clip(0, 255).astype(np.uint8)
        elif arr_f.min() >= -1.0 and arr_f.max() <= 1.0:
            arr = ((arr_f + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
        else:
            arr = arr_f.clip(0, 255).astype(np.uint8)
    return arr


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline visualize attention for vla_lib checkpoints.")
    parser.add_argument("--checkpoint_dir", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--env", type=str, default="libero", choices=[e.value for e in EnvMode])
    parser.add_argument("--model_type", type=str, default="pi05")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="attention_visualizations_vlalib")
    parser.add_argument("--layer_idx", type=int, default=-1)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--attention_camera", type=str, default="base_0_rgb")
    parser.add_argument("--max_episodes", type=int, default=50)
    parser.add_argument("--episode_id", type=int, default=None)
    args = parser.parse_args()

    skip_dims = load_action_norm_skip_dims_from_config(args.checkpoint_dir)
    asset_id = args.env
    policy = policy_config.create_trained_policy(
        checkpoint_dir=args.checkpoint_dir,
        env_type=args.env,
        model_type=args.model_type,
        asset_id=asset_id,
        device=args.device,
        action_norm_skip_dims=skip_dims,
        enable_attention_vis=False,
    )

    dataset, tasks = load_raw_dataset(args.data_path, args.episode_id)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    seen_eps: Set[int] = set()
    processed = 0
    core_model = policy._model.model  # noqa: SLF001

    for idx in range(len(dataset)):
        sample = dataset[idx]
        ep_idx = sample.get("episode_index", None)
        if isinstance(ep_idx, torch.Tensor):
            ep_idx = int(ep_idx.item())
        elif ep_idx is not None:
            ep_idx = int(ep_idx)
        else:
            ep_idx = idx

        if ep_idx in seen_eps:
            continue
        seen_eps.add(ep_idx)
        if processed >= args.max_episodes:
            break

        msg = process_data_to_msg(sample, args.env, tasks)
        if "observation/image" not in msg:
            logger.warning("Skip sample %d (episode=%s): missing observation/image", idx, ep_idx)
            continue

        # Build observation exactly through policy transform pipeline.
        inputs = {k: v.copy() if isinstance(v, np.ndarray) else v for k, v in msg.items()}
        inputs = policy._input_transform(inputs)  # noqa: SLF001
        inputs = policy._to_torch(inputs)  # noqa: SLF001
        observation = policy._prepare_observation(inputs)  # noqa: SLF001

        attn_inputs = attention_vis.extract_prefix_attention(core_model, observation, args.layer_idx)

        cam_list = list(observation.get("images", {}).keys())
        if not cam_list:
            logger.warning("Skip sample %d (episode=%s): no cameras in observation", idx, ep_idx)
            continue
        cam_name = args.attention_camera if args.attention_camera in cam_list else cam_list[0]
        cam_index = cam_list.index(cam_name)

        raw_img = msg["observation/image"]
        img_chw = to_chw_uint8(raw_img)
        image_shape = (img_chw.shape[1], img_chw.shape[2])

        heatmap = attention_vis.attention_to_heatmap(attn_inputs, cam_index, image_shape)

        ep_dir = output_dir / f"episode_{ep_idx}"
        ep_dir.mkdir(parents=True, exist_ok=True)
        out_path = ep_dir / "frame_0_vlalib.png"
        attention_vis.save_attention_overlay(img_chw, heatmap, out_path, alpha=args.alpha)

        processed += 1
        logger.info("Saved %s (episode=%s, camera=%s, layer_idx=%d)", out_path, ep_idx, cam_name, args.layer_idx)

    logger.info("Done. Processed episodes: %d. Output: %s", processed, output_dir)


if __name__ == "__main__":
    main()
