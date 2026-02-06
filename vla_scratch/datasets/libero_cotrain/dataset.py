from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Mapping, Tuple

import numpy as np
import torch
from lerobot.datasets.lerobot_dataset import (
    LeRobotDataset,
    LeRobotDatasetMetadata,
)

from vla_scratch.datasets.utils.paligemma_bbox_format import (
    paligemma_detect_answer,
    paligemma_detect_prompt,
    use_paligemma_tokens_enabled,
)
from vla_scratch.transforms.data_keys import (
    GENERATION_ANSWER_KEY,
    GENERATION_PROMPT_KEY,
    PROCESSED_ACTION_KEY,
    PROCESSED_IMAGE_KEY,
    PROCESSED_IMAGE_MASK_KEY,
    PROCESSED_STATE_KEY,
    TASK_KEY,
)
from vla_scratch.utils.paths import REPO_ROOT

if TYPE_CHECKING:
    from .config import LiberoCoTrainConfig


def _build_bbox_index(
    path: Path,
) -> Tuple[Dict[Tuple[int, int], int], List[Tuple[int, int]]]:
    """
    Build a (episode_index, frame_index) -> byte offset index for jsonl bboxes.
    Only stores offsets (and keys) to avoid loading bbox payloads into RAM.
    """
    index: Dict[Tuple[int, int], int] = {}
    keys: List[Tuple[int, int]] = []
    with path.open("rb") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            ep_idx = int(record["episode_index"])
            frame_idx = int(record["frame_index"])
            key = (ep_idx, frame_idx)
            if key not in index:
                index[key] = offset
                keys.append(key)
    return index, keys


def _resolve_episode_ids(meta: LeRobotDatasetMetadata) -> List[int]:
    episodes_obj = meta.episodes
    if hasattr(episodes_obj, "keys"):
        episode_ids = list(episodes_obj.keys())
    else:
        episode_ids = list(range(len(episodes_obj)))
    return sorted(int(x) for x in episode_ids)


class LiberoCoTrainDataset(torch.utils.data.Dataset):
    """
    LIBERO dataset wrapper aligned to bbox-cotrain pipeline.

    - Supports dual or front-only camera modes.
    - Uses actions from a configurable action_key (default: "actions").
    - Optional bbox-only filtering with on-disk bbox index.
    - Emits paligemma bbox prompts/answers when enabled.
    """

    def __init__(self, config: "LiberoCoTrainConfig"):
        self.action_horizon = action_horizon = config.action_horizon
        self.state_history = state_history = config.state_history
        self.action_key = str(getattr(config, "action_key", "actions"))

        root = getattr(config, "root_path", None)
        repo_id = config.repo_id
        meta_root = None
        if root:
            root_path = Path(root).expanduser()
            if not root_path.is_absolute():
                root_path = (REPO_ROOT / root_path).resolve()
            meta_root = root_path / repo_id

        meta = LeRobotDatasetMetadata(repo_id=repo_id, root=meta_root)
        fps = meta.fps

        features = meta.features
        self.cmd_keys: list[str] = [key for key in features.keys() if "cmd" in key]
        if self.action_key not in self.cmd_keys:
            self.cmd_keys.append(self.action_key)

        self.state_keys: list[str] = [
            key for key in features.keys() if "state" in key
        ]

        delta_timestamps = {}
        for key in self.cmd_keys:
            delta_timestamps[key] = (
                np.linspace(0, action_horizon - 1, action_horizon, dtype=int)
                / fps
            ).tolist()

        for key in self.state_keys:
            delta_timestamps[key] = (
                np.linspace(-state_history, 0, state_history + 1, dtype=int)
                / fps
            ).tolist()

        self.dataset = LeRobotDataset(
            repo_id=repo_id,
            root=meta_root,
            delta_timestamps=delta_timestamps,
            video_backend=config.video_backend,
        )
        assert fps == self.dataset.fps

        self.camera_mode = str(getattr(config, "camera_mode", "dual"))
        if self.camera_mode not in {"dual", "front_only"}:
            raise ValueError(
                f"Unsupported camera_mode={self.camera_mode!r}; expected 'dual' or 'front_only'."
            )

        self.bbox_only = bool(getattr(config, "bbox_only", False))
        self.remove_bbox = bool(getattr(config, "remove_bbox", False))
        assert not (self.bbox_only and self.remove_bbox), (
            "Cannot set both bbox_only and remove_bbox to True."
        )

        self._bbox_index: Dict[Tuple[int, int], int] = {}
        self._bbox_keys: List[Tuple[int, int]] = []
        self._bbox_path: Path | None = None
        self._bbox_file = None

        bbox_path = meta.root / "meta" / "bboxes.jsonl"
        if bbox_path.exists():
            self._bbox_index, self._bbox_keys = _build_bbox_index(bbox_path)
            self._bbox_path = bbox_path
        elif self.bbox_only:
            raise ValueError(
                f"bbox_only=True requires meta/bboxes.jsonl, but none found at {bbox_path}"
            )

        if self.bbox_only:
            episode_ids = _resolve_episode_ids(meta)
            episodes_obj = meta.episodes
            if hasattr(episodes_obj, "keys"):
                episode_lengths = [
                    int(episodes_obj[ep_id]["length"]) for ep_id in episode_ids
                ]
            else:
                episode_lengths = [
                    int(episodes_obj[ep_id]["length"]) for ep_id in episode_ids
                ]
            episode_start_indices = np.cumsum([0] + episode_lengths)[:-1]
            episode_to_start = dict(zip(episode_ids, episode_start_indices))
            self.filtered_indices = [
                episode_to_start[ep_idx] + frame_idx
                for ep_idx, frame_idx in self._bbox_keys
                if ep_idx in episode_to_start
            ]
            self.size = len(self.filtered_indices)
        else:
            self.filtered_indices = None
            self.size = len(self.dataset)

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int):
        if self.filtered_indices is not None:
            idx = int(self.filtered_indices[idx])
        item = self.dataset[idx]

        if self.camera_mode == "dual":
            if "image" in item and "wrist_image" in item:
                img = torch.stack([item["image"], item["wrist_image"]], dim=0)
            else:
                img = torch.stack(
                    [item["images.cam_front"], item["images.cam_wrist"]], dim=0
                )
        else:
            if "image" in item:
                img = item["image"].unsqueeze(0)
            else:
                img = item["images.cam_front"].unsqueeze(0)

        img = (img * 255).to(torch.uint8)
        img_mask = torch.ones((img.shape[0], 1), dtype=torch.bool)

        if "state" in item:
            state = item["state"]
        else:
            state = torch.cat(
                [
                    item["arm_state_cart_pos"],
                    item["arm_state_cart_rot"],
                    item["gripper_state_qpos"],
                ],
                dim=-1,
            )
        state = state[1:]

        if self.bbox_only:
            actions = None
        else:
            if self.action_key not in item:
                raise KeyError(
                    f"Expected action key '{self.action_key}' in dataset item."
                )
            actions = item[self.action_key]

        prompt = ""
        answer = ""
        if not self.remove_bbox and self._bbox_index:
            ep_idx_t = item.get("episode_index")
            frame_idx_t = item.get("frame_index")
            if ep_idx_t is not None and frame_idx_t is not None:
                ep_idx = int(ep_idx_t.item())
                frame_idx = int(frame_idx_t.item())
                record = self._read_bbox_record(ep_idx, frame_idx)
                if record is not None:
                    bbox = record.get("bbox") or []
                    labels = [d["label"] for d in bbox]
                    if use_paligemma_tokens_enabled():
                        prompt = paligemma_detect_prompt()
                        answer = paligemma_detect_answer(
                            [d["bbox_normalized"] for d in bbox], labels
                        )
                    else:
                        bbox_coords = [
                            [int(x * 1000) for x in d["bbox_normalized"]]
                            for d in bbox
                        ]
                        bbox_out = [
                            {"bbox_2d": coords, "label": label}
                            for coords, label in zip(bbox_coords, labels)
                        ]
                        prompt = (
                            "Please return bounding boxes for all task-relevant objects in JSON format as"
                            '[{"bbox_2d": [x1, y1, x2, y2], "label": "<object_name>"}]'
                        )
                        answer = json.dumps(bbox_out)

        processed = {
            PROCESSED_IMAGE_KEY: img,
            PROCESSED_IMAGE_MASK_KEY: img_mask,
            PROCESSED_STATE_KEY: state,
            PROCESSED_ACTION_KEY: actions,
            TASK_KEY: item.get("task"),
            GENERATION_PROMPT_KEY: prompt,
            GENERATION_ANSWER_KEY: answer,
        }
        return processed

    def _read_bbox_record(self, ep_idx: int, frame_idx: int) -> dict | None:
        if not self._bbox_index or self._bbox_path is None:
            return None
        offset = self._bbox_index.get((ep_idx, frame_idx))
        if offset is None:
            return None
        if self._bbox_file is None:
            self._bbox_file = open(self._bbox_path, "rb")
        self._bbox_file.seek(offset)
        line = self._bbox_file.readline()
        if not line:
            return None
        return json.loads(line)
