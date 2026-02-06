from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hydra.core.config_store import ConfigStore

from vla_scratch.datasets.config import DataConfig


@dataclass
class LiberoCoTrainConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.libero_cotrain.dataset.LiberoCoTrainDataset"
    repo_id: str = "libero_spatial_no_noops_lerobot"
    root_path: Optional[Path] = "data/libero_spatial_lerobot_v3.0/"
    norm_stats_path: Optional[str] = (
        "norm_stats/pi05_libero/libero_spatial_lerobot/libero_spatial_no_noops_lerobot"
    )
    bbox_only: bool = False
    remove_bbox: bool = False
    camera_mode: str = "dual"
    action_key: str = "actions"


libero_spatial_cotrain_config = LiberoCoTrainConfig()
libero_90_cotrain_config = LiberoCoTrainConfig(
    repo_id="libero_90_no_noops_lerobot",
    root_path="data/libero_90_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_90_lerobot/libero_90_no_noops_lerobot",
)
libero_goal_cotrain_config = LiberoCoTrainConfig(
    repo_id="libero_goal_no_noops_lerobot",
    root_path="data/libero_goal_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_goal_lerobot/libero_goal_no_noops_lerobot",
)
libero_object_cotrain_config = LiberoCoTrainConfig(
    repo_id="libero_object_no_noops_lerobot",
    root_path="data/libero_object_lerobot_v3.0",
    norm_stats_path="norm_stats/pi05_libero/libero_object_lerobot/libero_object_no_noops_lerobot",
)
libero_er_bbox_cotrain_config = LiberoCoTrainConfig(
    repo_id="er_libero_lerobot",
    root_path="data/er_libero_lerobot_v3.0",
    norm_stats_path=None,
    bbox_only=True,
    remove_bbox=False,
    camera_mode="front_only",
)


@dataclass
class LiberoMix120CoTrainConfig(DataConfig):
    _target_: str = (
        "vla_scratch.datasets.libero_cotrain_mix120.dataset.LiberoMix120CoTrainDataset"
    )
    # Component datasets
    libero_spatial: LiberoCoTrainConfig = libero_spatial_cotrain_config
    libero_goal: LiberoCoTrainConfig = libero_goal_cotrain_config
    libero_object: LiberoCoTrainConfig = libero_object_cotrain_config
    libero_90: LiberoCoTrainConfig = libero_90_cotrain_config

    norm_stats_path: str = (
        "norm_stats/pi05_libero/libero_mix_120_average_lerobot/libero_mix_120_average_no_noops_lerobot"
    )

    bbox_only: bool = False
    remove_bbox: bool = False
    camera_mode: str = "dual"
    action_key: str = "actions"


cs = ConfigStore.instance()
cs.store(name="libero-spatial-cotrain", node=libero_spatial_cotrain_config, group="data")
cs.store(name="libero-goal-cotrain", node=libero_goal_cotrain_config, group="data")
cs.store(name="libero-object-cotrain", node=libero_object_cotrain_config, group="data")
cs.store(name="libero-90-cotrain", node=libero_90_cotrain_config, group="data")
cs.store(name="libero-er-bbox", node=libero_er_bbox_cotrain_config, group="data")

libero_mix_120_cotrain_config = LiberoMix120CoTrainConfig()
cs.store(
    name="libero-mix-120-cotrain",
    node=libero_mix_120_cotrain_config,
    group="data",
)
