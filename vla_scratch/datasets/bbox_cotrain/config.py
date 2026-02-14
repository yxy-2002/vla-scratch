from hydra.core.config_store import ConfigStore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, List, Optional

from vla_scratch.datasets.config import DataConfig


@dataclass
class CoTrainConfig(DataConfig):
    _target_: str = "vla_scratch.datasets.bbox_cotrain.dataset.CoTrainDataset"
    repo_id: str = "horipse01/lerobot_merged"
    root_path: Optional[Path] = None

    bbox_only: bool = False
    remove_bbox: bool = False

    episodes: Optional[Any] = None
    # Regex filters applied to info.json splits keys (e.g., \".*banana.*\")
    splits: List[str] = field(default_factory=lambda: [".*"])

    norm_stats_path: str = "hf:elijahgalahad/norm_stats-bbox-cotrain"


train_cotrain_config = CoTrainConfig(
    root_path="/mnt/project_rlinf/yangxinye/vla-scratch/data/",
    repo_id="lerobot_merged_restricted",
    # norm_stats_path="/mnt/project_rlinf/yangxinye/vla-scratch/norm_stats/norm_stats-bbox-cotrain-train"
)
test_cotrain_config = CoTrainConfig(
    root_path="/mnt/project_rlinf/yangxinye/vla-scratch/data/",
    repo_id="lerobot_merged_restricted_val",
    # norm_stats_path="/mnt/project_rlinf/yangxinye/vla-scratch/norm_stats/norm_stats-bbox-cotrain-eval"
)

cs = ConfigStore.instance()
cs.store(name="bbox_cotrain_train", node=train_cotrain_config, group="data")
cs.store(name="bbox_cotrain_test", node=test_cotrain_config, group="data")
