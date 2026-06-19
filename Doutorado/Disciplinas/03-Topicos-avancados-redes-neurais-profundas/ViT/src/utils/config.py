from __future__ import annotations

import shutil
from pathlib import Path

import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open("r") as file:
        return yaml.safe_load(file)


def prepare_output_dir(config: dict, config_path: str | Path) -> Path:
    experiment_name = config.get("experiment_name", "default")
    output_dir = Path(config.get("output_root", "outputs")) / experiment_name
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, output_dir / "config.yaml")
    return output_dir
