from typing import Any

from pathlib import Path
import json
import yaml


def read_json(fpath: Path) -> Any:
    with open(fpath, 'r') as f:
        return json.load(f)


def read_yaml(fpath: Path) -> Any:
    with open(fpath, 'r') as f:
        return yaml.load(f, yaml.CLoader)


def write_yaml(data: Any, fpath: Path):
    with open(fpath, 'w') as f:
        yaml.dump(data, f)

