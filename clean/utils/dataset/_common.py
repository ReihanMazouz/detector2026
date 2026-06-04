import json
from pathlib import Path
from functools import lru_cache


@lru_cache(maxsize=32768)
def load_label_items(json_path):
    path = Path(json_path)
    if not path.exists():
        return []

    data = json.loads(path.read_text())
    if isinstance(data, dict):
        labels = data.get("labels", [])
        return labels if isinstance(labels, list) else []
    if isinstance(data, list):
        return data
    return []


@lru_cache(maxsize=32)
def load_class_index_to_name(dataset_root):
    path = Path(dataset_root) / "class_index_to_name.json"
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as handle:
        raw_mapping = json.load(handle)

    mapping = {}
    if isinstance(raw_mapping, dict):
        for key, value in raw_mapping.items():
            try:
                mapping[int(key)] = str(value)
            except (TypeError, ValueError):
                continue
    return mapping
