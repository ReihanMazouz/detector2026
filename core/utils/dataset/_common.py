import json
from pathlib import Path


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
