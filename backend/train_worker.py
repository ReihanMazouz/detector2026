from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
if str(WORKSPACE_ROOT) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_ROOT))

from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.models.tf_attn_yolo import TF_Attn_Yolo
from detector2026.core.models.yolov8 import YOLOv8
from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.utils.preprocess import preprocessing_num_channels


def _load_config(config_path: Path) -> Dict[str, Any]:
    return json.loads(config_path.read_text(encoding="utf-8"))


def _find_input_resolutions(data_dir: str, split: str = "train") -> List[Tuple[int, int]]:
    images_dir = Path(data_dir) / split / "data"
    example_pt = next(images_dir.glob("*.pt"), None)
    if example_pt is None:
        raise FileNotFoundError(f"Aucun .pt trouve dans {images_dir}")

    specs = torch.load(example_pt, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of Tensors, got {type(specs)} in {example_pt}")

    resolutions: List[Tuple[int, int]] = []
    for index, spec in enumerate(specs):
        if not torch.is_tensor(spec):
            raise ValueError(f"Element {index} in {example_pt} is a {type(spec)}, not a Tensor")
        if spec.ndim == 3:
            _, height, width = spec.shape
        elif spec.ndim == 2:
            height, width = spec.shape
        else:
            raise ValueError(f"Unexpected ndim={spec.ndim} for element {index} in {example_pt}")
        resolutions.append((height, width))
    return resolutions


def _build_model(config: Dict[str, Any]):
    model_id = str(config["model_id"])
    model_config = dict(config.get("model_config", {}))
    training = dict(config.get("training", {}))
    output_dir = str(config["output_dir"])
    device = str(model_config.get("device", "cuda:0"))
    reg_max = int(model_config.get("reg_max", 16))
    width_mult = float(model_config.get("width_mult", 0.5))
    num_classes = int(model_config.get("num_classes", 20))
    input_channels = preprocessing_num_channels(training.get("preprocessing", "spectrogram_psnr"))
    anisotropic = bool(model_config.get("anisotropic", False))
    p3_size = tuple(model_config.get("p3_size", [64, 64]))
    input_hw = None
    if training.get("dataset_mode") in {"specificres", "singleres"}:
        res_hw = model_config.get("res_hw", [256, 256])
        input_hw = (int(res_hw[0]), int(res_hw[1]))

    if model_id == "mr_yolo":
        input_resolutions = _find_input_resolutions(str(config["dataset_path"]))
        return MR_YOLO(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            input_resolutions=input_resolutions,
            in_ch=input_channels,
            width_mult=width_mult,
            backbone_mode=str(model_config.get("backbone_mode", "TFSep_pyramid")),
            outfusion_channels_mult=int(model_config.get("outfusion_channels_mult", 1)),
        )

    if model_id == "yolov8":
        return YOLOv8(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            in_ch=input_channels,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )

    if model_id == "yolov11":
        return YOLOv11(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            input_canals=input_channels,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )

    if model_id == "tf_attn_yolo":
        return TF_Attn_Yolo(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            input_canals=input_channels,
            width_mult=width_mult,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )

    raise ValueError(f"Unsupported model_id '{model_id}'.")


def _fit_kwargs(config: Dict[str, Any]) -> Dict[str, Any]:
    training = dict(config.get("training", {}))
    model_config = dict(config.get("model_config", {}))
    kwargs: Dict[str, Any] = {
        "data_dir": str(config["dataset_path"]),
        "epochs": int(training.get("epochs", 300)),
        "batch_size": int(training.get("batch_size", 64)),
        "lr": float(training.get("learning_rate", 1e-3)),
        "patience": int(training.get("patience", 30)),
        "dataset": str(training.get("dataset_mode", "fused")),
        "preprocessing": str(training.get("preprocessing", "spectrogram_psnr")),
    }
    if isinstance(training.get("preprocessing_kwargs"), dict):
        kwargs["preprocessing_kwargs"] = dict(training["preprocessing_kwargs"])

    dataset_mode = kwargs["dataset"]
    if dataset_mode in {"specificres", "singleres"}:
        kwargs["select_res"] = {
            "res_hw": (
                int(model_config.get("res_hw", [256, 256])[0]),
                int(model_config.get("res_hw", [256, 256])[1]),
            ),
            "res_key": str(model_config.get("res_key", "cfg512")),
        }
    return kwargs


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: train_worker.py <config.json>")

    config_path = Path(sys.argv[1]).resolve()
    config = _load_config(config_path)
    model = _build_model(config)
    fit_kwargs = _fit_kwargs(config)
    model.fit(**fit_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
