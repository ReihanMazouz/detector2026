from __future__ import annotations

import argparse
import gc
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import torch
from torch.utils.data import Dataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo import MR_YOLO
from detector2026.core.models.tf_attn_yolo import TF_Attn_Yolo
from detector2026.core.models.yolov11 import YOLOv11
from detector2026.core.models.yolov8 import YOLOv8
from detector2026.core.utils.dataset._common import load_label_items
from detector2026.core.utils.preprocess import build_preprocessor, preprocessing_num_channels


Resolution = Tuple[int, int]

DEFAULT_DATA_DIR = "/data/RAWSIM/RMA/rf_dataset_for_real_validation"
DEFAULT_OUTPUT_DIR_PARENT = "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation"
DEFAULT_DEVICE = "cuda:0"
DEFAULT_NUM_CLASSES = 20
DEFAULT_REG_MAX = 16
DEFAULT_EPOCHS = 100
DEFAULT_PATIENCE = 10
DEFAULT_BATCH_SIZE = 64
DEFAULT_LR = 1e-3
DEFAULT_PREPROCESSING = "none"
DEFAULT_NUM_WORKERS = None
DEFAULT_FULL_EVAL_EVERY = 5
DEFAULT_SAVE_LAST_EVERY = 5
DEFAULT_MONITOR = "val_loss"
DEFAULT_RES_KEYS = ["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"]

MR_WIDTH_MULT = {
    "n": 0.25,
    "s": 0.50,
    "m": 0.75,
}
# The local YOLOv11 and TF-Attn implementations expose width_mult only.
# The n/s/m mapping therefore follows width scaling in this repository.
YOLO11_WIDTH_MULT = {
    "n": 0.25,
    "s": 0.50,
    "m": 0.75,
    "l": 1.00,
}
YOLOV8_SCALE = {
    "n": {"width_mult": 0.25, "depth_mult": 0.33},
    "s": {"width_mult": 0.50, "depth_mult": 0.33},
    "m": {"width_mult": 0.75, "depth_mult": 0.67},
    "l": {"width_mult": 1.00, "depth_mult": 1.00},
}
MR_BACKBONE_MODE = "TFSep_pyramid"
MR_OUTFUSION_CHANNELS_MULT = 1


def find_input_resolutions(data_dir: str, split: str = "train") -> List[Resolution]:
    images_dir = Path(data_dir) / split / "data"
    example_pt = next(images_dir.glob("*.pt"), None)
    if example_pt is None:
        raise FileNotFoundError(f"Aucun .pt trouve dans {images_dir}")

    specs = torch.load(example_pt, map_location="cpu")
    if not isinstance(specs, list):
        raise ValueError(f"Expected a list of tensors in {example_pt}, got {type(specs)}")

    resolutions: List[Resolution] = []
    for index, spec in enumerate(specs):
        if not torch.is_tensor(spec):
            raise ValueError(f"Element {index} in {example_pt} is a {type(spec)}, not a tensor")
        if spec.ndim == 3:
            _, height, width = spec.shape
        elif spec.ndim == 2:
            height, width = spec.shape
        else:
            raise ValueError(f"Unexpected ndim={spec.ndim} for element {index} in {example_pt}")
        resolutions.append((height, width))

    return resolutions


def make_fused_subset_dataset(
    selected_indices: Sequence[int],
    selected_res_keys: Sequence[str],
) -> type[Dataset]:
    indices = tuple(int(index) for index in selected_indices)
    res_keys = tuple(str(res_key) for res_key in selected_res_keys)

    class YOLODatasetFusedSubset(Dataset):
        def __init__(
            self,
            data_dir,
            labels_dir,
            preprocessing="spectrogram_psnr",
            preprocessing_kwargs=None,
        ):
            self.data_paths = sorted(
                os.path.join(data_dir, filename)
                for filename in os.listdir(data_dir)
                if filename.lower().endswith(".pt")
            )
            self.labels_dir = labels_dir
            self.indices = indices
            self.res_keys = res_keys
            self.preprocess_tensor = build_preprocessor(preprocessing, preprocessing_kwargs)

        def __len__(self):
            return len(self.data_paths)

        @staticmethod
        def _to_chw(tensor):
            if tensor.ndim == 2:
                return tensor.unsqueeze(0)
            if tensor.ndim == 3:
                return tensor
            raise ValueError(f"Unexpected spec shape: {tuple(tensor.shape)}")

        def __getitem__(self, idx):
            data_path = self.data_paths[idx]
            raw_specs = torch.load(data_path, map_location="cpu")
            if not isinstance(raw_specs, list):
                raise ValueError(f"Expected a list of tensors in {data_path}, got {type(raw_specs)}")

            if any(index >= len(raw_specs) for index in self.indices):
                raise ValueError(
                    f"Requested indices {self.indices} but sample {data_path} only contains {len(raw_specs)} tensors."
                )

            selected_specs = [raw_specs[index] for index in self.indices]
            specs = [
                self._to_chw(self.preprocess_tensor(spec, cfg_key=res_key))
                for spec, res_key in zip(selected_specs, self.res_keys)
            ]

            base = os.path.splitext(os.path.basename(data_path))[0]
            label_items = load_label_items(os.path.join(self.labels_dir, f"{base}.json"))

            cls, bboxes, snrs = [], [], []
            psnrs_per_obj = []
            for item in label_items:
                cls.append(item["class"])
                bboxes.append([item["xc"], item["yc"], item["w"], item["h"]])
                snrs.append(float(item.get("snr", -1.0)))

                psnr = item.get("psnr")
                if isinstance(psnr, dict):
                    psnr_vec = [
                        float(psnr.get(res_key)) if psnr.get(res_key) is not None else -1.0
                        for res_key in self.res_keys
                    ]
                else:
                    psnr_vec = [-1.0] * len(self.res_keys)
                psnrs_per_obj.append(psnr_vec)

            cls_tensor = torch.tensor(cls, dtype=torch.float32)
            boxes_tensor = (
                torch.tensor(bboxes, dtype=torch.float32)
                if bboxes
                else torch.zeros((0, 4), dtype=torch.float32)
            )
            snr_tensor = torch.tensor(snrs, dtype=torch.float32)
            psnr_tensor = (
                torch.tensor(psnrs_per_obj, dtype=torch.float32)
                if psnrs_per_obj
                else torch.zeros((0, len(self.res_keys)), dtype=torch.float32)
            )

            return {
                "imgs": specs,
                "cls": cls_tensor,
                "bboxes": boxes_tensor,
                "snr": snr_tensor,
                "psnr": psnr_tensor,
                "img_idx": idx,
                "res_keys": list(self.res_keys),
            }

        @staticmethod
        def collate_fn(batch):
            imgs_lists = [item["imgs"] for item in batch]
            imgs_per_res = list(zip(*imgs_lists))
            imgs = [torch.stack(res_list, dim=0) for res_list in imgs_per_res]

            all_cls = [item["cls"] for item in batch]
            all_boxes = [item["bboxes"] for item in batch]
            all_snrs = [item["snr"] for item in batch]
            all_psnrs = [item["psnr"] for item in batch]

            targets = []
            for i, (cls, boxes, snr, psnr) in enumerate(zip(all_cls, all_boxes, all_snrs, all_psnrs)):
                if boxes.numel():
                    img_idx = torch.full((boxes.shape[0], 1), i, dtype=torch.float32)
                    row = torch.cat((img_idx, cls.unsqueeze(-1), boxes, snr.unsqueeze(-1), psnr), dim=1)
                    targets.append(row)

            targets = (
                torch.cat(targets, 0)
                if targets
                else torch.zeros(
                    (0, 7 + (all_psnrs[0].shape[1] if all_psnrs else 0)),
                    dtype=torch.float32,
                )
            )
            return imgs, targets, batch[0]["res_keys"]

    suffix = "_".join(str(index + 1) for index in indices)
    YOLODatasetFusedSubset.__name__ = f"YOLODatasetFusedSubset_{suffix}"
    return YOLODatasetFusedSubset


@dataclass(frozen=True)
class TrainingJob:
    label: str
    output_dir_name: str
    dataset: object
    model_builder: Callable[[str], torch.nn.Module]
    select_res: dict | None = None


def output_name_for_mr(scale: str, selected_res_keys: Sequence[str]) -> str:
    return f"mr_yolo_{scale}_fused_{'_'.join(selected_res_keys)}"


def output_name_for_tf_attn(scale: str, central_res_key: str) -> str:
    return f"tf_attn_yolo{scale}_specificres_{central_res_key}"


def output_name_for_yolov11(scale: str, central_res_key: str) -> str:
    return f"yolov11{scale}_specificres_{central_res_key}"


def output_name_for_yolov8(scale: str, central_res_key: str) -> str:
    return f"yolov8{scale}_specificres_{central_res_key}"


def build_jobs(
    input_resolutions: Sequence[Resolution],
    res_keys: Sequence[str],
    input_channels: int,
    device: str,
    num_classes: int,
    reg_max: int,
    central_res_key: str,
    central_res_hw: Resolution,
) -> List[TrainingJob]:
    if len(input_resolutions) != len(res_keys):
        raise ValueError(
            f"Mismatch between resolutions ({len(input_resolutions)}) and res_keys ({len(res_keys)})."
        )

    res_key_to_hw = dict(zip(res_keys, input_resolutions))
    res_key_to_index = {res_key: index for index, res_key in enumerate(res_keys)}

    def make_mr_subset(
        selected_res_keys: Sequence[str],
    ) -> tuple[type[Dataset], tuple[str, ...], tuple[Resolution, ...]]:
        missing = [res_key for res_key in selected_res_keys if res_key not in res_key_to_index]
        if missing:
            raise ValueError(f"Unknown resolution keys requested for MR subset: {missing}")

        indices = tuple(res_key_to_index[res_key] for res_key in selected_res_keys)
        dataset_cls = make_fused_subset_dataset(indices, selected_res_keys)
        selected_resolutions = tuple(res_key_to_hw[res_key] for res_key in selected_res_keys)
        return dataset_cls, tuple(selected_res_keys), selected_resolutions

    def build_mr(scale: str, selected_resolutions: Sequence[Resolution]) -> Callable[[str], torch.nn.Module]:
        return lambda output_dir: MR_YOLO(
            num_classes=num_classes,
            device=device,
            reg_max=reg_max,
            output_dir=output_dir,
            input_resolutions=list(selected_resolutions),
            in_ch=input_channels,
            width_mult=MR_WIDTH_MULT[scale],
            backbone_mode=MR_BACKBONE_MODE,
            outfusion_channels_mult=MR_OUTFUSION_CHANNELS_MULT,
        )

    def build_yolo11(scale: str) -> Callable[[str], torch.nn.Module]:
        return lambda output_dir: YOLOv11(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            input_canals=input_channels,
            width_mult=YOLO11_WIDTH_MULT[scale],
        )

    def build_tf_attn(scale: str) -> Callable[[str], torch.nn.Module]:
        return lambda output_dir: TF_Attn_Yolo(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            input_canals=input_channels,
            width_mult=YOLO11_WIDTH_MULT[scale],
        )

    def build_yolov8(scale: str) -> Callable[[str], torch.nn.Module]:
        return lambda output_dir: YOLOv8(
            output_dir=output_dir,
            num_classes=num_classes,
            reg_max=reg_max,
            device=device,
            in_ch=input_channels,
            width_mult=YOLOV8_SCALE[scale]["width_mult"],
            depth_mult=YOLOV8_SCALE[scale]["depth_mult"],
        )

    mr_subset_specs = [
        ("MR-YOLO n, all resolutions", tuple(res_keys)),
        ("MR-YOLO n, resolutions 256-512-1024", ("cfg256", "cfg512", "cfg1024")),
        ("MR-YOLO n, resolutions 128-512-2048", ("cfg128", "cfg512", "cfg2048")),
        ("MR-YOLO n, resolutions 256-1024", ("cfg256", "cfg1024")),
    ]

    mr_jobs = []
    for label, selected_res_keys in mr_subset_specs:
        dataset_cls, output_res_keys, selected_resolutions = make_mr_subset(selected_res_keys)
        mr_jobs.append(
            TrainingJob(
                label=label,
                output_dir_name=output_name_for_mr("n", output_res_keys),
                dataset=dataset_cls,
                model_builder=build_mr("n", selected_resolutions),
            )
        )

    all_res_dataset_cls, all_res_output_keys, all_resolutions = make_mr_subset(tuple(res_keys))
    mr_jobs.append(
        TrainingJob(
            label="MR-YOLO s, all resolutions",
            output_dir_name=output_name_for_mr("s", all_res_output_keys),
            dataset=all_res_dataset_cls,
            model_builder=build_mr("s", all_resolutions),
        )
    )

    yolov11s_extra_res_jobs = [
        TrainingJob(
            label=f"YOLOv11s, resolution {res_key}",
            output_dir_name=output_name_for_yolov11("s", res_key),
            dataset="specificres",
            model_builder=build_yolo11("s"),
            select_res={"res_hw": res_key_to_hw[res_key], "res_key": res_key},
        )
        for res_key in res_keys
        if res_key != central_res_key
    ]

    return [
        TrainingJob(
            label="YOLOv11n, central resolution",
            output_dir_name=output_name_for_yolov11("n", central_res_key),
            dataset="specificres",
            model_builder=build_yolo11("n"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="TF-Attn-YOLOn, central resolution",
            output_dir_name=output_name_for_tf_attn("n", central_res_key),
            dataset="specificres",
            model_builder=build_tf_attn("n"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        *mr_jobs,
        TrainingJob(
            label="YOLOv8n, central resolution",
            output_dir_name=output_name_for_yolov8("n", central_res_key),
            dataset="specificres",
            model_builder=build_yolov8("n"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv11s, central resolution",
            output_dir_name=output_name_for_yolov11("s", central_res_key),
            dataset="specificres",
            model_builder=build_yolo11("s"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        *yolov11s_extra_res_jobs,
        TrainingJob(
            label="TF-Attn-YOLOs, central resolution",
            output_dir_name=output_name_for_tf_attn("s", central_res_key),
            dataset="specificres",
            model_builder=build_tf_attn("s"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv8s, central resolution",
            output_dir_name=output_name_for_yolov8("s", central_res_key),
            dataset="specificres",
            model_builder=build_yolov8("s"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv11m, central resolution",
            output_dir_name=output_name_for_yolov11("m", central_res_key),
            dataset="specificres",
            model_builder=build_yolo11("m"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="TF-Attn-YOLOm, central resolution",
            output_dir_name=output_name_for_tf_attn("m", central_res_key),
            dataset="specificres",
            model_builder=build_tf_attn("m"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv8m, central resolution",
            output_dir_name=output_name_for_yolov8("m", central_res_key),
            dataset="specificres",
            model_builder=build_yolov8("m"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv11l, central resolution",
            output_dir_name=output_name_for_yolov11("l", central_res_key),
            dataset="specificres",
            model_builder=build_yolo11("l"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="TF-Attn-YOLOl, central resolution",
            output_dir_name=output_name_for_tf_attn("l", central_res_key),
            dataset="specificres",
            model_builder=build_tf_attn("l"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
        TrainingJob(
            label="YOLOv8l, central resolution",
            output_dir_name=output_name_for_yolov8("l", central_res_key),
            dataset="specificres",
            model_builder=build_yolov8("l"),
            select_res={"res_hw": central_res_hw, "res_key": central_res_key},
        ),
    ]


def cleanup_after_run():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_job(
    job: TrainingJob,
    data_dir: str,
    output_dir_parent: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    preprocessing: str,
    num_workers: int | None,
    full_eval_every: int,
    save_last_every: int,
    monitor: str,
):
    output_dir = output_dir_parent / job.output_dir_name
    if output_dir.exists():
        print(f"[SKIP] {job.label}: {output_dir} existe deja.")
        return

    print(f"[RUN ] {job.label}")
    print(f"       output_dir = {output_dir}")
    model = job.model_builder(str(output_dir))
    try:
        model.fit(
            data_dir=data_dir,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            patience=patience,
            dataset=job.dataset,
            preprocessing=preprocessing,
            select_res=job.select_res,
            num_workers=num_workers,
            full_eval_every=full_eval_every,
            save_last_every=save_last_every,
            monitor=monitor,
        )
    finally:
        del model
        cleanup_after_run()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run the requested training suite and skip experiments whose output_dir already exists."
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR, help="Dataset root directory.")
    parser.add_argument(
        "--output-dir-parent",
        default=DEFAULT_OUTPUT_DIR_PARENT,
        help="Parent directory that will contain all experiment folders.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, help="Training device, e.g. cpu, cuda:0.")
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--patience", type=int, default=DEFAULT_PATIENCE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--full-eval-every", type=int, default=DEFAULT_FULL_EVAL_EVERY)
    parser.add_argument("--save-last-every", type=int, default=DEFAULT_SAVE_LAST_EVERY)
    parser.add_argument("--monitor", default=DEFAULT_MONITOR)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only list the planned experiments and whether they will run or be skipped.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    output_dir_parent = Path(args.output_dir_parent)
    output_dir_parent.mkdir(parents=True, exist_ok=True)

    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != 5:
        raise ValueError(
            "Ce script attend exactement 5 resolutions dans le dataset. "
            f"Trouve {len(input_resolutions)}: {input_resolutions}"
        )

    res_keys = list(DEFAULT_RES_KEYS)
    central_index = 0
    central_res_key = res_keys[central_index]
    central_res_hw = input_resolutions[central_index]
    input_channels = preprocessing_num_channels(args.preprocessing)

    print("Resolutions detectees:")
    for index, (res_key, res_hw) in enumerate(zip(res_keys, input_resolutions), start=1):
        print(f"  {index}. {res_key}: {res_hw}")
    print(f"Resolution centrale pour les modeles uni-res = {central_res_key}: {central_res_hw}")
    print(f"Preprocessing = {args.preprocessing}")
    print(f"Input channels = {input_channels}")
    print(f"Num workers = {args.num_workers}")
    print(f"Full eval every = {args.full_eval_every}")
    print(f"Save last every = {args.save_last_every}")
    print(f"Monitor = {args.monitor}")

    jobs = build_jobs(
        input_resolutions=input_resolutions,
        res_keys=res_keys,
        input_channels=input_channels,
        device=args.device,
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        central_res_key=central_res_key,
        central_res_hw=central_res_hw,
    )

    print("\nSuite d'experiences:")
    for index, job in enumerate(jobs, start=1):
        status = "SKIP" if (output_dir_parent / job.output_dir_name).exists() else "RUN"
        print(f"  {index:02d}. [{status}] {job.label} -> {job.output_dir_name}")

    if args.dry_run:
        return

    for job in jobs:
        run_job(
            job=job,
            data_dir=args.data_dir,
            output_dir_parent=output_dir_parent,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            preprocessing=args.preprocessing,
            num_workers=args.num_workers,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor=args.monitor,
        )


if __name__ == "__main__":
    main()
