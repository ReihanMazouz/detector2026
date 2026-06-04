import os
import re

import torch
from torch.utils.data import Dataset

from ._common import load_label_items
from ..preprocess import build_preprocessor

_NUMERIC_RE = re.compile(r"(\d+)$")


class YOLODatasetFusedMultiRes(Dataset):
    def __init__(
        self,
        data_dir,
        labels_dir,
        res_keys=["cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048"],
        max_dim=1024,
        preprocessing="spectrogram_psnr",
        preprocessing_kwargs=None,
    ):
        self.data_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        )
        self.labels_dir = labels_dir
        self._res_keys = tuple(res_keys) if res_keys is not None else None
        self.max_dim = max_dim
        self.preprocess_tensor = build_preprocessor(preprocessing, preprocessing_kwargs)

    def __len__(self):
        return len(self.data_paths)

    @staticmethod
    def _numeric_key(key):
        match = _NUMERIC_RE.search(key)
        return int(match.group(1)) if match else float("inf")

    def _ensure_res_keys(self, psnr_dict):
        if self._res_keys is not None:
            return
        if not isinstance(psnr_dict, dict) or not psnr_dict:
            return
        keys = list(psnr_dict.keys())
        keys.sort(key=self._numeric_key)
        self._res_keys = keys

    @staticmethod
    def _to_chw(tensor):
        if tensor.ndim == 2:
            return tensor.unsqueeze(0)
        if tensor.ndim == 3:
            return tensor
        raise ValueError(f"Unexpected spec shape: {tensor.shape}")

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        raw_specs = torch.load(data_path, map_location="cpu")

        if not isinstance(raw_specs, list):
            raise ValueError(f"Expected a list of Tensors, got {type(raw_specs)}")

        base = os.path.splitext(os.path.basename(data_path))[0]
        label_items = load_label_items(os.path.join(self.labels_dir, f"{base}.json"))

        cls, bboxes, snrs = [], [], []
        psnrs_per_obj = []

        if self._res_keys is None:
            for item in label_items:
                psnr = item.get("psnr")
                if isinstance(psnr, dict) and psnr:
                    self._ensure_res_keys(psnr)
                    break

        if self._res_keys is not None and len(raw_specs) != len(self._res_keys):
            raise ValueError(
                f"Sample {data_path} has {len(raw_specs)} resolutions, "
                f"but res_keys defines {len(self._res_keys)}: {self._res_keys}."
            )
        sample_res_keys = list(self._res_keys) if self._res_keys is not None else [None] * len(raw_specs)

        specs = []
        active_res_keys = []
        for tensor, cfg_key in zip(raw_specs, sample_res_keys):
            if tensor.shape[-2] > self.max_dim or tensor.shape[-1] > self.max_dim:
                continue
            processed = self.preprocess_tensor(tensor, cfg_key=cfg_key)
            specs.append(self._to_chw(processed))
            if cfg_key is not None:
                active_res_keys.append(cfg_key)

        if not active_res_keys and self._res_keys is not None:
            active_res_keys = [
                key
                for tensor, key in zip(raw_specs, sample_res_keys)
                if tensor.shape[-2] <= self.max_dim and tensor.shape[-1] <= self.max_dim and key is not None
            ]

        for item in label_items:
            cls.append(item["class"])
            bboxes.append([item["xc"], item["yc"], item["w"], item["h"]])

            snr_val = item.get("snr")
            snrs.append(float(snr_val) if snr_val is not None else -1.0)

            vec = []
            psnr = item.get("psnr", {})
            if not active_res_keys:
                if isinstance(psnr, dict) and psnr:
                    local_keys = sorted(psnr.keys(), key=self._numeric_key)
                    vec = [float(psnr.get(key, -1.0)) for key in local_keys]
                    active_res_keys = local_keys
            else:
                vec = [
                    float(psnr.get(key)) if psnr.get(key) is not None else -1.0
                    for key in active_res_keys
                ]

            psnrs_per_obj.append(vec)

        cls = torch.tensor(cls, dtype=torch.float32)
        bboxes = (
            torch.tensor(bboxes, dtype=torch.float32)
            if bboxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        snrs = torch.tensor(snrs, dtype=torch.float32)

        num_res = len(active_res_keys)
        if not psnrs_per_obj:
            psnr_tensor = torch.zeros((0, num_res), dtype=torch.float32)
        else:
            if num_res == 0:
                num_res = len(psnrs_per_obj[0]) if psnrs_per_obj else 0
                active_res_keys = [f"cfg{i}" for i in range(num_res)]

            psnr_tensor = torch.tensor(psnrs_per_obj, dtype=torch.float32)
            if psnr_tensor.ndim != 2 or psnr_tensor.shape[1] != num_res:
                num_boxes = psnr_tensor.shape[0]
                fixed = torch.full((num_boxes, num_res), -1.0, dtype=torch.float32)
                cols = min(num_res, psnr_tensor.shape[1])
                if cols > 0:
                    fixed[:, :cols] = psnr_tensor[:, :cols]
                psnr_tensor = fixed

        return {
            "imgs": specs,
            "cls": cls,
            "bboxes": bboxes,
            "snr": snrs,
            "psnr": psnr_tensor,
            "img_idx": idx,
            "res_keys": active_res_keys,
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
        for i, (cls, boxes, snr, psnr) in enumerate(
            zip(all_cls, all_boxes, all_snrs, all_psnrs)
        ):
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

        res_keys = None
        for item in batch:
            if item.get("res_keys"):
                res_keys = tuple(item["res_keys"])
                break

        return imgs, targets, res_keys
