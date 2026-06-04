import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from ._common import load_label_items
from ..preprocess import build_preprocessor


class YOLODatasetSingleRes(Dataset):
    def __init__(
        self,
        data_dir: str,
        labels_dir: str,
        res_keys=["cfg512", "cfg256", "cfg128", "cfg1024", "cfg2048"],
        select_res="cfg256",
        target_length: int = 1024,
        min_box_size: float = 20.0,
        preprocessing="spectrogram_psnr",
        preprocessing_kwargs=None,
    ):
        self.data_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        )
        if not self.data_paths:
            raise FileNotFoundError(f"No .pt files found in {data_dir}")

        self.labels_dir = Path(labels_dir)
        self.res_keys = res_keys
        self.res_index = res_keys.index(select_res)
        self.res_key = select_res
        self.target_len = target_length
        self.min_box = min_box_size
        self.preprocess_tensor = build_preprocessor(preprocessing, preprocessing_kwargs)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        path = self.data_paths[idx]
        specs = torch.load(path, map_location="cpu")

        raw_spec = specs[self.res_index]
        spec = self.preprocess_tensor(raw_spec, cfg_key=self.res_key)

        _, height, width = spec.shape
        target_len = self.target_len

        dh, dw = target_len - height, target_len - width
        top, bottom = dh // 2, dh - dh // 2
        left, right = dw // 2, dw - dw // 2
        img = F.pad(spec, (left, right, top, bottom), value=0)

        stem = os.path.splitext(os.path.basename(path))[0]
        label_items = load_label_items(self.labels_dir / f"{stem}.json")

        cls_list, bbox_list, snr_list = [], [], []
        for item in label_items:
            x0, y0 = item["xc"], item["yc"]
            w0, h0 = item["w"], item["h"]

            x_abs = x0 * width + left
            y_abs = y0 * height + top
            w_abs = max(w0 * width, self.min_box)
            h_abs = max(h0 * height, self.min_box)

            x_abs = min(max(x_abs, 0), target_len)
            y_abs = min(max(y_abs, 0), target_len)
            w_abs = min(w_abs, target_len)
            h_abs = min(h_abs, target_len)

            bbox_list.append([x_abs / target_len, y_abs / target_len, w_abs / target_len, h_abs / target_len])
            cls_list.append(item["class"])
            snr_list.append(float(item.get("snr", -1.0)))

        if bbox_list:
            boxes = torch.tensor(bbox_list, dtype=torch.float32)
            classes = torch.tensor(cls_list, dtype=torch.float32).unsqueeze(1)
            snr_t = torch.tensor(snr_list, dtype=torch.float32).unsqueeze(1)
            img_idx = torch.zeros((len(boxes), 1), dtype=torch.float32)
            targets = torch.cat([img_idx, classes, boxes, snr_t], dim=1)
        else:
            targets = torch.zeros((0, 7), dtype=torch.float32)

        return {"img": img, "targets": targets}

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([item["img"] for item in batch], dim=0)

        all_targets = []
        for i, sample in enumerate(batch):
            targets = sample["targets"]
            if targets.numel() > 0:
                targets = targets.clone()
                targets[:, 0] = i
                all_targets.append(targets)

        targets = (
            torch.cat(all_targets, dim=0)
            if all_targets
            else torch.zeros((0, 7), dtype=torch.float32)
        )
        return imgs, targets, None
