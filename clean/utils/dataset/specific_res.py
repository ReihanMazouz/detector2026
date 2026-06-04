import os

import torch
from torch.utils.data import Dataset

from ._common import load_label_items
from ..preprocess import build_preprocessor


class YOLODatasetSpecificRes(Dataset):
    def __init__(
        self,
        data_dir,
        labels_dir,
        res_hw,
        res_key,
        preprocessing="spectrogram_psnr",
        preprocessing_kwargs=None,
    ):
        self.data_paths = sorted(
            os.path.join(data_dir, f)
            for f in os.listdir(data_dir)
            if f.lower().endswith(".pt")
        )
        self.labels_dir = labels_dir
        self.res_hw = tuple(res_hw)
        self.res_key = res_key
        self.preprocess_tensor = build_preprocessor(preprocessing, preprocessing_kwargs)

    def __len__(self):
        return len(self.data_paths)

    def _pick_specific_res(self, tensor_list):
        height, width = self.res_hw

        for tensor in tensor_list:
            if not isinstance(tensor, torch.Tensor):
                continue
            if tensor.ndim == 2 and tensor.shape == (height, width):
                return tensor.unsqueeze(0)
            if tensor.ndim == 3 and tensor.shape[-2:] == (height, width):
                return tensor

        raise ValueError(f"No tensor with resolution ({height},{width}) in this .pt file.")

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        tensor_list = torch.load(data_path, map_location="cpu")
        raw_spec = self._pick_specific_res(tensor_list)
        spec = self.preprocess_tensor(raw_spec, cfg_key=self.res_key)

        base = os.path.splitext(os.path.basename(data_path))[0]
        label_items = load_label_items(os.path.join(self.labels_dir, f"{base}.json"))

        cls, bboxes, snrs, psnrs = [], [], [], []
        for item in label_items:
            cls.append(item["class"])
            bboxes.append([item["xc"], item["yc"], item["w"], item["h"]])
            snrs.append(float(item.get("snr", -1.0)))

            psnr = item.get("psnr")
            if isinstance(psnr, dict):
                value = psnr.get(self.res_key, -1.0)
            else:
                value = psnr if psnr is not None else -1.0
            psnrs.append(float(value))

        cls_t = torch.tensor(cls, dtype=torch.float32)
        bboxes_t = (
            torch.tensor(bboxes, dtype=torch.float32)
            if bboxes
            else torch.zeros((0, 4), dtype=torch.float32)
        )
        snr_t = torch.tensor(snrs, dtype=torch.float32)
        psnr_t = (
            torch.tensor(psnrs, dtype=torch.float32).unsqueeze(1)
            if psnrs
            else torch.zeros((0, 1), dtype=torch.float32)
        )

        return {
            "specs": spec,
            "cls": cls_t,
            "bboxes": bboxes_t,
            "snr": snr_t,
            "psnr": psnr_t,
            "img_idx": idx,
            "res_key": self.res_key,
        }

    @staticmethod
    def collate_fn(batch):
        imgs = torch.stack([item["specs"] for item in batch], dim=0)

        all_cls = [item["cls"] for item in batch]
        all_boxes = [item["bboxes"] for item in batch]
        all_snrs = [item["snr"] for item in batch]
        all_psnrs = [item["psnr"] for item in batch]

        targets = []
        for i, (cls, boxes, snr, psnr) in enumerate(
            zip(all_cls, all_boxes, all_snrs, all_psnrs)
        ):
            if boxes.numel() > 0:
                img_idx = torch.full((boxes.size(0), 1), i, dtype=torch.float32)
                row = torch.cat(
                    [img_idx, cls.unsqueeze(1), boxes, snr.unsqueeze(1), psnr],
                    dim=1,
                )
                targets.append(row)

        targets = (
            torch.cat(targets, dim=0)
            if targets
            else torch.zeros((0, 8), dtype=torch.float32)
        )

        return imgs, targets, batch[0]["res_key"]
