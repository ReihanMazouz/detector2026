from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...nn.blocks import C3k2
from ...nn.convs import Conv
from ...utils.dataset import YOLODatasetSpecificRes
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from ...utils.loss import YOLODetectionLoss
from ...utils.rtdetr_loss import RTDETRLoss
from ..Head.detect import Detect
from ..Head.rtdetr import RTDETRHead
from ..base import _move_imgs_to_device, _resolve_num_workers, _supports_cuda
from ..yolov11 import YOLOv11


class YOLOv11P3Direct(YOLOv11):
    """YOLOv11n-style shallow backbone stopped at P3 with a single-level YOLO head."""

    def __init__(self, *args, tal_topk: int = 10, **kwargs):
        super().__init__(*args, **kwargs)

        for name in (
            "conv4",
            "c3_3",
            "conv5",
            "c3_4",
            "sppf",
            "attn",
            "upsample",
            "head_c3_1",
            "head_c3_2",
            "down_p3",
            "head_c3_3",
            "down_p4",
            "head_c3_4",
            "detect_one2one",
        ):
            if hasattr(self, name):
                delattr(self, name)

        c3 = int(256 * kwargs.get("width_mult", 0.25))
        self.strides = [8]
        self.detect = Detect(
            in_channels=[c3],
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
        )
        self.detect.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.criterion = YOLODetectionLoss(
            num_classes=self.num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            tal_topk=tal_topk,
            device=self.device,
        )
        self.to(self.device)

    def forward_features(self, x):
        x = self._prepare_input(x)
        x = self.conv1(x)
        self.debug_shape("conv1", x)
        x = self.conv2(x)
        self.debug_shape("conv2", x)
        f2 = self.c3_1(x)
        self.debug_shape("c3_1 (f2)", f2)
        x = self.conv3(f2)
        self.debug_shape("conv3", x)
        p3 = self.c3_2(x)
        self.debug_shape("p3", p3)
        return (p3,)

    def forward(self, x, head=None):
        del head
        return self.detect(*self.forward_features(x))


class YOLOv11P3RTDETR(YOLOv11P3Direct):
    """P3-only YOLOv11 backbone with a single-level RT-DETR one-to-one head."""

    def __init__(
        self,
        *args,
        hidden_dim: int = 128,
        num_queries: int = 100,
        num_decoder_layers: int = 6,
        num_heads: int = 8,
        num_decoder_points: int = 16,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        matcher_num_threads: int = 1,
        freeze_backbone: bool = False,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        c3 = int(256 * kwargs.get("width_mult", 0.25))
        self.detect_one2one = RTDETRHead(
            in_channels=[c3],
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=True,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learnt_init_query=False,
        )
        self.detect_one2one.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.criterion = RTDETRLoss(
            num_classes=self.num_classes,
            matcher_num_threads=matcher_num_threads,
        )
        self.freeze_backbone = bool(freeze_backbone)
        self._last_image_hw = self.input_hw
        self.to(self.device)

    def forward(self, x, head=None):
        del head
        self._last_image_hw = tuple(x.shape[-2:])
        image_size = self.input_hw if self.input_hw is not None else self._last_image_hw
        return self.detect_one2one(*self.forward_features(x), image_size=image_size)

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and self.freeze_backbone:
            for name in ("conv1", "conv2", "c3_1", "conv3", "c3_2", "detect"):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
            self.detect_one2one.train(True)
        return self

    def set_head_only_training(self):
        self.freeze_backbone = True
        for param in self.parameters():
            param.requires_grad = False
        for param in self.detect_one2one.parameters():
            param.requires_grad = True
        self.detect_one2one.train()

    def set_full_training(self):
        self.freeze_backbone = False
        for param in self.parameters():
            param.requires_grad = True
        self.detect.eval()

    def load_p3_yolo_weights(self, weights_path: str, device="cpu", eval_mode=False):
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        clean_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        try:
            self.detect_one2one.cv_dist.load_state_dict(self.detect.cv_dist.state_dict())
            self.detect_one2one.dfl.load_state_dict(self.detect.dfl.state_dict())
        except RuntimeError as exc:
            print(f"[WARN] Could not sync P3 YOLO DFL branch into RT-DETR head: {exc}")
        if eval_mode:
            self.eval()
        return missing, unexpected

    def loss_from_batch(self, outputs, targets):
        batch_size = outputs["pred_logits"].shape[0]
        target_list = targets_from_yolo_tensor(targets, batch_size, outputs["pred_logits"].device)
        return self.criterion(outputs, target_list)

    def postprocess(self, outputs, cls_out=None, feats=None, conf_thres=0.1, max_det=300, **kwargs):
        del cls_out, feats, kwargs
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probs = logits[..., : self.num_classes].sigmoid()
        scores, labels = probs.max(dim=-1)

        if self.input_hw is not None:
            image_h, image_w = float(self.input_hw[0]), float(self.input_hw[1])
        elif self._last_image_hw is not None:
            image_h, image_w = float(self._last_image_hw[0]), float(self._last_image_hw[1])
        else:
            image_h = image_w = 1.0

        results = []
        for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
            keep = scores_i >= float(conf_thres)
            if not keep.any():
                results.append(torch.zeros((0, 6), device=logits.device, dtype=logits.dtype))
                continue
            selected = boxes_i[keep]
            xc, yc, w, h = selected.unbind(-1)
            x1 = (xc - 0.5 * w).clamp(0.0, 1.0) * image_w
            y1 = (yc - 0.5 * h).clamp(0.0, 1.0) * image_h
            x2 = (xc + 0.5 * w).clamp(0.0, 1.0) * image_w
            y2 = (yc + 0.5 * h).clamp(0.0, 1.0) * image_h
            detections = torch.stack(
                (x1, y1, x2, y2, scores_i[keep], labels_i[keep].to(logits.dtype)),
                dim=1,
            )
            if detections.shape[0] > max_det:
                detections = detections[detections[:, 4].argsort(descending=True)[:max_det]]
            results.append(detections)
        return results

    def postprocess_for_metrics(self, outputs, conf_threshold=0.1, max_det=300, **kwargs):
        return self.postprocess(outputs, conf_thres=conf_threshold, max_det=max_det, **kwargs)

    def fit(
        self,
        data_dir,
        epochs=100,
        batch_size=32,
        lr=1e-4,
        patience=10,
        dataset="specificres",
        use_amp=True,
        preprocessing="none",
        preprocessing_kwargs=None,
        select_res=None,
        num_workers=None,
        persistent_workers=True,
        prefetch_factor=4,
        monitor="val_loss",
        save_last_every=5,
        full_eval_every=5,
        run_full_eval=True,
        **_,
    ):
        if dataset != "specificres":
            raise ValueError("YOLOv11P3RTDETR currently supports dataset='specificres'.")
        if not select_res or "res_hw" not in select_res or "res_key" not in select_res:
            raise ValueError("select_res={'res_hw': (H, W), 'res_key': 'cfgXXX'} is required.")
        if monitor != "val_loss":
            raise ValueError("YOLOv11P3RTDETR currently supports monitor='val_loss'.")

        self.input_hw = tuple(select_res["res_hw"])
        self._last_image_hw = self.input_hw
        train_dataset = YOLODatasetSpecificRes(
            data_dir=os.path.join(data_dir, "train/data"),
            labels_dir=os.path.join(data_dir, "train/labels_detect"),
            res_hw=self.input_hw,
            res_key=select_res["res_key"],
            preprocessing=preprocessing,
            preprocessing_kwargs=preprocessing_kwargs,
        )
        val_dataset = YOLODatasetSpecificRes(
            data_dir=os.path.join(data_dir, "val/data"),
            labels_dir=os.path.join(data_dir, "val/labels_detect"),
            res_hw=self.input_hw,
            res_key=select_res["res_key"],
            preprocessing=preprocessing,
            preprocessing_kwargs=preprocessing_kwargs,
        )

        pin_memory = _supports_cuda(self.device)
        resolved_workers = _resolve_num_workers(num_workers)
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "collate_fn": YOLODatasetSpecificRes.collate_fn,
            "num_workers": resolved_workers,
            "persistent_workers": bool(persistent_workers) and resolved_workers > 0,
        }
        if resolved_workers > 0:
            loader_kwargs["prefetch_factor"] = max(2, int(prefetch_factor))
        train_loader = DataLoader(train_dataset, shuffle=True, **loader_kwargs)
        val_loader = DataLoader(val_dataset, shuffle=False, **loader_kwargs)

        optimizer = torch.optim.AdamW((p for p in self.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler(enabled=bool(use_amp) and str(self.device).startswith("cuda"))

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.csv"
        extra_headers = []
        eval_runner = None
        if run_full_eval:
            from ...utils.dataset import load_class_index_to_name

            eval_runner = EvalRunner(
                output_dir=str(output_dir),
                cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=self.input_hw),
                class_index_to_name=load_class_index_to_name(data_dir),
            )
            extra_headers = eval_runner.extra_headers()

        with log_path.open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_loss", "loss_cls_val", "loss_bbox_val", "loss_giou_val", *extra_headers])

        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, int(epochs) + 1):
            start = time.perf_counter()
            train_loss, _ = self._run_rtdetr_epoch(train_loader, optimizer, scaler, train=True, desc=f"Epoch {epoch} P3 RTDETR train")
            val_loss, val_parts = self._run_rtdetr_epoch(val_loader, None, scaler, train=False, desc=f"Epoch {epoch} P3 RTDETR val")
            should_eval = bool(run_full_eval) and ((epoch % max(1, int(full_eval_every)) == 0) or epoch == int(epochs))
            extra_values = []
            if run_full_eval:
                if should_eval:
                    extra_values = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)["extra_values"]
                else:
                    extra_values = [None, None, *([float("nan")] * 7), None]

            with log_path.open("a", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow([
                    epoch,
                    train_loss,
                    val_loss,
                    val_parts.get("loss_cls", 0.0),
                    val_parts.get("loss_bbox", 0.0),
                    val_parts.get("loss_giou", 0.0),
                    *extra_values,
                ])

            if (epoch % max(1, int(save_last_every)) == 0) or epoch == int(epochs):
                torch.save(self.state_dict(), output_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(self.state_dict(), output_dir / "best.pt")
            else:
                bad_epochs += 1

            print(f"P3 RTDETR epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} time={time.perf_counter() - start:.1f}s")
            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
            if bad_epochs >= int(patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break

    def _run_rtdetr_epoch(self, loader, optimizer, scaler, train, desc):
        self.train(train)
        total_loss = 0.0
        parts_sum = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        amp_enabled = scaler.is_enabled() if scaler is not None else False
        context = torch.enable_grad() if train else torch.no_grad()
        with context:
            for imgs, targets, _ in tqdm(loader, desc=desc, unit="batch"):
                imgs = _move_imgs_to_device(imgs, self.device, non_blocking=_supports_cuda(self.device))
                targets = targets.to(self.device)
                if optimizer is not None:
                    optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=amp_enabled):
                    outputs = self(imgs)
                    loss, parts = self.loss_from_batch(outputs, targets)
                if train:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                total_loss += float(loss.detach().item())
                for key in parts_sum:
                    if key in parts:
                        parts_sum[key] += float(parts[key])
        num_batches = max(1, len(loader))
        return total_loss / num_batches, {key: value / num_batches for key, value in parts_sum.items()}
