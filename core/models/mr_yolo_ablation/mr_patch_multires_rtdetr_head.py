"""MR patch backbone (no fusion downsampling) with a multi-res RT-DETR head.

Each resolution keeps its natural patch-grid feature map; the RT-DETR
deformable decoder attends across all levels simultaneously instead of
working on a single fused 32×32 map.
"""
from __future__ import annotations

import csv
import os
import time
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils.dataset import YOLODatasetFusedMultiRes, load_class_index_to_name
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from ...utils.rtdetr_loss import RTDETRLoss
from ..Head.rtdetr import RTDETRHead
from ..base import BaseModel, _resolve_num_workers, _supports_cuda
from .mr_patch_backbone_yolo_one2many_head import (
    IsotropicRestrictedPatchBackbone,
    _sinusoidal_2d,
    _normalized_patch_centers,
)


def _move_to_device(imgs, device: str, non_blocking: bool = True):
    if isinstance(imgs, (list, tuple)):
        return [img.to(device, non_blocking=non_blocking) for img in imgs]
    return imgs.to(device, non_blocking=non_blocking)


# ── backbone ──────────────────────────────────────────────────────────────────

class IsotropicMultiResPatchBackbone(nn.Module):
    """Same encoder as IsotropicRestrictedPatchBackbone but without the fusion
    step.  Returns one (B, d_model, H_i, W_i) feature map per resolution at its
    natural patch-grid size instead of everything collapsed to p3_hw."""

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        in_ch: int = 1,
        d_model: int = 128,
        patch_size: int = 8,
        num_layers: int = 3,
        num_heads: int = 4,
        num_intra_points: int = 8,
        num_inter_neighbors: int = 8,
        dim_feedforward: int = 512,
        dropout: float = 0.0,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if d_model % 4 != 0:
            raise ValueError("d_model must be divisible by 4 for sinusoidal 2D encoding.")

        self.input_resolutions = [tuple(r) for r in input_resolutions]
        self.d_model = int(d_model)
        self.patch_size = int(patch_size)
        self.patch_shapes = self._compute_patch_shapes(self.input_resolutions, patch_size)

        # Reuse the same stem / embed / encoder architecture
        self.conv_stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
            )
            for _ in self.input_resolutions
        )
        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(16, d_model, kernel_size=patch_size, stride=patch_size)
            for _ in self.input_resolutions
        )
        self.res_embed = nn.Parameter(torch.zeros(len(self.input_resolutions), 1, d_model))
        nn.init.trunc_normal_(self.res_embed, std=0.02)

        from .mr_patch_backbone_yolo_one2many_head import IsotropicRestrictedPatchEncoderLayer
        self.encoder = nn.ModuleList(
            IsotropicRestrictedPatchEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_intra_points=num_intra_points,
                num_inter_neighbors=num_inter_neighbors,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_layers)
        )

        self.out_channels = tuple(d_model for _ in self.input_resolutions)
        self.out_strides = tuple(patch_size for _ in self.input_resolutions)

    @staticmethod
    def _compute_patch_shapes(
        input_resolutions: List[Tuple[int, int]],
        patch_size: int,
    ) -> List[Tuple[int, int]]:
        shapes = []
        for h, w in input_resolutions:
            if h % patch_size != 0 or w % patch_size != 0:
                raise ValueError(f"Resolution {(h, w)} not divisible by patch_size={patch_size}.")
            shapes.append((h // patch_size, w // patch_size))
        return shapes

    def _tokenize(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        tokens_by_res = []
        for i, (x, stem, embed, shape) in enumerate(
            zip(inputs, self.conv_stems, self.patch_embeds, self.patch_shapes)
        ):
            x = embed(stem(x))
            tokens = x.flatten(2).transpose(1, 2)
            pos = _sinusoidal_2d(shape[0], shape[1], self.d_model, x.device, x.dtype)
            tokens_by_res.append(tokens + pos + self.res_embed[i].to(dtype=tokens.dtype))
        return tokens_by_res

    def forward(self, inputs: List[torch.Tensor]) -> List[torch.Tensor]:
        tokens_by_res = self._tokenize(inputs)
        for layer in self.encoder:
            tokens_by_res = layer(tokens_by_res, self.patch_shapes)

        # Reshape each resolution back to its natural 2D feature map — no interpolation
        maps = []
        for tokens, shape in zip(tokens_by_res, self.patch_shapes):
            feature = tokens.transpose(1, 2).reshape(
                tokens.shape[0], self.d_model, shape[0], shape[1]
            )
            maps.append(feature)
        return maps  # List[(B, d_model, H_i/p, W_i/p)]


# ── model ─────────────────────────────────────────────────────────────────────

class MRPatchMultiResRTDETRHead(BaseModel):
    """IsotropicMultiResPatchBackbone + multi-level RT-DETR head.

    Each input resolution contributes its own feature map to the RT-DETR
    decoder; no information is lost to bilinear downsampling.

    For refinement, call load_backbone_weights() with a
    MRPatchBackboneYOLOOne2ManyHead or MRPatchBackboneRTDETRHead checkpoint.
    """

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        output_dir: str,
        num_classes: int = 20,
        reg_max: int = 16,
        device: str = "cuda:0",
        in_ch: int = 1,
        # ── backbone ────────────────────────────────────────────────────────
        d_model: int = 128,
        patch_size: int = 8,
        num_encoder_layers: int = 3,
        num_heads_backbone: int = 4,
        num_intra_points: int = 8,
        num_inter_neighbors: int = 8,
        dim_feedforward_backbone: int = 512,
        dropout: float = 0.0,
        # ── RT-DETR head ────────────────────────────────────────────────────
        hidden_dim: int = 128,
        num_queries: int = 100,
        num_decoder_layers: int = 2,
        num_heads_decoder: int = 8,
        num_decoder_points: int = 8,
        dim_feedforward_decoder: int = 1024,
        matcher_num_threads: int = 8,
    ):
        super().__init__(device=device, output_dir=output_dir)
        self.input_resolutions = [tuple(r) for r in input_resolutions]
        self.num_classes = int(num_classes)
        self.reg_max = int(reg_max)
        self._image_hw: Tuple[int, int] = (
            max(r[0] for r in self.input_resolutions),
            max(r[1] for r in self.input_resolutions),
        )

        self.backbone = IsotropicMultiResPatchBackbone(
            input_resolutions=self.input_resolutions,
            in_ch=in_ch,
            d_model=d_model,
            patch_size=patch_size,
            num_layers=num_encoder_layers,
            num_heads=num_heads_backbone,
            num_intra_points=num_intra_points,
            num_inter_neighbors=num_inter_neighbors,
            dim_feedforward=dim_feedforward_backbone,
            dropout=dropout,
        )

        # One feature map per resolution → num_levels = len(input_resolutions)
        self.strides = list(self.backbone.out_strides)
        self.detect_one2one = RTDETRHead(
            in_channels=list(self.backbone.out_channels),
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads_decoder,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=True,
            dim_feedforward=dim_feedforward_decoder,
            dropout=0.0,
            learnt_init_query=False,
        )
        self.detect_one2one.bias_init(image_size=max(self._image_hw))
        self.criterion = RTDETRLoss(
            num_classes=self.num_classes,
            matcher_num_threads=matcher_num_threads,
        )
        self.to(self.device)

    # ── forward ─────────────────────────────────────────────────────────────

    def forward(self, inputs: List[torch.Tensor]):
        feature_maps = self.backbone(inputs)  # List[(B, d_model, H_i, W_i)]
        return self.detect_one2one(*feature_maps, image_size=self._image_hw)

    # ── weight loading ───────────────────────────────────────────────────────

    def load_backbone_weights(self, weights_path: str, device: str = "cpu") -> Tuple[list, list]:
        """Load compatible backbone weights from any MRPatchBackbone* checkpoint."""
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        compatible = {
            k: v
            for k, v in state_dict.items()
            if k in model_state and model_state[k].shape == v.shape
        }
        missing, unexpected = self.load_state_dict(compatible, strict=False)
        return missing, unexpected

    # ── inference ────────────────────────────────────────────────────────────

    def loss_from_batch(self, outputs, targets):
        batch_size = outputs["pred_logits"].shape[0]
        target_list = targets_from_yolo_tensor(targets, batch_size, outputs["pred_logits"].device)
        return self.criterion(outputs, target_list)

    def postprocess(self, outputs, conf_thres: float = 0.1, max_det: int = 300, **_):
        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probs = logits[..., : self.num_classes].sigmoid()
        scores, labels = probs.max(dim=-1)
        image_h, image_w = float(self._image_hw[0]), float(self._image_hw[1])
        results = []
        for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
            keep = scores_i >= float(conf_thres)
            if not keep.any():
                results.append(torch.zeros((0, 6), device=logits.device, dtype=logits.dtype))
                continue
            sel = boxes_i[keep]
            xc, yc, w, h = sel.unbind(-1)
            x1 = (xc - 0.5 * w).clamp(0.0, 1.0) * image_w
            y1 = (yc - 0.5 * h).clamp(0.0, 1.0) * image_h
            x2 = (xc + 0.5 * w).clamp(0.0, 1.0) * image_w
            y2 = (yc + 0.5 * h).clamp(0.0, 1.0) * image_h
            dets = torch.stack(
                (x1, y1, x2, y2, scores_i[keep], labels_i[keep].to(logits.dtype)), dim=1
            )
            if dets.shape[0] > max_det:
                dets = dets[dets[:, 4].argsort(descending=True)[:max_det]]
            results.append(dets)
        return results

    def postprocess_for_metrics(self, outputs, conf_threshold: float = 0.1, max_det: int = 300, **kwargs):
        return self.postprocess(outputs, conf_thres=conf_threshold, max_det=max_det, **kwargs)

    # ── training loop ────────────────────────────────────────────────────────

    def fit(
        self,
        data_dir: str,
        epochs: int = 300,
        batch_size: int = 64,
        lr: float = 1e-4,
        patience: int = 10,
        preprocessing: str = "none",
        preprocessing_kwargs=None,
        res_keys: tuple = (),
        num_workers=None,
        prefetch_factor: int = 4,
        monitor: str = "val_loss",
        save_last_every: int = 5,
        full_eval_every: int = 5,
        run_full_eval: bool = True,
        use_amp: bool = True,
        **_,
    ):
        if monitor != "val_loss":
            raise ValueError("MRPatchMultiResRTDETRHead only supports monitor='val_loss'.")

        ds_kw = dict(
            res_keys=res_keys,
            preprocessing=preprocessing,
            preprocessing_kwargs=preprocessing_kwargs,
        )
        train_ds = YOLODatasetFusedMultiRes(
            data_dir=os.path.join(data_dir, "train/data"),
            labels_dir=os.path.join(data_dir, "train/labels_detect"),
            **ds_kw,
        )
        val_ds = YOLODatasetFusedMultiRes(
            data_dir=os.path.join(data_dir, "val/data"),
            labels_dir=os.path.join(data_dir, "val/labels_detect"),
            **ds_kw,
        )

        pin = _supports_cuda(self.device)
        nw = _resolve_num_workers(num_workers)
        dl_kw = dict(
            batch_size=batch_size,
            pin_memory=pin,
            collate_fn=YOLODatasetFusedMultiRes.collate_fn,
            num_workers=nw,
            persistent_workers=bool(nw > 0),
        )
        if nw > 0:
            dl_kw["prefetch_factor"] = max(2, int(prefetch_factor))
        train_loader = DataLoader(train_ds, shuffle=True, **dl_kw)
        val_loader = DataLoader(val_ds, shuffle=False, **dl_kw)

        optimizer = torch.optim.AdamW(
            (p for p in self.parameters() if p.requires_grad), lr=lr, weight_decay=1e-4
        )
        scaler = torch.cuda.amp.GradScaler(
            enabled=bool(use_amp) and str(self.device).startswith("cuda")
        )

        output_dir = Path(self.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        log_path = output_dir / "train_log.csv"

        eval_runner = None
        extra_headers: list = []
        if run_full_eval:
            eval_runner = EvalRunner(
                output_dir=str(output_dir),
                cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=self._image_hw),
                class_index_to_name=load_class_index_to_name(data_dir),
            )
            extra_headers = eval_runner.extra_headers()

        with log_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(
                ["epoch", "train_loss", "val_loss", "loss_cls_val", "loss_bbox_val", "loss_giou_val",
                 *extra_headers]
            )

        best_val = float("inf")
        bad_epochs = 0
        for epoch in range(1, int(epochs) + 1):
            t0 = time.perf_counter()
            train_loss, _ = self._run_epoch(train_loader, optimizer, scaler, train=True,
                                            desc=f"Epoch {epoch} train")
            val_loss, val_parts = self._run_epoch(val_loader, None, scaler, train=False,
                                                  desc=f"Epoch {epoch} val")

            should_eval = run_full_eval and (
                epoch % max(1, int(full_eval_every)) == 0 or epoch == int(epochs)
            )
            extra_values: list = []
            if run_full_eval:
                if should_eval:
                    extra_values = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)["extra_values"]
                else:
                    extra_values = [None, None, *([float("nan")] * 7), None]

            with log_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([
                    epoch, train_loss, val_loss,
                    val_parts.get("loss_cls", 0.0),
                    val_parts.get("loss_bbox", 0.0),
                    val_parts.get("loss_giou", 0.0),
                    *extra_values,
                ])

            if epoch % max(1, int(save_last_every)) == 0 or epoch == int(epochs):
                torch.save(self.state_dict(), output_dir / "last.pt")
            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(self.state_dict(), output_dir / "best.pt")
            else:
                bad_epochs += 1

            print(
                f"MR-Patch MultiRes RTDETR epoch {epoch}: "
                f"train={train_loss:.4f}  val={val_loss:.4f}  "
                f"time={time.perf_counter() - t0:.1f}s"
            )
            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
                TrainingPlots.plot_size_recalls(str(log_path), save_path=str(output_dir / "recall_size_curves.png"))
                TrainingPlots.plot_box_iou(str(log_path), save_path=str(output_dir / "box_iou_curves.png"))

            if bad_epochs >= int(patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break

    def _run_epoch(self, loader: DataLoader, optimizer, scaler, train: bool, desc: str):
        self.train(train)
        total_loss = 0.0
        parts_sum = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        amp_enabled = scaler.is_enabled() if scaler is not None else False
        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, targets, _ in tqdm(loader, desc=desc, unit="batch"):
                imgs = _move_to_device(imgs, self.device, non_blocking=_supports_cuda(self.device))
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
                for k in parts_sum:
                    if k in parts:
                        parts_sum[k] += float(parts[k])
        nb = max(1, len(loader))
        return total_loss / nb, {k: v / nb for k, v in parts_sum.items()}
