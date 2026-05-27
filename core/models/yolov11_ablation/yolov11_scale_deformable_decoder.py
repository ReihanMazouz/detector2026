from __future__ import annotations

import csv
import os
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils.dataset import YOLODatasetSpecificRes
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from ...utils.rtdetr_loss import RTDETRLoss
from ..Head.rtdetr import MLP, RTDETRDecoderLayer, RTDETRHead, inverse_sigmoid
from ..base import _move_imgs_to_device, _resolve_num_workers, _supports_cuda
from .yolov11_no_neck import YOLOv11NoNeck


class YOLOv11NoNeckScaleDeformableDecoder(YOLOv11NoNeck):
    """No-neck YOLOv11 backbone with a scale-specialized deformable detection head."""

    level_names = ("p3", "p4", "p5")

    def __init__(
        self,
        *args,
        hidden_dim: int = 128,
        query_counts: tuple[int, int, int] = (64, 32, 16),
        num_decoder_layers: int = 3,
        num_heads: int = 8,
        num_decoder_points: int = 16,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        matcher_num_threads: int = 1,
        freeze_backbone: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if len(query_counts) != 3:
            raise ValueError("query_counts must contain exactly three values for P3, P4 and P5.")

        width_mult = kwargs.get("width_mult", 0.25)
        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)
        channels = (c3, c4, c5)

        self.hidden_dim = int(hidden_dim)
        self.scale_decoders = nn.ModuleDict()
        for level_name, channels_l, stride_l, queries_l in zip(self.level_names, channels, self.strides, query_counts):
            self.scale_decoders[level_name] = RTDETRHead(
                in_channels=[channels_l],
                strides=[stride_l],
                num_classes=self.num_classes,
                reg_max=self.reg_max,
                hidden_dim=hidden_dim,
                num_queries=int(queries_l),
                num_decoder_layers=1,
                num_heads=num_heads,
                num_decoder_points=num_decoder_points,
                use_deformable_attention=True,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                learnt_init_query=False,
            )
            self.scale_decoders[level_name].bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)

        self.fusion_input_proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(channels_l, hidden_dim, kernel_size=1, bias=False), nn.BatchNorm2d(hidden_dim))
            for channels_l in channels
        )
        self.fusion_layers = nn.ModuleList(
            RTDETRDecoderLayer(
                hidden_dim=hidden_dim,
                num_heads=num_heads,
                num_levels=3,
                num_points=num_decoder_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(max(0, int(num_decoder_layers) - 1))
        )
        self.fusion_score_heads = nn.ModuleList(nn.Linear(hidden_dim, self.num_classes) for _ in self.fusion_layers)
        self.fusion_bbox_heads = nn.ModuleList(MLP(hidden_dim, hidden_dim, 4, num_layers=3) for _ in self.fusion_layers)
        self.fusion_query_pos_head = MLP(4, 2 * hidden_dim, hidden_dim, num_layers=2)
        self._reset_fusion_heads()

        self.query_counts = tuple(int(value) for value in query_counts)
        self.num_decoder_layers = int(num_decoder_layers)
        self.num_decoder_points = int(num_decoder_points)
        self.freeze_backbone = bool(freeze_backbone)
        self.criterion = RTDETRLoss(num_classes=self.num_classes, matcher_num_threads=matcher_num_threads)
        self._last_image_hw = self.input_hw
        self.to(self.device)

    def _reset_fusion_heads(self):
        import math

        prior_prob = 0.01
        bias_value = -math.log((1.0 - prior_prob) / prior_prob)
        for cls_head, box_head in zip(self.fusion_score_heads, self.fusion_bbox_heads):
            nn.init.constant_(cls_head.bias, bias_value)
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    def forward(self, x, head=None):
        del head
        self._last_image_hw = tuple(x.shape[-2:])
        image_size = self.input_hw if self.input_hw is not None else self._last_image_hw
        features = self.forward_features(x)
        per_level = {}
        level_queries = []
        level_boxes = []
        for level_name, decoder, feature in zip(self.level_names, self.scale_decoders.values(), features):
            level_output = decoder(feature, image_size=image_size, return_decoder_states=True)
            per_level[level_name] = level_output
            level_queries.append(level_output["decoder_states"][-1])
            level_boxes.append(level_output["pred_boxes"].detach())

        query = torch.cat(level_queries, dim=1)
        reference_boxes = torch.cat(level_boxes, dim=1)
        memory, shapes = self._fusion_memory(features)

        fusion_outputs = []
        for layer, cls_head, box_head in zip(self.fusion_layers, self.fusion_score_heads, self.fusion_bbox_heads):
            query = layer(
                query,
                reference_boxes,
                memory,
                shapes,
                query_pos=self.fusion_query_pos_head(reference_boxes),
            )
            bbox_delta = box_head(query)
            pred_boxes = torch.sigmoid(bbox_delta + inverse_sigmoid(reference_boxes))
            pred_logits = cls_head(query)
            fusion_outputs.append({"pred_logits": pred_logits, "pred_boxes": pred_boxes})
            reference_boxes = pred_boxes.detach() if self.training else pred_boxes

        if fusion_outputs:
            final_output = fusion_outputs[-1]
            final_output["aux_outputs"] = fusion_outputs[:-1]
        else:
            final_output = {
                "pred_logits": torch.cat([per_level[name]["pred_logits"] for name in self.level_names], dim=1),
                "pred_boxes": torch.cat([per_level[name]["pred_boxes"] for name in self.level_names], dim=1),
                "aux_outputs": [],
            }
        final_output["per_level"] = per_level
        return final_output

    def _fusion_memory(self, features):
        tokens = []
        shapes = []
        for projection, feature in zip(self.fusion_input_proj, features):
            projected = projection(feature)
            height, width = projected.shape[2:]
            tokens.append(projected.flatten(2).transpose(1, 2))
            shapes.append((height, width))
        return torch.cat(tokens, dim=1), shapes

    def train(self, mode=True):
        nn.Module.train(self, mode)
        if mode and self.freeze_backbone:
            for name in (
                "conv1",
                "conv2",
                "c3_1",
                "conv3",
                "c3_2",
                "conv4",
                "c3_3",
                "conv5",
                "c3_4",
                "sppf",
                "attn",
                "detect",
                "detect_one2one",
            ):
                module = getattr(self, name, None)
                if module is not None:
                    module.eval()
            self.scale_decoders.train(True)
        return self

    def set_decoder_only_training(self):
        self.freeze_backbone = True
        for param in self.parameters():
            param.requires_grad = False
        for param in self.scale_decoders.parameters():
            param.requires_grad = True
        for param in self.fusion_input_proj.parameters():
            param.requires_grad = True
        for param in self.fusion_layers.parameters():
            param.requires_grad = True
        for param in self.fusion_score_heads.parameters():
            param.requires_grad = True
        for param in self.fusion_bbox_heads.parameters():
            param.requires_grad = True
        for param in self.fusion_query_pos_head.parameters():
            param.requires_grad = True
        self.detect.eval()
        self.scale_decoders.train()
        self.fusion_input_proj.train()
        self.fusion_layers.train()
        self.fusion_score_heads.train()
        self.fusion_bbox_heads.train()
        self.fusion_query_pos_head.train()

    def load_yolov11_weights(self, weights_path: str, device="cpu", eval_mode=False):
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        clean_state = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and model_state[key].shape == value.shape
        }
        missing, unexpected = self.load_state_dict(clean_state, strict=False)
        for level_index, level_name in enumerate(self.level_names):
            try:
                decoder = self.scale_decoders[level_name]
                decoder.cv_dist[0].load_state_dict(self.detect.cv_dist[level_index].state_dict())
                decoder.dfl.load_state_dict(self.detect.dfl.state_dict())
            except RuntimeError as exc:
                print(f"[WARN] Could not sync YOLO bbox branch into {level_name} decoder: {exc}")
        for level_index, projection in enumerate(self.fusion_input_proj):
            scale_projection = self.scale_decoders[self.level_names[level_index]].input_proj[0]
            try:
                projection.load_state_dict(scale_projection.state_dict())
            except RuntimeError as exc:
                print(f"[WARN] Could not sync fusion projection for {self.level_names[level_index]}: {exc}")
        if eval_mode:
            self.eval()
        return missing, unexpected

    def _targets_for_level(self, targets: torch.Tensor, level_index: int) -> torch.Tensor:
        if targets.numel() == 0:
            return targets
        if self.input_hw is None:
            raise ValueError("input_hw must be set before scale-specific target assignment.")
        image_h, image_w = float(self.input_hw[0]), float(self.input_hw[1])
        width_px = targets[:, 4].clamp(min=0.0) * image_w
        height_px = targets[:, 5].clamp(min=0.0) * image_h
        object_scale = torch.sqrt((width_px * height_px).clamp(min=0.0))

        stride_values = torch.as_tensor(
            [max(float(stride[0]), float(stride[1])) if isinstance(stride, (tuple, list)) else float(stride) for stride in self.strides],
            device=targets.device,
            dtype=targets.dtype,
        )
        boundary_34 = torch.sqrt(stride_values[0] * stride_values[1])
        boundary_45 = torch.sqrt(stride_values[1] * stride_values[2])
        if level_index == 0:
            mask = object_scale < boundary_34
        elif level_index == 1:
            mask = (object_scale >= boundary_34) & (object_scale < boundary_45)
        else:
            mask = object_scale >= boundary_45
        return targets[mask]

    def loss_from_batch(self, outputs, targets):
        batch_size = outputs["pred_logits"].shape[0]
        total = outputs["pred_logits"].sum() * 0.0
        parts_sum = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        for level_index, level_name in enumerate(self.level_names):
            level_targets = self._targets_for_level(targets, level_index)
            target_list = targets_from_yolo_tensor(level_targets, batch_size, outputs["pred_logits"].device)
            loss, parts = self.criterion(outputs["per_level"][level_name], target_list)
            total = total + loss
            for key in parts_sum:
                parts_sum[key] += float(parts.get(key, 0.0))
        all_targets = targets_from_yolo_tensor(targets, batch_size, outputs["pred_logits"].device)
        loss, parts = self.criterion(
            {
                "pred_logits": outputs["pred_logits"],
                "pred_boxes": outputs["pred_boxes"],
                "aux_outputs": outputs.get("aux_outputs", []),
            },
            all_targets,
        )
        total = total + loss
        for key in parts_sum:
            parts_sum[key] += float(parts.get(key, 0.0))
        normalizer = len(self.level_names) + 1
        return total, {key: value / normalizer for key, value in parts_sum.items()}

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
            detections = torch.stack((x1, y1, x2, y2, scores_i[keep], labels_i[keep].to(logits.dtype)), dim=1)
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
            raise ValueError("YOLOv11NoNeckScaleDeformableDecoder supports dataset='specificres'.")
        if not select_res or "res_hw" not in select_res or "res_key" not in select_res:
            raise ValueError("select_res={'res_hw': (H, W), 'res_key': 'cfgXXX'} is required.")
        if monitor != "val_loss":
            raise ValueError("YOLOv11NoNeckScaleDeformableDecoder currently supports monitor='val_loss'.")

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
            train_loss, _ = self._run_scale_decoder_epoch(train_loader, optimizer, scaler, train=True, desc=f"Epoch {epoch} scale decoder train")
            val_loss, val_parts = self._run_scale_decoder_epoch(val_loader, None, scaler, train=False, desc=f"Epoch {epoch} scale decoder val")
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

            print(f"Scale decoder epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} time={time.perf_counter() - start:.1f}s")
            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
            if bad_epochs >= int(patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break

    def _run_scale_decoder_epoch(self, loader, optimizer, scaler, train, desc):
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
