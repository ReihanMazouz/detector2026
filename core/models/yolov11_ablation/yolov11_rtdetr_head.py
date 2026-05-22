import torch
from tqdm import tqdm

from ..Head.rtdetr import RTDETRHead
from ..base import _move_imgs_to_device, _supports_cuda
from ..yolov11 import YOLOv11
from ...utils.loss import YOLODetectionLoss
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.rtdetr_loss import RTDETRLoss


class YOLOv11RTDETRHead(YOLOv11):
    """YOLOv11 with a YOLO one-to-many head and an RT-DETR-like one-to-one head."""

    def __init__(
        self,
        output_dir,
        num_classes=80,
        strides=None,
        reg_max=16,
        device="cuda:0",
        input_canals=1,
        width_mult=0.25,
        debug=False,
        anisotropic=False,
        p3_size=(64, 64),
        input_hw=None,
        hidden_dim=256,
        num_queries=300,
        num_decoder_layers=3,
        num_heads=8,
        num_decoder_points=4,
        use_deformable_attention=False,
        dim_feedforward=1024,
        dropout=0.0,
        learnt_init_query=False,
    ):
        super().__init__(
            output_dir=output_dir,
            num_classes=num_classes,
            strides=strides,
            reg_max=reg_max,
            device=device,
            input_canals=input_canals,
            width_mult=width_mult,
            debug=debug,
            anisotropic=anisotropic,
            p3_size=p3_size,
            input_hw=input_hw,
        )

        c3 = int(256 * width_mult)
        c4 = int(512 * width_mult)
        c5 = int(1024 * width_mult)
        self.detect_one2one = RTDETRHead(
            in_channels=[c3, c4, c5],
            strides=self.strides,
            num_classes=self.num_classes,
            reg_max=self.reg_max,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            num_decoder_layers=num_decoder_layers,
            num_heads=num_heads,
            num_decoder_points=num_decoder_points,
            use_deformable_attention=use_deformable_attention,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            learnt_init_query=learnt_init_query,
        )
        self.detect_one2one.bias_init(image_size=self.input_hw if self.input_hw is not None else 1024)
        self.criterion_one2many = YOLODetectionLoss(
            num_classes=num_classes,
            strides=self.strides,
            reg_max=self.reg_max,
            device=self.device,
        )
        self.criterion_one2one = RTDETRLoss(num_classes=self.num_classes)
        self.criterion = self.criterion_one2many
        self.active_head = "one2many"
        self._last_image_hw = self.input_hw
        self.to(self.device)

    def forward(self, x, head=None):
        self._last_image_hw = tuple(x.shape[-2:])
        p3_out, p4_out2, p5_out = self.forward_features(x)
        head = self.active_head if head is None else head

        if head == "one2many":
            return self.detect(p3_out, p4_out2, p5_out)
        if head == "one2one":
            image_size = self.input_hw if self.input_hw is not None else self._last_image_hw
            return self.detect_one2one(p3_out, p4_out2, p5_out, image_size=image_size)
        raise ValueError("head must be 'one2many' or 'one2one'.")

    def training_forward(self, imgs):
        return self(imgs, head=self.active_head)

    def get_training_criterion(self):
        if self.active_head == "one2one":
            return self.criterion_one2one
        return self.criterion

    def loss_from_batch(self, outputs, targets):
        if self.active_head != "one2one":
            raise ValueError("loss_from_batch is only for the RT-DETR one-to-one head.")
        batch_size = outputs["pred_logits"].shape[0]
        target_list = targets_from_yolo_tensor(targets, batch_size, outputs["pred_logits"].device)
        return self.criterion_one2one(outputs, target_list)

    def postprocess(self, outputs, cls_out=None, feats=None, conf_thres=0.1, max_det=300, **kwargs):
        if self.active_head != "one2one":
            return super().postprocess(outputs, cls_out, feats, conf_thres=conf_thres, max_det=max_det, **kwargs)

        logits = outputs["pred_logits"]
        boxes = outputs["pred_boxes"]
        probs = logits.softmax(-1)[..., : self.num_classes]
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

            selected_boxes = boxes_i[keep]
            xc, yc, w, h = selected_boxes.unbind(-1)
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

    def train_one2one_head_only(self, sync_from_one2many=False, **_):
        """Freeze backbone, neck and one-to-many head, then train only the RT-DETR one-to-one head."""
        if sync_from_one2many:
            self.sync_one2one_from_one2many()
        self.active_head = "one2one"
        self.criterion = self.criterion_one2one
        for param in self.parameters():
            param.requires_grad = False
        for param in self.detect_one2one.parameters():
            param.requires_grad = True
        self.detect.eval()
        self.detect_one2one.train()

    def use_one2many_head(self):
        self.active_head = "one2many"
        self.criterion = self.criterion_one2many

    def use_one2one_head(self):
        self.active_head = "one2one"
        self.criterion = self.criterion_one2one

    def train(self, mode=True):
        super().train(mode)
        if self.active_head == "one2one":
            self._set_frozen_parts_eval()
            self.detect_one2one.train(mode)
        return self

    def fit(self, *args, **kwargs):
        if self.active_head != "one2one":
            return super().fit(*args, **kwargs)
        return self._fit_one2one_rtdetr(*args, **kwargs)

    def _fit_one2one_rtdetr(
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
        monitor="val_loss",
        save_last_every=1,
        full_eval_every=5,
        run_full_eval=True,
        **_,
    ):
        import csv
        import os
        import time
        from pathlib import Path

        from torch.utils.data import DataLoader

        from ...utils.dataset import YOLODatasetSpecificRes
        from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
        from ..base import _resolve_num_workers

        if dataset != "specificres":
            raise ValueError("YOLOv11RTDETRHead one-to-one fine-tuning currently supports dataset='specificres'.")
        if not select_res or "res_hw" not in select_res or "res_key" not in select_res:
            raise ValueError("select_res={'res_hw': (H, W), 'res_key': 'cfgXXX'} is required.")
        if monitor != "val_loss":
            raise ValueError("YOLOv11RTDETRHead one-to-one fine-tuning currently supports monitor='val_loss'.")

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
        resolved_num_workers = _resolve_num_workers(num_workers)
        loader_kwargs = {
            "batch_size": batch_size,
            "pin_memory": pin_memory,
            "collate_fn": YOLODatasetSpecificRes.collate_fn,
            "num_workers": resolved_num_workers,
            "persistent_workers": bool(persistent_workers) and resolved_num_workers > 0,
        }
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
            train_loss, _ = self._run_one2one_epoch(
                train_loader,
                optimizer,
                scaler,
                train=True,
                desc=f"Epoch {epoch} RTDETR train",
            )
            val_loss, val_parts = self._run_one2one_epoch(val_loader, None, scaler, train=False, desc=f"Epoch {epoch} RTDETR val")
            should_eval = bool(run_full_eval) and (
                (epoch % max(1, int(full_eval_every)) == 0) or epoch == int(epochs)
            )
            if should_eval:
                eval_result = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)
                extra_values = eval_result["extra_values"]
            else:
                extra_values = [None, None, float("nan"), float("nan"), float("nan"), None] if run_full_eval else []

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

            print(
                f"RTDETR one2one epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} "
                f"time={time.perf_counter() - start:.1f}s"
            )
            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_dir / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_dir / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_dir / "avg_recall_curves.png"))
            if bad_epochs >= int(patience):
                print(f"Early stopping on val_loss after {bad_epochs} epochs without improvement.")
                break

    def _run_one2one_epoch(self, loader, optimizer, scaler, train, desc):
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

    def _set_frozen_parts_eval(self):
        for name, module in self.named_children():
            if name != "detect_one2one":
                module.eval()

    def sync_one2one_from_one2many(self):
        self.detect_one2one.dfl.load_state_dict(self.detect.dfl.state_dict())

    def load_yolov11_weights(self, weights_path: str, device="cpu", eval_mode=True):
        """Load compatible weights from a standard YOLOv11 checkpoint."""
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith("detect_one2one."):
                continue
            if key in model_state and model_state[key].shape == value.shape:
                clean_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(clean_state_dict, strict=False)
        try:
            self.sync_one2one_from_one2many()
        except RuntimeError as exc:
            print(f"[WARN] Could not sync one2one DFL weights from YOLOv11 checkpoint: {exc}")
        if eval_mode:
            self.eval()
        return missing_keys, unexpected_keys

    def load_weights(self, weights_path: str, device="cpu", eval_mode=True):
        state_dict = torch.load(weights_path, map_location=device)
        model_state = self.state_dict()
        clean_state_dict = {}
        for key, value in state_dict.items():
            if key in model_state and model_state[key].shape == value.shape:
                clean_state_dict[key] = value

        missing_keys, unexpected_keys = self.load_state_dict(clean_state_dict, strict=False)
        if eval_mode:
            self.eval()
        return missing_keys, unexpected_keys
