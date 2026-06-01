import os
import csv
import copy
import math
import time
from contextlib import nullcontext
from tqdm import tqdm
import json
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchinfo import summary

from collections import defaultdict

from ..utils.dataset import (
    YOLODatasetFusedMultiRes,
    YOLODatasetSpecificRes,
    YOLODatasetSingleRes,
    load_class_index_to_name,
)
from ..utils.display_outputs import plot_batch_with_boxes, plot_batch_matched_boxes, plot_predicted_boxes_batch
from ..utils.training_functions import should_stop_early_from_csv
from ..utils.metrics import match_boxes_iou, ConfusionMatrix, box_iou
from ..utils.post_process import non_max_suppression 
from ..utils.tal import make_anchors, dist2bbox

from ..utils.evaluate import EvalRunner, EvalConfig, MetricsLogger, TrainingPlots

def _to_device(obj, device):
    if torch.is_tensor(obj):
        return obj.to(device)
    if isinstance(obj, (list, tuple)):
        return [ _to_device(o, device) for o in obj ]
    if isinstance(obj, dict):
        return {k: _to_device(v, device) for k, v in obj.items()}
    return obj


def _supports_cuda(device):
    return device.type == "cuda" and torch.cuda.is_available()


def _device_synchronize(device):
    if _supports_cuda(device):
        torch.cuda.synchronize(device)


def _move_imgs_to_device(imgs, device, non_blocking=False):
    if isinstance(imgs, list):
        return [
            img.to(device, dtype=torch.float32, non_blocking=non_blocking)
            for img in imgs
        ]
    return imgs.to(device, dtype=torch.float32, non_blocking=non_blocking)


def _resolve_num_workers(num_workers):
    if num_workers is not None:
        return max(0, int(num_workers))

    cpu_count = os.cpu_count() or 1
    if cpu_count <= 2:
        return 0
    return min(4, cpu_count - 1)


def _as_hw(value):
    if value is None:
        return None
    if isinstance(value, int):
        return (value, value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return (int(value[0]), int(value[1]))
    return None


def _max_hw(resolutions):
    if not resolutions:
        return None
    hw = [_as_hw(res) for res in resolutions]
    hw = [res for res in hw if res is not None]
    if not hw:
        return None
    return (max(h for h, _ in hw), max(w for _, w in hw))


def _tensor_hw(tensor):
    if not torch.is_tensor(tensor) or tensor.ndim < 2:
        return None
    return (int(tensor.shape[-2]), int(tensor.shape[-1]))


def _sample_img_size(sample):
    if isinstance(sample, dict):
        for key in ("imgs", "specs", "img"):
            if key in sample:
                return _sample_img_size(sample[key])
        return None

    if torch.is_tensor(sample):
        return _tensor_hw(sample)

    if isinstance(sample, (list, tuple)):
        tensor_shapes = [_tensor_hw(item) for item in sample]
        tensor_shapes = [shape for shape in tensor_shapes if shape is not None]
        if tensor_shapes:
            return _max_hw(tensor_shapes)

    return None


def _resolve_eval_img_size(model, dataset, fallback=None):
    if hasattr(model, "input_resolutions"):
        img_size = _max_hw(getattr(model, "input_resolutions"))
        if img_size is not None:
            return img_size

    if hasattr(dataset, "res_hw"):
        img_size = _as_hw(getattr(dataset, "res_hw"))
        if img_size is not None:
            return img_size

    if hasattr(dataset, "target_len"):
        target_len = int(getattr(dataset, "target_len"))
        return (target_len, target_len)

    try:
        if len(dataset) > 0:
            img_size = _sample_img_size(dataset[0])
            if img_size is not None:
                return img_size
    except Exception:
        pass

    return _as_hw(fallback) or (1024, 1024)


class _NoOpGradScaler:
    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        return None

    def step(self, optimizer):
        optimizer.step()

    def update(self):
        return None


def _build_grad_scaler(enabled):
    if not enabled:
        return _NoOpGradScaler()
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler(enabled=True)
    if hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "GradScaler"):
        return torch.cuda.amp.GradScaler(enabled=True)
    return _NoOpGradScaler()


def _autocast_context(device, enabled):
    if not enabled:
        return nullcontext()
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type=device.type, enabled=True)
    if device.type == "cuda" and hasattr(torch.cuda, "amp") and hasattr(torch.cuda.amp, "autocast"):
        return torch.cuda.amp.autocast(enabled=True)
    return nullcontext()


def _anomaly_context(enabled):
    if enabled:
        return torch.autograd.detect_anomaly(check_nan=True)
    return nullcontext()

class BaseModel(nn.Module):
    def __init__(self, device="cuda:0", output_dir="outputs"):
        super().__init__()
        requested_device = torch.device(device)
        if requested_device.type == "cuda" and not torch.cuda.is_available():
            self.device = torch.device("cpu")
        else:
            self.device = requested_device
        self.to(self.device)

        self.name = self.__class__.__name__
        self.history = []

        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)


    def save_model_summary(self, model, output_dir, filename="model_summary.txt"):
        """
        Sauvegarde un résumé du modèle dans un fichier texte.
        - model : le modèle PyTorch (hérite de nn.Module)
        - output_dir : dossier où enregistrer le résumé
        - input_shapes : liste des shapes des entrées simulées
        - filename : nom du fichier à écrire
        """
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, filename)

        try:
            input_channels = int(getattr(getattr(model, "conv1", None), "conv", None).in_channels) if hasattr(getattr(model, "conv1", None), "conv") else 1
            if hasattr(model, "input_resolutions"):
                input_shapes = [(1, input_channels, int(height), int(width)) for height, width in model.input_resolutions]
            elif getattr(model, "input_hw", None) is not None:
                height, width = model.input_hw
                input_shapes = [(1, input_channels, int(height), int(width))]
            else:
                input_shapes = [(1, input_channels, 256, 256)]
            dummy_input = [torch.randn(shape).to(model.device) for shape in input_shapes]
            input_data = (dummy_input,) if len(dummy_input) > 1 else dummy_input[0]

            # Résumé structuré avec torchinfo
            model_summary = summary(
                model,
                input_data=input_data,
                depth=3,
                col_names=("input_size", "output_size", "num_params"),
                verbose=0
            )

            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model_summary))

            print(f"[summary] Model summary saved to {path}")

        except Exception as e:
            # Fallback : print str(model)
            with open(path, "w", encoding="utf-8") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model))
                f.write(f"\n\ntorchinfo.summary failed: {e}")

            print(f"[warning] Fallback model summary saved to {path}")

    def _check_dataset_dirs(self, data_dir, split, dataset_type):
        data_path = os.path.join(data_dir, split, "data")
        labels_path = os.path.join(data_dir, split, "labels_detect")

        if not os.path.isdir(data_path):
            raise FileNotFoundError(
                f"[❌ DATASET ERROR] Missing directory: {data_path} "
                f"(dataset='{dataset_type}', split='{split}')"
            )

        if not os.path.isdir(labels_path):
            raise FileNotFoundError(
                f"[❌ DATASET ERROR] Missing directory: {labels_path} "
                f"(dataset='{dataset_type}', split='{split}')"
            )

        if len(os.listdir(data_path)) == 0:
            raise RuntimeError(
                f"[❌ DATASET ERROR] Empty data directory: {data_path}"
            )

        if len(os.listdir(labels_path)) == 0:
            raise RuntimeError(
                f"[❌ DATASET ERROR] Empty labels directory: {labels_path}"
            )


    def fit(self,
        data_dir,
        epochs=100,
        batch_size=32,
        lr=1e-3,
        patience=5,
        debug=False, 
        dataset = 'fused',
        use_amp=True, 
        select_res = None,
        preprocessing="none",
        preprocessing_kwargs=None,
        num_workers=None,
        persistent_workers=True,
        full_eval_every=1,
        save_last_every=5,
        monitor="val_loss",
        validate=True,
        minimal_outputs=False,
        run_full_eval=True):
        """
        Fonction d'apprentissage du modèle.
        """

        pid = os.getpid()
        amp_enabled = bool(use_amp and _supports_cuda(self.device))
        gpu_name = torch.cuda.get_device_name(self.device) if _supports_cuda(self.device) else "CPU"
        print(f"[🚀] Initializing training on device: {self.device} ({gpu_name}) as {pid}")
        full_eval_every = max(1, int(full_eval_every))
        save_last_every = max(1, int(save_last_every))
        plot_every = full_eval_every
        validate = bool(validate)
        minimal_outputs = bool(minimal_outputs)
        run_full_eval = bool(run_full_eval)
        monitor = str(monitor).strip()
        if monitor.lower() == "map50:95":
            monitor = "map50_95"
        elif monitor.lower() == "map50:50":
            monitor = "map50"
        if not run_full_eval and monitor not in {"val_loss", "train_loss"}:
            raise ValueError("run_full_eval=False only supports monitor='val_loss' or monitor='train_loss'.")
        if not validate and monitor != "train_loss":
            print(f"[ℹ] Validation disabled; switching monitor from '{monitor}' to 'train_loss'.")
            monitor = "train_loss"
        monitor_mode = "min" if "loss" in monitor.lower() else "max"

        if not minimal_outputs:
            self.save_model_summary(self, self.output_dir)

        # ---------------- jeux de données ----------------
        DATASETS = {
            "fused": YOLODatasetFusedMultiRes,
            "specificres": YOLODatasetSpecificRes,
            "singleres": YOLODatasetSingleRes,
        }

        dataset_name = dataset.lower() if isinstance(dataset, str) else dataset

        if isinstance(dataset_name, str):
            YOLODataset = DATASETS.get(dataset_name)
            if YOLODataset is None:
                 raise ValueError(f"Unknown dataset type '{dataset}', "
                                  f"choose one of {list(DATASETS)}")
        else:
            YOLODataset = dataset_name

        img_size = None

        self._check_dataset_dirs(data_dir, "train", dataset_name)
        if validate:
            self._check_dataset_dirs(data_dir, "val", dataset_name)

        if dataset_name == "singleres":
            selected_res = select_res.get("res_key") if isinstance(select_res, dict) else select_res
            train_dataset = YOLODataset(
                os.path.join(data_dir, "train/data"),
                os.path.join(data_dir, "train/labels_detect"),
                select_res=selected_res,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )
            val_dataset = None
            if validate:
                val_dataset = YOLODataset(
                    os.path.join(data_dir, "val/data"),
                    os.path.join(data_dir, "val/labels_detect"),
                    select_res=selected_res,
                    preprocessing=preprocessing,
                    preprocessing_kwargs=preprocessing_kwargs,
                )

        elif dataset_name == "specificres":
            # expected: select_res = { "res_hw": (H, W), "res_key": "cfgXXX" }
            res_hw  = select_res.get("res_hw", None)
            res_key = select_res.get("res_key", None)

            if res_hw is None or res_key is None:
                raise ValueError("select_res must contain 'res_hw' and 'res_key'.")

            train_dataset = YOLODataset(
                data_dir=os.path.join(data_dir, "train/data"),
                labels_dir=os.path.join(data_dir, "train/labels_detect"),
                res_hw=res_hw,
                res_key=res_key,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )

            val_dataset = None
            if validate:
                val_dataset = YOLODataset(
                    data_dir=os.path.join(data_dir, "val/data"),
                    labels_dir=os.path.join(data_dir, "val/labels_detect"),
                    res_hw=res_hw,
                    res_key=res_key,
                    preprocessing=preprocessing,
                    preprocessing_kwargs=preprocessing_kwargs,
                )

        else:
            fused_res_keys = None
            if isinstance(select_res, dict):
                fused_res_keys = select_res.get("res_keys")
            if fused_res_keys is None:
                fused_res_keys = getattr(self, "res_keys", None)
            fused_res_keys = tuple(fused_res_keys) if fused_res_keys is not None else None
            train_dataset = YOLODataset(
                os.path.join(data_dir, "train/data"),
                os.path.join(data_dir, "train/labels_detect"),
                res_keys=fused_res_keys,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )
            val_dataset = None
            if validate:
                val_dataset = YOLODataset(
                    os.path.join(data_dir, "val/data"),
                    os.path.join(data_dir, "val/labels_detect"),
                    res_keys=fused_res_keys,
                    preprocessing=preprocessing,
                    preprocessing_kwargs=preprocessing_kwargs,
                )
            if fused_res_keys is not None:
                print(f"[info] Fused MR resolution order: {list(fused_res_keys)}")

        img_size = _resolve_eval_img_size(self, val_dataset or train_dataset, fallback=img_size)

        pin_memory = _supports_cuda(self.device)
        resolved_num_workers = _resolve_num_workers(num_workers)
        persistent_workers = bool(persistent_workers) and resolved_num_workers > 0

        train_loader_kwargs = {
            "batch_size": batch_size,
            "shuffle": True,
            "pin_memory": pin_memory,
            "collate_fn": train_dataset.collate_fn,
            "num_workers": resolved_num_workers,
            "persistent_workers": persistent_workers,
        }
        val_loader_kwargs = None
        if validate:
            val_loader_kwargs = {
                "batch_size": batch_size,
                "shuffle": False,
                "pin_memory": pin_memory,
                "collate_fn": val_dataset.collate_fn,
                "num_workers": resolved_num_workers,
                "persistent_workers": persistent_workers,
            }
        if resolved_num_workers > 0:
            train_loader_kwargs["prefetch_factor"] = 2
            if val_loader_kwargs is not None:
                val_loader_kwargs["prefetch_factor"] = 2

        train_loader = DataLoader(train_dataset, **train_loader_kwargs)
        val_loader = DataLoader(val_dataset, **val_loader_kwargs) if validate else None
        print(
            f"[ℹ] DataLoader config | num_workers={resolved_num_workers} | "
            f"pin_memory={pin_memory} | persistent_workers={persistent_workers}"
        )
        print(
            f"[ℹ] Training cadence | full_eval_every={full_eval_every} | "
            f"plot_every={plot_every} | save_last_every={save_last_every}"
        )
        print(f"[ℹ] Validation | {'enabled' if validate else 'disabled'}")
        print(f"[ℹ] Full detection metrics during training | {'enabled' if run_full_eval else 'disabled'}")
        print(f"[ℹ] Eval image size | {img_size}")
        print(f"[ℹ] Monitor | {monitor} ({monitor_mode})")

        # # → noise_loader conditionnel
        # noise_images = os.path.join(data_dir, "noise/images")
        # noise_labels = os.path.join(data_dir, "noise/labels")
        # if os.path.isdir(noise_images) and os.path.isdir(noise_labels):
        #     noise_dataset = YOLODataset(noise_images, noise_labels)
        #     noise_loader = DataLoader(
        #         noise_dataset, batch_size=batch_size, shuffle=False,
        #         pin_memory=True, collate_fn=noise_dataset.collate_fn)
        #     print(f"[ℹ] Noise dataset détecté, {len(noise_dataset)} échantillons")
        # else:
        #     noise_loader = None
        #     print("[⚠] Pas de noise dataset, PFA/Défaut de seuil activé")
            
        # ---------------- logs & checkpoints -------------
        os.makedirs(self.output_dir, exist_ok=True)
        log_path  = os.path.join(self.output_dir, "train_log.csv")
        best_path = os.path.join(self.output_dir, "best.pt")
        last_path = os.path.join(self.output_dir, "last.pt")

        logger = None if minimal_outputs else MetricsLogger(log_path)

        eval_runner = None
        extra_headers = []
        if run_full_eval:
            eval_runner = EvalRunner(
                output_dir=self.output_dir,
                cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=img_size),
                class_index_to_name=load_class_index_to_name(data_dir),
            )
            extra_headers = eval_runner.extra_headers()

        # ---------------- opti & loss --------------------
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = _build_grad_scaler(amp_enabled)
        criterion = self.criterion
        best_monitor_value = float("inf") if monitor_mode == "min" else float("-inf")
        epochs_without_improvement = 0

        # =================================================
        for epoch in range(1, epochs + 1):
            print(f"\n📚 Epoch {epoch}/{epochs}")
            self.train()

            # ---------- Entraînement ----------
            loss_box_train = loss_cls_train = loss_dfl_train = running_train_loss = 0.0
            first_display = True
            train_data_time = 0.0
            train_step_time = 0.0

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} 🔧 Training", unit="batch")
            batch_wait_start = time.perf_counter()
            for imgs, targets, res_keys in train_pbar:
                if hasattr(self, "res_keys") and res_keys is not None and tuple(res_keys) != tuple(self.res_keys):
                    raise ValueError(
                        f"Batch resolution order mismatch: dataset returned {tuple(res_keys)}, "
                        f"model expects {tuple(self.res_keys)}."
                    )
                batch_ready_time = time.perf_counter()
                train_data_time += batch_ready_time - batch_wait_start

                step_start = time.perf_counter()
                imgs = _move_imgs_to_device(imgs, self.device, non_blocking=pin_memory)

                targets = _to_device(targets, self.device)

                batch = {
                    "batch_idx": targets[:, 0].long(),
                    "cls": targets[:, 1].unsqueeze(1).long(),
                    "bboxes": targets[:, 2:6],
                    "snr": targets[:, 6].unsqueeze(1)
                }


                optimizer.zero_grad()
                with _autocast_context(self.device, amp_enabled):
                    dist_out, clsobj_out = self(imgs)
                    # for dist, cls in zip(dist_out, clsobj_out):
                    #     print('dist_out.shape === ',dist.shape)
                    #     print('clsobj_out.shape === ',cls.shape)
                    feats = dist_out
                    pred_scores = torch.cat([x.flatten(2).permute(0, 2, 1) for x in clsobj_out], dim=1)
                    pred_distri = torch.cat([x.flatten(2).permute(0, 2, 1) for x in dist_out], dim=1)
                    loss, loss_dict, debug_data = criterion(pred_distri, pred_scores, batch, feats=feats)
                    if not torch.isfinite(loss):
                        pred_scores_finite = pred_scores[torch.isfinite(pred_scores)]
                        pred_distri_finite = pred_distri[torch.isfinite(pred_distri)]
                        score_stats = (
                            f"min={pred_scores_finite.min().item():.6g} max={pred_scores_finite.max().item():.6g}"
                            if pred_scores_finite.numel()
                            else "all non-finite"
                        )
                        distr_stats = (
                            f"min={pred_distri_finite.min().item():.6g} max={pred_distri_finite.max().item():.6g}"
                            if pred_distri_finite.numel()
                            else "all non-finite"
                        )
                        raise FloatingPointError(
                            "Non-finite YOLO loss before optimizer.step: "
                            f"loss={loss.item()} parts={loss_dict} "
                            f"pred_scores({score_stats}) pred_distri({distr_stats}) "
                            f"num_targets={int(targets.shape[0])}"
                        )

                loss_box_train += loss_dict[0]
                loss_cls_train += loss_dict[1]
                loss_dfl_train += loss_dict[2]
                running_train_loss += loss.item()

                # backward avec AMP
                scaled_loss = scaler.scale(loss)
                with _anomaly_context(getattr(self, "_detect_anomaly", False)):
                    scaled_loss.backward()
                scaler.unscale_(optimizer)
                trainable_params = [p for p in self.parameters() if p.requires_grad and p.grad is not None]
                grad_clip_norm = getattr(self, "_grad_clip_norm", None)
                max_norm = float(grad_clip_norm) if grad_clip_norm is not None else float("inf")
                try:
                    total_grad_norm = nn.utils.clip_grad_norm_(
                        trainable_params,
                        max_norm=max_norm,
                        error_if_nonfinite=True,
                    )
                except RuntimeError as exc:
                    bad_grads = []
                    for name, param in self.named_parameters():
                        if param.grad is None:
                            continue
                        finite_mask = torch.isfinite(param.grad)
                        if finite_mask.all():
                            continue
                        grad = param.grad.detach()
                        finite = grad[finite_mask]
                        stats = (
                            f"finite_min={finite.min().item():.6g} finite_max={finite.max().item():.6g}"
                            if finite.numel()
                            else "all_grad_values_non_finite"
                        )
                        bad_grads.append(
                            f"{name}: shape={tuple(grad.shape)} "
                            f"nan={(torch.isnan(grad)).sum().item()} "
                            f"inf={(torch.isinf(grad)).sum().item()} {stats}"
                        )
                        if len(bad_grads) >= 8:
                            break
                    raise FloatingPointError(
                        "Non-finite gradient before optimizer.step: "
                        f"loss={loss.item()} parts={loss_dict} "
                        f"bad_grads={bad_grads}"
                    ) from exc
                scaler.step(optimizer)
                scaler.update()
                if getattr(self, "_check_finite_after_step", False):
                    for name, param in self.named_parameters():
                        if not torch.isfinite(param).all():
                            raise FloatingPointError(
                                "Non-finite parameter after optimizer.step: "
                                f"{name} grad_norm={float(total_grad_norm):.6g} "
                                f"loss={loss.item()} parts={loss_dict}"
                            )
                _device_synchronize(self.device)
                train_step_time += time.perf_counter() - step_start

                if first_display and debug:
                    plot_batch_matched_boxes(
                        imgs=imgs,
                        gt_boxes_list=[d["gt_boxes"] for d in debug_data],
                        pred_boxes_list=[d["task_selected_pred_boxes_abs"] for d in debug_data],
                        anchors_list=[d["task_selected_anchor_points_abs"] for d in debug_data],
                        save_path=os.path.join(self.output_dir, f"task_align_epoch{epoch:02d}_batch.png"), 
                        optimal_shape=img_size
                    )

                    # plot_predicted_boxes_batch(
                    #     imgs,
                    #     batch_pred_boxes=[d["pred_bboxes_abs"].detach().cpu() for d in debug_data],
                    #     save_path=os.path.join(self.output_dir, f"predictions_epoch{epoch:02d}_batch.png"),
                    #     max_boxes=200
                    # )

                    # processed_outputs = self.postprocess(dist_out, clsobj_out, feats)
                    # processed_targets = []
                    # for pred in processed_outputs:
                    #     if pred is not None and len(pred) > 0:
                    #         boxes = pred[:, [5, 0, 1, 2, 3]]
                    #     else:
                    #         boxes = torch.zeros((0, 5))
                    #     processed_targets.append(boxes)

                    # plot_batch_with_boxes(
                    #     imgs[:len(processed_outputs)],
                    #     processed_targets,
                    #     class_names=getattr(self, 'class_names', None),
                    #     save_path=os.path.join(self.output_dir, f"postprocessed_epoch{epoch:02d}.png"),
                    #     max_batch_size=1, 
                    #     optimal_shape=img_size
                    # )
                    first_display = False

                train_pbar.set_postfix(loss=loss.item())
                batch_wait_start = time.perf_counter()


            n_train_batches = max(1, len(train_loader))
            loss_box_train /= n_train_batches
            loss_cls_train /= n_train_batches
            loss_dfl_train  /= n_train_batches
            train_loss = running_train_loss / n_train_batches

            # ---------- Validation ----------
            loss_box_val = loss_cls_val = loss_dfl_val = val_loss = float("nan")
            val_data_time = 0.0
            val_step_time = 0.0
            n_val_batches = 1

            if validate:
                self.eval()
                loss_box_val = loss_cls_val = loss_dfl_val = running_val_loss = 0.0

                val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} 🧪 Validation", unit="batch")
                with torch.no_grad():
                    batch_wait_start = time.perf_counter()
                    for imgs, targets, res_keys in val_pbar:
                        if hasattr(self, "res_keys") and res_keys is not None and tuple(res_keys) != tuple(self.res_keys):
                            raise ValueError(
                                f"Validation batch resolution order mismatch: dataset returned {tuple(res_keys)}, "
                                f"model expects {tuple(self.res_keys)}."
                            )
                        batch_ready_time = time.perf_counter()
                        val_data_time += batch_ready_time - batch_wait_start

                        step_start = time.perf_counter()
                        imgs = _move_imgs_to_device(imgs, self.device, non_blocking=pin_memory)

                        targets = _to_device(targets, self.device)
                        batch = {
                            "batch_idx": targets[:, 0].long(),
                            "cls": targets[:, 1].unsqueeze(1).long(),
                            "bboxes": targets[:, 2:6],
                            "snr": targets[:, 6].unsqueeze(1)
                        }

                        with _autocast_context(self.device, amp_enabled):
                            dist_out, clsobj_out = self(imgs)
                            feats = dist_out
                            pred_scores = torch.cat([x.flatten(2).permute(0, 2, 1) for x in clsobj_out], dim=1)
                            pred_distri = torch.cat([x.flatten(2).permute(0, 2, 1) for x in dist_out], dim=1)
                            val_loss_batch, loss_dict_val, _ = criterion(pred_distri, pred_scores, batch, feats=feats)
                            if not torch.isfinite(val_loss_batch):
                                raise FloatingPointError(
                                    "Non-finite YOLO validation loss: "
                                    f"loss={val_loss_batch.item()} parts={loss_dict_val} "
                                    f"num_targets={int(targets.shape[0])}"
                                )
                        _device_synchronize(self.device)
                        val_step_time += time.perf_counter() - step_start

                        running_val_loss += val_loss_batch.item()
                        loss_box_val += loss_dict_val[0]
                        loss_cls_val += loss_dict_val[1]
                        loss_dfl_val += loss_dict_val[2]

                        val_pbar.set_postfix(val_loss=val_loss_batch.item())
                        batch_wait_start = time.perf_counter()

                n_val_batches = max(1, len(val_loader))
                val_loss = running_val_loss / n_val_batches
                loss_box_val /= n_val_batches
                loss_cls_val /= n_val_batches
                loss_dfl_val  /= n_val_batches

            monitor_value = None
            if monitor == "val_loss":
                monitor_value = val_loss
            elif monitor == "train_loss":
                monitor_value = train_loss

            should_run_full_eval = (
                run_full_eval
                and
                validate
                and (
                    (epoch % full_eval_every == 0)
                    or (epoch == epochs)
                    or (monitor == "val_loss" and not hasattr(self, "_best_monitor_value"))
                )
            )
            if should_run_full_eval:
                result = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)
                ev = result["extra_values"]
            else:
                ev = [None, None, *([float("nan")] * 7), None]
                result = {
                    "did_eval": False,
                    "extra_headers": extra_headers,
                    "extra_values": ev if run_full_eval else [],
                    "json_path": None,
                    "full_metrics": None,
                }

            if monitor == "map50_95":
                monitor_value = result["extra_values"][1]
            elif monitor == "map50":
                monitor_value = result["extra_values"][0]

            if not hasattr(self, "_best_monitor_value"):
                self._best_monitor_value = float("inf") if monitor_mode == "min" else float("-inf")

            is_best_monitor = False
            if monitor_value is not None:
                if monitor_mode == "min":
                    is_best_monitor = monitor_value < self._best_monitor_value
                else:
                    is_best_monitor = monitor_value > self._best_monitor_value

            improved_for_early_stop = False
            if monitor_value is not None:
                if monitor_mode == "min":
                    improved_for_early_stop = monitor_value < best_monitor_value
                else:
                    improved_for_early_stop = monitor_value > best_monitor_value
                if improved_for_early_stop:
                    best_monitor_value = float(monitor_value)
                    epochs_without_improvement = 0
                else:
                    epochs_without_improvement += 1

            print(
                f"📉 Summary Epoch {epoch:02d} | "
                f"Train: {train_loss:.4f} (box={loss_box_train:.3f}, cls={loss_cls_train:.3f}, dfl={loss_dfl_train:.3f}) | "
                f"Val:   {val_loss:.4f} (box={loss_box_val:.3f}, cls={loss_cls_val:.3f}, dfl={loss_dfl_val:.3f}) | "
                f"mAP50={ev[0] if ev[0] is not None else 'NA'} | "
                f"mAP50_95={ev[1] if ev[1] is not None else 'NA'} | "
                f"avgRec(low/med/high)={ev[2]:.3f}/{ev[3]:.3f}/{ev[4]:.3f}"
            )
            print(
                f"⏱ Epoch {epoch:02d} timings | "
                f"train data={train_data_time / n_train_batches:.4f}s/batch | "
                f"train step={train_step_time / n_train_batches:.4f}s/batch | "
                f"val data={val_data_time / n_val_batches:.4f}s/batch | "
                f"val step={val_step_time / n_val_batches:.4f}s/batch"
            )

            # ---------- Logging CSV (colonnes de base + extras EvalRunner) ----------
            if logger is not None:
                logger.log(
                    epoch=epoch,
                    train_loss=float(train_loss), val_loss=float(val_loss),
                    loss_box_train=float(loss_box_train), loss_cls_train=float(loss_cls_train), loss_dfl_train=float(loss_dfl_train),
                    loss_box_val=float(loss_box_val),     loss_cls_val=float(loss_cls_val),     loss_dfl_val=float(loss_dfl_val),
                    extra_headers=result["extra_headers"],
                    extra_values=result["extra_values"],
                )

            # --- CHECKPOINT & PLOTS ---
            if (epoch % save_last_every == 0) or (epoch == epochs):
                torch.save(self.state_dict(), last_path)

            if (not minimal_outputs) and ((epoch % plot_every == 0) or (epoch == epochs)):
                TrainingPlots.plot_losses(log_path, save_path=os.path.join(self.output_dir, "loss_curves.png"))
                if run_full_eval:
                    TrainingPlots.plot_maps(log_path,   save_path=os.path.join(self.output_dir, "map_curves.png"))
                    TrainingPlots.plot_avg_recalls(log_path, save_path=os.path.join(self.output_dir, "avg_recall_curves.png"))
                    TrainingPlots.plot_size_recalls(log_path, save_path=os.path.join(self.output_dir, "recall_size_curves.png"))
                    TrainingPlots.plot_box_iou(log_path, save_path=os.path.join(self.output_dir, "box_iou_curves.png"))

            if is_best_monitor and not minimal_outputs:
                self._best_monitor_value = monitor_value
                torch.save(self.state_dict(), best_path)
                print(f"💾 Best model ({monitor}={monitor_value:.4f}) saved.")

            if minimal_outputs:
                if epochs_without_improvement >= int(patience):
                    print(f"⛔️ Early stopping déclenché sur {monitor} (aucune amélioration ≥ {patience} epochs).")
                    break
            else:
                if should_stop_early_from_csv(log_path, patience=patience, monitor=monitor, mode=monitor_mode):
                    print(f"⛔️ Early stopping déclenché sur {monitor} (aucune amélioration ≥ {patience} epochs).")
                    break

        print("✅ Entraînement terminé.")


    def evaluate(self, val_loader, conf_thresh, return_confmat=True, debug=False):
        self.eval()
        total_gt, total_tp = 0, 0
        num_classes = getattr(self, "num_classes", 1)
        confusion = ConfusionMatrix(nc=num_classes, iou_thres=0.5)
        snr_metrics = defaultdict(lambda: {"total_tp": 0, "total_gt": 0})

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="🔍 Evaluating", unit="batch")
            for imgs, targets in val_pbar:
                # Envoi sur device
                imgs = _move_imgs_to_device(imgs, self.device)
                targets = _to_device(targets, self.device)

                # Inférence
                dist_out, clsobj_out = self(imgs)
                feats = dist_out
                preds = self.postprocess(dist_out, clsobj_out, feats, conf_thres=conf_thresh)

                for i, pred in enumerate(preds):
                    # Extraction des cibles pour l'image i
                    target = targets[targets[:, 0] == i]
                    gt_classes = target[:, 1] if len(target) > 0 else torch.zeros((0,), device=imgs.device)
                    snrs = target[:, 6].cpu() if len(target) > 0 else torch.tensor([])

                    # On fixe la résolution (ici 1024×1024)
                    img_h, img_w = 1024, 1024

                    # Conversion GT relative → absolue
                    if len(target) > 0:
                        gt_rel = target[:, 2:6]  # [xc, yc, w, h] relatifs
                        x_c, y_c, w_r, h_r = gt_rel.unbind(dim=1)
                        x1 = (x_c - w_r/2) * img_w
                        y1 = (y_c - h_r/2) * img_h
                        x2 = (x_c + w_r/2) * img_w
                        y2 = (y_c + h_r/2) * img_h
                        gt_boxes_abs = torch.stack([x1, y1, x2, y2], dim=1)
                    else:
                        gt_boxes_abs = torch.zeros((0, 4), device=imgs.device)

                    # Mise à jour du total GT et des GT par bin SNR
                    if len(target) > 0:
                        total_gt += len(gt_classes)
                        for j in range(len(gt_boxes_abs)):
                            snr_bin = int(snrs[j].item())
                            snr_metrics[snr_bin]["total_gt"] += 1

                    # Préparation des détections
                    if len(pred) > 0:
                        pred_boxes   = pred[:, :4]
                        pred_classes = pred[:, 5].long()

                        # Confusion matrix (toutes détections vs tous GT)
                        det_np = torch.cat([pred_boxes,   pred_classes.unsqueeze(1)], dim=1).cpu().numpy()
                        gt_np  = torch.cat([gt_boxes_abs, gt_classes.unsqueeze(1)],      dim=1).cpu().numpy()
                        confusion.process(det_np, gt_np)

                        # --------------- MATCHING IoU ---------------
                        # Renvoie [(pred_idx, gt_idx, iou), ...]
                        matches = match_boxes_iou(pred_boxes, gt_boxes_abs, iou_thresh=0.1)

                        # Nombre de vrais positifs
                        tp = len(matches)
                        total_tp += tp

                        # On incrémente les TP par bin SNR et on stocke les IoU si besoin
                        for _, gt_idx, iou in matches:
                            snr_bin = int(snrs[gt_idx].item())
                            snr_metrics[snr_bin]["total_tp"] += 1
                            # si vous voulez garder la liste des IoU, vous pouvez :
                            # stats_by_snr[snr_bin]["ious"].append(iou)

        # === Calcul des métriques globales ===
        confmat = confusion.matrix
        tp_vec = confmat.diagonal()
        fp_vec = confmat.sum(0) - tp_vec
        fn_vec = confmat.sum(1) - tp_vec

        precision = (tp_vec / (tp_vec + fp_vec + 1e-6)).mean().item()
        recall    = (tp_vec / (tp_vec + fn_vec + 1e-6)).mean().item()
        f1        = 2 * precision * recall / (precision + recall + 1e-6)
        accuracy  = tp_vec.sum().item() / (confmat.sum().item() + 1e-6)
        bal_acc   = ((tp_vec / (tp_vec + fn_vec + 1e-6)) +
                    (tp_vec / (tp_vec + fp_vec + 1e-6))).mean().item()
        pd        = total_tp / (total_gt + 1e-6)

        pd_by_snr = {
            bin_name: v["total_tp"] / (v["total_gt"] + 1e-6)
            for bin_name, v in snr_metrics.items()
        }

        metrics = dict(
            conf_thresh_pfa=conf_thresh,
            pd=pd,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            balanced_accuracy=bal_acc,
            map50=0.0,  # à calculer si besoin
            pd_by_snr=pd_by_snr
        )
        if return_confmat:
            metrics["confusion_matrix"] = confmat

        return metrics



    def predict(self, image_tensor, to_plot=False, conf_threshold=0.1, labels=None, optimal_shape=(1024,1024), iou_thres=0.1, iou_same_box = 0.9):
        self.eval()
        image_tensor = _move_imgs_to_device(image_tensor, self.device)

        with torch.no_grad():
            dist_out, clsobj_out = self(image_tensor)
            feats = dist_out
            processed_output = self.postprocess(dist_out, clsobj_out, feats, conf_thres=conf_threshold, iou_thres=iou_thres, iou_same_box=iou_same_box)

            if to_plot:
                processed_targets = []
                for pred in processed_output:
                    if pred is not None and len(pred) > 0:
                        boxes = pred[:, [5, 0, 1, 2, 3]]  # cls, x1, y1, x2, y2
                    else:
                        boxes = torch.zeros((0, 5))
                    processed_targets.append(boxes)

                # 💡 Ajout de `labels` ici
                plot_batch_with_boxes(
                    feats=image_tensor,  # images d'entrée
                    targets=processed_targets,  # prédictions (rouge)
                    class_names=getattr(self, 'class_names', None),
                    save_path=to_plot,
                    max_batch_size=3,
                    labels=labels,  # ← ground truth (vert)
                    optimal_shape=optimal_shape
                )

            return processed_output, dist_out, clsobj_out


    def postprocess(self, dist_out, cls_out, feats, conf_thres=0.1, iou_thres=0.1, iou_same_box=0.9, without_nms=False):
        """
        Postprocessing like YOLOv11 without objectness, with NMS.
        """
        # dist_out et cls_out sont des listes de (B, C, H, W)
        pred_dist = torch.cat([x.flatten(2) for x in dist_out], dim=2).permute(0, 2, 1)  # (B, N, 4*reg_max)
        pred_cls  = torch.cat([x.flatten(2) for x in cls_out],  dim=2).permute(0, 2, 1)  # (B, N, C)
        B, N, _ = pred_dist.shape

        # (3) Anchors
        anchor_points, stride_tensor = make_anchors(feats, self.strides)
        anchor_points = anchor_points.to(pred_dist.device)
        stride_tensor = stride_tensor.to(pred_dist.device)
        stride_tensor_boxes = torch.cat([stride_tensor, stride_tensor], dim=1)

        # (4) DFL Projection — ⚠️ Cast `proj` to same dtype & device as `pred_dist`
        proj = torch.arange(self.reg_max, dtype=torch.float, device=pred_dist.device)
        proj = proj.to(dtype=pred_dist.dtype)  # AMP compatibility
        pred_ltrb = pred_dist.view(B, N, 4, self.reg_max).softmax(3).matmul(proj)

        # (5) Convertir les distances en boîtes
        pred_bboxes = dist2bbox(pred_ltrb, anchor_points, xywh=False)  # (B, N, 4)
        pred_bboxes_abs = pred_bboxes * stride_tensor_boxes  # (B, N, 4)

        # (6) Score des classes
        cls_scores = pred_cls.sigmoid()  # (B, N, C)

        # Convertir xyxy → xywh avant concat
        x1y1 = pred_bboxes_abs[..., :2]
        x2y2 = pred_bboxes_abs[..., 2:4]
        xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        pred_bboxes_xywh = torch.cat([xy, wh], dim=-1)  # (B, N, 4)

        # (7) Empilement [x, y, w, h, conf1, conf2, ..., confN]
        pred_final = torch.cat([pred_bboxes_xywh, cls_scores], dim=2)  # (B, N, 4 + C)
        prediction = pred_final.permute(0, 2, 1)  # (B, 4+C, N)

        if without_nms:
            return prediction

        # (8) NMS
        results = non_max_suppression(
            prediction=prediction,
            conf_thres=conf_thres,
            iou_thres=iou_thres,
            nc=self.num_classes,
            in_place=True,
            multi_label=True
        )

        # ========================================================
        #  (9) CLASS-AGNOSTIC MERGE-NMS
        #      -> keeps only the highest-class prediction
        #         when two boxes overlap strongly
        # ========================================================
        final_results = []

        for det in results:                     # det: (num_det, 6) : [x1,y1,x2,y2,score,class]
            if det is None or len(det) == 0:
                final_results.append(det)
                continue

            boxes  = det[:, :4]
            scores = det[:, 4]
            labels = det[:, 5]

            # Compute pairwise IoU
            iou_matrix = box_iou(boxes, boxes)

            keep = []
            removed = torch.zeros(len(det), dtype=torch.bool, device=det.device)

            for i in range(len(det)):
                if removed[i]:
                    continue

                # Find all boxes overlapping strongly
                ious = iou_matrix[i]   # IoU with all
                dup_idx = (ious > iou_same_box).nonzero(as_tuple=True)[0]

                best = dup_idx[scores[dup_idx].argmax()]

                keep.append(best.item())

                # Mark the others as removed
                dup_idx = dup_idx.tolist()
                for j in dup_idx:
                    if j != best:
                        removed[j] = True

            keep = sorted(list(set(keep)))
            final_results.append(det[keep])

        return final_results



    def load_weights(self, weights_path: str, device="cpu", eval_mode=True):
        state_dict = torch.load(weights_path, map_location=device)

        model_state = self.state_dict()
        clean_state_dict = {}

        for k, v in state_dict.items():
            # garder la clé seulement si elle existe ET si la shape correspond
            if k in model_state and model_state[k].shape == v.shape:
                clean_state_dict[k] = v
            # else:
            #     print(f"[skip] incompatible key: {k}  "
            #         f"checkpoint={tuple(v.shape)}, model={tuple(model_state[k].shape) if k in model_state else 'missing'}")

        missing_keys, unexpected_keys = self.load_state_dict(clean_state_dict, strict=False)

        if eval_mode:
            self.eval()

        return missing_keys, unexpected_keys
