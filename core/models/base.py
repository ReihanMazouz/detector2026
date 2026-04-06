import os
import csv
import copy
import math
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


class _NoOpGradScaler:
    def scale(self, loss):
        return loss

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
            dummy_input = [torch.randn(shape).to(model.device) for shape in self.input_resolutions]
            input_data = dummy_input if len(dummy_input) > 1 else dummy_input[0]

            # Résumé structuré avec torchinfo
            model_summary = summary(
                model,
                input_data=input_data,
                depth=3,
                col_names=("input_size", "output_size", "num_params"),
                verbose=0
            )

            with open(path, "w") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model_summary))

            print(f"[📄] Model summary saved to {path}")

        except Exception as e:
            # Fallback : print str(model)
            with open(path, "w") as f:
                f.write(f"# Model: {model.__class__.__name__}\n")
                f.write(f"# Device: {model.device}\n")
                f.write(str(model))
                f.write(f"\n\n⚠️ torchinfo.summary failed: {e}")

            print(f"[⚠] Fallback model summary saved to {path}")

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
        preprocessing="spectrogram_psnr",
        preprocessing_kwargs=None):
        """
        Fonction d'apprentissage du modèle.
        """

        pid = os.getpid()
        amp_enabled = bool(use_amp and _supports_cuda(self.device))
        gpu_name = torch.cuda.get_device_name(self.device) if _supports_cuda(self.device) else "CPU"
        print(f"[🚀] Initializing training on device: {self.device} ({gpu_name}) as {pid}")

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

        img_size = (1024,1024)

        for split in ("train", "val"):
            self._check_dataset_dirs(data_dir, split, dataset_name)

        if dataset_name == "singleres":
            selected_res = select_res.get("res_key") if isinstance(select_res, dict) else select_res
            train_dataset = YOLODataset(
                os.path.join(data_dir, "train/data"),
                os.path.join(data_dir, "train/labels_detect"),
                select_res=selected_res,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )
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
            img_size = res_hw

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

            val_dataset = YOLODataset(
                data_dir=os.path.join(data_dir, "val/data"),
                labels_dir=os.path.join(data_dir, "val/labels_detect"),
                res_hw=res_hw,
                res_key=res_key,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )

        else:
            train_dataset = YOLODataset(
                os.path.join(data_dir, "train/data"),
                os.path.join(data_dir, "train/labels_detect"),
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )
            val_dataset = YOLODataset(
                os.path.join(data_dir, "val/data"),
                os.path.join(data_dir, "val/labels_detect"),
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )

        pin_memory = _supports_cuda(self.device)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=pin_memory, collate_fn=train_dataset.collate_fn)
        val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False, pin_memory=pin_memory, collate_fn=val_dataset.collate_fn)

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

        logger = MetricsLogger(log_path)

        eval_runner = EvalRunner(
            output_dir=self.output_dir,
            cfg=EvalConfig(iou_thresh=0.5, fa_target=0.01, img_size=img_size)
        )

        # ---------------- opti & loss --------------------
        optimizer = optim.Adam(self.parameters(), lr=lr)
        scaler = _build_grad_scaler(amp_enabled)
        criterion = self.criterion

        # =================================================
        for epoch in range(1, epochs + 1):
            print(f"\n📚 Epoch {epoch}/{epochs}")
            self.train()

            # ---------- Entraînement ----------
            loss_box_train = loss_cls_train = loss_dfl_train = running_train_loss = 0.0
            first_display = True

            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch} 🔧 Training", unit="batch")
            for imgs, targets, res_keys in train_pbar:

                # print('len(imgs) === ', len(imgs))
                # print('imgs[0].shape == ', imgs[0].shape)
                
                if isinstance(imgs, list):
                    imgs = [img.to(self.device) for img in imgs]
                else:
                    imgs = imgs.to(self.device)

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

                loss_box_train += loss_dict[0]
                loss_cls_train += loss_dict[1]
                loss_dfl_train += loss_dict[2]
                running_train_loss += loss.item()

                # backward avec AMP
                scaled_loss = scaler.scale(loss)
                scaled_loss.backward()
                scaler.step(optimizer)
                scaler.update()

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


            n_train_batches = max(1, len(train_loader))
            loss_box_train /= n_train_batches
            loss_cls_train /= n_train_batches
            loss_dfl_train  /= n_train_batches
            train_loss = running_train_loss / n_train_batches

            # ---------- Validation ----------
            self.eval()
            loss_box_val = loss_cls_val = loss_dfl_val = running_val_loss = 0.0

            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch} 🧪 Validation", unit="batch")
            with torch.no_grad():
                for imgs, targets, res_keys in val_pbar:
                    if isinstance(imgs, list):
                        imgs = [img.to(self.device) for img in imgs]
                    else:
                        imgs = imgs.to(self.device)
    
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

                    running_val_loss += val_loss_batch.item()
                    loss_box_val += loss_dict_val[0]
                    loss_cls_val += loss_dict_val[1]
                    loss_dfl_val += loss_dict_val[2]

                    val_pbar.set_postfix(val_loss=val_loss_batch.item())

            n_val_batches = max(1, len(val_loader))
            val_loss = running_val_loss / n_val_batches
            loss_box_val /= n_val_batches
            loss_cls_val /= n_val_batches
            loss_dfl_val  /= n_val_batches

            # ---------- Évaluation complète via EvalRunner (chaque epoch) ----------
            result = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)
            # extra_values = [map50, map50_95, avg_low, avg_med, avg_high, json_path]
            ev = result["extra_values"]

            print(
                f"📉 Summary Epoch {epoch:02d} | "
                f"Train: {train_loss:.4f} (box={loss_box_train:.3f}, cls={loss_cls_train:.3f}, dfl={loss_dfl_train:.3f}) | "
                f"Val:   {val_loss:.4f} (box={loss_box_val:.3f}, cls={loss_cls_val:.3f}, dfl={loss_dfl_val:.3f}) | "
                f"mAP50={ev[0] if ev[0] is not None else 'NA'} | "
                f"mAP50_95={ev[1] if ev[1] is not None else 'NA'} | "
                f"avgRec(low/med/high)={ev[2]:.3f}/{ev[3]:.3f}/{ev[4]:.3f}"
            )

            # ---------- Logging CSV (colonnes de base + extras EvalRunner) ----------
            logger.log(
                epoch=epoch,
                train_loss=float(train_loss), val_loss=float(val_loss),
                loss_box_train=float(loss_box_train), loss_cls_train=float(loss_cls_train), loss_dfl_train=float(loss_dfl_train),
                loss_box_val=float(loss_box_val),     loss_cls_val=float(loss_cls_val),     loss_dfl_val=float(loss_dfl_val),
                extra_headers=result["extra_headers"],
                extra_values=result["extra_values"],
            )

            # --- CHECKPOINT & PLOTS ---
            torch.save(self.state_dict(), last_path)

            # Plots “camera-ready”
            TrainingPlots.plot_losses(log_path, save_path=os.path.join(self.output_dir, "loss_curves.png"))
            TrainingPlots.plot_maps(log_path,   save_path=os.path.join(self.output_dir, "map_curves.png"))
            TrainingPlots.plot_avg_recalls(log_path, save_path=os.path.join(self.output_dir, "avg_recall_curves.png"))

            # Best (val_loss)
            current_map5095 = result["extra_values"][1]
            current_map50 = result["extra_values"][0]
            if not hasattr(self, "_best_map5095"):
                self._best_map5095 = float("-inf")

            if current_map5095 is not None and current_map5095 > self._best_map5095:
                self._best_map5095 = current_map5095
                torch.save(self.state_dict(), best_path)  # ou os.path.join(self.output_dir, "best_map5095.pt")
                print(f"💾 Best model (mAP50:95={current_map5095:.4f}) (mAP50={current_map50}) saved.")

            # Early stopping sur mAP50:95 (mode='max')
            if should_stop_early_from_csv(log_path, patience=patience, monitor="val_loss", mode="min"):
                print(f"⛔️ Early stopping déclenché sur mAP50:95 (aucune amélioration ≥ {patience} epochs).")
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
                if isinstance(imgs, list):
                    imgs = [img.to(self.device) for img in imgs]
                else:
                    imgs = imgs.to(self.device)
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
        if isinstance(image_tensor, list):
            image_tensor = [img.to(self.device) for img in image_tensor]
        else:
            image_tensor = image_tensor.to(self.device)

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
