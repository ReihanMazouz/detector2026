"""
MRViTPatchDetector — ViT multi-résolution end-to-end avec décodeur à queries.

Pipeline :
  conv stem (16 ch, 3×3, BN, GELU)  par résolution
  → patch embed (8×8, stride 8)     par résolution
  → encodage position sinusoïdal 2D + embedding de résolution appris
  → encodeur déformable : num_encoder_layers couches de self-attention déformable
                          multi-niveaux (une résolution = un niveau)
  → décodeur : num_decoder_layers couches (self-attn + cross-attn déformable + FFN)
               appliquées sur num_queries queries appris
  → têtes classes + boîtes par couche (raffinement itératif par sigmoid)
  Loss : RTDETRLoss (matching Hungarian one-to-one, varifocal, L1, GIoU)
"""
from __future__ import annotations

import csv
import math
import os
import time
from contextlib import nullcontext
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ...utils.dataset import YOLODatasetFusedMultiRes, load_class_index_to_name
from ...utils.detr_loss import targets_from_yolo_tensor
from ...utils.evaluate import EvalConfig, EvalRunner, TrainingPlots
from ...utils.rtdetr_loss import RTDETRLoss
from ..Head.rtdetr import MLP, MSDeformAttn, RTDETRDecoderLayer, inverse_sigmoid
from ..base import BaseModel, _move_imgs_to_device, _resolve_num_workers, _supports_cuda


# ─────────────────────────────────────────────────────────────────────────────
# Utilitaires
# ─────────────────────────────────────────────────────────────────────────────

def _sinusoidal_2d(h: int, w: int, d_model: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """
    Encodage de position sinusoïdal 2D. Retourne [1, h*w, d_model].

    Les d_model/2 premiers dims encodent y, les d_model/2 suivants encodent x.
    Requiert d_model % 4 == 0.
    """
    half = d_model // 2
    omega = torch.arange(half // 2, dtype=dtype, device=device)
    omega = 1.0 / (10000.0 ** (2.0 * omega / half))

    y_pos = torch.arange(h, dtype=dtype, device=device)
    x_pos = torch.arange(w, dtype=dtype, device=device)

    pe_y = torch.outer(y_pos, omega)                         # [h, half/2]
    pe_x = torch.outer(x_pos, omega)                         # [w, half/2]
    pe_y = torch.cat([pe_y.sin(), pe_y.cos()], dim=-1)       # [h, half]
    pe_x = torch.cat([pe_x.sin(), pe_x.cos()], dim=-1)       # [w, half]

    pe = torch.cat([
        pe_y.unsqueeze(1).expand(-1, w, -1),                 # [h, w, half]
        pe_x.unsqueeze(0).expand(h, -1, -1),                 # [h, w, half]
    ], dim=-1)                                               # [h, w, d_model]
    return pe.reshape(1, h * w, d_model)


def _encoder_reference_points(
    spatial_shapes: list[Tuple[int, int]],
    num_levels: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """
    Point de référence de chaque token = sa propre position normalisée,
    répétée pour chaque niveau. Retourne [1, N_total, num_levels, 2].
    """
    refs = []
    for H, W in spatial_shapes:
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        xy = torch.stack([xx, yy], dim=-1).reshape(-1, 2)    # [H*W, 2]
        refs.append(xy)
    ref = torch.cat(refs, dim=0)                             # [N_total, 2]
    ref = ref.unsqueeze(1).expand(-1, num_levels, -1)        # [N_total, num_levels, 2]
    return ref.unsqueeze(0)                                  # [1, N_total, num_levels, 2]


def _token_reference_boxes(
    spatial_shapes: list[Tuple[int, int]],
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Initial xywh boxes for encoder tokens, normalized in the common latent grid."""
    refs = []
    for H, W in spatial_shapes:
        y = torch.linspace(0.5 / H, 1.0 - 0.5 / H, H, device=device, dtype=dtype)
        x = torch.linspace(0.5 / W, 1.0 - 0.5 / W, W, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        wh = torch.empty(H, W, 2, device=device, dtype=dtype)
        wh[..., 0] = 1.0 / W
        wh[..., 1] = 1.0 / H
        refs.append(torch.cat([torch.stack([xx, yy], dim=-1), wh], dim=-1).reshape(-1, 4))
    return torch.cat(refs, dim=0).unsqueeze(0)


# ─────────────────────────────────────────────────────────────────────────────
# Couche d'encodeur déformable
# ─────────────────────────────────────────────────────────────────────────────

class _DeformableEncoderLayer(nn.Module):
    """Self-attention déformable multi-niveaux + FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_levels: int,
        num_points: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()
        self.self_attn = MSDeformAttn(d_model, num_levels, num_heads, num_points)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)

    def forward(
        self,
        src: torch.Tensor,                        # [B, N_total, d_model]
        ref_points: torch.Tensor,                 # [1, N_total, num_levels, 2]
        spatial_shapes: list[Tuple[int, int]],
    ) -> torch.Tensor:
        ref = ref_points.expand(src.shape[0], -1, -1, -1)
        src2 = self.self_attn(src, ref, src, spatial_shapes)
        src = self.norm1(src + self.drop1(src2))
        src2 = self.linear2(self.drop2(F.relu(self.linear1(src))))
        return self.norm2(src + self.drop3(src2))


# ─────────────────────────────────────────────────────────────────────────────
# Modèle principal
# ─────────────────────────────────────────────────────────────────────────────

class MRViTPatchDetector(BaseModel):
    """
    Détecteur ViT multi-résolution end-to-end.

    Chaque résolution d'entrée est tokenisée indépendamment (conv stem + patch
    embed 8×8). Les tokens de toutes les résolutions sont concaténés et traités
    par un encodeur déformable. Un décodeur à 100 queries prédit les boîtes.
    """

    def __init__(
        self,
        input_resolutions: List[Tuple[int, int]],
        output_dir: str,
        num_classes: int = 20,
        device: str = "cuda:0",
        in_ch: int = 1,
        d_model: int = 256,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_queries: int = 100,
        patch_grid_hw: Tuple[int, int] = (32, 32),
        num_heads: int = 8,
        num_encoder_points: int = 16,
        num_decoder_points: int = 16,
        dim_feedforward: int = 1024,
        dropout: float = 0.0,
        matcher_num_threads: int = 1,
    ):
        super().__init__(device=device, output_dir=output_dir)
        if not input_resolutions:
            raise ValueError("input_resolutions must not be empty.")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads}).")
        if d_model % 4 != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by 4 (PE sinusoïdal 2D).")

        self.input_resolutions = list(input_resolutions)
        self.num_resolutions = len(input_resolutions)
        self.num_classes = int(num_classes)
        self.in_ch = int(in_ch)
        self.d_model = int(d_model)
        self.num_queries = int(num_queries)
        self.patch_grid_hw = tuple(int(value) for value in patch_grid_hw)
        if len(self.patch_grid_hw) != 2 or self.patch_grid_hw[0] <= 0 or self.patch_grid_hw[1] <= 0:
            raise ValueError(f"patch_grid_hw must be a positive (H, W) tuple, got {patch_grid_hw}.")
        self.patch_sizes = self._patch_sizes_for_grid(self.input_resolutions, self.patch_grid_hw)

        # ── 1. Tokenisation ───────────────────────────────────────────────
        # Conv stem léger par résolution : extrait 16 feature maps de bas niveau
        self.conv_stems = nn.ModuleList(
            nn.Sequential(
                nn.Conv2d(in_ch, 16, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(16),
                nn.GELU(),
            )
            for _ in input_resolutions
        )
        # Patch embed anisotrope : chaque résolution produit la même grille latente.
        # Ex. grille 32x32 : 256x256 -> 8x8, 64x1024 -> 2x32.
        self.patch_embeds = nn.ModuleList(
            nn.Conv2d(16, d_model, kernel_size=patch, stride=patch)
            for patch in self.patch_sizes
        )
        # Embedding de résolution appris (distingue les niveaux dans l'encodeur)
        self.res_embed = nn.Parameter(torch.zeros(self.num_resolutions, 1, d_model))
        nn.init.trunc_normal_(self.res_embed, std=0.02)

        # ── 2. Encodeur déformable ─────────────────────────────────────────
        self.encoder = nn.ModuleList(
            _DeformableEncoderLayer(
                d_model=d_model,
                num_heads=num_heads,
                num_levels=self.num_resolutions,
                num_points=num_encoder_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_encoder_layers)
        )

        # ── 3. Décodeur RT-DETR-like ──────────────────────────────────────
        # Les queries ne sont pas purement apprises : on sélectionne les top-k
        # tokens encodeur par score de classe, puis leurs boîtes deviennent les
        # références initiales du decoder.
        self.enc_output = nn.Sequential(nn.Linear(d_model, d_model), nn.LayerNorm(d_model))
        self.enc_score_head = nn.Linear(d_model, num_classes)
        self.enc_bbox_head = MLP(d_model, d_model, 4, num_layers=3)

        # Couches décodeur (une par couche, pas de partage de poids)
        self.decoder_layers = nn.ModuleList(
            RTDETRDecoderLayer(
                hidden_dim=d_model,
                num_heads=num_heads,
                num_levels=self.num_resolutions,
                num_points=num_decoder_points,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
            )
            for _ in range(num_decoder_layers)
        )

        # Têtes de prédiction : une par couche décodeur (supervision auxiliaire)
        self.class_heads = nn.ModuleList(
            nn.Linear(d_model, num_classes) for _ in range(num_decoder_layers)
        )
        self.box_heads = nn.ModuleList(
            MLP(d_model, d_model, 4, num_layers=3) for _ in range(num_decoder_layers)
        )
        # Positional embedding des queries calculé depuis les boîtes de référence
        self.query_pos_head = MLP(4, 2 * d_model, d_model, num_layers=2)
        self._reset_prediction_heads()

        # ── 4. Loss ────────────────────────────────────────────────────────
        self.criterion = RTDETRLoss(
            num_classes=num_classes,
            matcher_num_threads=matcher_num_threads,
        )
        self.to(self.device)

    def _reset_prediction_heads(self) -> None:
        prior_prob = 0.01
        bias = -math.log((1.0 - prior_prob) / prior_prob)
        nn.init.constant_(self.enc_score_head.bias, bias)
        nn.init.constant_(self.enc_bbox_head.layers[-1].weight, 0.0)
        nn.init.constant_(self.enc_bbox_head.layers[-1].bias, 0.0)
        for cls_head, box_head in zip(self.class_heads, self.box_heads):
            nn.init.constant_(cls_head.bias, bias)
            nn.init.constant_(box_head.layers[-1].weight, 0.0)
            nn.init.constant_(box_head.layers[-1].bias, 0.0)

    # ── Forward ───────────────────────────────────────────────────────────

    @staticmethod
    def _patch_sizes_for_grid(
        input_resolutions: List[Tuple[int, int]],
        patch_grid_hw: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        grid_h, grid_w = patch_grid_hw
        patch_sizes = []
        for resolution in input_resolutions:
            height, width = resolution
            if height % grid_h != 0 or width % grid_w != 0:
                raise ValueError(
                    f"Resolution {resolution} is not divisible by patch grid {patch_grid_hw}."
                )
            patch_sizes.append((height // grid_h, width // grid_w))
        return patch_sizes

    def _validate_inputs(self, inputs: List[torch.Tensor]) -> None:
        if len(inputs) != self.num_resolutions:
            raise ValueError(f"Expected {self.num_resolutions} inputs, got {len(inputs)}.")
        for index, (x, resolution) in enumerate(zip(inputs, self.input_resolutions)):
            if x.dim() != 4:
                raise ValueError(f"Input #{index} must be 4D, got {x.dim()}D.")
            if x.shape[1] != self.in_ch:
                raise ValueError(f"Input #{index} has {x.shape[1]} channels, expected {self.in_ch}.")
            if tuple(x.shape[-2:]) != tuple(resolution):
                raise ValueError(
                    f"Input #{index} has shape {tuple(x.shape[-2:])}, expected {tuple(resolution)}."
                )

    def _tokenize(
        self, inputs: List[torch.Tensor]
    ) -> Tuple[torch.Tensor, list[Tuple[int, int]]]:
        """
        Conv stem + patch embed + PE sinusoïdal 2D + res embed.
        Retourne (memory [B, N_total, D], spatial_shapes).
        """
        all_tokens = []
        spatial_shapes = []
        for i, (x, stem, embed) in enumerate(zip(inputs, self.conv_stems, self.patch_embeds)):
            x = stem(x)                                          # [B, 16, H, W]
            x = embed(x)                                         # [B, D, grid_h, grid_w]
            H, W = x.shape[-2:]
            if (H, W) != self.patch_grid_hw:
                raise RuntimeError(
                    f"Patch embed for input #{i} produced {(H, W)}, expected {self.patch_grid_hw}."
                )
            spatial_shapes.append((H, W))
            tokens = x.flatten(2).transpose(1, 2)                # [B, H*W, D]
            pe = _sinusoidal_2d(H, W, self.d_model, x.device, x.dtype)
            tokens = tokens + pe + self.res_embed[i].to(dtype=tokens.dtype)
            all_tokens.append(tokens)
        return torch.cat(all_tokens, dim=1), spatial_shapes

    def forward(self, inputs: List[torch.Tensor]) -> dict:
        # 1. Tokenisation multi-résolution
        self._validate_inputs(inputs)
        memory, spatial_shapes = self._tokenize(inputs)          # [B, N_total, D]

        # 2. Encodeur déformable (self-attention sur tous les tokens)
        ref_points = _encoder_reference_points(
            spatial_shapes, self.num_resolutions, memory.device, memory.dtype
        )
        for layer in self.encoder:
            memory = layer(memory, ref_points, spatial_shapes)

        # 3. Sélection RT-DETR-like des queries depuis les tokens encodeur
        enc_features = self.enc_output(memory)
        enc_logits = self.enc_score_head(enc_features)
        token_ref_boxes = _token_reference_boxes(spatial_shapes, memory.device, memory.dtype)
        token_ref_boxes = token_ref_boxes.expand(memory.shape[0], -1, -1)
        enc_boxes = torch.sigmoid(self.enc_bbox_head(enc_features) + inverse_sigmoid(token_ref_boxes))
        object_scores = enc_logits[..., : self.num_classes].max(dim=-1).values
        num_queries = min(self.num_queries, enc_features.shape[1])
        topk_indices = torch.topk(object_scores, num_queries, dim=1).indices
        query = enc_features.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, self.d_model))
        enc_top_boxes = enc_boxes.gather(1, topk_indices.unsqueeze(-1).expand(-1, -1, 4))
        ref_boxes = enc_top_boxes.detach()
        enc_top_logits = enc_logits.gather(
            1,
            topk_indices.unsqueeze(-1).expand(-1, -1, self.num_classes),
        )
        query = query.detach() if self.training else query

        pred_logits_per_layer = []
        pred_boxes_per_layer = []

        for layer, cls_head, box_head in zip(self.decoder_layers, self.class_heads, self.box_heads):
            query_pos = self.query_pos_head(ref_boxes)
            query = layer(query, ref_boxes, memory, spatial_shapes, query_pos=query_pos)

            delta = box_head(query)
            pred_boxes = torch.sigmoid(delta + inverse_sigmoid(ref_boxes))
            pred_logits = cls_head(query)

            pred_logits_per_layer.append(pred_logits)
            pred_boxes_per_layer.append(pred_boxes)
            ref_boxes = pred_boxes.detach() if self.training else pred_boxes

        aux_outputs = [{"pred_logits": enc_top_logits, "pred_boxes": enc_top_boxes}]
        aux_outputs.extend(
            {"pred_logits": lg, "pred_boxes": bx}
            for lg, bx in zip(pred_logits_per_layer[:-1], pred_boxes_per_layer[:-1])
        )

        return {
            "pred_logits": pred_logits_per_layer[-1],
            "pred_boxes": pred_boxes_per_layer[-1],
            "aux_outputs": aux_outputs,
        }

    # ── Loss ──────────────────────────────────────────────────────────────

    def loss_from_batch(
        self, outputs: dict, targets: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        B = outputs["pred_logits"].shape[0]
        target_list = targets_from_yolo_tensor(targets, B, outputs["pred_logits"].device)
        return self.criterion(outputs, target_list)

    # ── Post-traitement ───────────────────────────────────────────────────

    def postprocess(self, outputs: dict, conf_thres: float = 0.1, max_det: int = 300, **_) -> List[torch.Tensor]:
        logits = outputs["pred_logits"]   # [B, Q, num_classes]
        boxes = outputs["pred_boxes"]     # [B, Q, 4] xywh normalisé

        scores, labels = logits.sigmoid().max(dim=-1)

        max_h = max(h for h, _ in self.input_resolutions)
        max_w = max(w for _, w in self.input_resolutions)

        results = []
        for boxes_i, scores_i, labels_i in zip(boxes, scores, labels):
            keep = scores_i >= float(conf_thres)
            if not keep.any():
                results.append(torch.zeros((0, 6), device=logits.device, dtype=logits.dtype))
                continue
            b = boxes_i[keep]
            xc, yc, w, h = b.unbind(-1)
            x1 = (xc - 0.5 * w).clamp(0.0, 1.0) * max_w
            y1 = (yc - 0.5 * h).clamp(0.0, 1.0) * max_h
            x2 = (xc + 0.5 * w).clamp(0.0, 1.0) * max_w
            y2 = (yc + 0.5 * h).clamp(0.0, 1.0) * max_h
            det = torch.stack(
                (x1, y1, x2, y2, scores_i[keep], labels_i[keep].to(logits.dtype)), dim=1
            )
            if det.shape[0] > max_det:
                det = det[det[:, 4].argsort(descending=True)[:max_det]]
            results.append(det)
        return results

    def postprocess_for_metrics(self, outputs, conf_threshold: float = 0.1, max_det: int = 300, **kwargs) -> List[torch.Tensor]:
        return self.postprocess(outputs, conf_thres=conf_threshold, max_det=max_det)

    # ── Entraînement ──────────────────────────────────────────────────────

    def fit(
        self,
        data_dir: str,
        epochs: int = 300,
        batch_size: int = 32,
        lr: float = 1e-4,
        patience: int = 100,
        dataset: str = "fused",
        use_amp: bool = True,
        select_res: dict | None = None,
        preprocessing: str = "none",
        preprocessing_kwargs: dict | None = None,
        num_workers: int | None = None,
        persistent_workers: bool = True,
        full_eval_every: int = 5,
        save_last_every: int = 5,
        monitor: str = "val_loss",
        run_full_eval: bool = True,
        **_,
    ) -> None:
        if dataset != "fused":
            raise ValueError("MRViTPatchDetector requires dataset='fused'.")
        if monitor != "val_loss":
            raise ValueError("MRViTPatchDetector only supports monitor='val_loss'.")

        fused_res_keys = select_res.get("res_keys") if isinstance(select_res, dict) else None

        def _make_loader(split: str, shuffle: bool) -> DataLoader:
            ds = YOLODatasetFusedMultiRes(
                os.path.join(data_dir, f"{split}/data"),
                os.path.join(data_dir, f"{split}/labels_detect"),
                res_keys=tuple(fused_res_keys) if fused_res_keys else None,
                preprocessing=preprocessing,
                preprocessing_kwargs=preprocessing_kwargs,
            )
            n_workers = _resolve_num_workers(num_workers)
            kwargs: dict = dict(
                batch_size=batch_size,
                shuffle=shuffle,
                pin_memory=_supports_cuda(self.device),
                collate_fn=ds.collate_fn,
                num_workers=n_workers,
                persistent_workers=bool(persistent_workers) and n_workers > 0,
            )
            if n_workers > 0:
                kwargs["prefetch_factor"] = 4
            return DataLoader(ds, **kwargs)

        train_loader = _make_loader("train", shuffle=True)
        val_loader = _make_loader("val", shuffle=False)

        amp_enabled = bool(use_amp) and _supports_cuda(self.device)
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=1e-4)
        scaler = torch.cuda.amp.GradScaler() if amp_enabled else None

        output_path = Path(self.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        log_path = output_path / "train_log.csv"

        eval_runner = None
        extra_headers: list = []
        if run_full_eval:
            eval_runner = EvalRunner(
                output_dir=str(output_path),
                cfg=EvalConfig(
                    iou_thresh=0.5,
                    fa_target=0.01,
                    img_size=(
                        max(h for h, _ in self.input_resolutions),
                        max(w for _, w in self.input_resolutions),
                    ),
                ),
                class_index_to_name=load_class_index_to_name(data_dir),
            )
            extra_headers = eval_runner.extra_headers()

        with log_path.open("w", newline="") as f:
            csv.writer(f).writerow([
                "epoch", "train_loss", "val_loss",
                "loss_cls_val", "loss_bbox_val", "loss_giou_val",
                *extra_headers,
            ])

        best_val = float("inf")
        bad_epochs = 0

        for epoch in range(1, int(epochs) + 1):
            t0 = time.perf_counter()
            train_loss, _ = self._run_epoch(
                train_loader, optimizer, scaler, amp_enabled,
                train=True, desc=f"Epoch {epoch} train",
            )
            val_loss, val_parts = self._run_epoch(
                val_loader, None, scaler, amp_enabled,
                train=False, desc=f"Epoch {epoch} val",
            )

            should_eval = run_full_eval and (
                epoch % max(1, int(full_eval_every)) == 0 or epoch == int(epochs)
            )
            extra_values: list = []
            if run_full_eval:
                if should_eval:
                    extra_values = eval_runner.run(epoch=epoch, model=self, val_loader=val_loader)["extra_values"]
                else:
                    extra_values = [None, None, *([float("nan")] * 7), None]

            with log_path.open("a", newline="") as f:
                csv.writer(f).writerow([
                    epoch, train_loss, val_loss,
                    val_parts.get("loss_cls", 0.0),
                    val_parts.get("loss_bbox", 0.0),
                    val_parts.get("loss_giou", 0.0),
                    *extra_values,
                ])

            if epoch % max(1, int(save_last_every)) == 0 or epoch == int(epochs):
                torch.save(self.state_dict(), output_path / "last.pt")

            if val_loss < best_val:
                best_val = val_loss
                bad_epochs = 0
                torch.save(self.state_dict(), output_path / "best.pt")
            else:
                bad_epochs += 1

            print(f"Epoch {epoch}: train={train_loss:.4f} val={val_loss:.4f} ({time.perf_counter() - t0:.1f}s)")

            if run_full_eval and should_eval:
                TrainingPlots.plot_losses(str(log_path), save_path=str(output_path / "loss_curves.png"))
                TrainingPlots.plot_maps(str(log_path), save_path=str(output_path / "map_curves.png"))
                TrainingPlots.plot_avg_recalls(str(log_path), save_path=str(output_path / "avg_recall_curves.png"))
                TrainingPlots.plot_size_recalls(str(log_path), save_path=str(output_path / "recall_size_curves.png"))
                TrainingPlots.plot_box_iou(str(log_path), save_path=str(output_path / "box_iou_curves.png"))

            if bad_epochs >= int(patience):
                print(f"Early stopping after {bad_epochs} epochs without improvement.")
                break

    def _run_epoch(
        self,
        loader: DataLoader,
        optimizer,
        scaler,
        amp_enabled: bool,
        train: bool,
        desc: str,
    ) -> Tuple[float, dict]:
        self.train(train)
        total_loss = 0.0
        parts_sum = {"loss_cls": 0.0, "loss_bbox": 0.0, "loss_giou": 0.0}
        non_blocking = _supports_cuda(self.device)
        autocast = torch.cuda.amp.autocast if amp_enabled else nullcontext

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for imgs, targets, _ in tqdm(loader, desc=desc, unit="batch"):
                imgs = _move_imgs_to_device(imgs, self.device, non_blocking=non_blocking)
                targets = targets.to(self.device)

                if optimizer is not None:
                    optimizer.zero_grad(set_to_none=True)

                with autocast():
                    outputs = self(imgs)
                    loss, parts = self.loss_from_batch(outputs, targets)

                if train:
                    if scaler is not None:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()

                total_loss += float(loss.detach())
                for k in parts_sum:
                    parts_sum[k] += float(parts.get(k, 0.0))

        n = max(1, len(loader))
        return total_loss / n, {k: v / n for k, v in parts_sum.items()}
