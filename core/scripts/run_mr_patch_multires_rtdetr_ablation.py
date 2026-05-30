from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import torch

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))

from detector2026.core.models.mr_yolo_ablation import MRPatchMultiResRTDETRHead  # noqa: E402
from detector2026.core.scripts.train_benchmark_suite import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_DEVICE,
    DEFAULT_NUM_CLASSES,
    DEFAULT_PREPROCESSING,
    DEFAULT_REG_MAX,
    DEFAULT_RES_KEYS,
    find_input_resolutions,
)
from detector2026.core.utils.preprocess import preprocessing_num_channels  # noqa: E402


DEFAULT_OUTPUT_ROOT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/mr_yolo_ablation"
)
DEFAULT_RUN_NAME = "mr_patch_multires_rtdetr_head_frozen"
DEFAULT_BACKBONE_CHECKPOINT = (
    "/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation/"
    "mr_yolo_ablation/mr_patch_backbone_yolo_one2many_head/best.pt"
)


def parse_hw(value: str) -> tuple[int, int]:
    if "," in value:
        left, right = value.split(",", 1)
    elif "x" in value.lower():
        left, right = value.lower().split("x", 1)
    else:
        raise argparse.ArgumentTypeError("Expected H,W or HxW.")
    return int(left), int(right)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Refine the pre-trained MRPatchBackboneYOLOOne2ManyHead encoder with a multi-level "
            "RT-DETR head. Each resolution keeps its natural patch-grid feature map; the "
            "loaded backbone is frozen and only the RT-DETR head is trained."
        )
    )
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--backbone-checkpoint",
        default=DEFAULT_BACKBONE_CHECKPOINT,
        help="Path to the pre-trained MRPatchBackboneYOLOOne2ManyHead best.pt.",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE)
    parser.add_argument("--num-classes", type=int, default=DEFAULT_NUM_CLASSES)
    parser.add_argument("--reg-max", type=int, default=DEFAULT_REG_MAX)
    parser.add_argument("--preprocessing", default=DEFAULT_PREPROCESSING)
    # Training schedule
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--full-eval-every", type=int, default=5)
    parser.add_argument("--save-last-every", type=int, default=5)
    # Backbone architecture (must match the checkpoint if one is provided)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--num-encoder-layers", type=int, default=3)
    parser.add_argument("--num-heads-backbone", type=int, default=4)
    parser.add_argument("--num-intra-points", type=int, default=8)
    parser.add_argument("--num-inter-neighbors", type=int, default=8)
    parser.add_argument("--dim-feedforward-backbone", type=int, default=512)
    parser.add_argument("--dropout", type=float, default=0.0)
    # RT-DETR head
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-queries", type=int, default=100)
    parser.add_argument("--num-decoder-layers", type=int, default=6)
    parser.add_argument("--num-heads-decoder", type=int, default=8)
    parser.add_argument("--num-decoder-points", type=int, default=8)
    parser.add_argument("--dim-feedforward-decoder", type=int, default=1024)
    parser.add_argument("--matcher-num-threads", type=int, default=8)
    # Control
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def output_is_complete(output_dir: Path) -> bool:
    return (output_dir / "best.pt").exists() or (output_dir / "last.pt").exists()


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def freeze_backbone(model: MRPatchMultiResRTDETRHead) -> int:
    frozen = 0
    for param in model.backbone.parameters():
        param.requires_grad = False
        frozen += param.numel()
    model._freeze_backbone = True
    model.backbone.eval()
    return frozen


def main() -> None:
    args = parse_args()
    input_resolutions = find_input_resolutions(args.data_dir)
    if len(input_resolutions) != len(DEFAULT_RES_KEYS):
        raise ValueError(
            f"Expected {len(DEFAULT_RES_KEYS)} resolutions, found {len(input_resolutions)}."
        )

    input_channels = preprocessing_num_channels(args.preprocessing)
    backbone_ckpt = Path(args.backbone_checkpoint)
    output_dir = Path(args.output_root) / args.run_name

    patch_shapes = [(h // args.patch_size, w // args.patch_size) for h, w in input_resolutions]
    total_tokens = sum(h * w for h, w in patch_shapes)

    num_levels = len(input_resolutions)

    print("MRPatchMultiResRTDETRHead frozen-backbone refinement")
    print(f"  output_dir          = {output_dir}")
    print(f"  device              = {args.device}")
    print(f"  backbone_checkpoint = {backbone_ckpt}  exists={backbone_ckpt.is_file()}")
    print(f"  input_channels      = {input_channels}")
    print(f"  epochs={args.epochs}  patience={args.patience}  lr={args.lr}")
    print(f"  num_levels={num_levels}  total_tokens_per_image={total_tokens}")
    print("  patch grids per resolution:")
    for res_key, res_hw, shape in zip(DEFAULT_RES_KEYS, input_resolutions, patch_shapes):
        print(f"    {res_key}: {res_hw} → {shape[0]}×{shape[1]} = {shape[0]*shape[1]} tokens")
    print("  backbone:")
    print(f"    d_model={args.d_model}  patch_size={args.patch_size}  encoder_layers={args.num_encoder_layers}")
    print(f"    num_heads={args.num_heads_backbone}  intra_points={args.num_intra_points}  inter_neighbors={args.num_inter_neighbors}")
    print("    trainable=False")
    print("  RT-DETR head:")
    print(f"    hidden_dim={args.hidden_dim}  num_queries={args.num_queries}  num_levels={num_levels}")
    print(f"    num_decoder_layers={args.num_decoder_layers}  num_heads={args.num_heads_decoder}  num_decoder_points={args.num_decoder_points}")

    if args.dry_run:
        return
    if output_is_complete(output_dir) and not args.overwrite:
        print(f"[SKIP] checkpoint already present in {output_dir}")
        return
    if not backbone_ckpt.is_file():
        raise FileNotFoundError(
            f"Backbone checkpoint not found: {backbone_ckpt}\n"
            "Train MRPatchBackboneYOLOOne2ManyHead first or pass --backbone-checkpoint."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    model = MRPatchMultiResRTDETRHead(
        input_resolutions=list(input_resolutions),
        output_dir=str(output_dir),
        num_classes=args.num_classes,
        reg_max=args.reg_max,
        device=args.device,
        in_ch=input_channels,
        d_model=args.d_model,
        patch_size=args.patch_size,
        num_encoder_layers=args.num_encoder_layers,
        num_heads_backbone=args.num_heads_backbone,
        num_intra_points=args.num_intra_points,
        num_inter_neighbors=args.num_inter_neighbors,
        dim_feedforward_backbone=args.dim_feedforward_backbone,
        dropout=args.dropout,
        hidden_dim=args.hidden_dim,
        num_queries=args.num_queries,
        num_decoder_layers=args.num_decoder_layers,
        num_heads_decoder=args.num_heads_decoder,
        num_decoder_points=args.num_decoder_points,
        dim_feedforward_decoder=args.dim_feedforward_decoder,
        matcher_num_threads=args.matcher_num_threads,
    )
    try:
        missing, unexpected = model.load_backbone_weights(str(backbone_ckpt), device=args.device)
        frozen_params = freeze_backbone(model)
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"[OK] backbone weights loaded — missing={len(missing)}  unexpected={len(unexpected)}")
        print(f"[OK] backbone frozen — frozen_params={frozen_params}  trainable_params={trainable_params}")

        model.fit(
            data_dir=args.data_dir,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            patience=args.patience,
            preprocessing=args.preprocessing,
            res_keys=tuple(DEFAULT_RES_KEYS),
            num_workers=args.num_workers,
            prefetch_factor=args.prefetch_factor,
            full_eval_every=args.full_eval_every,
            save_last_every=args.save_last_every,
            monitor="val_loss",
        )
    finally:
        del model
        cleanup()

    print(f"\n[DONE] outputs: {output_dir}")


if __name__ == "__main__":
    main()
