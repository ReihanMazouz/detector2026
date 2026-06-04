# Clean code and documentation

This directory contains the core detection models, building blocks, and utilities used my thesis. 

---

## Repository Structure

```
clean/
├── models/              Core detection architectures
│   ├── yolov8.py        YOLOv8 baseline
│   ├── yolov11.py       YOLOv11 baseline (main single-res model)
│   ├── yolov12.py       YOLOv12-turbo (area-attention blocks)
│   ├── tf_attn_yolo.py  YOLOv11 + transformer attention in backbone
│   ├── mr_yolo.py       Multi-resolution YOLO (main multi-res model)
│   ├── detr.py          DETR (transformer encoder-decoder)
│   ├── resnet.py        ResNet classifiers (18/34/50/101)
│   ├── base.py          BaseModel: unified training loop, checkpointing
│   ├── anisotropic_utils.py  Anisotropic stride schedule helpers
│   ├── Backbones/       Standalone backbone modules
│   ├── Head/            Detection head modules
│   └── Neck/            Neck / feature pyramid modules
├── nn/
│   ├── blocks.py        C2f, C3k2, SPPF, C2PSA, A2C2f, DFL, …
│   └── convs.py         Conv, DWConv, CBAM, RepConv, …
├── utils/
│   ├── dataset/         Dataset loaders (single-res, multi-res, fused)
│   ├── loss/            Loss functions (YOLO, one-to-one, DFL, SNR-aware)
│   ├── preprocess/      Input preprocessing (spectrogram, complex spectrum)
│   ├── fusion_uni_res/  Post-hoc NMS fusion & oracle upper bound
│   ├── evaluate.py      EvalRunner, metrics logging, training plots
│   ├── metrics.py       mAP, precision/recall, confusion matrix
│   ├── tal.py           Task-Aligned Assigner
│   ├── detr_loss.py     DETR matching & loss
│   ├── rtdetr_loss.py   RT-DETR loss
│   └── …
├── train.py             Unified training script (all models)
└── evaluate.py          Unified evaluation script (all models)
```

---

## Models

### Single-resolution YOLO family

All three models share the same YOLO one-to-many detection head (P3/P4/P5 pyramid, DFL regression, TAL assignment). They inherit `BaseModel.fit()`.

| Model | Key feature | Constructor |
|---|---|---|
| `YOLOv8` | C2f blocks, depth scaling | `YOLOv8(output_dir, num_classes, reg_max, device, in_ch, width_mult, depth_mult)` |
| `YOLOv11` | C3k2 + C2PSA attention | `YOLOv11(output_dir, num_classes, reg_max, device, input_canals, width_mult)` |
| `YOLOv12` | A2C2f area-attention blocks | `YOLOv12(output_dir, num_classes, reg_max, device, in_ch, width_mult, depth_mult)` |
| `TF_Attn_Yolo` | YOLOv11 + TFSep attention in backbone | `TF_Attn_Yolo(output_dir, num_classes, reg_max, device, input_canals, width_mult)` |

Scale shortcuts (width / depth multipliers):

| Scale | `width_mult` | `depth_mult` |
|---|---|---|
| n (nano) | 0.25 | 0.33 |
| s (small) | 0.50 | 0.33 |
| m (medium) | 0.75 | 0.67 |
| l (large) | 1.00 | 1.00 |

### MR-YOLO (multi-resolution)

Processes multiple spectrograms at different resolutions simultaneously. Each resolution is handled by an independent branch backbone (`MR_TF_Backbone`), then the branch features are fused before the shared P3/P4/P5 neck and detection head.

```python
from detector2026.clean.models import MR_YOLO

model = MR_YOLO(
    input_resolutions=[(512, 512), (256, 1024), (128, 2048)],
    output_dir="runs/mr_yolo_n",
    num_classes=20,
    reg_max=16,
    device="cuda:0",
    in_ch=1,
    width_mult=0.25,
    backbone_mode="TFSep_pyramid",   # recommended
)
```

Backbone modes: `F`, `pyramid`, `TFSep_pyramid` (default), `TFSep_pyramid_up`.

### DETR

Standard encoder-decoder transformer with Hungarian matching and one-to-one assignment. Has its own `fit()` method (only supports `dataset="specificres"`).

```python
from detector2026.clean.models import DETR

model = DETR(
    output_dir="runs/detr_s",
    num_classes=20,
    device="cuda:0",
    input_channels=1,
    width_mult=0.50,
    hidden_dim=256,
    num_queries=100,
    num_encoder_layers=2,
    num_decoder_layers=3,
)
```

---

## Training

### Using `train.py` (recommended)

```bash
# YOLOv11 nano — single resolution 256×256
python train.py \
    --model yolov11 --scale n \
    --data-dir /data/rf_dataset \
    --res-key cfg512 --res-hw 256 256 \
    --epochs 100 --batch-size 32 --lr 1e-3

# MR-YOLO nano — all five resolutions
python train.py \
    --model mr_yolo --scale n \
    --data-dir /data/rf_dataset \
    --res-keys cfg512 cfg256 cfg128 cfg1024 cfg2048 \
    --epochs 100 --batch-size 32

# DETR small
python train.py \
    --model detr --scale s \
    --data-dir /data/rf_dataset \
    --res-key cfg512 --res-hw 256 256 \
    --lr 1e-4 --monitor map50

# YOLOv8 nano, monitor mAP50:95
python train.py \
    --model yolov8 --scale n \
    --data-dir /data/rf_dataset \
    --res-key cfg512 --res-hw 256 256 \
    --monitor map50_95 --full-eval-every 5
```

Key options:

| Flag | Default | Description |
|---|---|---|
| `--model` | — | `yolov8 / yolov11 / yolov12 / tf_attn / mr_yolo / detr` |
| `--scale` | `n` | `n / s / m / l` |
| `--data-dir` | — | Dataset root (must contain `train/` and `val/` splits) |
| `--res-key` | `cfg512` | Resolution key for single-res models |
| `--res-hw H W` | auto | Input size in pixels; auto-detected if omitted |
| `--res-keys` | — | Ordered resolution keys for MR-YOLO |
| `--epochs` | 100 | Max training epochs |
| `--patience` | 10 | Early stopping patience |
| `--monitor` | `val_loss` | Metric to monitor: `val_loss / map50 / map50_95` |
| `--output-dir-parent` | `runs` | Parent folder for experiment outputs |
| `--dry-run` | — | Print config without training |

### Programmatic API

```python
from detector2026.clean.models import YOLOv11

model = YOLOv11(
    output_dir="runs/my_experiment",
    num_classes=20,
    reg_max=16,
    device="cuda:0",
    input_canals=1,
    width_mult=0.25,
)

model.fit(
    data_dir="/data/rf_dataset",
    epochs=100,
    batch_size=32,
    lr=1e-3,
    patience=10,
    dataset="specificres",
    select_res={"res_hw": (256, 256), "res_key": "cfg512"},
    monitor="val_loss",
    full_eval_every=5,
)
```

---

## Evaluation

### Using `evaluate.py`

```bash
# Evaluate YOLOv11n on the val split
python evaluate.py \
    --model yolov11 --scale n \
    --checkpoint runs/yolov11_n_specificres_cfg512/best.pt \
    --data-dir /data/rf_dataset \
    --res-key cfg512 --res-hw 256 256

# Evaluate MR-YOLO on the test split
python evaluate.py \
    --model mr_yolo --scale n \
    --checkpoint runs/mr_yolo_n_fused_cfg512_cfg256/best.pt \
    --data-dir /data/rf_dataset \
    --res-keys cfg512 cfg256 \
    --split test --output-json results/mr_yolo_test.json
```

Results are written to a JSON file containing mAP50, mAP50:95, precision, recall, and per-SNR-bin metrics.

### Programmatic API

```python
import torch
from torch.utils.data import DataLoader
from detector2026.clean.models import YOLOv11
from detector2026.clean.utils.analysing_results import dataset_analysis_with_metrics
from detector2026.clean.utils.dataset import YOLODatasetSpecificRes, load_class_index_to_name

model = YOLOv11(output_dir="runs/my_experiment", num_classes=20, device="cpu", width_mult=0.25)
model.load_weights("runs/my_experiment/best.pt", device="cpu", eval_mode=True)

dataset = YOLODatasetSpecificRes(
    data_dir="/data/rf_dataset/val/data",
    labels_dir="/data/rf_dataset/val/labels_detect",
    res_hw=(256, 256),
    res_key="cfg512",
)
loader = DataLoader(dataset, batch_size=16, collate_fn=dataset.collate_fn)

metrics = dataset_analysis_with_metrics(
    model=model,
    val_loader=loader,
    iou_thresh=0.5,
    fa=0.01,
    img_size=(256, 256),
    class_index_to_name=load_class_index_to_name("/data/rf_dataset"),
)
print(metrics["map_stats"])
```

---

## Dataset format

Each dataset split must follow the structure:

```
<data_dir>/
├── train/
│   ├── data/           .pt files — each is a list of tensors, one per resolution
│   └── labels_detect/  .json files with YOLO-format annotations
├── val/
│   ├── data/
│   └── labels_detect/
└── test/               (optional)
    ├── data/
    └── labels_detect/
```

Each `.json` label file contains a list of objects with fields:
- `class` (int), `xc`, `yc`, `w`, `h` (normalized), `snr` (float), `psnr` (dict keyed by res_key)

---

## Building blocks

### `nn/blocks.py`

| Block | Description |
|---|---|
| `C2f` | Cross-stage partial bottleneck (YOLOv8 style) |
| `C3k2` | C3k2 bottleneck with optional attention (YOLOv11 style) |
| `A2C2f` | Area-attention C2f (YOLOv12 style) |
| `SPPF` | Spatial Pyramid Pooling — Fast |
| `C2PSA` | C2f with Parallel Self-Attention |
| `TFSepBlock` | Separable transformer block with DW-Conv |
| `DFL` | Distribution Focal Loss projection head |

### `nn/convs.py`

`Conv`, `DWConv`, `GhostConv`, `ChannelAttention`, `SpatialAttention`, `CBAM`, `RepConv`

### `models/Backbones/`

| Module | Description |
|---|---|
| `BranchBackbone` | Per-resolution encoder with anisotropic strides |
| `MR_Backbone` | Multi-resolution branches with configurable fusion |
| `MR_TF_Backbone` | Multi-resolution branches with transformer blocks (default for MR-YOLO) |
| `SwinBackbone` | Hierarchical Swin Transformer (4 stages) |
| `DATBackbone` | Deformable Attention Transformer (Xia et al. 2022) |

### `models/Head/`

| Module | Description |
|---|---|
| `Detect` | YOLO one-to-many head (DFL regression + focal cls) |
| `One2OneDetect` | YOLO one-to-one head for hybrid training |
| `RTDETRHead` | RT-DETR head with deformable cross-attention decoder |

### `models/Neck/`

| Module | Description |
|---|---|
| `RTDETRHybridEncoderNeck` | Transformer on P5 + FPN/PAN fusion |
| `TransformerPyramidNeck` | Dense or deformable self-attention on P3/P4/P5 tokens |
| `DeformablePyramidNeck` | Deformable attention variant of the above |

---

## Training outputs

Each experiment folder (`output_dir`) contains:

```
<output_dir>/
├── best.pt              Best checkpoint (by monitored metric)
├── last.pt              Latest checkpoint
├── train_log.csv        Per-epoch metrics log
├── model_summary.txt    torchinfo model summary
├── loss_curves.png      Training / validation loss plots
├── map_curves.png       mAP50 / mAP50:95 curves
├── avg_recall_curves.png  Average recall by SNR bin
└── eval_epoch_*.json    Full detection metrics per evaluation epoch
```
