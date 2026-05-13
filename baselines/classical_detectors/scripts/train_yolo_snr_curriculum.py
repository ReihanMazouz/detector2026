from __future__ import annotations

import argparse
import copy
import csv
import inspect
import json
import os
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable

import numpy as np
import torch


os.environ.setdefault("MPLBACKEND", "Agg")


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SIMULATOR_ROOT = PROJECT_ROOT.parent / "ICML2026DataSimulator"
SIMULATOR_EXAMPLES = SIMULATOR_ROOT / "examples"

for candidate in (PROJECT_ROOT, SIMULATOR_ROOT, SIMULATOR_EXAMPLES):
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)

from rf_data_generation import ACQUISITION_TIME, F_E, SIGNAL_DEFS, STFT_CFGS, build_class_index_to_name  # type: ignore  # noqa: E402
from simulator.multi_res_generation_with_seg import (  # type: ignore  # noqa: E402
    FIXED_DISTANCE,
    LIGHT_SPEED,
    NOISE_POWER,
    generate_and_store_spectrum_multi,
)

from core.models.mr_yolo import MR_YOLO  # noqa: E402
from core.models.tf_attn_yolo import TF_Attn_Yolo  # noqa: E402
from core.models.yolov11 import YOLOv11  # noqa: E402
from core.utils.preprocess import preprocessing_num_channels  # noqa: E402
from baselines.classical_detectors.evaluation.deep_waveform_sweep import (  # noqa: E402
    DeepModelSpec,
    DeepWaveformSweepConfig,
    run_deep_waveform_snr_sweep,
)


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "runs" / "classical_detectors" / "yolo_snr_curriculum"
DEFAULT_WORK_DIR = PROJECT_ROOT / "runs" / "classical_detectors" / "tmp" / "yolo_snr_curriculum_dataset"
DEFAULT_BENCHMARK_WEIGHTS_ROOT = Path("/data/RAWSIM/RMA/training_folder/rf_dataset_for_real_validation")
DEFAULT_VAL_ROOT = Path("/data/RAWSIM/RMA/rf_single_emitter_real_validation")
DEFAULT_RES_HW = {
    "cfg128": (64, 1024),
    "cfg256": (128, 512),
    "cfg512": (256, 256),
    "cfg1024": (512, 128),
    "cfg2048": (1024, 64),
}
DEFAULT_RES_KEYS = ("cfg128", "cfg256", "cfg512", "cfg1024", "cfg2048")
DEFAULT_BENCHMARK_MR_RES_KEYS = DEFAULT_RES_KEYS
DEFAULT_YOLO_VN_WIDTH_MULT = 0.25
DEFAULT_MR_VN_WIDTH_MULT = 0.25
DEFAULT_FIT_TRAIN_RATIO = 0.9
DEFAULT_MODEL_FAMILIES = {
    "yolov11vn": "yolov11",
    "tf_attn_yolovn": "tf_attn_yolo",
    "mr_yolovn": "mr_yolo",
}


def _log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def _snr_sequence(start: int, end: int, step: int) -> list[int]:
    if step <= 0:
        raise ValueError("SNR step must be strictly positive.")
    if start >= end:
        return list(range(start, end - 1, -step))
    return list(range(start, end + 1, step))


def _draw_fp_with_bw(rng: np.random.Generator, bw: float, fp_min: float, fp_max: float) -> float:
    low = fp_min + bw / 2.0
    high = fp_max - bw / 2.0
    if low >= high:
        return 0.5 * (fp_min + fp_max)
    return float(rng.uniform(low, high))


def _erp_for_snr_db(snr_db: float, fp: float) -> float:
    wavelength = LIGHT_SPEED / float(fp)
    return float(
        (10 ** (float(snr_db) / 10.0) * NOISE_POWER * (4.0 * np.pi * FIXED_DISTANCE) ** 2)
        / wavelength**2
    )


def _build_single_emitter_signal_defs(selected_waveforms: set[str] | None) -> dict:
    waveforms = {}
    for label, definition in SIGNAL_DEFS["waveforms"].items():
        if label == "none" or float(definition.get("p", 0.0)) <= 0.0:
            continue
        if selected_waveforms is not None and label not in selected_waveforms:
            continue
        copied = dict(definition)
        copied["p"] = 1.0
        waveforms[label] = copied

    if not waveforms:
        raise ValueError("No waveform selected for generation.")

    return {
        "waveforms": waveforms,
        "interferences": {
            "none": {
                "duration_min": 0.0,
                "duration_max": 0.0,
                "bandwidth_min": 0.0,
                "bandwidth_max": 0.0,
                "p": 0.0,
            }
        },
    }


def _sample_base_waveform(*, rng: np.random.Generator, label: str, definition: dict) -> dict:
    pw = float(rng.uniform(definition["duration_min"], definition["duration_max"]))
    bw = float(rng.uniform(definition["bandwidth_min"], definition["bandwidth_max"]))
    fp = _draw_fp_with_bw(rng, bw, 0.1 * F_E / 2.0, 0.9 * F_E / 2.0)
    pri = float(rng.uniform(min(2.0 * pw, 1e3 / LIGHT_SPEED), 10.0 * pw))
    waveform = {
        "waveform_label": label,
        "waveform_type": str(definition.get("modulation", label)),
        "erp": 0.0,
        "pw": pw,
        "fe": F_E,
        "fp": fp,
        "bandwidth": bw,
        "numberOfCycles": 1,
        "pri": pri,
    }
    if waveform["waveform_type"] == "FMCW_TRI":
        cycle_duration = definition.get("cycle_duration")
        if cycle_duration is None:
            cycle_duration = float(rng.uniform(pw / 50.0, pw / 2.0))
        waveform["cycle_duration"] = float(cycle_duration)
        waveform["start_direction"] = int(rng.choice([-1, 1]))
    return waveform


def _build_fixed_snr_scenarios(
    *,
    signal_defs: dict,
    scenarios_per_waveform: int,
    snr_db: float,
    seed: int,
) -> list[list[dict]]:
    scenarios: list[list[dict]] = []
    active_waveforms = list(signal_defs["waveforms"].items())
    group_id = 0

    for waveform_label, definition in active_waveforms:
        for scenario_index in range(int(scenarios_per_waveform)):
            scenario_seed = int(seed + group_id * 1009)
            acquisition_seed = int(seed + group_id * 10007)
            rng = np.random.default_rng(scenario_seed)
            base_waveform = _sample_base_waveform(rng=rng, label=waveform_label, definition=definition)
            waveform_type = str(base_waveform["waveform_type"])

            waveform = copy.deepcopy(base_waveform)
            waveform["waveform_id"] = f"{waveform_label}_g{group_id:05d}_snr{snr_db:+05.1f}"
            waveform["erp"] = _erp_for_snr_db(float(snr_db), waveform["fp"])
            waveform["target_snr_db"] = float(snr_db)
            waveform["scenario_group_id"] = group_id
            waveform["sweep_snr_db"] = float(snr_db)
            waveform["acquisition_seed"] = acquisition_seed

            emitter = {
                "label": "Emitter_1",
                "emitter_type": "radar",
                "x": FIXED_DISTANCE,
                "y": 0.0,
                "waveforms": [waveform],
                "target_snr_db": float(snr_db),
                "scenario_group_id": group_id,
                "sweep_snr_db": float(snr_db),
                "acquisition_seed": acquisition_seed,
                "scenario_waveform_label": waveform_label,
                "scenario_waveform_type": waveform_type,
                "scenario_index_per_waveform": scenario_index,
            }
            scenarios.append([emitter])
            group_id += 1

    return scenarios


def _parse_waveforms(raw: str) -> set[str] | None:
    if str(raw).strip().lower() in {"", "all"}:
        return None
    return {item.strip() for item in str(raw).split(",") if item.strip()}


def _parse_res_keys(raw: str) -> tuple[str, ...]:
    res_keys = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not res_keys:
        raise ValueError("At least one MR resolution key must be provided.")
    unknown = [key for key in res_keys if key not in DEFAULT_RES_HW]
    if unknown:
        raise ValueError(f"Unknown MR resolution key(s) {unknown}. Expected one of {list(DEFAULT_RES_KEYS)}.")
    duplicates = sorted({key for key in res_keys if res_keys.count(key) > 1})
    if duplicates:
        raise ValueError(f"Duplicate MR resolution key(s): {duplicates}.")
    return res_keys


def _parse_model_names(raw: str) -> tuple[str, ...]:
    if str(raw).strip().lower() in {"", "all"}:
        return tuple(DEFAULT_MODEL_FAMILIES)
    model_names = tuple(item.strip() for item in str(raw).split(",") if item.strip())
    if not model_names:
        raise ValueError("At least one model must be selected.")
    unknown = [name for name in model_names if name not in DEFAULT_MODEL_FAMILIES]
    if unknown:
        raise ValueError(f"Unknown model name(s) {unknown}. Expected one of {list(DEFAULT_MODEL_FAMILIES)}.")
    duplicates = sorted({name for name in model_names if model_names.count(name) > 1})
    if duplicates:
        raise ValueError(f"Duplicate model name(s): {duplicates}.")
    return model_names


def _benchmark_weight_defaults(root: Path, res_key: str) -> dict[str, Path]:
    mr_suffix = "_".join(DEFAULT_BENCHMARK_MR_RES_KEYS)
    return {
        "yolov11vn": root / f"yolov11n_specificres_{res_key}" / "best.pt",
        "tf_attn_yolovn": root / f"tf_attn_yolon_specificres_{res_key}" / "best.pt",
        "mr_yolovn": root / f"mr_yolo_n_fused_{mr_suffix}" / "best.pt",
    }


def _optional_path(value: str | Path | None) -> Path | None:
    if value is None:
        return None
    raw = str(value).strip()
    if raw.lower() in {"", "none", "null"}:
        return None
    return Path(raw)


def _generate_yolo_dataset(
    *,
    output_dir: Path,
    snr_db: float,
    seed: int,
    scenarios_per_waveform: int,
    fit_train_ratio: float,
    waveforms: str,
    generation_preprocessing: str,
) -> dict[int, str]:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.parent.mkdir(parents=True, exist_ok=True)

    signal_defs = _build_single_emitter_signal_defs(_parse_waveforms(waveforms))
    class_index_to_name = build_class_index_to_name(signal_defs)
    scenarios = _build_fixed_snr_scenarios(
        signal_defs=signal_defs,
        scenarios_per_waveform=scenarios_per_waveform,
        snr_db=snr_db,
        seed=seed,
    )

    _log(
        f"generating YOLO dataset snr={snr_db:g} seed={seed} "
        f"samples={len(scenarios)} fit_train_ratio={fit_train_ratio:g}"
    )
    generate_and_store_spectrum_multi(
        scenarios=scenarios,
        base_path=str(output_dir),
        split_train_test=True,
        train_ratio=float(fit_train_ratio),
        acquisition_time=ACQUISITION_TIME,
        stft_cfgs=STFT_CFGS,
        seed=seed,
        pow2_resize=False,
        preprocessing=generation_preprocessing,
        store_complex_spectrum=False,
        save_raw_data=True,
        representation="stft",
        class_index_to_name=class_index_to_name,
    )
    return class_index_to_name


@dataclass(frozen=True)
class ModelJob:
    name: str
    family: str
    device: str
    output_dir: Path
    previous_weights: Path | None


def _build_model(
    *,
    job: ModelJob,
    num_classes: int,
    input_channels: int,
    res_key: str,
    res_keys: tuple[str, ...],
    width_mult: float,
    mr_width_mult: float,
    mr_backbone_mode: str,
    outfusion_channels_mult: int,
) -> torch.nn.Module:
    if job.family == "yolov11":
        return YOLOv11(
            output_dir=str(job.output_dir),
            num_classes=num_classes,
            device=job.device,
            input_canals=input_channels,
            width_mult=width_mult,
            input_hw=DEFAULT_RES_HW[res_key],
        )
    if job.family == "tf_attn_yolo":
        return TF_Attn_Yolo(
            output_dir=str(job.output_dir),
            num_classes=num_classes,
            device=job.device,
            input_canals=input_channels,
            width_mult=width_mult,
            input_hw=DEFAULT_RES_HW[res_key],
        )
    if job.family == "mr_yolo":
        model = MR_YOLO(
            input_resolutions=[DEFAULT_RES_HW[key] for key in res_keys],
            output_dir=str(job.output_dir),
            num_classes=num_classes,
            device=job.device,
            in_ch=input_channels,
            width_mult=mr_width_mult,
            backbone_mode=mr_backbone_mode,
            outfusion_channels_mult=outfusion_channels_mult,
        )
        model.res_keys = tuple(res_keys)
        return model
    raise ValueError(f"Unknown model family: {job.family}")


def _train_model_job(
    *,
    job: ModelJob,
    dataset_root: Path,
    class_index_to_name: dict[int, str],
    args: argparse.Namespace,
) -> dict:
    _log(f"[{job.name}] start on {job.device} output={job.output_dir}")
    job.output_dir.mkdir(parents=True, exist_ok=True)

    input_channels = preprocessing_num_channels(args.preprocessing)
    model = _build_model(
        job=job,
        num_classes=len(class_index_to_name),
        input_channels=input_channels,
        res_key=args.res_key,
        res_keys=_parse_res_keys(args.mr_res_keys),
        width_mult=float(args.width_mult),
        mr_width_mult=float(args.mr_width_mult),
        mr_backbone_mode=args.mr_backbone_mode,
        outfusion_channels_mult=int(args.outfusion_channels_mult),
    )
    if job.previous_weights is not None and job.previous_weights.is_file():
        missing, unexpected = model.load_weights(str(job.previous_weights), device=job.device, eval_mode=False)
        _log(f"[{job.name}] loaded transfer weights={job.previous_weights} missing={len(missing)} unexpected={len(unexpected)}")

    dataset_type = "fused" if job.family == "mr_yolo" else "specificres"
    select_res = None
    if dataset_type == "specificres":
        select_res = {"res_hw": DEFAULT_RES_HW[args.res_key], "res_key": args.res_key}
    elif dataset_type == "fused":
        select_res = {"res_keys": _parse_res_keys(args.mr_res_keys)}

    model.fit(
        data_dir=str(dataset_root),
        epochs=int(args.epochs),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        patience=int(args.patience),
        dataset=dataset_type,
        select_res=select_res,
        preprocessing=args.preprocessing,
        num_workers=int(args.num_workers),
        persistent_workers=bool(args.persistent_workers),
        use_amp=not bool(args.no_amp),
        full_eval_every=int(args.full_eval_every),
        save_last_every=int(args.save_last_every),
        monitor=args.monitor,
        run_full_eval=bool(args.train_full_eval),
    )
    best_path = job.output_dir / "best.pt"
    last_path = job.output_dir / "last.pt"
    transfer_path = last_path if last_path.is_file() else best_path
    _log(
        f"[{job.name}] done last={last_path if last_path.is_file() else None}"
    )
    return {
        "name": job.name,
        "family": job.family,
        "device": job.device,
        "output_dir": str(job.output_dir),
        "best_path": str(best_path) if best_path.is_file() else None,
        "last_path": str(last_path) if last_path.is_file() else None,
        "transfer_path": str(transfer_path) if transfer_path.is_file() else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv11-vn, TF-Attn-YOLO-vn and MR-YOLO-vn with an SNR curriculum.")
    parser.add_argument("--train-snr-start", type=int, default=0)
    parser.add_argument("--train-snr-end", type=int, default=-30)
    parser.add_argument("--train-snr-step", type=int, default=1)
    parser.add_argument("--seed-start", type=int, default=400)
    parser.add_argument("--scenarios-per-waveform", type=int, default=500)
    parser.add_argument(
        "--fit-train-ratio",
        type=float,
        default=DEFAULT_FIT_TRAIN_RATIO,
        help="Fraction of generated stage scenarios used for training. The remainder is used for stage validation.",
    )
    parser.add_argument(
        "--first-snr-scenarios-per-waveform",
        type=int,
        default=1000,
        help="Number of scenarios per waveform for the first SNR stage. Later stages use --scenarios-per-waveform.",
    )
    parser.add_argument("--waveforms", default="all")
    parser.add_argument("--work-dir", type=Path, default=DEFAULT_WORK_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--val-root", type=Path, default=DEFAULT_VAL_ROOT)
    parser.add_argument(
        "--benchmark-weights-root",
        type=Path,
        default=DEFAULT_BENCHMARK_WEIGHTS_ROOT,
        help="Root containing train_benchmark_suite.py output folders used as default initial weights.",
    )
    parser.add_argument("--init-yolov11-weights", default=None, help="Initial YOLOv11 weights. Default: benchmark yolov11n best.pt.")
    parser.add_argument("--init-tf-attn-weights", default=None, help="Initial TF-Attn-YOLO weights. Default: benchmark tf_attn_yolon best.pt.")
    parser.add_argument("--init-mr-yolo-weights", default=None, help="Initial MR-YOLO weights. Default: benchmark mr_yolo_n all-res best.pt.")
    parser.add_argument(
        "--models",
        default="all",
        help=(
            "Comma-separated model names to train/evaluate. "
            f"Use any of {','.join(DEFAULT_MODEL_FAMILIES)} or 'all'."
        ),
    )
    parser.add_argument("--res-key", default="cfg512")
    parser.add_argument("--mr-res-keys", default=",".join(DEFAULT_RES_KEYS))
    parser.add_argument(
        "--generation-preprocessing",
        default="log_snr_estimated",
        help="Preprocessing used by the simulator when writing spectra to disk.",
    )
    parser.add_argument(
        "--preprocessing",
        default="none",
        help="Preprocessing applied by the core Dataset at training time. Use 'none' when generated tensors are already preprocessed.",
    )
    parser.add_argument("--width-mult", type=float, default=DEFAULT_YOLO_VN_WIDTH_MULT)
    parser.add_argument("--mr-width-mult", type=float, default=DEFAULT_MR_VN_WIDTH_MULT)
    parser.add_argument("--mr-backbone-mode", default="TFSep_pyramid")
    parser.add_argument("--outfusion-channels-mult", type=int, default=1)
    parser.add_argument("--devices", default="cuda:0,cuda:1")
    parser.add_argument("--max-parallel-models", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--monitor", default="val_loss")
    parser.add_argument("--full-eval-every", type=int, default=1)
    parser.add_argument(
        "--train-full-eval",
        action="store_true",
        help="Run full detection metrics during training epochs. By default, curriculum training uses validation loss only.",
    )
    parser.add_argument("--save-last-every", type=int, default=5)
    parser.add_argument("--pfa", type=float, default=1e-2)
    parser.add_argument("--noise-trials", type=int, default=1000)
    parser.add_argument("--noise-variance", type=float, default=1.0)
    parser.add_argument("--eval-seed", type=int, default=444)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--eval-snr-start", type=int, default=-30)
    parser.add_argument("--eval-snr-end", type=int, default=0)
    parser.add_argument("--eval-snr-step", type=int, default=1)
    parser.add_argument("--eval-snr-values", default=None, help="Comma-separated override, e.g. '-30,-20,-10,0,10,20,30'.")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--persistent-workers", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no-amp", action="store_true")
    return parser.parse_args()


def _assign_devices(model_names: tuple[str, ...], devices: tuple[str, ...]) -> dict[str, str]:
    if not devices:
        return {name: "cpu" for name in model_names}
    return {name: devices[index % len(devices)] for index, name in enumerate(model_names)}


def _parse_eval_snr_values(raw: str | None, *, start: int, end: int, step: int) -> tuple[float, ...]:
    if raw:
        return tuple(float(item.strip()) for item in raw.split(",") if item.strip())
    return tuple(float(value) for value in _snr_sequence(start, end, step))


def _weighted_mean(items: list[dict], value_key: str, weight_key: str) -> float | None:
    total_weight = 0.0
    total_value = 0.0
    for item in items:
        value = item.get(value_key)
        weight = float(item.get(weight_key, 0) or 0)
        if value is None or not np.isfinite(float(value)) or weight <= 0:
            continue
        total_value += float(value) * weight
        total_weight += weight
    return float(total_value / total_weight) if total_weight > 0 else None


def _diagonal_rows(rows: list[dict]) -> list[dict]:
    return [
        row
        for row in rows
        if abs(float(row["train_snr"]) - float(row["eval_snr"])) < 1e-9
    ]


def _deep_model_spec(**kwargs) -> DeepModelSpec:
    supported_fields = set(getattr(DeepModelSpec, "__dataclass_fields__", {}))
    if not supported_fields:
        signature = inspect.signature(DeepModelSpec)
        accepts_kwargs = any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values())
        supported_fields = set(signature.parameters) if not accepts_kwargs else set(kwargs)
    filtered = {key: value for key, value in kwargs.items() if key in supported_fields}
    try:
        return DeepModelSpec(**filtered)
    except TypeError as exc:
        message = str(exc)
        unexpected = "unexpected keyword argument "
        if unexpected not in message:
            raise
        bad_key = message.split(unexpected, 1)[1].strip().strip("'\"")
        filtered.pop(bad_key, None)
        return DeepModelSpec(**filtered)


def _evaluate_stage_models(
    *,
    args: argparse.Namespace,
    train_snr: float,
    stage_results: list[dict],
    class_index_to_name: dict[int, str],
) -> tuple[list[dict], list[dict]]:
    class_index_to_name_path = Path(args.output_dir) / "class_index_to_name.json"
    class_index_to_name_path.write_text(
        json.dumps({str(key): value for key, value in class_index_to_name.items()}, indent=2),
        encoding="utf-8",
    )

    eval_specs = []
    for result in stage_results:
        weights_path = Path(str(result.get("best_path") or result.get("transfer_path") or result.get("last_path") or ""))
        if not weights_path.is_file():
            raise FileNotFoundError(f"Missing checkpoint for evaluation: {weights_path}")
        eval_specs.append(
            _deep_model_spec(
                name=f"{result['name']}_train_snr_{float(train_snr):+06.1f}",
                family=str(result["family"]),
                weights_path=weights_path,
                scale="n",
                res_key=str(args.res_key),
                res_keys=_parse_res_keys(args.mr_res_keys),
                preprocessing=str(args.generation_preprocessing),
                num_classes=len(class_index_to_name),
                backbone_mode=str(args.mr_backbone_mode),
                outfusion_channels_mult=int(args.outfusion_channels_mult),
                class_index_to_name_path=class_index_to_name_path,
                class_index_to_name=class_index_to_name,
            )
        )

    payload = run_deep_waveform_snr_sweep(
        DeepWaveformSweepConfig(
            dataset_root=Path(args.val_root),
            models=tuple(eval_specs),
            pfa=float(args.pfa),
            noise_trials=int(args.noise_trials),
            seed=int(args.eval_seed),
            noise_variance=float(args.noise_variance),
            snr_values_db=_parse_eval_snr_values(
                args.eval_snr_values,
                start=int(args.eval_snr_start),
                end=int(args.eval_snr_end),
                step=int(args.eval_snr_step),
            ),
            waveforms=tuple() if str(args.waveforms).lower() == "all" else tuple(_parse_waveforms(args.waveforms)),
            device=str(args.devices).split(",")[0].strip() if str(args.devices).strip() else "cpu",
            batch_size=int(args.eval_batch_size or args.batch_size),
            progress_log=_log,
        )
    )

    rows = []
    waveform_rows = []
    for result in stage_results:
        detector_name = f"{result['name']}_train_snr_{float(train_snr):+06.1f}"
        detector_payload = payload["detectors"][detector_name]
        by_snr = detector_payload.get("by_snr", [])
        by_characterization = detector_payload.get("by_characterization", [])
        threshold_payload = detector_payload.get("threshold", {})
        snr_values = sorted({float(item["snr_db"]) for item in by_snr})
        for eval_snr in snr_values:
            snr_items = [item for item in by_snr if float(item["snr_db"]) == eval_snr]
            char_items = [item for item in by_characterization if float(item["snr_db"]) == eval_snr]
            n_samples = int(sum(int(item.get("n_samples", 0)) for item in snr_items))
            n_detected = int(sum(int(item.get("n_detected", 0)) for item in char_items))
            pd_weighted = _weighted_mean(snr_items, "pd", "n_samples")
            rows.append(
                {
                    "model": result["name"],
                    "train_snr": float(train_snr),
                    "eval_snr": float(eval_snr),
                    "pd": float(pd_weighted) if pd_weighted is not None else float("nan"),
                    "n_samples": n_samples,
                    "threshold": float(threshold_payload.get("threshold", float("nan"))),
                    "empirical_pfa": float(threshold_payload.get("empirical_pfa", float("nan"))),
                    "classification_accuracy_on_detected": _weighted_mean(char_items, "classification_accuracy_on_detected", "n_detected"),
                    "mean_xc_abs_error_on_detected": _weighted_mean(char_items, "mean_center_x_abs_error_on_detected", "n_detected"),
                    "mean_yc_abs_error_on_detected": _weighted_mean(char_items, "mean_center_y_abs_error_on_detected", "n_detected"),
                    "mean_w_abs_error_on_detected": _weighted_mean(char_items, "mean_width_abs_error_on_detected", "n_detected"),
                    "mean_h_abs_error_on_detected": _weighted_mean(char_items, "mean_height_abs_error_on_detected", "n_detected"),
                    "mean_iou_on_detected": _weighted_mean(char_items, "mean_iou_on_detected", "n_detected"),
                    "n_detected": n_detected,
                }
            )
            waveform_labels = sorted({str(item.get("waveform_label", "")) for item in snr_items if item.get("waveform_label") is not None})
            for waveform_label in waveform_labels:
                waveform_snr_items = [
                    item
                    for item in snr_items
                    if str(item.get("waveform_label", "")) == waveform_label
                ]
                waveform_char_items = [
                    item
                    for item in char_items
                    if str(item.get("waveform_label", "")) == waveform_label
                ]
                waveform_n_samples = int(sum(int(item.get("n_samples", 0)) for item in waveform_snr_items))
                waveform_n_detected = int(sum(int(item.get("n_detected", 0)) for item in waveform_char_items))
                waveform_pd = _weighted_mean(waveform_snr_items, "pd", "n_samples")
                waveform_rows.append(
                    {
                        "model": result["name"],
                        "waveform_label": waveform_label,
                        "train_snr": float(train_snr),
                        "eval_snr": float(eval_snr),
                        "pd": float(waveform_pd) if waveform_pd is not None else float("nan"),
                        "n_samples": waveform_n_samples,
                        "threshold": float(threshold_payload.get("threshold", float("nan"))),
                        "empirical_pfa": float(threshold_payload.get("empirical_pfa", float("nan"))),
                        "classification_accuracy_on_detected": _weighted_mean(waveform_char_items, "classification_accuracy_on_detected", "n_detected"),
                        "mean_xc_abs_error_on_detected": _weighted_mean(waveform_char_items, "mean_center_x_abs_error_on_detected", "n_detected"),
                        "mean_yc_abs_error_on_detected": _weighted_mean(waveform_char_items, "mean_center_y_abs_error_on_detected", "n_detected"),
                        "mean_w_abs_error_on_detected": _weighted_mean(waveform_char_items, "mean_width_abs_error_on_detected", "n_detected"),
                        "mean_h_abs_error_on_detected": _weighted_mean(waveform_char_items, "mean_height_abs_error_on_detected", "n_detected"),
                        "mean_iou_on_detected": _weighted_mean(waveform_char_items, "mean_iou_on_detected", "n_detected"),
                        "n_detected": waveform_n_detected,
                    }
                )
    return rows, waveform_rows


def _write_eval_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["waveform_label"] if any("waveform_label" in row for row in rows) else []
    fieldnames += [
        "model",
        "train_snr",
        "eval_snr",
        "pd",
        "n_samples",
        "threshold",
        "empirical_pfa",
        "classification_accuracy_on_detected",
        "mean_xc_abs_error_on_detected",
        "mean_yc_abs_error_on_detected",
        "mean_w_abs_error_on_detected",
        "mean_h_abs_error_on_detected",
        "mean_iou_on_detected",
        "n_detected",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _plot_pd_curves(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str | None, float], list[dict]] = {}
    for row in rows:
        waveform_label = str(row["waveform_label"]) if "waveform_label" in row else None
        grouped.setdefault((str(row["model"]), waveform_label, float(row["train_snr"])), []).append(row)
    has_waveform_labels = any(key[1] is not None for key in grouped)
    fig, axis = plt.subplots(figsize=(12.0, 7.0) if has_waveform_labels else (11.0, 6.5))
    for (model_name, waveform_label, train_snr), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: float(item["eval_snr"]))
        label = f"{model_name} train {train_snr:g} dB"
        if waveform_label is not None:
            label = f"{model_name} {waveform_label} train {train_snr:g} dB"
        axis.plot(
            [float(item["eval_snr"]) for item in items],
            [float(item["pd"]) for item in items],
            marker="o",
            linewidth=1.2,
            markersize=2.5,
            label=label,
        )
    axis.set_xlabel("Eval SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_diagonal_pd_by_waveform(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[tuple[str, str], list[dict]] = {}
    for row in _diagonal_rows(rows):
        grouped.setdefault((str(row["model"]), str(row["waveform_label"])), []).append(row)

    fig, axis = plt.subplots(figsize=(11.0, 6.5))
    for (model_name, waveform_label), items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: float(item["eval_snr"]))
        axis.plot(
            [float(item["eval_snr"]) for item in items],
            [float(item["pd"]) for item in items],
            marker="o",
            linewidth=1.4,
            markersize=3.0,
            label=f"{model_name} {waveform_label}",
        )
    axis.set_xlabel("Training/evaluation SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="x-small", ncol=2)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def _plot_diagonal_pd_curves(rows: list[dict], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    grouped: dict[str, list[dict]] = {}
    for row in _diagonal_rows(rows):
        grouped.setdefault(str(row["model"]), []).append(row)

    fig, axis = plt.subplots(figsize=(9.0, 5.5))
    for model_name, items in sorted(grouped.items()):
        items = sorted(items, key=lambda item: float(item["eval_snr"]))
        axis.plot(
            [float(item["eval_snr"]) for item in items],
            [float(item["pd"]) for item in items],
            marker="o",
            linewidth=1.5,
            markersize=3.0,
            label=model_name,
        )
    axis.set_xlabel("Training/evaluation SNR (dB)")
    axis.set_ylabel("Pd")
    axis.set_ylim(-0.02, 1.02)
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize="small")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    if not (Path(args.val_root) / "manifest.json").is_file():
        raise FileNotFoundError(f"Missing validation manifest: {Path(args.val_root) / 'manifest.json'}")
    if args.res_key not in DEFAULT_RES_HW:
        raise ValueError(f"Unknown --res-key {args.res_key}. Expected one of {sorted(DEFAULT_RES_HW)}.")
    if not 0.0 < float(args.fit_train_ratio) < 1.0:
        raise ValueError("--fit-train-ratio must be in (0, 1).")
    mr_res_keys = _parse_res_keys(args.mr_res_keys)
    if mr_res_keys != DEFAULT_RES_KEYS:
        _log(f"MR resolution order override: {mr_res_keys}")
    else:
        _log(f"MR resolution order: {mr_res_keys}")
    for key in mr_res_keys:
        if key not in DEFAULT_RES_HW:
            raise ValueError(f"Unknown MR resolution key {key}. Expected one of {sorted(DEFAULT_RES_HW)}.")
    train_snr_values = _snr_sequence(args.train_snr_start, args.train_snr_end, args.train_snr_step)
    if len(train_snr_values) > 61:
        raise ValueError("Training SNR sequence is longer than the locked seed range 400..460.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    devices = tuple(item.strip() for item in str(args.devices).split(",") if item.strip())
    model_names = _parse_model_names(args.models)
    model_families = DEFAULT_MODEL_FAMILIES
    _log(f"Selected models: {model_names}")
    model_devices = _assign_devices(model_names, devices)
    benchmark_defaults = _benchmark_weight_defaults(args.benchmark_weights_root, args.res_key)
    initial_weights: dict[str, Path | None] = {
        "yolov11vn": _optional_path(args.init_yolov11_weights) or benchmark_defaults["yolov11vn"],
        "tf_attn_yolovn": _optional_path(args.init_tf_attn_weights) or benchmark_defaults["tf_attn_yolovn"],
        "mr_yolovn": _optional_path(args.init_mr_yolo_weights) or benchmark_defaults["mr_yolovn"],
    }
    for model_name, weights_path in initial_weights.items():
        if weights_path is not None and weights_path.is_file():
            _log(f"[{model_name}] initial benchmark weights: {weights_path}")
        elif weights_path is not None:
            _log(f"[{model_name}] initial weights not found, training from scratch: {weights_path}")
            initial_weights[model_name] = None
    previous_weights: dict[str, Path | None] = dict(initial_weights)
    all_eval_rows: list[dict] = []
    all_waveform_eval_rows: list[dict] = []

    for step_index, train_snr in enumerate(train_snr_values):
        seed = int(args.seed_start + step_index)
        if seed > 460:
            raise ValueError(f"Seed {seed} exceeds locked maximum seed 460.")

        scenarios_per_waveform = (
            int(args.first_snr_scenarios_per_waveform)
            if step_index == 0
            else int(args.scenarios_per_waveform)
        )
        stage_dataset_dir = args.work_dir / f"snr_{train_snr:+04d}"
        class_index_to_name = _generate_yolo_dataset(
            output_dir=stage_dataset_dir,
            snr_db=float(train_snr),
            seed=seed,
            scenarios_per_waveform=scenarios_per_waveform,
            fit_train_ratio=float(args.fit_train_ratio),
            waveforms=args.waveforms,
            generation_preprocessing=args.generation_preprocessing,
        )
        _log(f"stage snr={train_snr} classes={class_index_to_name}")

        jobs = [
            ModelJob(
                name=name,
                family=model_families[name],
                device=model_devices[name],
                output_dir=args.output_dir / name / f"snr_{train_snr:+04d}",
                previous_weights=previous_weights[name],
            )
            for name in model_names
        ]

        max_workers = max(1, min(int(args.max_parallel_models), len(jobs)))
        stage_results = []
        if max_workers == 1:
            for job in jobs:
                result = _train_model_job(
                    job=job,
                    dataset_root=stage_dataset_dir,
                    class_index_to_name=class_index_to_name,
                    args=args,
                )
                stage_results.append(result)
                if result["transfer_path"]:
                    previous_weights[str(result["name"])] = Path(str(result["transfer_path"]))
        else:
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                futures = [
                    executor.submit(
                        _train_model_job,
                        job=job,
                        dataset_root=stage_dataset_dir,
                        class_index_to_name=class_index_to_name,
                        args=args,
                    )
                    for job in jobs
                ]
                for future in as_completed(futures):
                    result = future.result()
                    stage_results.append(result)
                    if result["transfer_path"]:
                        previous_weights[str(result["name"])] = Path(str(result["transfer_path"]))

        _log(f"evaluating stage train_snr={train_snr} on validation dataset")
        stage_eval_rows, stage_waveform_eval_rows = _evaluate_stage_models(
            args=args,
            train_snr=float(train_snr),
            stage_results=stage_results,
            class_index_to_name=class_index_to_name,
        )
        all_eval_rows.extend(stage_eval_rows)
        all_waveform_eval_rows.extend(stage_waveform_eval_rows)
        csv_path = args.output_dir / "yolo_snr_curriculum_pd_characterization.csv"
        _write_eval_csv(all_eval_rows, csv_path)
        _log(f"updated {csv_path}")
        diagonal_csv_path = args.output_dir / "yolo_snr_curriculum_pd_characterization_diagonal.csv"
        _write_eval_csv(_diagonal_rows(all_eval_rows), diagonal_csv_path)
        _log(f"updated {diagonal_csv_path}")
        waveform_csv_path = args.output_dir / "yolo_snr_curriculum_pd_characterization_by_waveform.csv"
        _write_eval_csv(all_waveform_eval_rows, waveform_csv_path)
        _log(f"updated {waveform_csv_path}")
        diagonal_waveform_csv_path = args.output_dir / "yolo_snr_curriculum_pd_characterization_diagonal_by_waveform.csv"
        _write_eval_csv(_diagonal_rows(all_waveform_eval_rows), diagonal_waveform_csv_path)
        _log(f"updated {diagonal_waveform_csv_path}")
        if not args.no_plot:
            plot_path = args.output_dir / "yolo_snr_curriculum_pd_vs_snr.png"
            _plot_pd_curves(all_eval_rows, plot_path)
            _log(f"updated {plot_path}")
            diagonal_plot_path = args.output_dir / "yolo_snr_curriculum_pd_diagonal.png"
            _plot_diagonal_pd_curves(all_eval_rows, diagonal_plot_path)
            _log(f"updated {diagonal_plot_path}")
            waveform_plot_path = args.output_dir / "yolo_snr_curriculum_pd_by_waveform_vs_snr.png"
            _plot_pd_curves(all_waveform_eval_rows, waveform_plot_path)
            _log(f"updated {waveform_plot_path}")
            diagonal_waveform_plot_path = args.output_dir / "yolo_snr_curriculum_pd_diagonal_by_waveform.png"
            _plot_diagonal_pd_by_waveform(all_waveform_eval_rows, diagonal_waveform_plot_path)
            _log(f"updated {diagonal_waveform_plot_path}")

        if stage_dataset_dir.exists():
            shutil.rmtree(stage_dataset_dir)
            _log(f"removed dataset {stage_dataset_dir}")


if __name__ == "__main__":
    main()
