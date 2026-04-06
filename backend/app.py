from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from .config import APP_NAME, APP_VERSION, DEFAULT_BACKEND_PORT, DEFAULT_FRONTEND_URL, PROJECT_ROOT
from .datasets import (
    dataset_stats as build_dataset_stats,
    list_dataset_examples,
    resolve_dataset_path,
    sample_preview_payload,
    training_dataset_info,
)
from .evaluation import (
    evaluation_confusion_matrices,
    evaluation_f1_stats,
    evaluation_recall_snr,
    evaluation_run_details,
    list_evaluation_runs,
)
from .inference import artifacts_preview_payload
from .path_picker import NativePathPickerError, open_native_path_picker
from .training import (
    cancel_training_run,
    list_training_runs,
    start_training_run,
    system_status_payload,
    training_models_payload,
)


app = FastAPI(title=APP_NAME, version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/connection/info")
def connection_info() -> Dict[str, str]:
    return {
        "app_name": APP_NAME,
        "version": APP_VERSION,
        "project_root": str(PROJECT_ROOT),
        "frontend_url": DEFAULT_FRONTEND_URL,
        "backend_port": str(DEFAULT_BACKEND_PORT),
        "auth_mode": "local-demo",
        "default_username": "admin",
    }


@app.get("/dataset/stats")
def dataset_stats(path: str, split: str = "train") -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(path)
    return build_dataset_stats(dataset_path, split=split)


@app.get("/dataset/examples")
def dataset_examples(path: str, split: str = "train", offset: int = 0, limit: int = 200) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(path)
    return list_dataset_examples(dataset_path, split=split, offset=offset, limit=limit)


@app.get("/dataset/example")
def dataset_example(path: str, split: str = "train", sample_id: str = "", cfg_index: int = 0) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(path)
    if not sample_id:
        raise HTTPException(status_code=400, detail="sample_id is required.")
    return sample_preview_payload(dataset_path, split=split, sample_id=sample_id, cfg_index=cfg_index)


@app.get("/artifacts/preview")
def artifacts_preview(
    path: str,
    split: str = "train",
    sample_id: str = "",
    cfg_index: int = 0,
    checkpoint_path: str = "",
    conf_thres: float = 0.1,
    iou_thres: float = 0.1,
) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(path)
    if not sample_id:
        raise HTTPException(status_code=400, detail="sample_id is required.")
    if not checkpoint_path:
        raise HTTPException(status_code=400, detail="checkpoint_path is required.")
    return artifacts_preview_payload(
        dataset_path,
        split=split,
        sample_id=sample_id,
        cfg_index=cfg_index,
        checkpoint_path=checkpoint_path,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
    )


@app.get("/training/models")
def training_models() -> Dict[str, Any]:
    return training_models_payload()


@app.get("/training/dataset-info")
def training_dataset_info_route(path: str) -> Dict[str, Any]:
    dataset_path = resolve_dataset_path(path)
    return training_dataset_info(dataset_path)


@app.get("/training/runs")
def training_runs() -> Dict[str, Any]:
    return list_training_runs()


@app.get("/system/status")
def system_status() -> Dict[str, Any]:
    return system_status_payload()


@app.post("/system/path-picker")
async def system_path_picker(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")

    try:
        selected_path = open_native_path_picker(
            kind=str(payload.get("kind", "directory")),
            title=str(payload.get("title", "Choisir un chemin")),
            initial_path=str(payload.get("path", "")),
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except NativePathPickerError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {"path": selected_path or "", "cancelled": selected_path is None}


@app.get("/evaluation/runs")
def evaluation_runs() -> Dict[str, Any]:
    return list_evaluation_runs()


@app.get("/evaluation/run")
def evaluation_run(path: str) -> Dict[str, Any]:
    return evaluation_run_details(path)


@app.get("/evaluation/run/recall-snr")
def evaluation_run_recall_snr(path: str, epoch: int) -> Dict[str, Any]:
    return evaluation_recall_snr(path, epoch=epoch)


@app.get("/evaluation/run/f1-stats")
def evaluation_run_f1_stats(path: str, epoch: int) -> Dict[str, Any]:
    return evaluation_f1_stats(path, epoch=epoch)


@app.get("/evaluation/run/confusion-matrices")
def evaluation_run_confusion_matrices(path: str, epoch: int) -> Dict[str, Any]:
    return evaluation_confusion_matrices(path, epoch=epoch)


@app.post("/training/start")
def training_start(payload: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise HTTPException(status_code=400, detail="Request body must be valid JSON.")
    return start_training_run(payload)


@app.post("/training/{run_id}/cancel")
def training_cancel(run_id: str) -> Dict[str, Any]:
    try:
        return cancel_training_run(run_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=f"Unknown run '{run_id}'.") from error


if __name__ == "__main__":
    uvicorn.run("backend.app:app", host="0.0.0.0", port=DEFAULT_BACKEND_PORT, reload=True)
