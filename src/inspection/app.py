"""FastAPI application: upload management, calibration and anomaly detection."""

from __future__ import annotations

import base64
import uuid

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles

from . import config, synth
from .features import get_extractor
from .memorybank import nearest_distances
from .preprocessing import ImageValidationError, load_image, upsample_distance_map
from .store import InspectionStore

app = FastAPI(title="工业表面异常定位", version="0.1.0")
store = InspectionStore()
app.mount("/static", StaticFiles(directory=str(config.WEB_DIR / "static")), name="static")


@app.middleware("http")
async def enforce_upload_limit(request: Request, call_next):
    if request.method == "POST" and request.url.path.startswith("/api/"):
        length = request.headers.get("content-length")
        if length and int(length) > config.MAX_UPLOAD_BYTES:
            return JSONResponse(
                {"detail": f"上传文件过大：上限 {config.MAX_UPLOAD_BYTES // 1024 // 1024}MB"},
                status_code=413,
            )
    return await call_next(request)


def _validate_group(group: str) -> str:
    if group not in {"reference", "calibration"}:
        raise HTTPException(status_code=400, detail="group 必须是 reference 或 calibration")
    return group


async def _read_loaded_async(file: UploadFile):
    data = await file.read()
    if len(data) > config.MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="上传文件超过 10MB 限制")
    suffix = "" if file.filename is None else "." + file.filename.rsplit(".", 1)[-1].lower()
    try:
        return load_image(data=data, suffix=suffix, content_type=file.content_type or "")
    except ImageValidationError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


def index_page():
    return FileResponse(config.WEB_DIR / "index.html")


app.add_api_route("/", index_page, methods=["GET"])


@app.get("/api/status")
def status():
    return store.status()


@app.get("/api/preprocessing")
def preprocessing():
    return {
        "image_size": config.IMAGE_SIZE,
        "resize": "完整图双线性缩放到 224x224，不做中心裁剪",
        "normalization": "ImageNet mean/std",
        "encoder": "冻结 ResNet18 (IMAGENET1K_V1)，仅 CPU 推理",
        "features": "layer2(128) 对齐 14x14 后与 layer3(256) 拼接，局部描述子 384 维",
        "memory_bank": "贪心最远点选择，至多 256 项，不构造全样本距离矩阵",
        "score": "每个局部到记忆库的最近欧氏距离，最大值为图像分数",
        "threshold": "独立校准正常图分数的 95% 线性分位数，严格大于阈值才判异常",
        "heatmap": "距离图双线性还原原图尺寸；颜色上限固定为阈值两倍，阈值为 0 时使用校准最高分",
        "max_upload_bytes": config.MAX_UPLOAD_BYTES,
        "allowed_formats": sorted(config.ALLOWED_SUFFIXES),
        "side_limits": [config.MIN_SIDE, config.MAX_SIDE],
    }


@app.get("/api/samples/{group}")
def list_samples(group: str):
    _validate_group(group)
    return store.list_samples(group)


@app.get("/api/samples/{group}/{sample_id}/image")
def sample_image(group: str, sample_id: str):
    _validate_group(group)
    if sample_id not in store.samples or store.samples[sample_id].group != group:
        raise HTTPException(status_code=404, detail="样本不存在")
    return FileResponse(store.sample_path(sample_id, group), media_type="image/png")


@app.post("/api/samples/{group}")
async def upload_sample(group: str, file: UploadFile = File(...)):
    _validate_group(group)
    loaded = await _read_loaded_async(file)
    extractor = get_extractor()
    descriptors = extractor(loaded.tensor)
    with store._lock:
        duplicate = store.find_duplicate(loaded.content_hash)
        if duplicate is not None:
            raise HTTPException(
                status_code=409,
                detail=f"与已存在的 {duplicate.group} 样本内容相同（参考组与校准组不可包含相同图像）",
            )
        preview_bank = (
            None if store.state.memory_bank is None else store.state.memory_bank.clone()
        )
    score = (
        float(nearest_distances(descriptors.squeeze(0), preview_bank).max())
        if preview_bank is not None
        else 0.0
    )
    sample_id = uuid.uuid4().hex
    try:
        sample = store.add_sample(
            sample_id=sample_id,
            group=group,
            filename=file.filename or f"{group}.png",
            content_hash=loaded.content_hash,
            original=loaded.original,
            descriptors=descriptors,
            score=score,
        )
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {
        "id": sample.id,
        "filename": sample.filename,
        "group": sample.group,
        "score": sample.score,
        "status": store.status(),
    }


@app.delete("/api/samples/{group}/{sample_id}")
def remove_sample(group: str, sample_id: str):
    _validate_group(group)
    if not store.delete_sample(sample_id):
        raise HTTPException(status_code=404, detail="样本不存在")
    return {"ok": True, "status": store.status()}


@app.post("/api/detect")
async def detect(file: UploadFile = File(...)):
    loaded = await _read_loaded_async(file)
    descriptors = get_extractor()(loaded.tensor)
    try:
        result = store.detect(descriptors)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    width, height = loaded.original.size
    restored = upsample_distance_map(result["distance_map"], (height, width))
    raw_bytes = np.ascontiguousarray(restored.astype(np.float32)).tobytes()
    return {
        "score": result["score"],
        "threshold": result["threshold"],
        "is_anomaly": result["is_anomaly"],
        "color_upper_bound": result["color_upper_bound"],
        "state_version": result["state_version"],
        "image_width": width,
        "image_height": height,
        "distance_map": {
            "encoding": "base64 little-endian float32",
            "height": height,
            "width": width,
            "data": base64.b64encode(raw_bytes).decode("ascii"),
        },
    }


@app.get("/api/examples")
def list_examples():
    return {
        kind: {"variants": spec["variants"], "purpose": spec["purpose"]}
        for kind, spec in synth.EXAMPLES.items()
    }


@app.get("/api/examples/{kind}/{variant}.png")
def example_image(kind: str, variant: int):
    try:
        data = synth.example_png(kind, variant)
    except KeyError:
        raise HTTPException(status_code=404, detail="示例不存在") from None
    return Response(content=data, media_type="image/png")
