"""Persistent storage for samples, descriptors, memory bank and calibration."""

from __future__ import annotations

import json
import os
import shutil
import threading
from dataclasses import dataclass, field
from pathlib import Path

import torch
from PIL import Image

from . import config
from .memorybank import (
    build_memory_bank,
    heatmap_upper_bound,
    linear_quantile,
    nearest_distances,
)

STATE_FILE = "state.pt"
META_FILE = "meta.json"


@dataclass
class Sample:
    id: str
    group: str  # "reference" | "calibration"
    filename: str
    content_hash: str
    original_size: list[int]
    descriptors: torch.Tensor  # 196 x 384
    score: float


@dataclass
class DetectorState:
    version: int = 0
    memory_bank: torch.Tensor | None = None
    threshold: float = 0.0
    color_upper_bound: float = config.DISTANCE_FLOOR
    calibration_scores: torch.Tensor = field(default_factory=lambda: torch.empty(0))
    reference_count: int = 0
    calibration_count: int = 0

    @property
    def ready(self) -> bool:
        return self.memory_bank is not None


class InspectionStore:
    def __init__(self, data_dir: Path | None = None) -> None:
        self.data_dir = Path(data_dir or config.DATA_DIR)
        self.ref_dir = self.data_dir / "reference"
        self.cal_dir = self.data_dir / "calibration"
        for directory in (self.ref_dir, self.cal_dir):
            directory.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.samples: dict[str, Sample] = {}
        self.state = DetectorState()
        self._load()

    def sample_path(self, sample_id: str, group: str, ext: str = ".png") -> Path:
        base = self.ref_dir if group == "reference" else self.cal_dir
        return base / f"{sample_id}{ext}"

    def list_samples(self, group: str) -> list[dict]:
        with self._lock:
            return [
                {
                    "id": sample.id,
                    "filename": sample.filename,
                    "group": sample.group,
                    "score": sample.score,
                    "original_size": sample.original_size,
                }
                for sample in self.samples.values()
                if sample.group == group
            ]

    def status(self) -> dict:
        with self._lock:
            state = self.state
            return {
                "reference_count": sum(s.group == "reference" for s in self.samples.values()),
                "calibration_count": sum(
                    s.group == "calibration" for s in self.samples.values()
                ),
                "ready": state.ready,
                "memory_bank_size": 0 if state.memory_bank is None else state.memory_bank.shape[0],
                "threshold": state.threshold,
                "color_upper_bound": state.color_upper_bound,
                "state_version": state.version,
            }

    def find_duplicate(self, content_hash: str) -> Sample | None:
        return next(
            (sample for sample in self.samples.values() if sample.content_hash == content_hash),
            None,
        )

    def add_sample(
        self,
        sample_id: str,
        group: str,
        filename: str,
        content_hash: str,
        original: Image.Image,
        descriptors: torch.Tensor,
        score: float,
    ) -> Sample:
        with self._lock:
            if self.find_duplicate(content_hash) is not None:
                raise ValueError("该图像内容已存在：参考组与校准组均不得包含相同图像")
            path = self.sample_path(sample_id, group)
            original.save(path, format="PNG")
            sample = Sample(
                id=sample_id,
                group=group,
                filename=filename,
                content_hash=content_hash,
                original_size=list(original.size),
                descriptors=descriptors.squeeze(0).cpu(),
                score=score,
            )
            self.samples[sample_id] = sample
            self._rebuild_and_persist()
            return sample

    def delete_sample(self, sample_id: str) -> bool:
        with self._lock:
            sample = self.samples.pop(sample_id, None)
            if sample is None:
                return False
            path = self.sample_path(sample_id, sample.group)
            if path.exists():
                path.unlink()
            self._rebuild_and_persist()
            return True

    def detect(self, descriptors: torch.Tensor) -> dict:
        # Snapshot one consistent state version; never mix two versions.
        with self._lock:
            state = self.state
            if state.memory_bank is None:
                raise RuntimeError("记忆库尚未建立：请先上传参考正常图并完成校准")
            memory_bank = state.memory_bank.clone()
            threshold = state.threshold
            color_bound = state.color_upper_bound
            version = state.version

        distances = nearest_distances(descriptors.squeeze(0), memory_bank)
        score = float(distances.max().item())
        return {
            "score": score,
            "threshold": threshold,
            "is_anomaly": score > threshold,
            "color_upper_bound": color_bound,
            "distance_map": distances.view(config.GRID_SIZE, config.GRID_SIZE).cpu(),
            "state_version": version,
        }

    def _rebuild_and_persist(self) -> None:
        """Build a new state candidate; keep the old state if any step fails."""
        references = [s for s in self.samples.values() if s.group == "reference"]
        calibrations = [s for s in self.samples.values() if s.group == "calibration"]
        old_state = self.state
        try:
            candidate = DetectorState(
                version=old_state.version + 1,
                threshold=old_state.threshold,
                color_upper_bound=old_state.color_upper_bound,
                calibration_scores=old_state.calibration_scores,
            )
            if references:
                descriptor_matrix = torch.cat([s.descriptors for s in references], dim=0)
                candidate.memory_bank = build_memory_bank(descriptor_matrix)
                scores = torch.tensor(
                    [
                        float(nearest_distances(s.descriptors, candidate.memory_bank).max())
                        for s in calibrations
                    ],
                    dtype=torch.float32,
                )
                candidate.calibration_scores = scores
                candidate.threshold = linear_quantile(scores) if scores.numel() else 0.0
                candidate.color_upper_bound = heatmap_upper_bound(
                    candidate.threshold, scores
                )
                candidate.reference_count = len(references)
                candidate.calibration_count = len(calibrations)
            self._persist_atomic(candidate)
            self.state = candidate
        except Exception:
            self.state = old_state  # rebuild failure keeps the previous library
            raise

    def _persist_atomic(self, state: DetectorState) -> None:
        payload = {
            "samples": {
                sid: {
                    "id": s.id,
                    "group": s.group,
                    "filename": s.filename,
                    "content_hash": s.content_hash,
                    "original_size": s.original_size,
                    "score": s.score,
                    "descriptors": s.descriptors,
                }
                for sid, s in self.samples.items()
            },
            "memory_bank": state.memory_bank,
            "threshold": state.threshold,
            "color_upper_bound": state.color_upper_bound,
            "calibration_scores": state.calibration_scores,
            "reference_count": state.reference_count,
            "calibration_count": state.calibration_count,
            "preprocessing": {
                "input_size": config.IMAGE_SIZE,
                "resize": "whole-image bilinear to 224x224, no crop",
                "normalization": "ImageNet mean/std",
                "features": [
                    "resnet18.layer2(128) aligned to 14x14",
                    "resnet18.layer3(256) at 14x14",
                ],
                "descriptor_dim": 384,
                "grid_size": config.GRID_SIZE,
                "max_memory_items": config.MAX_MEMORY_ITEMS,
                "selection": "deterministic greedy farthest point",
                "score": "max local euclidean nearest-neighbour distance",
                "threshold_percentile": config.PERCENTILE,
            },
        }
        temporary = self.data_dir / f"{STATE_FILE}.tmp"
        torch.save(payload, temporary)
        temporary.replace(self.data_dir / STATE_FILE)
        meta = {
            "format_version": 1,
            "description": "Memory bank, calibration threshold and preprocessing metadata",
        }
        meta_tmp = self.data_dir / f"{META_FILE}.tmp"
        with open(meta_tmp, "w", encoding="utf-8") as handle:
            json.dump(meta, handle, ensure_ascii=False, indent=2)
        meta_tmp.replace(self.data_dir / META_FILE)

    def _load(self) -> None:
        path = self.data_dir / STATE_FILE
        if not path.exists():
            return
        try:
            payload = torch.load(path, map_location="cpu", weights_only=False)
            for sid, raw in payload["samples"].items():
                group = raw["group"]
                if not self.sample_path(sid, group).exists():
                    raise FileNotFoundError(f"missing sample image {sid}")
                self.samples[sid] = Sample(
                    id=raw["id"],
                    group=group,
                    filename=raw["filename"],
                    content_hash=raw["content_hash"],
                    original_size=raw["original_size"],
                    descriptors=raw["descriptors"].float(),
                    score=raw["score"],
                )
            bank = payload["memory_bank"]
            self.state = DetectorState(
                version=0,
                memory_bank=None if bank is None else bank.float(),
                threshold=float(payload["threshold"]),
                color_upper_bound=float(
                    payload.get("color_upper_bound", config.DISTANCE_FLOOR)
                ),
                calibration_scores=payload.get("calibration_scores", torch.empty(0)).float(),
                reference_count=int(payload.get("reference_count", 0)),
                calibration_count=int(payload.get("calibration_count", 0)),
            )
        except Exception as exc:
            quarantine = self.data_dir / f"{STATE_FILE}.corrupt"
            if quarantine.exists():
                os.remove(quarantine)
            shutil.move(path, quarantine)
            raise RuntimeError(
                f"记忆库恢复失败，已将损坏文件隔离为 {quarantine.name}：{exc}"
            ) from exc
