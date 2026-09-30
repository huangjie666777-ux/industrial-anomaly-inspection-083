"""End-to-end API tests with real ResNet18 features on CPU."""

import importlib

import pytest
from fastapi.testclient import TestClient

from inspection import app as app_module
from inspection import config, synth

client = TestClient(app_module.app)


def _upload(group, kind, variant):
    response = client.post(
        f"/api/samples/{group}",
        files={
            "file": (
                f"{kind}_{variant}.png",
                synth.example_png(kind, variant),
                "image/png",
            )
        },
    )
    return response


def test_full_workflow():
    for variant in range(4):
        response = _upload("reference", "normal_ref", variant)
        assert response.status_code == 200, response.text
    for variant in range(4):
        response = _upload("calibration", "normal_cal", variant)
        assert response.status_code == 200, response.text

    status = client.get("/api/status").json()
    assert status["reference_count"] == 4
    assert status["calibration_count"] == 4
    assert status["ready"] is True
    assert status["memory_bank_size"] <= 256
    assert status["threshold"] >= 0

    # A fresh normal example should be below or near the threshold.
    normal = client.post(
        "/api/detect",
        files={"file": ("ok.png", synth.example_png("normal_ref", 4), "image/png")},
    )
    assert normal.status_code == 200, normal.text
    normal_body = normal.json()
    assert set(normal_body["distance_map"]["encoding"])
    assert normal_body["distance_map"]["width"] == 256

    # Anomaly examples should score above the calibrated normal threshold.
    anomaly_scores = []
    for kind in ("scratch", "foreign_object"):
        for variant in range(3):
            response = client.post(
                "/api/detect",
                files={
                    "file": (
                        f"{kind}.png",
                        synth.example_png(kind, variant),
                        "image/png",
                    )
                },
            )
            assert response.status_code == 200, response.text
            body = response.json()
            assert body["color_upper_bound"] >= body["threshold"]
            anomaly_scores.append(body["score"])
    assert min(anomaly_scores) > status["threshold"]


def test_duplicate_content_rejected_across_groups():
    response = _upload("reference", "normal_cal", 0)
    assert response.status_code == 409


def test_bad_format_rejected():
    response = client.post(
        "/api/samples/reference",
        files={"file": ("x.gif", b"GIF89a", "image/gif")},
    )
    assert response.status_code == 400


def test_delete_rebuilds_bank():
    before = client.get("/api/status").json()
    samples = client.get("/api/samples/calibration").json()
    response = client.delete(f"/api/samples/calibration/{samples[0]['id']}")
    assert response.status_code == 200
    after = response.json()["status"]
    assert after["calibration_count"] == before["calibration_count"] - 1


def test_persistence_across_store_restart(tmp_path, monkeypatch):
    # Rebuild a store pointed at the existing on-disk data directory.
    from inspection.store import InspectionStore

    restarted = InspectionStore(config.DATA_DIR)
    status = restarted.status()
    assert status["ready"] is True
    assert status["reference_count"] == 4
    assert status["calibration_count"] == 3
