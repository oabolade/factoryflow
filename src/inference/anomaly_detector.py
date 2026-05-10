"""Score a 512-point FFT window with MOMENT and derive an anomaly score + RUL.

MOMENT operates on patches of 8 timesteps. A 512-point window therefore yields
64 patches, exactly matching the architecture's expected sequence length.

Anomaly scoring strategy:
    score = normalized reconstruction MSE between the model's output and input.
    A module-level calibration max is updated online so that early-demo windows
    don't all score 1.0 — the score is bounded to [0, 1].

RUL estimate:
    Heuristic mapping from anomaly score to remaining useful life in hours,
    anchored on RUL_ALERT_HOURS from .env. Above the alert threshold the RUL
    decays linearly toward 0; below it, RUL stays at the alert value.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np
import structlog
import torch

from src.inference.model_loader import get_model

log = structlog.get_logger()

WINDOW_SIZE = 512
PATCH_SIZE = 8
ANOMALY_THRESHOLD = float(os.getenv("ANOMALY_THRESHOLD", "0.75"))
RUL_ALERT_HOURS = float(os.getenv("RUL_ALERT_HOURS", "48"))

_calibration_max: float = 1e-6  # running max of raw MSE seen so far


@dataclass
class AnomalyResult:
    score: float  # in [0, 1]; >ANOMALY_THRESHOLD = action required
    rul_hours: float  # estimated remaining useful life
    confidence: float  # in [0, 1]; rises as calibration matures
    raw_mse: float
    latency_ms: float

    def as_dict(self) -> dict[str, float]:
        return {
            "score": round(self.score, 4),
            "rul_hours": round(self.rul_hours, 2),
            "confidence": round(self.confidence, 3),
            "raw_mse": round(self.raw_mse, 6),
            "latency_ms": round(self.latency_ms, 2),
        }


def _estimate_rul(score: float) -> float:
    if score <= ANOMALY_THRESHOLD:
        return RUL_ALERT_HOURS
    # Linear decay from alert threshold (full RUL) to score=1.0 (zero RUL).
    span = max(1e-6, 1.0 - ANOMALY_THRESHOLD)
    fraction_remaining = max(0.0, (1.0 - score) / span)
    return round(RUL_ALERT_HOURS * fraction_remaining, 2)


def _to_tensor(window: np.ndarray, bundle) -> torch.Tensor:
    if window.shape[-1] != WINDOW_SIZE:
        raise ValueError(
            f"anomaly_detector expects {WINDOW_SIZE}-point window, got {window.shape}"
        )
    arr = window.astype(np.float32, copy=False)
    # MOMENT expects shape (batch, n_channels, seq_len).
    tensor = torch.from_numpy(arr).reshape(1, 1, WINDOW_SIZE)
    return tensor.to(bundle.device.torch_device).to(bundle.dtype)


def _calibration_update(raw_mse: float) -> tuple[float, float]:
    global _calibration_max
    _calibration_max = max(_calibration_max, raw_mse)
    score = float(np.clip(raw_mse / _calibration_max, 0.0, 1.0))
    # Confidence proxy: how saturated calibration is. Low when _calibration_max
    # is still tiny (early demo windows); high once we've seen real spikes.
    confidence = float(np.clip(_calibration_max / 1.0, 0.05, 1.0))
    return score, confidence


def detect(window: np.ndarray) -> AnomalyResult:
    bundle = get_model()
    started = time.perf_counter()

    try:
        x = _to_tensor(window, bundle)
        with torch.no_grad():
            output = bundle.model(x_enc=x)
        reconstruction = getattr(output, "reconstruction", None)
        if reconstruction is None:
            # MOMENTPipeline returns an object with .reconstruction; fall back to indexing.
            reconstruction = output[0] if hasattr(output, "__getitem__") else output
        diff = (reconstruction.float() - x.float()).pow(2).mean()
        raw_mse = float(diff.item())
    except Exception as exc:
        log.error(
            "inference_failed",
            component="inference.anomaly_detector",
            error=str(exc),
        )
        raise

    score, confidence = _calibration_update(raw_mse)
    rul = _estimate_rul(score)
    latency_ms = (time.perf_counter() - started) * 1000.0

    log.info(
        "inference_complete",
        component="inference.anomaly_detector",
        score=round(score, 4),
        rul_hours=rul,
        raw_mse=round(raw_mse, 6),
        latency_ms=round(latency_ms, 2),
        device=bundle.device.torch_device,
    )

    return AnomalyResult(
        score=score,
        rul_hours=rul,
        confidence=confidence,
        raw_mse=raw_mse,
        latency_ms=latency_ms,
    )


def reset_calibration() -> None:
    global _calibration_max
    _calibration_max = 1e-6
