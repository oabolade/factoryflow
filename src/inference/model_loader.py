"""Load MOMENT-1-large once and cache as a module-level singleton.

The momentfm wrapper hard-pins old transformers/numpy in its package metadata,
but the actual code works on modern stacks — install with ``--no-deps``.

Public API:
    get_model() -> MomentBundle
        Returns the cached (model, device_info) bundle, loading on first call.
    reset_model()
        Drops the cached model — useful in tests.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import structlog
import torch

from src.inference.rocm_check import DeviceInfo, detect_device

log = structlog.get_logger()

MODEL_NAME = os.getenv("MOMENT_MODEL", "AutonLab/MOMENT-1-large")
TASK_NAME = "reconstruction"  # MOMENT anomaly detection uses reconstruction error


@dataclass
class MomentBundle:
    model: Any
    device: DeviceInfo
    dtype: torch.dtype


_bundle: MomentBundle | None = None


def _select_dtype(device: DeviceInfo) -> torch.dtype:
    # fp16 only pays off on real GPUs; CPU/MPS prefer fp32 for stability.
    if device.backend in ("rocm", "cuda"):
        return torch.float16
    return torch.float32


def _load() -> MomentBundle:
    from momentfm import MOMENTPipeline  # imported lazily; heavy dep

    device = detect_device()
    dtype = _select_dtype(device)

    log.info(
        "model_load_start",
        component="inference.model_loader",
        model=MODEL_NAME,
        device=device.torch_device,
        dtype=str(dtype),
    )
    started = time.perf_counter()

    try:
        model = MOMENTPipeline.from_pretrained(
            MODEL_NAME,
            model_kwargs={"task_name": TASK_NAME},
        )
        model.init()
        model.to(device.torch_device).to(dtype)
        model.eval()
    except Exception as exc:
        log.error(
            "model_load_failed",
            component="inference.model_loader",
            model=MODEL_NAME,
            error=str(exc),
        )
        raise

    elapsed_ms = (time.perf_counter() - started) * 1000.0
    log.info(
        "model_load_complete",
        component="inference.model_loader",
        model=MODEL_NAME,
        device=device.torch_device,
        load_ms=round(elapsed_ms, 1),
    )
    return MomentBundle(model=model, device=device, dtype=dtype)


def get_model() -> MomentBundle:
    global _bundle
    if _bundle is None:
        _bundle = _load()
    return _bundle


def reset_model() -> None:
    global _bundle
    _bundle = None
