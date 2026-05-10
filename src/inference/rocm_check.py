"""Verify the inference device available to FactoryFlow.

Run as a module: ``python -m src.inference.rocm_check``

Selection order (override with ``AMD_DEVICE`` env var):
    1. AMD ROCm GPU (torch built with ROCm exposes ``torch.version.hip``)
    2. NVIDIA CUDA GPU (useful for local dev on non-AMD boxes)
    3. Apple MPS (MacBook fallback)
    4. CPU
"""
from __future__ import annotations

import os
import platform
from dataclasses import dataclass

import structlog
import torch

log = structlog.get_logger()


@dataclass
class DeviceInfo:
    backend: str  # "rocm" | "cuda" | "mps" | "cpu"
    torch_device: str  # value to pass to ``.to(...)``
    name: str
    vram_gb: float | None
    runtime_version: str | None

    def as_dict(self) -> dict[str, object]:
        return {
            "backend": self.backend,
            "torch_device": self.torch_device,
            "name": self.name,
            "vram_gb": self.vram_gb,
            "runtime_version": self.runtime_version,
        }


def _detect_rocm() -> DeviceInfo | None:
    if not torch.cuda.is_available():
        return None
    hip_version = getattr(torch.version, "hip", None)
    if not hip_version:
        return None
    props = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        backend="rocm",
        torch_device="cuda",  # ROCm exposes the CUDA-compatible API
        name=props.name,
        vram_gb=round(props.total_memory / 1024**3, 1),
        runtime_version=hip_version,
    )


def _detect_cuda() -> DeviceInfo | None:
    if not torch.cuda.is_available():
        return None
    if getattr(torch.version, "hip", None):
        return None  # already handled by ROCm path
    props = torch.cuda.get_device_properties(0)
    return DeviceInfo(
        backend="cuda",
        torch_device="cuda",
        name=props.name,
        vram_gb=round(props.total_memory / 1024**3, 1),
        runtime_version=torch.version.cuda,
    )


def _detect_mps() -> DeviceInfo | None:
    mps = getattr(torch.backends, "mps", None)
    if mps is None or not mps.is_available():
        return None
    return DeviceInfo(
        backend="mps",
        torch_device="mps",
        name=f"Apple MPS ({platform.processor() or platform.machine()})",
        vram_gb=None,
        runtime_version=None,
    )


def _detect_cpu() -> DeviceInfo:
    return DeviceInfo(
        backend="cpu",
        torch_device="cpu",
        name=platform.processor() or platform.machine() or "CPU",
        vram_gb=None,
        runtime_version=None,
    )


def detect_device() -> DeviceInfo:
    """Return the best available device, honoring ``AMD_DEVICE`` override."""
    override = os.getenv("AMD_DEVICE", "").strip().lower()
    if override == "cpu":
        return _detect_cpu()

    info = _detect_rocm() or _detect_cuda() or _detect_mps() or _detect_cpu()
    log.info(
        "device_detected",
        component="inference.rocm_check",
        backend=info.backend,
        torch_device=info.torch_device,
        name=info.name,
        vram_gb=info.vram_gb,
        runtime_version=info.runtime_version,
        torch_version=torch.__version__,
    )
    return info


def main() -> None:
    info = detect_device()
    print(f"torch:           {torch.__version__}")
    print(f"backend:         {info.backend}")
    print(f"torch_device:    {info.torch_device}")
    print(f"name:            {info.name}")
    print(f"vram_gb:         {info.vram_gb}")
    print(f"runtime_version: {info.runtime_version}")
    if info.backend == "rocm":
        print("✓ AMD ROCm GPU detected — ready for MI300X demo run.")
    elif info.backend in ("cuda", "mps"):
        print(f"⚠ Running on {info.backend} — fine for local dev, swap to ROCm for the demo.")
    else:
        print("⚠ CPU only — inference will be slow; use for unit tests only.")


if __name__ == "__main__":
    main()
