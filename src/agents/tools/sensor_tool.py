"""CrewAI tool: read the latest sensor window and run anomaly detection.

The tool keeps a process-local simulator + last-result cache so successive
agent calls within one Crew kickoff see consistent state. The MCP server is
the public-facing stream for the Gradio UI; agents use this in-process path
for tighter latency and easier orchestration.
"""
from __future__ import annotations

import json

import numpy as np
import structlog
from crewai.tools import tool

from src.inference.anomaly_detector import AnomalyResult, detect
from src.sensor.simulator import BearingFaultSimulator

log = structlog.get_logger()

_simulator = BearingFaultSimulator()
_last: tuple[str, AnomalyResult] | None = None


def force_state(state: str) -> None:
    """Demo helper — switch the in-process simulator state from the UI."""
    _simulator.set_state(state)


def latest() -> tuple[str, AnomalyResult] | None:
    """Return the most recent (state, result) pair without re-running inference."""
    return _last


@tool("read_sensor_anomaly")
def read_sensor_anomaly() -> str:
    """Read the next vibration window from the in-process simulator,
    run MOMENT anomaly detection on it, and return the result as JSON."""
    global _last
    try:
        window = _simulator.generate_window()
        fft = np.array(window.fft_window, dtype=np.float32)
        result = detect(fft)
    except Exception as exc:
        log.error(
            "sensor_tool_failed",
            component="agents.tools.sensor_tool",
            error=str(exc),
        )
        return json.dumps({"ok": False, "error": str(exc)})

    _last = (_simulator.state, result)
    payload = {
        "ok": True,
        "state_label": _simulator.state,
        "dominant_freq_hz": window.dominant_freq_hz,
        "rms_velocity": window.rms_velocity,
        **result.as_dict(),
    }
    log.info(
        "sensor_tool_ok",
        component="agents.tools.sensor_tool",
        state=_simulator.state,
        score=payload["score"],
        rul_hours=payload["rul_hours"],
    )
    return json.dumps(payload)
