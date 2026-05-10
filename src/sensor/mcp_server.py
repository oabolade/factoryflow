"""MCP server exposing the bearing-fault simulator over SSE on port 8765.

Resources
- ``sensor://vibration/latest``  — most recent window as JSON
- ``sensor://vibration/stream``  — alias for ``latest`` (single-shot read; clients poll)
- ``sensor://vibration/history`` — last 60 windows as a JSON array

Tools
- ``set_state(state)`` — force the simulator into a named state
- ``get_stats()``      — current RMS, dominant frequency, sample count
"""

from __future__ import annotations

import asyncio
import json
from collections import deque
from typing import Deque

import structlog
from mcp.server.fastmcp import FastMCP

from src.sensor.simulator import BearingFaultSimulator, SensorWindow

log = structlog.get_logger()

EMIT_INTERVAL_SECONDS: float = 5.0
HISTORY_SIZE: int = 60
SERVER_PORT: int = 8765

_simulator: BearingFaultSimulator = BearingFaultSimulator()
_history: Deque[SensorWindow] = deque(maxlen=HISTORY_SIZE)
_emit_task: asyncio.Task | None = None

mcp: FastMCP = FastMCP("factoryflow-sensor")
mcp.settings.host = "0.0.0.0"
mcp.settings.port = SERVER_PORT


def _emit_once() -> SensorWindow:
    window = _simulator.generate_window()
    _history.append(window)
    return window


async def _emit_loop() -> None:
    log.info(
        "emit_loop_start",
        component="sensor.mcp_server",
        interval_s=EMIT_INTERVAL_SECONDS,
    )
    # Seed immediately so the first read after startup is non-empty.
    _emit_once()
    while True:
        try:
            await asyncio.sleep(EMIT_INTERVAL_SECONDS)
            _emit_once()
        except asyncio.CancelledError:
            log.info(
                "emit_loop_cancelled",
                component="sensor.mcp_server",
            )
            raise
        except Exception as exc:  # pragma: no cover — keep loop alive
            log.error(
                "emit_loop_error",
                component="sensor.mcp_server",
                error=str(exc),
            )


async def _ensure_emit_loop() -> None:
    global _emit_task
    if _emit_task is None or _emit_task.done():
        _emit_task = asyncio.create_task(_emit_loop())


@mcp.resource("sensor://vibration/latest")
async def latest_window() -> str:
    """Return the most recent sensor window as JSON."""
    await _ensure_emit_loop()
    if not _history:
        _emit_once()
    return json.dumps(_history[-1].to_dict())


@mcp.resource("sensor://vibration/stream")
async def stream_window() -> str:
    """Single-shot read of the latest window (clients poll for streaming)."""
    return await latest_window()


@mcp.resource("sensor://vibration/history")
async def history_windows() -> str:
    """Return the last ``HISTORY_SIZE`` windows as a JSON array."""
    await _ensure_emit_loop()
    return json.dumps([w.to_dict() for w in _history])


@mcp.tool()
async def set_state(state: str) -> str:
    """Force the simulator into ``normal``, ``degrading``, or ``imminent_failure``."""
    try:
        _simulator.set_state(state)
    except ValueError as exc:
        log.warning(
            "set_state_invalid",
            component="sensor.mcp_server",
            requested=state,
            error=str(exc),
        )
        return json.dumps({"ok": False, "error": str(exc)})
    return json.dumps(
        {
            "ok": True,
            "state": _simulator.state,
            "degradation_level": round(_simulator.degradation_level, 3),
        }
    )


@mcp.tool()
async def get_stats() -> str:
    """Return current RMS, dominant frequency, and total samples emitted."""
    await _ensure_emit_loop()
    if not _history:
        _emit_once()
    latest = _history[-1]
    return json.dumps(
        {
            "state": _simulator.state,
            "degradation_level": round(_simulator.degradation_level, 3),
            "dominant_freq_hz": latest.dominant_freq_hz,
            "rms_velocity": latest.rms_velocity,
            "sample_count": len(_history),
            "last_timestamp": latest.timestamp,
        }
    )


def main() -> None:
    log.info(
        "server_start",
        component="sensor.mcp_server",
        port=SERVER_PORT,
        transport="sse",
    )
    mcp.run(transport="sse")


if __name__ == "__main__":
    main()
