"""CrewAI tool: map a vibration fault signature to a replacement part SKU.

The mapping is keyed on the dominant frequency (Hz) of the FFT window, which
is the standard way to identify rotating-machinery faults. The Engineer Agent
calls ``identify_part`` with the dict produced by ``read_sensor_anomaly``.
"""
from __future__ import annotations

import json

import structlog
from crewai.tools import tool

log = structlog.get_logger()

# (label, low_hz, high_hz, sku, description)
FAULT_TABLE = [
    ("bearing_fault", 70.0, 110.0, "SKU-BRG-6205",
     "6205-2RS deep-groove ball bearing — BPFO band 85 Hz at 1800 RPM"),
    ("gear_mesh_fault", 250.0, 400.0, "SKU-GBX-HELICAL-32T",
     "32-tooth helical gear — mesh frequency band"),
    ("imbalance", 25.0, 35.0, "SKU-BAL-WEIGHT-KIT",
     "Rotor balancing weight kit — 1x running speed indicates imbalance"),
]
DEFAULT_SKU = ("unknown_fault", "SKU-DIAGNOSTIC-KIT",
               "Unidentified fault — dispatch diagnostic kit for manual inspection")


@tool("identify_part")
def identify_part(sensor_payload_json: str) -> str:
    """Given the JSON payload from read_sensor_anomaly, return the SKU
    of the recommended replacement part along with urgency level."""
    try:
        payload = json.loads(sensor_payload_json)
    except json.JSONDecodeError as exc:
        log.error(
            "parts_lookup_bad_input",
            component="agents.tools.parts_lookup",
            error=str(exc),
        )
        return json.dumps({"ok": False, "error": f"invalid JSON: {exc}"})

    dominant_hz = float(payload.get("dominant_freq_hz", 0.0))
    score = float(payload.get("score", 0.0))
    rul_hours = float(payload.get("rul_hours", 9999.0))

    matched = None
    for label, lo, hi, sku, desc in FAULT_TABLE:
        if lo <= dominant_hz <= hi:
            matched = (label, sku, desc)
            break
    if matched is None:
        matched = DEFAULT_SKU

    if rul_hours <= 12:
        urgency = "critical"
    elif rul_hours <= 48:
        urgency = "high"
    elif score >= 0.5:
        urgency = "elevated"
    else:
        urgency = "routine"

    out = {
        "ok": True,
        "fault_label": matched[0],
        "part_sku": matched[1],
        "part_description": matched[2],
        "anomaly_score": score,
        "rul_hours": rul_hours,
        "dominant_freq_hz": dominant_hz,
        "urgency": urgency,
    }
    log.info(
        "parts_lookup_ok",
        component="agents.tools.parts_lookup",
        sku=out["part_sku"],
        urgency=urgency,
        dominant_hz=dominant_hz,
    )
    return json.dumps(out)
