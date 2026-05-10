"""Engineer Agent — diagnoses failures from sensor data and identifies parts.

Inputs (via tools): the latest anomaly result + dominant frequency.
Outputs: a structured JSON blob the Procurement Agent consumes downstream:
    {part_sku, part_description, anomaly_score, rul_hours, urgency, fault_label}
"""
from __future__ import annotations

from crewai import Agent, Task

from src.agents.llm_config import get_llm
from src.agents.tools.parts_lookup import identify_part
from src.agents.tools.sensor_tool import read_sensor_anomaly


def build_engineer_agent() -> Agent:
    return Agent(
        role="Reliability Engineer",
        goal=(
            "Monitor vibration sensor data and identify which replacement part is "
            "needed if the anomaly score exceeds the action threshold."
        ),
        backstory=(
            "Veteran maintenance engineer with deep experience in rotating-machinery "
            "diagnostics. Reads FFT spectra and connects dominant frequencies to "
            "specific failure modes (bearings, gear meshes, imbalance)."
        ),
        tools=[read_sensor_anomaly, identify_part],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def build_engineer_task(agent: Agent) -> Task:
    return Task(
        description=(
            "1. Call read_sensor_anomaly to fetch the latest sensor window and "
            "anomaly score.\n"
            "2. Pass the JSON output to identify_part to get the recommended SKU.\n"
            "3. If anomaly_score >= 0.75 OR rul_hours <= 48, classify the situation "
            "as actionable and return the identify_part JSON verbatim.\n"
            "4. Otherwise return a JSON object with part_sku set to null and "
            "urgency set to 'routine' so procurement is skipped."
        ),
        expected_output=(
            "A single JSON object with keys: part_sku, part_description, "
            "anomaly_score, rul_hours, urgency, fault_label. Do not wrap in markdown."
        ),
        agent=agent,
    )
