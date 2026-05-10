"""CrewAI orchestrator wiring Engineer → Procurement as a sequential Crew.

Run as a module for a one-shot end-to-end test:
    python -m src.agents.orchestrator
"""
from __future__ import annotations

import json
import os

import structlog
from crewai import Crew, Process

from src.agents.engineer_agent import build_engineer_agent, build_engineer_task
from src.agents.procurement_agent import (
    build_procurement_agent,
    build_procurement_task,
)
from src.agents.tools import sensor_tool

log = structlog.get_logger()


def build_crew() -> Crew:
    engineer = build_engineer_agent()
    procurement = build_procurement_agent()
    eng_task = build_engineer_task(engineer)
    proc_task = build_procurement_task(procurement)
    proc_task.context = [eng_task]  # procurement reads engineer's output
    return Crew(
        agents=[engineer, procurement],
        tasks=[eng_task, proc_task],
        process=Process.sequential,
        verbose=True,
    )


def run_cycle(force_state: str | None = None) -> dict:
    """Run a single Engineer→Procurement cycle and return the merged result."""
    if force_state:
        sensor_tool.force_state(force_state)
    crew = build_crew()
    result = crew.kickoff()
    payload = {
        "engineer": _safe_json(result.tasks_output[0].raw if result.tasks_output else ""),
        "procurement": _safe_json(
            result.tasks_output[1].raw if len(result.tasks_output) > 1 else ""
        ),
    }
    log.info(
        "crew_cycle_complete",
        component="agents.orchestrator",
        forced_state=force_state or "none",
    )
    return payload


def _safe_json(raw: str) -> dict:
    try:
        return json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return {"raw": raw}


def main() -> None:
    state = os.getenv("DEMO_FORCE_STATE", "imminent_failure")
    out = run_cycle(force_state=state)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
