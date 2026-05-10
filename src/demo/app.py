"""FactoryFlow Gradio demo — single-page judge-facing UI.

Run: PYTHONPATH=. python -m src.demo.app
Then open http://localhost:7860

Layout (4 panels):
    1. Sensor feed — live anomaly score line chart, polled every 2s
    2. Inference panel — score gauge + RUL countdown
    3. Agent activity log — scrolling text of CrewAI agent steps + auth/pay
    4. Procurement result — supplier card with auth + payment status
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Generator

import gradio as gr
import numpy as np
import structlog
from dotenv import load_dotenv

from src.agents.orchestrator import run_cycle
from src.agents.tools import sensor_tool
from src.auth.proxlock import request_authorization
from src.demo.components import DemoState, format_gauge, format_supplier_card
from src.inference.anomaly_detector import detect
from src.payments.x402_client import execute_purchase
from src.sensor.simulator import BearingFaultSimulator

load_dotenv()
log = structlog.get_logger()

POLL_INTERVAL_S = 2.0

_demo_simulator = BearingFaultSimulator()  # used by the auto-poller only


def _poll_sensor(state: DemoState) -> DemoState:
    window = _demo_simulator.generate_window()
    fft = np.array(window.fft_window, dtype=np.float32)
    result = detect(fft)
    state.last_state_label = _demo_simulator.state
    state.last_dominant_hz = window.dominant_freq_hz
    state.append_score(result.score, result.rul_hours)
    return state


def _set_state(label: str, state: DemoState) -> DemoState:
    _demo_simulator.set_state(label)
    sensor_tool.force_state(label)  # keep agent's in-process simulator in sync
    state.log(f"sensor state forced → {label}")
    return state


def on_poll(state: DemoState):
    state = _poll_sensor(state)
    df = state.score_dataframe()
    last_score = state.score_points[-1][1] if state.score_points else 0.0
    last_rul = state.score_points[-1][2] if state.score_points else 0.0
    gauge = format_gauge(last_score, last_rul, state.last_state_label, state.last_dominant_hz)
    return state, df, gauge


def on_force_state(label: str, state: DemoState):
    state = _set_state(label, state)
    return state, state.log_text()


def on_run_cycle(state: DemoState) -> Generator[tuple[Any, ...], None, None]:
    """Generator: streams updates to log + supplier card as the pipeline runs."""
    state.log("▶ kicking off CrewAI cycle (Engineer → Procurement)")
    yield state, state.log_text(), format_supplier_card(state)

    try:
        crew_out = run_cycle(force_state=None)
    except Exception as exc:
        state.log(f"✗ Crew failed: {exc}")
        yield state, state.log_text(), format_supplier_card(state)
        return

    eng = crew_out.get("engineer", {})
    proc = crew_out.get("procurement", {})
    state.log(f"engineer → SKU={eng.get('part_sku')} urgency={eng.get('urgency')}")
    state.log(f"procurement → {proc.get('selected_supplier')} @ ${proc.get('unit_price_usd')}")
    state.procurement = proc
    yield state, state.log_text(), format_supplier_card(state)

    if not proc.get("selected_supplier"):
        state.log("· no procurement action required, stopping")
        yield state, state.log_text(), format_supplier_card(state)
        return

    sku = proc.get("part_sku") or eng.get("part_sku") or ""
    amount = float(proc.get("unit_price_usd") or 0.0)
    supplier = proc.get("selected_supplier") or ""
    purchase_url = proc.get("purchase_url") or ""

    state.log(f"requesting Proxlock authorization for ${amount:.2f}…")
    yield state, state.log_text(), format_supplier_card(state)
    auth = asyncio.run(request_authorization(sku, amount))
    state.auth = auth.as_dict()
    state.log(f"auth → authorized={auth.authorized} approver={auth.approver}")
    yield state, state.log_text(), format_supplier_card(state)

    if not auth.authorized:
        state.log("✗ authorization denied — payment skipped")
        yield state, state.log_text(), format_supplier_card(state)
        return

    state.log("executing X402 payment…")
    yield state, state.log_text(), format_supplier_card(state)
    pay = asyncio.run(execute_purchase(sku, amount, supplier, purchase_url, auth))
    state.payment = pay.as_dict()
    state.log(f"payment → status={pay.status} txn={pay.transaction_id}")
    yield state, state.log_text(), format_supplier_card(state)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="FactoryFlow — Autonomous Predictive Maintenance") as app:
        gr.Markdown(
            "# FactoryFlow\n"
            "Sensor → MOMENT anomaly detection (AMD GPU) → CrewAI agents → "
            "Proxlock auth → X402 payment. End-to-end autonomous procurement."
        )
        state = gr.State(DemoState())

        with gr.Row():
            with gr.Column(scale=2):
                gr.Markdown("### Vibration sensor — anomaly score (live)")
                chart = gr.LinePlot(
                    x="tick",
                    y="anomaly_score",
                    x_title="window #",
                    y_title="anomaly score",
                    y_lim=[0.0, 1.0],
                    height=260,
                    show_label=False,
                )
            with gr.Column(scale=1):
                gauge = gr.Markdown(format_gauge(0.0, 48.0, "normal", 0.0))
                with gr.Row():
                    btn_normal = gr.Button("Normal", size="sm")
                    btn_degrade = gr.Button("Degrading", size="sm")
                    btn_fail = gr.Button("Imminent failure", size="sm", variant="stop")

        with gr.Row():
            with gr.Column():
                gr.Markdown("### Agent activity log")
                log_box = gr.Textbox(
                    value="",
                    lines=14,
                    max_lines=14,
                    interactive=False,
                    show_label=False,
                )
                run_btn = gr.Button("▶ Run agent cycle", variant="primary")
            with gr.Column():
                gr.Markdown("### Procurement")
                card = gr.Markdown(format_supplier_card(DemoState()))

        timer = gr.Timer(POLL_INTERVAL_S)
        timer.tick(on_poll, inputs=[state], outputs=[state, chart, gauge])

        btn_normal.click(on_force_state, inputs=[gr.State("normal"), state],
                         outputs=[state, log_box])
        btn_degrade.click(on_force_state, inputs=[gr.State("degrading"), state],
                          outputs=[state, log_box])
        btn_fail.click(on_force_state, inputs=[gr.State("imminent_failure"), state],
                       outputs=[state, log_box])

        run_btn.click(on_run_cycle, inputs=[state], outputs=[state, log_box, card])

    return app


def main() -> None:
    app = build_app()
    app.queue().launch(server_name="0.0.0.0", server_port=7860, show_error=True)


if __name__ == "__main__":
    main()
