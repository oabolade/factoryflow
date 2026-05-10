"""Reusable Gradio components and small formatting helpers for the demo UI."""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

import pandas as pd

HISTORY_WINDOWS = 60  # last N polled points shown in the chart


@dataclass
class DemoState:
    score_points: deque[tuple[int, float, float]] = field(
        default_factory=lambda: deque(maxlen=HISTORY_WINDOWS)
    )  # (tick, score, rul_hours)
    tick: int = 0
    agent_log: list[str] = field(default_factory=list)
    last_state_label: str = "normal"
    last_dominant_hz: float = 0.0
    procurement: dict[str, Any] | None = None
    auth: dict[str, Any] | None = None
    payment: dict[str, Any] | None = None

    def append_score(self, score: float, rul_hours: float) -> None:
        self.tick += 1
        self.score_points.append((self.tick, score, rul_hours))

    def score_dataframe(self) -> pd.DataFrame:
        if not self.score_points:
            return pd.DataFrame({"tick": [], "anomaly_score": []})
        ticks, scores, _ = zip(*self.score_points)
        return pd.DataFrame({"tick": list(ticks), "anomaly_score": list(scores)})

    def log(self, msg: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        self.agent_log.append(f"[{ts}] {msg}")
        if len(self.agent_log) > 200:
            del self.agent_log[: len(self.agent_log) - 200]

    def log_text(self) -> str:
        return "\n".join(self.agent_log[-40:])


def format_gauge(score: float, rul_hours: float, state_label: str, dominant_hz: float) -> str:
    bar_len = 24
    filled = int(round(score * bar_len))
    bar = "█" * filled + "░" * (bar_len - filled)
    severity = "NORMAL"
    if score >= 0.85:
        severity = "IMMINENT FAILURE"
    elif score >= 0.75:
        severity = "ACTION REQUIRED"
    elif score >= 0.5:
        severity = "DEGRADING"
    return (
        f"### Live Inference\n"
        f"```\n"
        f"score   {score:0.3f}  [{bar}]\n"
        f"rul     {rul_hours:0.1f} h\n"
        f"state   {state_label}\n"
        f"dom_hz  {dominant_hz:0.1f}\n"
        f"status  {severity}\n"
        f"```"
    )


def format_supplier_card(state: DemoState) -> str:
    proc = state.procurement
    auth = state.auth
    pay = state.payment
    if proc is None:
        return "_No procurement cycle has been run yet._"

    if not proc.get("selected_supplier"):
        return f"**No procurement action.** Reason: `{proc.get('reason', 'not specified')}`"

    lines = [
        "### Procurement",
        f"- **Supplier:** {proc.get('selected_supplier')}",
        f"- **SKU:** `{proc.get('part_sku')}`",
        f"- **Price:** ${float(proc.get('unit_price_usd', 0.0)):.2f}",
        f"- **Delivery:** {proc.get('delivery_days')} days",
        f"- **Stock:** {proc.get('stock_status')}",
        f"- **URL:** {proc.get('purchase_url')}",
    ]
    if auth:
        lines.append("")
        lines.append("### Authorization (Proxlock)")
        lines.append(f"- **Authorized:** {auth.get('authorized')}")
        lines.append(f"- **Approver:** `{auth.get('approver') or '—'}`")
        lines.append(f"- **Reason:** {auth.get('reason')}")
    if pay:
        lines.append("")
        lines.append("### Payment (X402)")
        lines.append(f"- **Status:** {pay.get('status')}")
        lines.append(f"- **Transaction:** `{pay.get('transaction_id')}`")
        lines.append(f"- **Amount:** ${float(pay.get('amount_usd', 0.0)):.2f}")
        if pay.get("receipt_url"):
            lines.append(f"- **Receipt:** {pay.get('receipt_url')}")
    return "\n".join(lines)
