"""Proxlock authorization gate for autonomous procurement.

In ``DEMO_MODE=true`` the call returns a mock approval after a 3s delay so the
UI sequence looks identical to a real Proxlock unlock. In live mode it POSTs
to the Proxlock API; failures are surfaced (never silently approved).
"""
from __future__ import annotations

import asyncio
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import structlog

from src.auth.budget_config import (
    AUTHORIZED_APPROVERS,
    AUTO_APPROVE_LIMIT_USD,
    HARD_BUDGET_CEILING_USD,
)

log = structlog.get_logger()

PROXLOCK_BASE_URL = os.getenv("PROXLOCK_BASE_URL", "https://api.proxlock.io/v1")
DEMO_APPROVAL_DELAY_S = 3.0


@dataclass
class AuthResult:
    authorized: bool
    approver: str
    timestamp: str
    reason: str

    def as_dict(self) -> dict[str, object]:
        return {
            "authorized": self.authorized,
            "approver": self.approver,
            "timestamp": self.timestamp,
            "reason": self.reason,
        }


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _budget_check(amount_usd: float) -> tuple[bool, str]:
    if amount_usd <= 0:
        return False, "amount must be positive"
    if amount_usd > HARD_BUDGET_CEILING_USD:
        return False, f"exceeds hard ceiling ${HARD_BUDGET_CEILING_USD:.0f}"
    return True, "within budget"


async def request_authorization(
    part_sku: str,
    amount_usd: float,
    requester: str = "factoryflow_agent",
) -> AuthResult:
    """Ask Proxlock to authorize an autonomous purchase. Returns AuthResult."""
    log.info(
        "auth_request",
        component="auth.proxlock",
        sku=part_sku,
        amount_usd=amount_usd,
        requester=requester,
    )

    ok, reason = _budget_check(amount_usd)
    if not ok:
        log.warning(
            "auth_rejected_budget",
            component="auth.proxlock",
            sku=part_sku,
            amount_usd=amount_usd,
            reason=reason,
        )
        return AuthResult(False, approver="", timestamp=_now(), reason=reason)

    if amount_usd <= AUTO_APPROVE_LIMIT_USD:
        return AuthResult(
            authorized=True,
            approver="auto_approval",
            timestamp=_now(),
            reason=f"under auto-approve limit ${AUTO_APPROVE_LIMIT_USD:.0f}",
        )

    if _is_demo_mode():
        await asyncio.sleep(DEMO_APPROVAL_DELAY_S)
        approver = AUTHORIZED_APPROVERS[0] if AUTHORIZED_APPROVERS else "demo_approver"
        log.info(
            "auth_demo_approved",
            component="auth.proxlock",
            sku=part_sku,
            amount_usd=amount_usd,
            approver=approver,
        )
        return AuthResult(
            authorized=True,
            approver=approver,
            timestamp=_now(),
            reason="demo mode mock approval",
        )

    return await _live_request(part_sku, amount_usd, requester)


async def _live_request(part_sku: str, amount_usd: float, requester: str) -> AuthResult:
    api_key = os.environ["PROXLOCK_API_KEY"]
    device_id = os.environ["PROXLOCK_DEVICE_ID"]
    body = {
        "device_id": device_id,
        "requester": requester,
        "budget_action": {
            "sku": part_sku,
            "amount_usd": amount_usd,
            "currency": "USD",
        },
    }
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{PROXLOCK_BASE_URL}/authorize",
                headers={"Authorization": f"Bearer {api_key}"},
                json=body,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.error(
            "auth_live_failed",
            component="auth.proxlock",
            sku=part_sku,
            error=str(exc),
        )
        return AuthResult(False, approver="", timestamp=_now(), reason=f"proxlock error: {exc}")

    return AuthResult(
        authorized=bool(data.get("authorized", False)),
        approver=str(data.get("approver", "")),
        timestamp=str(data.get("timestamp", _now())),
        reason=str(data.get("reason", "")),
    )
