"""X402 programmable-payment client for autonomous purchase execution.

In ``DEMO_MODE=true`` the call logs the payload and returns a mock transaction
that is visually identical to a real one in the UI. In live mode it POSTs to
the X402 payment endpoint with the merchant credentials from .env.
"""
from __future__ import annotations

import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

import httpx
import structlog

from src.auth.proxlock import AuthResult

log = structlog.get_logger()

X402_BASE_URL = os.getenv("X402_BASE_URL", "https://api.x402.xyz/v1")


@dataclass
class PaymentResult:
    transaction_id: str
    status: str  # "confirmed" | "simulated" | "failed"
    amount_usd: float
    timestamp: str
    receipt_url: str
    error: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "transaction_id": self.transaction_id,
            "status": self.status,
            "amount_usd": self.amount_usd,
            "timestamp": self.timestamp,
            "receipt_url": self.receipt_url,
            "error": self.error,
        }


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


async def execute_purchase(
    part_sku: str,
    amount_usd: float,
    supplier: str,
    purchase_url: str,
    auth: AuthResult,
) -> PaymentResult:
    """Execute the payment after Proxlock has authorized it."""
    if not auth.authorized:
        log.warning(
            "payment_blocked",
            component="payments.x402_client",
            sku=part_sku,
            reason="auth not granted",
        )
        return PaymentResult(
            transaction_id="",
            status="failed",
            amount_usd=amount_usd,
            timestamp=_now(),
            receipt_url="",
            error="authorization not granted",
        )

    payload = {
        "merchant_id": os.getenv("X402_MERCHANT_ID", "demo_merchant"),
        "amount_usd": amount_usd,
        "currency": "USD",
        "metadata": {
            "part_sku": part_sku,
            "supplier": supplier,
            "purchase_url": purchase_url,
            "approver": auth.approver,
            "approved_at": auth.timestamp,
        },
    }

    if _is_demo_mode():
        txn_id = f"sim_{uuid.uuid4().hex[:12]}"
        log.info(
            "payment_simulated",
            component="payments.x402_client",
            transaction_id=txn_id,
            sku=part_sku,
            amount_usd=amount_usd,
            supplier=supplier,
        )
        return PaymentResult(
            transaction_id=txn_id,
            status="simulated",
            amount_usd=amount_usd,
            timestamp=_now(),
            receipt_url=f"https://demo.factoryflow.local/receipts/{txn_id}",
        )

    return await _live_charge(payload, amount_usd)


async def _live_charge(payload: dict, amount_usd: float) -> PaymentResult:
    api_key = os.environ["X402_API_KEY"]
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{X402_BASE_URL}/charges",
                headers={"Authorization": f"Bearer {api_key}"},
                json=payload,
            )
            resp.raise_for_status()
            data = resp.json()
    except httpx.HTTPError as exc:
        log.error(
            "payment_live_failed",
            component="payments.x402_client",
            error=str(exc),
        )
        return PaymentResult(
            transaction_id="",
            status="failed",
            amount_usd=amount_usd,
            timestamp=_now(),
            receipt_url="",
            error=str(exc),
        )

    return PaymentResult(
        transaction_id=str(data.get("transaction_id", "")),
        status=str(data.get("status", "confirmed")),
        amount_usd=float(data.get("amount_usd", amount_usd)),
        timestamp=str(data.get("timestamp", _now())),
        receipt_url=str(data.get("receipt_url", "")),
    )
