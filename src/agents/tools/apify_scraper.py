"""CrewAI tool: query suppliers for a part SKU via Apify, with demo fixture.

In ``DEMO_MODE=true`` (or when ``APIFY_API_TOKEN`` is missing) the tool returns
the hardcoded fixture from memory.md so the agent loop runs offline. In live
mode it calls the Apify ``apify/web-scraper`` actor synchronously and parses
the dataset for ``{supplier, unit_price_usd, delivery_days, url}``.
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import structlog
from crewai.tools import tool

log = structlog.get_logger()

APIFY_FIXTURE: dict[str, list[dict[str, Any]]] = {
    "SKU-BRG-6205": [
        {
            "supplier": "BearingPoint Industrial",
            "unit_price_usd": 47.00,
            "delivery_days": 2,
            "stock_status": "in_stock",
            "url": "https://bearingpoint.example.com/6205-2RS",
        },
        {
            "supplier": "GlobalBearings.com",
            "unit_price_usd": 39.50,
            "delivery_days": 5,
            "stock_status": "in_stock",
            "url": "https://globalbearings.example.com/catalog/6205",
        },
        {
            "supplier": "FastParts Express",
            "unit_price_usd": 62.00,
            "delivery_days": 1,
            "stock_status": "low_stock",
            "url": "https://fastparts.example.com/bearings/6205-2RS",
        },
    ],
    "SKU-GBX-HELICAL-32T": [
        {
            "supplier": "GearWorks Direct",
            "unit_price_usd": 312.00,
            "delivery_days": 4,
            "stock_status": "in_stock",
            "url": "https://gearworks.example.com/helical-32t",
        },
    ],
    "SKU-BAL-WEIGHT-KIT": [
        {
            "supplier": "VibraTech Supplies",
            "unit_price_usd": 89.00,
            "delivery_days": 3,
            "stock_status": "in_stock",
            "url": "https://vibratech.example.com/balance-kit",
        },
    ],
}

CACHE_TTL_SECONDS = 60.0
_cache: dict[str, tuple[float, list[dict[str, Any]]]] = {}


def _is_demo_mode() -> bool:
    return os.getenv("DEMO_MODE", "true").lower() in ("1", "true", "yes")


def _from_fixture(sku: str) -> list[dict[str, Any]]:
    return APIFY_FIXTURE.get(sku, [])


def _from_apify(sku: str) -> list[dict[str, Any]]:
    from apify_client import ApifyClient  # imported lazily

    token = os.environ["APIFY_API_TOKEN"]
    actor_id = os.getenv("APIFY_ACTOR_ID", "apify/web-scraper")
    client = ApifyClient(token)
    run_input = {"sku": sku, "maxPagesPerCrawl": 5}
    run = client.actor(actor_id).call(run_input=run_input, timeout_secs=90)
    items = list(client.dataset(run["defaultDatasetId"]).iterate_items())
    return [
        {
            "supplier": str(it.get("supplier", "Unknown")),
            "unit_price_usd": float(it.get("price", 0.0)),
            "delivery_days": int(it.get("delivery_days", 7)),
            "stock_status": str(it.get("stock", "unknown")),
            "url": str(it.get("url", "")),
        }
        for it in items
    ]


@tool("scrape_suppliers")
def scrape_suppliers(part_sku: str) -> str:
    """Return ranked supplier offers for the given part SKU as JSON.
    Uses Apify in live mode, demo fixture when DEMO_MODE=true."""
    sku = part_sku.strip()
    cached = _cache.get(sku)
    if cached and (time.time() - cached[0]) < CACHE_TTL_SECONDS:
        log.info(
            "apify_cache_hit",
            component="agents.tools.apify_scraper",
            sku=sku,
            offers=len(cached[1]),
        )
        return json.dumps({"ok": True, "sku": sku, "offers": cached[1], "source": "cache"})

    demo_mode = _is_demo_mode() or not os.getenv("APIFY_API_TOKEN")
    try:
        offers = _from_fixture(sku) if demo_mode else _from_apify(sku)
    except Exception as exc:
        log.error(
            "apify_scrape_failed",
            component="agents.tools.apify_scraper",
            sku=sku,
            error=str(exc),
        )
        return json.dumps({"ok": False, "error": str(exc), "sku": sku})

    _cache[sku] = (time.time(), offers)
    log.info(
        "apify_scrape_ok",
        component="agents.tools.apify_scraper",
        sku=sku,
        offers=len(offers),
        source="fixture" if demo_mode else "apify",
    )
    return json.dumps({
        "ok": True,
        "sku": sku,
        "offers": offers,
        "source": "fixture" if demo_mode else "apify",
    })
