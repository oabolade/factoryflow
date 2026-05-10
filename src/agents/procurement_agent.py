"""Procurement Agent — selects the best supplier for a given part SKU.

Inputs: Engineer Agent's JSON output (part_sku, rul_hours, urgency).
Outputs: JSON with the selected supplier + price + delivery + purchase URL,
which the auth + payments layer consumes next.
"""
from __future__ import annotations

from crewai import Agent, Task

from src.agents.llm_config import get_llm
from src.agents.tools.apify_scraper import scrape_suppliers


def build_procurement_agent() -> Agent:
    return Agent(
        role="Industrial Procurement Specialist",
        goal=(
            "Select the best supplier offer for a given part SKU, balancing unit "
            "price against delivery time within the remaining-useful-life window."
        ),
        backstory=(
            "Procurement lead for a small-batch manufacturer. Optimizes for total "
            "cost of downtime: a slightly more expensive part that arrives in time "
            "beats a cheap one that arrives after failure."
        ),
        tools=[scrape_suppliers],
        llm=get_llm(),
        verbose=True,
        allow_delegation=False,
    )


def build_procurement_task(agent: Agent) -> Task:
    return Task(
        description=(
            "Read the Engineer Agent's structured output from context. If part_sku "
            "is null or urgency is 'routine', return a JSON object with "
            "selected_supplier set to null and reason set to 'no action required'.\n"
            "Otherwise:\n"
            "1. Call scrape_suppliers with the part_sku.\n"
            "2. Filter offers whose delivery_days exceed rul_hours / 24 — those "
            "would arrive after failure.\n"
            "3. From the remaining offers, pick the cheapest unit_price_usd. If "
            "urgency is 'critical' and any in_stock offer arrives within 24h, "
            "prefer fastest delivery over cheapest price.\n"
            "4. Return JSON with: selected_supplier, unit_price_usd, delivery_days, "
            "stock_status, purchase_url, part_sku, reason."
        ),
        expected_output=(
            "A single JSON object with keys: selected_supplier, unit_price_usd, "
            "delivery_days, stock_status, purchase_url, part_sku, reason. "
            "No markdown wrappers."
        ),
        agent=agent,
    )
