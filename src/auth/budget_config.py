"""Budget thresholds and authorized approver list for Proxlock auth gate."""
from __future__ import annotations

import os

# Maximum unit price (USD) that can be auto-approved without human signoff
AUTO_APPROVE_LIMIT_USD = float(os.getenv("AUTO_APPROVE_LIMIT_USD", "100"))

# Hard ceiling — purchases above this are always rejected, even with approval
HARD_BUDGET_CEILING_USD = float(os.getenv("HARD_BUDGET_CEILING_USD", "5000"))

# Comma-separated list of usernames authorized to approve via Proxlock
AUTHORIZED_APPROVERS = [
    name.strip()
    for name in os.getenv("AUTHORIZED_APPROVERS", "factory_lead,maintenance_supervisor").split(",")
    if name.strip()
]
