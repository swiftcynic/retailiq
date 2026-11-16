"""Transparent inventory *simulation*; M5 does not include on-hand inventory."""
from __future__ import annotations

import math


def recommend_order(forecast_7d: float, stock_on_hand: int, lead_time_days: int,
                    safety_days: int = 3, on_order: int = 0, backorders: int = 0) -> dict:
    """Simple deterministic coverage policy, not a probabilistic stockout forecast.

    Assumes the seven-day forecast converts to a uniform daily demand rate.
    Recommended orders are integer units. A negative prediction is rejected.
    """
    if not math.isfinite(float(forecast_7d)) or forecast_7d < 0:
        raise ValueError("Forecast must be a finite nonnegative number")
    if any(not isinstance(v, int) or isinstance(v, bool) or v < 0 for v in
           (stock_on_hand, lead_time_days, safety_days, on_order, backorders)):
        raise ValueError("Inventory parameters must be nonnegative integers")
    daily_rate = float(forecast_7d) / 7.0
    lead_demand = daily_rate * lead_time_days
    safety_stock = daily_rate * safety_days
    inventory_position = stock_on_hand + on_order - backorders
    order_qty = max(0, math.ceil(lead_demand + safety_stock - inventory_position - 1e-10))
    cover_days = (stock_on_hand / daily_rate) if daily_rate > 0 else None
    return {"daily_demand": daily_rate, "lead_time_demand": lead_demand,
            "safety_stock": safety_stock, "inventory_position": inventory_position,
            "recommended_order": order_qty, "stock_cover_days": cover_days}
