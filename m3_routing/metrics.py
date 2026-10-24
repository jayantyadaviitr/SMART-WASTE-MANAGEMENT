"""Distances and savings. Nothing here is a hardcoded demo number."""

from __future__ import annotations

import math

# Diesel combustion factor used only if the user leaves the sidebar default.
# Shown in the UI as an assumption.
KG_CO2_PER_LITRE_DIESEL = 2.68


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlmb = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def path_length_km(coords: list[tuple[float, float]]) -> float:
    if len(coords) < 2:
        return 0.0
    return sum(haversine_km(coords[i], coords[i + 1]) for i in range(len(coords) - 1))


def overflow_incidents(zones: list[dict], visited_ids: set[int], overflow_pct: float = 95.0) -> list[dict]:
    return [
        z
        for z in zones
        if float(z["combined"]) >= overflow_pct and int(z["id"]) not in visited_ids
    ]


def scorecard(
    baseline_km: float,
    dynamic_km: float,
    zones: list[dict],
    dynamic_stops: list[dict],
    km_per_litre: float,
    diesel_inr_per_litre: float,
    overflow_pct: float = 95.0,
    collection_days: int = 300,
) -> dict:
    km_saved = baseline_km - dynamic_km
    litres_saved = km_saved / km_per_litre if km_per_litre else 0.0
    inr_saved = litres_saved * diesel_inr_per_litre
    pct = (100.0 * km_saved / baseline_km) if baseline_km else 0.0
    visited = {int(z["id"]) for z in dynamic_stops}
    skipped = [z for z in zones if int(z["id"]) not in visited]
    overflows = overflow_incidents(zones, visited, overflow_pct)
    return {
        "baseline_km": baseline_km,
        "dynamic_km": dynamic_km,
        "km_saved": km_saved,
        "pct_km_saved": pct,
        "litres_saved": litres_saved,
        "inr_saved": inr_saved,
        "zones_skipped": len(skipped),
        "skipped_names": [z["name"] for z in skipped],
        "overflow_incidents": len(overflows),
        "overflow_names": [z["name"] for z in overflows],
        "co2_kg_saved": litres_saved * KG_CO2_PER_LITRE_DIESEL,
        "annual_litres_saved": litres_saved * collection_days,
        "annual_co2_kg": litres_saved * KG_CO2_PER_LITRE_DIESEL * collection_days,
        "annual_inr": inr_saved * collection_days,
        "assumptions": {
            "km_per_litre": km_per_litre,
            "diesel_inr_per_litre": diesel_inr_per_litre,
            "kg_co2_per_litre": KG_CO2_PER_LITRE_DIESEL,
            "collection_days": collection_days,
            "distance": "haversine straight-line unless OSRM draws the line",
        },
    }
