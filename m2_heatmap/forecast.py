"""Predicted fill % per zone. All numbers come from this formula, not a table of answers.

fill_stream = clip(base_stream × population_weight × time_multiplier × event_multiplier, 0, 100)

Combined fill used for map colour and routing:
  combined = clip(0.55 * wet + 0.45 * dry, 0, 100)
Shown in the UI so judges can see it is not a hidden constant.
"""

from __future__ import annotations

from copy import deepcopy

import pandas as pd

from paths import ZONES_CSV

TIME_SLOTS = ("morning", "afternoon", "festival")

# Multipliers by zone type. Morning = wet spike in hostels/homes.
# Afternoon = dry spike in academic/market. Festival ~2.5× on markets.
TIME_MULT = {
    "morning": {
        "campus residential": {"wet": 1.70, "dry": 0.70},
        "campus academic": {"wet": 0.80, "dry": 1.05},
        "campus/town edge": {"wet": 1.00, "dry": 0.90},
        "market": {"wet": 1.10, "dry": 1.15},
        "residential": {"wet": 1.55, "dry": 0.75},
        "market/residential": {"wet": 1.35, "dry": 1.05},
        "transit": {"wet": 1.20, "dry": 1.10},
    },
    "afternoon": {
        "campus residential": {"wet": 0.90, "dry": 1.10},
        "campus academic": {"wet": 0.85, "dry": 1.55},
        "campus/town edge": {"wet": 0.90, "dry": 1.20},
        "market": {"wet": 1.05, "dry": 1.65},
        "residential": {"wet": 1.00, "dry": 1.05},
        "market/residential": {"wet": 1.00, "dry": 1.45},
        "transit": {"wet": 1.10, "dry": 1.50},
    },
    "festival": {
        "campus residential": {"wet": 1.20, "dry": 1.15},
        "campus academic": {"wet": 1.10, "dry": 1.30},
        "campus/town edge": {"wet": 1.20, "dry": 1.40},
        "market": {"wet": 2.50, "dry": 2.50},
        "residential": {"wet": 1.25, "dry": 1.20},
        "market/residential": {"wet": 2.10, "dry": 2.20},
        "transit": {"wet": 1.60, "dry": 1.80},
    },
}

DEFAULT_MULT = {"wet": 1.0, "dry": 1.0}
WET_WEIGHT = 0.55
DRY_WEIGHT = 0.45
INTERCEPT_DRY_FRACTION = 0.65  # kabadiwala takes 65% of that zone's dry pile


def load_zones(path=ZONES_CSV) -> pd.DataFrame:
    df = pd.read_csv(path)
    required = {"id", "name", "type", "lat", "lon", "population_weight", "base_wet", "base_dry"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"zones.csv missing columns: {missing}")
    if df[["lat", "lon"]].isna().any().any():
        raise ValueError("zones.csv has empty lat/lon. Fill verified coordinates before mapping.")
    return df


def collection_zones(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["type"].str.lower() != "depot"].copy().reset_index(drop=True)


def depot_row(df: pd.DataFrame) -> pd.Series:
    depots = df[df["type"].str.lower() == "depot"]
    if depots.empty:
        raise ValueError("zones.csv needs one row with type=depot")
    return depots.iloc[0]


def _clip(x: float) -> float:
    return float(max(0.0, min(100.0, x)))


def predict_zone(row: pd.Series, slot: str) -> dict:
    slot = slot if slot in TIME_MULT else "morning"
    mult = TIME_MULT[slot].get(str(row["type"]).strip(), DEFAULT_MULT)
    w = float(row["population_weight"])
    wet = _clip(float(row["base_wet"]) * w * float(mult["wet"]))
    dry = _clip(float(row["base_dry"]) * w * float(mult["dry"]))
    combined = _clip(WET_WEIGHT * wet + DRY_WEIGHT * dry)
    return {
        "id": int(row["id"]),
        "name": str(row["name"]),
        "type": str(row["type"]),
        "lat": float(row["lat"]),
        "lon": float(row["lon"]),
        "wet": wet,
        "dry": dry,
        "combined": combined,
        "time_mult_wet": float(mult["wet"]),
        "time_mult_dry": float(mult["dry"]),
    }


def predict_all(df: pd.DataFrame, slot: str) -> list[dict]:
    return [predict_zone(row, slot) for _, row in collection_zones(df).iterrows()]


def apply_intercept(rows: list[dict], zone_id: int, fraction: float = INTERCEPT_DRY_FRACTION) -> list[dict]:
    out = deepcopy(rows)
    for r in out:
        if int(r["id"]) == int(zone_id):
            r["dry"] = _clip(r["dry"] * (1.0 - fraction))
            r["combined"] = _clip(WET_WEIGHT * r["wet"] + DRY_WEIGHT * r["dry"])
            r["intercepted"] = True
            r["intercept_fraction"] = fraction
    return out


def colour_for_fill(fill: float, green_max: float = 60.0, yellow_max: float = 85.0) -> str:
    if fill < green_max:
        return "green"
    if fill < yellow_max:
        return "orange"
    return "red"
