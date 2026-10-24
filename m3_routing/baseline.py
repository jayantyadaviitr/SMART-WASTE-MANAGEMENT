"""Fixed municipal loop: depot → zones in CSV id order → depot."""

from __future__ import annotations

from m3_routing.metrics import haversine_km, path_length_km


def ordered_stops(zones: list[dict]) -> list[dict]:
    return sorted(zones, key=lambda z: int(z["id"]))


def baseline_tour(depot: dict, zones: list[dict]) -> dict:
    stops = ordered_stops(zones)
    coords = [(depot["lat"], depot["lon"])] + [(s["lat"], s["lon"]) for s in stops] + [
        (depot["lat"], depot["lon"])
    ]
    km = path_length_km(coords)
    return {
        "name": "Fixed loop (all zones, CSV order)",
        "stops": stops,
        "coords": coords,
        "km": km,
        "solver": "sequence",
        "skipped": [],
    }
