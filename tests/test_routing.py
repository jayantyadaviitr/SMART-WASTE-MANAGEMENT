from m2_heatmap.forecast import depot_row, load_zones, predict_all
from m3_routing.baseline import baseline_tour
from m3_routing.dynamic import dynamic_tour
from m3_routing.metrics import haversine_km, scorecard


def test_haversine_known_distance():
    # Roughly 1 degree latitude ~ 111.2 km
    km = haversine_km((29.0, 77.0), (30.0, 77.0))
    assert 110 < km < 112.5


def test_dynamic_drops_low_fill():
    df = load_zones()
    depot = depot_row(df)
    depot_d = {"id": int(depot["id"]), "name": str(depot["name"]), "lat": float(depot["lat"]), "lon": float(depot["lon"])}
    rows = predict_all(df, "morning")
    rows[0]["combined"] = 20.0
    rows[1]["combined"] = 90.0
    for r in rows[2:]:
        r["combined"] = 10.0
    dyn = dynamic_tour(depot_d, rows, fill_threshold=60.0)
    ids = {int(s["id"]) for s in dyn["stops"]}
    assert rows[0]["id"] not in ids
    assert rows[1]["id"] in ids


def test_scorecard_is_computed():
    df = load_zones()
    depot = depot_row(df)
    depot_d = {"id": int(depot["id"]), "name": str(depot["name"]), "lat": float(depot["lat"]), "lon": float(depot["lon"])}
    rows = predict_all(df, "festival")
    base = baseline_tour(depot_d, rows)
    dyn = dynamic_tour(depot_d, rows, fill_threshold=60.0)
    card = scorecard(base["km"], dyn["km"], rows, dyn["stops"], km_per_litre=8.0, diesel_inr_per_litre=94.0)
    assert card["baseline_km"] == base["km"]
    assert card["dynamic_km"] == dyn["km"]
    assert abs(card["km_saved"] - (base["km"] - dyn["km"])) < 1e-9
    assert card["baseline_km"] > 0
