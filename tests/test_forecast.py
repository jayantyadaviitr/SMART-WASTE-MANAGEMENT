from m2_heatmap.forecast import INTERCEPT_DRY_FRACTION, apply_intercept, load_zones, predict_all


def test_zones_file_has_ten_plus_depot():
    df = load_zones()
    assert (df["type"] == "depot").sum() == 1
    assert (df["type"] != "depot").sum() == 10
    assert df[["lat", "lon"]].notna().all().all()


def test_festival_raises_market_fill():
    df = load_zones()
    morning = {r["id"]: r for r in predict_all(df, "morning")}
    fest = {r["id"]: r for r in predict_all(df, "festival")}
    sabzi = [i for i, r in morning.items() if "Sabzi" in r["name"]][0]
    assert fest[sabzi]["combined"] > morning[sabzi]["combined"]


def test_intercept_lowers_dry_and_combined():
    df = load_zones()
    rows = predict_all(df, "afternoon")
    zid = rows[0]["id"]
    before = rows[0]["dry"]
    after = apply_intercept(rows, zid, INTERCEPT_DRY_FRACTION)
    got = next(r for r in after if r["id"] == zid)
    assert got["dry"] < before
    assert got["combined"] <= rows[0]["combined"]
