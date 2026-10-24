from m2_heatmap.dispatch import build_whatsapp_body, send_whatsapp


def test_message_contains_maps_and_zone():
    zone = {
        "id": 10,
        "name": "Sabzi Mandi (wholesale)",
        "lat": 29.8765,
        "lon": 77.889068,
        "dry": 80.0,
    }
    body = build_whatsapp_body(zone, kg_per_100pct=80.0)
    assert "Sabzi Mandi" in body
    assert "29.8765" in body
    assert "64.0 kg" in body  # 80% of 80 kg


def test_whatsapp_dry_run_never_raises(monkeypatch):
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("TWILIO_ACCOUNT_SID", "")
    out = send_whatsapp("hello", dry_run=True)
    assert out["ok"] is True
    assert out["mode"] == "dry-run"
    assert out["body"] == "hello"
