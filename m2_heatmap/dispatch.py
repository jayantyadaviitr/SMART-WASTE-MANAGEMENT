"""WhatsApp / voice alerts with a dry-run fallback so the demo never crashes."""

from __future__ import annotations

import os

from dotenv import load_dotenv

from paths import ROOT


def _env() -> None:
    load_dotenv(ROOT / ".env")


def google_maps_link(lat: float, lon: float) -> str:
    return f"https://www.google.com/maps?q={lat},{lon}"


def build_whatsapp_body(zone: dict, kg_per_100pct: float = 80.0) -> str:
    """kg_per_100pct is an assumption (sidebar): dry kg ≈ dry% / 100 * kg_per_100pct."""
    dry_kg = zone["dry"] / 100.0 * kg_per_100pct
    link = google_maps_link(zone["lat"], zone["lon"])
    return (
        f"Roorkee Smart Waste Loop — Kabadiwala intercept\n"
        f"Zone {zone['id']}: {zone['name']}\n"
        f"Predicted dry fill: {zone['dry']:.0f}% (simulated model)\n"
        f"Estimated dry recyclables: {dry_kg:.1f} kg (assumption {kg_per_100pct:.0f} kg at 100% fill)\n"
        f"Please collect dry waste first.\n"
        f"Map: {link}"
    )


def send_whatsapp(body: str, dry_run: bool | None = None) -> dict:
    """Send via Twilio if keys exist; otherwise return the exact preview."""
    _env()
    if dry_run is None:
        dry_run = os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes"}
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    frm = os.getenv("TWILIO_WHATSAPP_FROM", "").strip()
    to = os.getenv("TWILIO_WHATSAPP_TO", "").strip()
    payload = {"body": body, "from": frm, "to": to}

    if dry_run or not sid or not token or not frm or not to:
        return {
            "ok": True,
            "mode": "dry-run",
            "reason": "DRY_RUN is on, or Twilio keys/numbers are missing. Showing the exact message that would be sent.",
            **payload,
        }
    try:
        from twilio.rest import Client

        client = Client(sid, token)
        msg = client.messages.create(from_=frm, to=to, body=body)
        return {"ok": True, "mode": "twilio", "sid": msg.sid, **payload}
    except Exception as exc:
        return {
            "ok": False,
            "mode": "dry-run",
            "reason": f"Twilio failed ({exc}). Falling back to on-screen preview.",
            **payload,
        }


def send_voice(spoken: str, dry_run: bool | None = None) -> dict:
    _env()
    if dry_run is None:
        dry_run = os.getenv("DRY_RUN", "true").lower() in {"1", "true", "yes"}
    sid = os.getenv("TWILIO_ACCOUNT_SID", "").strip()
    token = os.getenv("TWILIO_AUTH_TOKEN", "").strip()
    frm = os.getenv("TWILIO_VOICE_FROM", "").strip()
    to = os.getenv("TWILIO_VOICE_TO", "").strip()
    payload = {"spoken": spoken, "from": frm, "to": to}
    if dry_run or not sid or not token or not frm or not to:
        return {
            "ok": True,
            "mode": "dry-run",
            "reason": "Voice dry-run (no keys or DRY_RUN=true).",
            **payload,
        }
    try:
        from twilio.rest import Client
        from twilio.twiml.voice_response import VoiceResponse

        client = Client(sid, token)
        twiml = VoiceResponse()
        twiml.say(spoken, language="en-IN")
        call = client.calls.create(to=to, from_=frm, twiml=str(twiml))
        return {"ok": True, "mode": "twilio", "sid": call.sid, **payload}
    except Exception as exc:
        return {
            "ok": False,
            "mode": "dry-run",
            "reason": f"Voice call failed ({exc}).",
            **payload,
        }
