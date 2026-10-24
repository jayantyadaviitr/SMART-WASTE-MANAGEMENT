"""Roorkee Smart Waste Loop — Streamlit demo (laptop / projector).

Run:
    python -m venv .venv
    .venv\\Scripts\\activate
    pip install -r requirements.txt
    python -m m1_audio.train
    streamlit run app.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from m1_audio.capture import load_wav, record_on_impact
from m1_audio.classify import classify_wave, load_metrics, load_model
from m1_audio.features import fft_spectrum
from m1_audio.train import train_and_save
from m2_heatmap.dispatch import build_whatsapp_body, send_voice, send_whatsapp
from m2_heatmap.forecast import (
    DRY_WEIGHT,
    INTERCEPT_DRY_FRACTION,
    TIME_SLOTS,
    WET_WEIGHT,
    apply_intercept,
    depot_row,
    load_zones,
    predict_all,
)
from m2_heatmap.map_view import fill_map, route_map
from m3_routing.baseline import baseline_tour
from m3_routing.dynamic import dynamic_tour, osrm_geometry
from m3_routing.metrics import scorecard
from paths import AUDIO_DIR as SAMPLES

st.set_page_config(
    page_title="Roorkee Smart Waste Loop",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    html, body, [class*="css"] { font-size: 18px; }
    h1 { font-size: 2.1rem !important; }
    h2 { font-size: 1.5rem !important; }
    .stMetric label { font-size: 1rem !important; }
    div[data-testid="stMetricValue"] { font-size: 1.8rem !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

SLOT_LABELS = {
    "morning": "Morning 8 AM",
    "afternoon": "Afternoon 2 PM",
    "festival": "Festival / Mandi day",
}


def _ensure_state():
    if "slot" not in st.session_state:
        st.session_state.slot = "afternoon"
    if "intercepted_ids" not in st.session_state:
        st.session_state.intercepted_ids = []
    if "fill_source" not in st.session_state:
        st.session_state.fill_source = "Forecast model (simulated)"
    if "manual_fill" not in st.session_state:
        st.session_state.manual_fill = {}
    if "seg_boost" not in st.session_state:
        st.session_state.seg_boost = {}  # zone_id -> {wet, dry}
    if "use_osrm" not in st.session_state:
        st.session_state.use_osrm = False
    if "last_alert" not in st.session_state:
        st.session_state.last_alert = None
    if "last_class" not in st.session_state:
        st.session_state.last_class = None
    if "model_ready" not in st.session_state:
        st.session_state.model_ready = False


def demo_reset():
    st.session_state.slot = "festival"
    st.session_state.intercepted_ids = []
    st.session_state.fill_source = "Forecast model (simulated)"
    st.session_state.manual_fill = {}
    st.session_state.seg_boost = {}
    st.session_state.last_alert = None
    st.session_state.last_class = None


def current_rows(df) -> list[dict]:
    rows = predict_all(df, st.session_state.slot)
    for zid in st.session_state.intercepted_ids:
        rows = apply_intercept(rows, zid, INTERCEPT_DRY_FRACTION)
    boost = st.session_state.seg_boost
    for r in rows:
        b = boost.get(int(r["id"]), {})
        r["wet"] = min(100.0, r["wet"] + float(b.get("wet", 0)))
        r["dry"] = min(100.0, r["dry"] + float(b.get("dry", 0)))
        r["combined"] = min(100.0, WET_WEIGHT * r["wet"] + DRY_WEIGHT * r["dry"])
    if st.session_state.fill_source.startswith("Manual"):
        for r in rows:
            key = f"fill_{r['id']}"
            if key in st.session_state.manual_fill:
                r["combined"] = float(st.session_state.manual_fill[key])
                r["wet"] = r["combined"]
                r["dry"] = r["combined"]
    return rows


def sidebar(df):
    st.sidebar.title("Roorkee Smart Waste Loop")
    st.sidebar.caption("Sociothon 2026 demo · all savings computed live")
    if st.sidebar.button("Demo mode: reset known scenario", type="primary"):
        demo_reset()
        st.rerun()

    st.sidebar.markdown("### Assumptions (edit these)")
    km_l = st.sidebar.number_input("Diesel km per litre (assumption)", 2.0, 20.0, 8.0, 0.5)
    price = st.sidebar.number_input("Diesel ₹ per litre (assumption)", 50.0, 150.0, 94.0, 1.0)
    days = st.sidebar.number_input("Collection days per year (assumption)", 50, 365, 300, 1)
    kg_full = st.sidebar.number_input("Kg at 100% dry fill (assumption)", 10.0, 300.0, 80.0, 5.0)
    green = st.sidebar.slider("Green below %", 20, 80, 60)
    yellow = st.sidebar.slider("Yellow below % (else red)", 50, 95, 85)
    thresh = st.sidebar.slider("Dynamic route visits fill ≥ %", 20, 90, 60)
    overflow = st.sidebar.slider("Overflow if unvisited fill ≥ %", 70, 100, 95)
    st.session_state.use_osrm = st.sidebar.checkbox(
        "Try OSRM road geometry (falls back to straight lines)",
        value=st.session_state.use_osrm,
    )
    dry_run = st.sidebar.checkbox("Force Twilio dry-run preview", value=True)

    st.sidebar.markdown("### Fill source")
    st.session_state.fill_source = st.sidebar.radio(
        "How should Routing get fill %?",
        ["Forecast model (simulated)", "Manual sliders (simulated)"],
        index=0 if "Forecast" in st.session_state.fill_source else 1,
    )
    zones = df[df["type"] != "depot"]
    if st.session_state.fill_source.startswith("Manual"):
        st.sidebar.caption("These sliders are SIMULATED. They are not sensor readings.")
        for _, row in zones.iterrows():
            key = f"fill_{int(row['id'])}"
            default = float(st.session_state.manual_fill.get(key, 70.0))
            st.session_state.manual_fill[key] = st.sidebar.slider(
                f"{int(row['id'])} {row['name'][:28]}", 0, 100, int(default), key=f"s_{key}"
            )

    st.sidebar.markdown("### Honesty panel")
    st.sidebar.info(
        "Fill % and kg are a **simulated model**. Distances are computed from "
        "`data/zones.csv` (verify lat/lon on Google Maps). "
        "Audio accuracy is whatever the trained model measures — we do not invent 90%."
    )
    return {
        "km_l": km_l,
        "price": price,
        "days": int(days),
        "kg_full": kg_full,
        "green": float(green),
        "yellow": float(yellow),
        "thresh": float(thresh),
        "overflow": float(overflow),
        "dry_run": dry_run,
    }


def tab_segregation(rows, cfg):
    st.header("1 · Acoustic segregation")
    st.write(
        "An item hits a plate. We record **1.5 s at 22,050 Hz**, pull peak frequency, "
        "spectral centroid, ZCR and 13 MFCCs, then a Random Forest labels it. "
        "If confidence is **under 70%**, it goes to a **manual-check** slot."
    )
    col_a, col_b = st.columns([1, 1])

    model = load_model()
    metrics = load_metrics()
    if model is None:
        with col_a:
            st.warning("No model file yet.")
            if st.button("Train on sample clips (synthetic if you have not recorded yet)"):
                with st.spinner("Training…"):
                    m = train_and_save()
                st.success(f"Held-out accuracy {m['accuracy']:.0%} on {m['n_test']} clips.")
                st.rerun()
        return

    with col_a:
        st.subheader("Live drop or fallback clip")
        if st.button("Listen for an impact (mic)"):
            audio, status = record_on_impact()
            if audio is None:
                st.error(f"Mic fallback: {status}. Use a sample clip instead.")
            else:
                st.session_state.last_class = {"audio": audio, "src": "live mic"}
        samples = []
        for folder in sorted(SAMPLES.iterdir()):
            if folder.is_dir():
                samples.extend(sorted(folder.glob("*.wav"))[:2])
        names = [str(p.relative_to(SAMPLES)) for p in samples]
        if names:
            pick = st.selectbox("Play sample clip (fallback)", names)
            if st.button("Classify selected sample"):
                path = SAMPLES / pick
                audio = load_wav(path)
                src = "sample clip (synthetic)" if path.name.startswith("sim_") else "your recording"
                st.session_state.last_class = {"audio": audio, "src": src, "path": str(path)}
        up = st.file_uploader("Or upload a WAV", type=["wav"])
        if up is not None and st.button("Classify upload"):
            import io
            import soundfile as sf
            import numpy as np
            import tempfile

            data, sr = sf.read(io.BytesIO(up.getvalue()), always_2d=False)
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                sf.write(tmp.name, data, sr)
                audio = load_wav(tmp.name)
            st.session_state.last_class = {"audio": audio, "src": "uploaded (may be real)"}

        zone_ids = [r["id"] for r in rows]
        feed_zone = st.selectbox(
            "Feed this classification into a zone (Phase 5 wiring)",
            zone_ids,
            format_func=lambda i: next(r["name"] for r in rows if r["id"] == i),
        )

    with col_b:
        result = st.session_state.last_class
        if not result:
            st.info("Drop an item, or play a sample clip.")
            return
        audio = result["audio"]
        out = classify_wave(audio, model=model)
        st.markdown(
            f"<div style='padding:16px;border-radius:12px;background:{out['color']};"
            f"color:white;font-size:1.4rem;font-weight:700'>{out['label']}</div>",
            unsafe_allow_html=True,
        )
        st.metric("Confidence", f"{out['confidence']:.0%}")
        st.write(f"Source: **{result['src']}** · Suggested bin: **{out['bin']}**")
        if out["manual"]:
            st.warning(out["reason"] or "Low confidence → manual check.")
        freqs, spec = fft_spectrum(audio)
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.plot(freqs, spec, color="#111")
        ax.set_xlim(0, 8000)
        ax.set_xlabel("Hz")
        ax.set_ylabel("Magnitude")
        ax.set_title("Live FFT spectrum")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close(fig)

        if st.button("Add this drop to selected zone's mix"):
            boost = st.session_state.seg_boost.setdefault(int(feed_zone), {"wet": 0.0, "dry": 0.0})
            if out["raw_class"] == "wet":
                boost["wet"] += 4.0
            elif out["raw_class"] in {"plastic", "paper", "metal", "glass"}:
                boost["dry"] += 4.0
            st.session_state.seg_boost[int(feed_zone)] = boost
            st.success("Heatmap / routing will use the updated mix (simulated increment).")

    st.subheader("Held-out accuracy (real number from your current files)")
    if metrics:
        st.write(
            f"Accuracy **{metrics['accuracy']:.0%}** on {metrics['n_test']} held-out clips "
            f"({metrics.get('note', '')})"
        )
        labels = metrics["labels"]
        cm = pd.DataFrame(metrics["confusion_matrix"], index=labels, columns=labels)
        st.dataframe(cm, use_container_width=True)
    st.caption(
        "Frequency ranges in the brief (wet 80–500 Hz, metal 2.5–6 kHz, …) are a "
        "**hypothesis**. This screen shows what the model actually scored."
    )


def tab_heatmap(rows, cfg, df):
    st.header("2 · Predictive heatmap + kabadiwala intercept")
    st.write(
        "Fill is **simulated**: `base × population_weight × time_multiplier`, "
        f"combined as `{WET_WEIGHT}×wet + {DRY_WEIGHT}×dry`. "
        "Not a sensor."
    )
    labels = [SLOT_LABELS[s] for s in TIME_SLOTS]
    idx = TIME_SLOTS.index(st.session_state.slot) if st.session_state.slot in TIME_SLOTS else 1
    chosen = st.select_slider("Time of day / event", options=labels, value=labels[idx])
    st.session_state.slot = TIME_SLOTS[labels.index(chosen)]

    fmap = fill_map(
        rows,
        green_max=cfg["green"],
        yellow_max=cfg["yellow"],
        title=f"{SLOT_LABELS[st.session_state.slot]} · simulated fill",
    )
    st_folium(fmap, width=1100, height=520)

    table = pd.DataFrame(rows)[["id", "name", "type", "wet", "dry", "combined"]]
    table = table.rename(columns={"combined": "combined_fill_pct", "wet": "wet_pct", "dry": "dry_pct"})
    st.dataframe(table, use_container_width=True, hide_index=True)

    st.subheader("Kabadiwala intercept")
    st.write(
        f"Removes **{INTERCEPT_DRY_FRACTION:.0%} of that zone's dry fill** "
        "(simulated scrap pickup). Routing then skips more green zones."
    )
    names = {r["id"]: r["name"] for r in rows}
    pick = st.selectbox("Zone to intercept", list(names.keys()), format_func=lambda i: f"{i} · {names[i]}")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Intercept dry waste", type="primary"):
            if pick not in st.session_state.intercepted_ids:
                st.session_state.intercepted_ids.append(pick)
            st.rerun()
    with c2:
        if st.button("Send WhatsApp alert"):
            zone = next(r for r in rows if r["id"] == pick)
            body = build_whatsapp_body(zone, kg_per_100pct=cfg["kg_full"])
            st.session_state.last_alert = send_whatsapp(body, dry_run=cfg["dry_run"])
    with c3:
        if st.button("Optional voice call"):
            zone = next(r for r in rows if r["id"] == pick)
            spoken = (
                f"Alert. Dry waste intercept for {zone['name']}. "
                f"Predicted dry fill {zone['dry']:.0f} percent."
            )
            st.session_state.last_alert = send_voice(spoken, dry_run=cfg["dry_run"])

    alert = st.session_state.last_alert
    if alert:
        st.markdown(f"**Alert mode:** `{alert.get('mode')}`")
        st.info(alert.get("reason", ""))
        if "body" in alert:
            st.code(alert["body"])
        if "spoken" in alert:
            st.code(alert["spoken"])
        if alert.get("mode") == "twilio" and alert.get("ok"):
            st.success(f"Twilio accepted the message. sid={alert.get('sid')}")

    if st.session_state.intercepted_ids:
        st.caption("Intercepted zone ids: " + ", ".join(str(i) for i in st.session_state.intercepted_ids))


def tab_routing(rows, cfg, df):
    st.header("3 · Fixed loop vs dynamic truck route")
    st.write(
        "Left: the truck visits **every** zone in CSV order. "
        f"Right: only zones with combined fill **≥ {cfg['thresh']:.0f}%**. "
        "Kilometres are computed from coordinates (haversine). Nothing is hardcoded."
    )
    dep = depot_row(df)
    depot = {"id": int(dep["id"]), "name": str(dep["name"]), "lat": float(dep["lat"]), "lon": float(dep["lon"])}
    base = baseline_tour(depot, rows)
    dyn = dynamic_tour(depot, rows, fill_threshold=cfg["thresh"])

    base_path, base_src = base["coords"], "straight-line"
    dyn_path, dyn_src = dyn["coords"], "straight-line"
    if st.session_state.use_osrm:
        base_path, base_src = osrm_geometry(base["coords"])
        dyn_path, dyn_src = osrm_geometry(dyn["coords"])

    card = scorecard(
        base["km"],
        dyn["km"],
        rows,
        dyn["stops"],
        km_per_litre=cfg["km_l"],
        diesel_inr_per_litre=cfg["price"],
        overflow_pct=cfg["overflow"],
        collection_days=cfg["days"],
    )

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Fixed loop km", f"{card['baseline_km']:.2f}")
    m2.metric("Dynamic km", f"{card['dynamic_km']:.2f}")
    m3.metric("Km saved", f"{card['km_saved']:.2f}", f"{card['pct_km_saved']:.1f}%")
    m4.metric("Diesel litres saved", f"{card['litres_saved']:.2f}")
    m5.metric("Overflow incidents", f"{card['overflow_incidents']}")

    left, right = st.columns(2)
    with left:
        st.caption(f"Geometry: {base_src} · solver: {base['solver']}")
        st_folium(
            route_map(depot, base["stops"], base_path, "Fixed loop", "#64748b"),
            width=540,
            height=480,
            key="map_base",
        )
    with right:
        st.caption(f"Geometry: {dyn_src} · solver: {dyn['solver']}")
        st_folium(
            route_map(depot, dyn["stops"], dyn_path, "Dynamic loop", "#0f766e"),
            width=540,
            height=480,
            key="map_dyn",
        )

    st.subheader("What changed")
    st.write(
        f"Zones skipped: **{card['zones_skipped']}** — "
        + (", ".join(card["skipped_names"]) if card["skipped_names"] else "none")
    )
    if card["overflow_names"]:
        st.error(
            "Unvisited and ≥ overflow threshold: " + ", ".join(card["overflow_names"])
        )
    else:
        st.success("No overflow incidents at the current threshold.")

    st.subheader("Impact if this pattern repeats")
    i1, i2, i3 = st.columns(3)
    i1.metric("Annual litres saved (computed)", f"{card['annual_litres_saved']:.0f}")
    i2.metric("Annual kg CO₂ avoided (computed)", f"{card['annual_co2_kg']:.0f}")
    i3.metric("Annual ₹ diesel (computed)", f"{card['annual_inr']:.0f}")
    st.json(card["assumptions"])
    st.caption(
        "Change any fill slider or the time-of-day control on the Heatmap tab. "
        "This scorecard must move. If it does not, something is hardcoded — it is not."
    )


def main():
    _ensure_state()
    try:
        df = load_zones()
    except Exception as exc:
        st.error(f"Cannot read data/zones.csv: {exc}")
        st.stop()
        return

    cfg = sidebar(df)
    rows = current_rows(df)

    st.title("Roorkee Smart Waste Loop")
    st.write(
        "Today a municipal truck drives the same loop every day. "
        "We sort by **sound**, **predict** dry-waste spikes, **alert kabadiwalas**, "
        "and send the truck only where fill is actually critical."
    )

    t1, t2, t3 = st.tabs(["Segregation", "Heatmap", "Routing"])
    with t1:
        tab_segregation(rows, cfg)
    with t2:
        tab_heatmap(rows, cfg, df)
    with t3:
        tab_routing(rows, cfg, df)


if __name__ == "__main__":
    main()
