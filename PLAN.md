# Sociothon 2026: Roorkee Smart Waste Loop

Working title: **Roorkee Smart Waste Loop**

Three modules: **(1)** Acoustic waste segregation → **(2)** Roorkee 10-zone predictive heatmap + kabadiwala alerts → **(3)** AI dynamic truck routing vs fixed route.

## How to run (copy-paste)

On Windows PowerShell, from this folder:

```
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
python -m m1_audio.train
streamlit run app.py
```

Then open the URL Streamlit prints (usually http://localhost:8501).

- Click **Demo mode: reset known scenario** before a judge walk-through.
- Coordinates live in `data/zones.csv`. Check each lat/lon on Google Maps and edit the file if a pin is wrong. The app never invents coordinates.
- Secrets go in `.env` (copy `.env.example`). Leave `DRY_RUN=true` if you have no Twilio keys.

## Pitch

Today a municipal truck drives the same loop every day. We make the loop smart: bins sort waste by **sound**, we **predict** where dry recyclables will pile up, **alert kabadiwalas** to collect them first, and the truck only visits what is actually critical.

## Honest scope

| Part | Real at demo | Simulated at demo |
|---|---|---|
| Module 1 audio | Live mic, FFT/MFCC, classifier | Cardboard chute; `sim_*.wav` clips until you record real drops |
| Module 2 | Map, slider, intercept, Twilio or dry-run preview | Fill % and kg from the formula, not sensors |
| Module 3 | Route optimisation on `zones.csv` coordinates; live km | Diesel km/l, price, CO₂ factor listed in the sidebar |

## Cursor rules

See `.cursor/rules/project.mdc`. Short version: one Streamlit app, one phase at a time when iterating, never hardcode savings, never invent coordinates, always fallback (mic / OSRM / Twilio).

## Formulae (all computed)

- `fill_stream = clip(base × population_weight × time_multiplier, 0, 100)`
- `combined = 0.55 × wet + 0.45 × dry`
- Morning: wet spike in hostels/homes. Afternoon: dry spike in academic/market. Festival: ~2.5× on markets.
- Kabadiwala intercept: remove 65% of that zone’s dry fill.
- Dynamic route: visit zones with combined fill ≥ threshold (default 60%).
- Distance: haversine km; optional OSRM road shape with straight-line fallback.
- Litres saved = km saved / (km per litre from sidebar).

## 10 zones + depot

Campus 1–4, community 5–10. Pins were filled from public listings and **must be verified**.

1. Rajendra / Cautley Bhawan hostels  
2. Ravindra / Govind Bhawan  
3. Academic zone (Main Building)  
4. Canal edge / Saraswati Mandir  
5. Civil Lines Market  
6. Ramnagar  
7. Amber Talab  
8. Subhash Nagar  
9. Roorkee Railway Station  
10. Sabzi Mandi (wholesale)  
Depot: IIT Roorkee (`type=depot`)

## Module 1 recording (humans, not code)

Classes: wet, plastic, paper, metal, glass. 30+ drops each, 1.5 s, same mic and plate. Put WAVs in `data/audio_samples/<class>/` then run `python -m m1_audio.train`. Keep the &lt;70% confidence → manual check path.

## Four-minute demo

| Time | What the audience sees |
|---|---|
| 0:00–0:30 | Fixed loops overflow; kabadiwalas unused |
| 0:30–1:15 | Segregation tab: sample or live drop; mixed → amber |
| 1:15–2:00 | Heatmap: Morning → Afternoon → Festival |
| 2:00–2:40 | Intercept + WhatsApp (real or dry-run preview) |
| 2:40–3:30 | Routing: side-by-side maps + live scorecard |
| 3:30–4:00 | Assumptions, BOM (₹30 piezo + ESP32), roadmap |

Backups: sample clips, dry-run WhatsApp, straight-line routes, hotspot, second laptop.

## Prior art (ideas only — do not copy unlicensed code)

Studied: abdullah-engg/Smart-Waste-Segregation (MIT, camera not audio), belanasaikiran/Smart-Waste-Bin-Sorter, dimssrmdn01/smart-campus-waste-analytics, manetasgiwrgos-cyber/smartcity-route-optimization, arora-amit37/cvrptw_clarke_wright, Arijeet2580/Waste_Management_System.

**Differentiator:** acoustic sort + kabadiwala intercept + dynamic vs fixed route in one demo.

## Tests

```
python -m pytest -q
```

## Acceptance

- Phase 0: `streamlit run app.py` shows three tabs; `zones.csv` has 10 zones + depot.
- Phase 1: change a fill slider; both routes and the scorecard update.
- Phase 2: time slider recolours the map; intercept lowers dry fill.
- Phase 3: WhatsApp sends or shows the exact dry-run message.
- Phase 4: sample clip → spectrum + label + confidence; low confidence → manual check.
- Phase 5–6: segregation can boost a zone; Demo mode resets; OSRM optional.
