"""Folium maps: zone fill colours and truck routes as polylines."""

from __future__ import annotations

import folium

from m2_heatmap.forecast import colour_for_fill


def _centre(rows: list[dict]) -> tuple[float, float]:
    if not rows:
        return 29.8660, 77.8946
    return (
        sum(r["lat"] for r in rows) / len(rows),
        sum(r["lon"] for r in rows) / len(rows),
    )


def fill_map(
    rows: list[dict],
    green_max: float = 60.0,
    yellow_max: float = 85.0,
    title: str = "Predicted fill",
) -> folium.Map:
    lat, lon = _centre(rows)
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [29.8660, 77.8946],
        tooltip="Roorkee map-centre hint (verify zones)",
        icon=folium.Icon(color="gray", icon="info-sign"),
    ).add_to(m)
    for r in rows:
        color = colour_for_fill(r["combined"], green_max, yellow_max)
        popup = (
            f"<b>{r['name']}</b><br>"
            f"Type: {r['type']}<br>"
            f"Wet fill: {r['wet']:.1f}% (simulated model)<br>"
            f"Dry fill: {r['dry']:.1f}% (simulated model)<br>"
            f"Combined: {r['combined']:.1f}% = 0.55×wet + 0.45×dry"
        )
        folium.CircleMarker(
            [r["lat"], r["lon"]],
            radius=11,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.85,
            popup=popup,
            tooltip=f"{r['id']} {r['name']} ({r['combined']:.0f}%)",
        ).add_to(m)
    m.get_root().html.add_child(folium.Element(f"<div style='padding:6px;font-size:16px'><b>{title}</b></div>"))
    return m


def route_map(
    depot: dict,
    stops: list[dict],
    path_coords: list[tuple[float, float]],
    title: str,
    line_color: str,
) -> folium.Map:
    pts = [{"lat": depot["lat"], "lon": depot["lon"]}] + stops
    lat, lon = _centre(pts)
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="OpenStreetMap")
    folium.Marker(
        [depot["lat"], depot["lon"]],
        tooltip=f"Depot: {depot['name']}",
        icon=folium.Icon(color="black", icon="home"),
    ).add_to(m)
    for i, s in enumerate(stops, start=1):
        folium.CircleMarker(
            [s["lat"], s["lon"]],
            radius=9,
            color=line_color,
            fill=True,
            fill_color=line_color,
            popup=f"<b>Stop {i}</b><br>{s['name']}<br>fill {s['combined']:.1f}%",
            tooltip=f"{i}. {s['name']}",
        ).add_to(m)
        folium.Marker(
            [s["lat"], s["lon"]],
            icon=folium.DivIcon(
                html=f'<div style="font-size:14px;font-weight:700;color:#111">{i}</div>'
            ),
        ).add_to(m)
    if len(path_coords) >= 2:
        folium.PolyLine(path_coords, color=line_color, weight=5, opacity=0.85).add_to(m)
    m.get_root().html.add_child(folium.Element(f"<div style='padding:6px;font-size:16px'><b>{title}</b></div>"))
    return m
