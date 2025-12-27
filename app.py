import datetime as dt
import io
import logging
import random
import wave

import folium
import networkx as nx
import numpy as np
import pandas as pd
import streamlit as st
from folium.plugins import AntPath
from geopy.distance import geodesic
from streamlit_folium import st_folium


logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(message)s")

SANTA = "\U0001F385"
SNOWFLAKE = "\u2744\ufe0f"
SPARKLES = "\u2728"
GIFT = "\U0001F381"

st.set_page_config(
    page_title="Quantum Santa's Path Optimizer",
    page_icon=SANTA,
    layout="wide",
    initial_sidebar_state="expanded",
)


CITY_DATA = [
    {"name": "North Pole", "lat": 90.0, "lon": 0.0, "tz": 0.0, "icon": SNOWFLAKE, "base_risk": 0.05},
    {"name": "New York", "lat": 40.7128, "lon": -74.0060, "tz": -5.0, "icon": "\U0001F5FD", "base_risk": 0.30},
    {"name": "London", "lat": 51.5074, "lon": -0.1278, "tz": 0.0, "icon": "\U0001F3F0", "base_risk": 0.25},
    {"name": "Tokyo", "lat": 35.6762, "lon": 139.6503, "tz": 9.0, "icon": "\U0001F5FC", "base_risk": 0.35},
    {"name": "Sydney", "lat": -33.8688, "lon": 151.2093, "tz": 10.0, "icon": "\U0001F998", "base_risk": 0.20},
    {"name": "Paris", "lat": 48.8566, "lon": 2.3522, "tz": 1.0, "icon": "\U0001F950", "base_risk": 0.28},
    {"name": "Cairo", "lat": 30.0444, "lon": 31.2357, "tz": 2.0, "icon": "\U0001F9FF", "base_risk": 0.32},
    {"name": "Rio de Janeiro", "lat": -22.9068, "lon": -43.1729, "tz": -3.0, "icon": "\U0001F334", "base_risk": 0.33},
    {"name": "Cape Town", "lat": -33.9249, "lon": 18.4241, "tz": 2.0, "icon": "\U0001F427", "base_risk": 0.22},
    {"name": "Moscow", "lat": 55.7558, "lon": 37.6176, "tz": 3.0, "icon": "\U0001F9CA", "base_risk": 0.27},
    {"name": "Mumbai", "lat": 19.0760, "lon": 72.8777, "tz": 5.5, "icon": "\U0001F54C", "base_risk": 0.38},
    {"name": "Singapore", "lat": 1.3521, "lon": 103.8198, "tz": 8.0, "icon": "\U0001F981", "base_risk": 0.31},
    {"name": "Los Angeles", "lat": 34.0522, "lon": -118.2437, "tz": -8.0, "icon": "\U0001F3AC", "base_risk": 0.29},
    {"name": "Mexico City", "lat": 19.4326, "lon": -99.1332, "tz": -6.0, "icon": "\U0001F32E", "base_risk": 0.34},
    {"name": "Toronto", "lat": 43.6532, "lon": -79.3832, "tz": -5.0, "icon": "\U0001F341", "base_risk": 0.26},
]


def build_city_df():
    return pd.DataFrame(CITY_DATA).set_index("name")


def distance_km(city_a, city_b):
    return geodesic((city_a["lat"], city_a["lon"]), (city_b["lat"], city_b["lon"])).km


def build_distance_matrix(city_df):
    names = list(city_df.index)
    n = len(names)
    dist = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = distance_km(city_df.loc[names[i]], city_df.loc[names[j]])
            dist[i, j] = d
            dist[j, i] = d
    return dist, names


def route_length(route, dist):
    total = 0.0
    for i in range(len(route) - 1):
        total += dist[route[i], route[i + 1]]
    return total


def nearest_neighbor_route(dist):
    n = dist.shape[0]
    unvisited = set(range(1, n))
    route = [0]
    while unvisited:
        last = route[-1]
        next_city = min(unvisited, key=lambda x: dist[last, x])
        route.append(next_city)
        unvisited.remove(next_city)
    route.append(0)
    return route


def christofides_route(dist):
    n = dist.shape[0]
    g = nx.Graph()
    for i in range(n):
        for j in range(i + 1, n):
            g.add_edge(i, j, weight=dist[i, j])
    cycle = nx.algorithms.approximation.traveling_salesman_problem(
        g,
        weight="weight",
        cycle=True,
        method=nx.algorithms.approximation.christofides,
    )
    return rotate_cycle_start(cycle, 0)


def rotate_cycle_start(route, start_index):
    if route[0] == start_index and route[-1] == start_index:
        return route
    if start_index in route:
        idx = route.index(start_index)
        rotated = route[idx:] + route[1:idx + 1]
        if rotated[0] != start_index:
            rotated = [start_index] + rotated
        if rotated[-1] != start_index:
            rotated.append(start_index)
        return rotated
    return [start_index] + route + [start_index]


def compute_arrivals(route, city_df, dist, start_dt, speed_kmph):
    arrivals = []
    elapsed_hours = 0.0
    names = list(city_df.index)
    for idx, city_idx in enumerate(route):
        city_name = names[city_idx]
        city = city_df.loc[city_name]
        arrival_utc = start_dt + dt.timedelta(hours=elapsed_hours)
        local_time = arrival_utc + dt.timedelta(hours=city["tz"])
        arrivals.append(
            {
                "city": city_name,
                "order": idx + 1,
                "arrival_utc": arrival_utc,
                "local_time": local_time,
            }
        )
        if idx < len(route) - 1:
            leg_km = dist[route[idx], route[idx + 1]]
            elapsed_hours += leg_km / speed_kmph
    return arrivals


def predict_awake_probability(base_prob, hour):
    if 23 <= hour or hour < 5:
        time_factor = 0.3
    elif 5 <= hour < 8:
        time_factor = 0.7
    elif 8 <= hour < 18:
        time_factor = 1.2
    else:
        time_factor = 0.8
    return min(0.95, base_prob * time_factor)


def compute_route_metrics(route, city_df, dist, start_dt, speed_kmph):
    arrivals = compute_arrivals(route, city_df, dist, start_dt, speed_kmph)
    risks = []
    for item in arrivals:
        city = city_df.loc[item["city"]]
        hour = item["local_time"].hour
        risk = predict_awake_probability(city["base_risk"], hour)
        item["risk"] = risk
        risks.append(risk)
    total_distance = route_length(route, dist)
    avg_risk = float(np.mean(risks)) if risks else 0.0
    return total_distance, avg_risk, arrivals


def route_edge_similarity(route_a, route_b):
    edges_a = {(route_a[i], route_a[i + 1]) for i in range(len(route_a) - 1)}
    edges_b = {(route_b[i], route_b[i + 1]) for i in range(len(route_b) - 1)}
    if not edges_a:
        return 0.0
    return len(edges_a & edges_b) / len(edges_a)


def quantum_inspired_tsp(dist, start_dt, strength, city_df, speed_kmph):
    base_route = nearest_neighbor_route(dist)
    candidates = [base_route]
    rng = np.random.default_rng()
    num_candidates = int(10 + 20 * strength)
    for _ in range(num_candidates):
        perm = base_route[1:-1]
        perm = perm.copy()
        swaps = max(1, int(strength * len(perm)))
        for _ in range(swaps):
            i, j = rng.integers(0, len(perm), size=2)
            perm[i], perm[j] = perm[j], perm[i]
        candidate = [0] + perm + [0]
        candidates.append(candidate)

    costs = []
    for route in candidates:
        dist_km, avg_risk, _ = compute_route_metrics(route, city_df, dist, start_dt, speed_kmph)
        costs.append(dist_km * (1.0 + avg_risk))

    best_idx = int(np.argmin(costs))
    worst_idx = int(np.argmax(costs))
    best_route = candidates[best_idx]
    worst_route = candidates[worst_idx]

    amplitudes = []
    for route, cost in zip(candidates, costs):
        weight = 1.0 / (1.0 + cost)
        sim_best = route_edge_similarity(route, best_route)
        sim_worst = route_edge_similarity(route, worst_route)
        interference = (1.0 + 0.5 * sim_best) * (1.0 - 0.3 * sim_worst * strength)
        amplitudes.append(max(1e-6, weight * interference))

    amplitudes = np.array(amplitudes)
    amplitudes = amplitudes / amplitudes.sum()

    if random.random() < 0.25 * strength:
        worse_pool = np.argsort(costs)[-max(2, len(candidates) // 4):]
        pick = int(rng.choice(worse_pool))
        return candidates[pick]

    pick = int(rng.choice(len(candidates), p=amplitudes))
    return candidates[pick]


def build_map(city_df, route, arrivals):
    names = list(city_df.index)
    map_center = [city_df["lat"].mean(), city_df["lon"].mean()]
    fmap = folium.Map(location=map_center, zoom_start=1, tiles="CartoDB dark_matter")

    risk_lookup = {item["city"]: item["risk"] for item in arrivals}
    coords = []
    for order, city_idx in enumerate(route):
        city_name = names[city_idx]
        city = city_df.loc[city_name]
        coords.append((city["lat"], city["lon"]))
        risk = risk_lookup.get(city_name, 0.0)
        color = "#2ecc71" if risk < 0.2 else "#f1c40f" if risk < 0.5 else "#e74c3c"
        popup = (
            f"<b>{order + 1}. {city_name}</b><br>"
            f"Local time: {arrivals[order]['local_time'].strftime('%H:%M')}<br>"
            f"Awake risk: {risk:.0%}"
        )
        folium.CircleMarker(
            location=(city["lat"], city["lon"]),
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            popup=popup,
        ).add_to(fmap)

    folium.PolyLine(coords, weight=3, color="#d4af37", opacity=0.9).add_to(fmap)
    AntPath(coords, color="#e6f1ff", weight=2, delay=800).add_to(fmap)
    return fmap


def santa_summary(route, city_df, total_distance, avg_risk):
    names = list(city_df.index)
    path = " -> ".join(names[i] for i in route)
    return f"Route: {path}\nDistance: {total_distance:.1f} km\nAvg awake risk: {avg_risk:.0%}"


def render_quantum_cards():
    cards = [
        ("Superposition", "Explore many candidate routes at once to mimic quantum states."),
        ("Tunneling", "Occasionally accept worse routes to escape local minima."),
        ("Interference", "Reinforce good paths and dampen weak ones via similarity weighting."),
    ]
    cols = st.columns(3)
    for col, (title, body) in zip(cols, cards):
        with col:
            st.markdown(
                f"""
                <div class="quantum-card">
                    <h4>{title}</h4>
                    <p>{body}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_css(christmas_mode):
    glow = "glowBorder 4s ease-in-out infinite alternate" if christmas_mode else "none"
    st.markdown(
        f"""
        <style>
        :root {{
            --red: #8b0000;
            --gold: #d4af37;
            --blue1: #0a192f;
            --blue2: #1a2a6c;
            --ice: #e6f1ff;
        }}
        html, body, [class*="css"] {{
            font-family: "Cinzel", "Georgia", serif;
        }}
        .stApp {{
            background: linear-gradient(135deg, var(--blue1), var(--blue2));
            color: var(--ice);
        }}
        section.main > div {{
            animation: {glow};
            border: 1px solid rgba(212, 175, 55, 0.2);
            border-radius: 16px;
            padding: 1.25rem;
            background: rgba(10, 25, 47, 0.45);
            backdrop-filter: blur(6px);
        }}
        h1, h2, h3 {{
            color: var(--ice);
        }}
        .metric-card {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 14px;
            padding: 1rem;
            text-align: center;
            box-shadow: 0 6px 18px rgba(0, 0, 0, 0.25);
        }}
        .metric-card h3 {{
            margin-bottom: 0.25rem;
            color: var(--gold);
        }}
        .metric-card p {{
            margin: 0;
            font-size: 1.3rem;
            color: var(--ice);
        }}
        .quantum-card {{
            background: rgba(255, 255, 255, 0.08);
            border: 1px solid rgba(212, 175, 55, 0.35);
            border-radius: 12px;
            padding: 0.75rem 1rem;
            height: 100%;
        }}
        .quantum-card h4 {{
            margin: 0 0 0.5rem 0;
            color: var(--gold);
        }}
        div.stButton > button {{
            background: linear-gradient(120deg, #d4af37, #f7d774);
            color: #2b1b00;
            font-weight: 700;
            border: none;
            padding: 0.6rem 1.4rem;
            border-radius: 999px;
            font-size: 1.1rem;
        }}
        div.stButton > button:hover {{
            filter: brightness(1.05);
        }}
        @keyframes glowBorder {{
            from {{ box-shadow: 0 0 12px rgba(212, 175, 55, 0.2); }}
            to {{ box-shadow: 0 0 24px rgba(212, 175, 55, 0.45); }}
        }}
        .snowflake {{
            position: fixed;
            top: -10px;
            color: rgba(230, 241, 255, 0.8);
            user-select: none;
            z-index: 9999;
            animation: fall 10s linear infinite;
        }}
        @keyframes fall {{
            0% {{ transform: translateY(-10px); opacity: 0; }}
            10% {{ opacity: 1; }}
            100% {{ transform: translateY(110vh); opacity: 0; }}
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if christmas_mode:
        flakes = "".join(
            f'<div class="snowflake" style="left:{i * 8}%; animation-delay:{i * 0.6}s;">{SNOWFLAKE}</div>'
            for i in range(12)
        )
        st.markdown(flakes, unsafe_allow_html=True)


def render_share_button(text):
    escaped = text.replace("\\", "\\\\").replace("\n", "\\n").replace("'", "\\'").replace('"', '\\"')
    st.components.v1.html(
        f"""
        <div style="display:flex;gap:8px;align-items:center;">
          <button onclick="navigator.clipboard.writeText('{escaped}')"
            style="background:#d4af37;color:#2b1b00;font-weight:700;border:none;padding:10px 16px;border-radius:999px;">
            Copy Route Summary
          </button>
          <span style="color:#e6f1ff;font-size:0.9rem;">Ready to share {GIFT}</span>
        </div>
        """,
        height=60,
    )


@st.cache_data(show_spinner=False)
def generate_bell_audio():
    sample_rate = 22050
    duration = 1.0
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    tone = 0.45 * np.sin(2 * np.pi * 880 * t) * np.exp(-3 * t)
    audio = np.int16(tone * 32767)
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio.tobytes())
    return buffer.getvalue()


def main():
    today = dt.datetime.utcnow().date()
    christmas_mode = today.month == 12 and today.day in (24, 25)
    render_css(christmas_mode)

    st.title(f"{SANTA} Quantum Santa's Path Optimizer")
    st.write("Plan Santa's global gift route with quantum-inspired optimization and risk awareness.")

    city_df = build_city_df()
    dist_check, _ = build_distance_matrix(city_df)
    health_ok = len(city_df.index) >= 15 and np.isfinite(dist_check).all()

    left, right = st.columns([0.3, 0.7], gap="large")

    with left:
        st.subheader("Control Deck")
        city_names = list(city_df.index)
        city_labels = {name: f"{city_df.loc[name]['icon']} {name}" for name in city_names}
        selected = st.multiselect(
            "Select 3-15 cities (North Pole required)",
            options=city_names,
            default=["North Pole", "New York", "London", "Tokyo", "Sydney"],
            format_func=lambda x: city_labels[x],
        )
        algo = st.selectbox(
            "Optimization mode",
            ["Quantum-Inspired", "Classic (Christofides)", "Classic (Greedy)"],
        )
        strength = st.slider("Quantum strength", min_value=0.0, max_value=1.0, value=0.7, step=0.05)
        start_time = st.time_input("Departure time (UTC)", value=dt.time(22, 30))
        speed = st.slider("Santa speed (km/h)", 300.0, 1500.0, 900.0, step=50.0)

        st.markdown("### Quantum Concepts")
        render_quantum_cards()

        enable_sound = st.checkbox("Enable bell sound (Christmas mode)", value=False)
        optimize_clicked = st.button(f"{SPARKLES} OPTIMIZE ROUTE", use_container_width=True, key="optimize")

        st.caption(f"Health check: {'OK' if health_ok else 'Issues detected'}")

    if len(selected) < 3:
        st.error("Pick at least 3 cities to begin the optimization.")
        return
    if len(selected) > 15:
        st.error("Please select 15 cities or fewer for a responsive experience.")
        return
    if "North Pole" not in selected:
        st.error("North Pole must be included as the starting point.")
        return

    selected = ["North Pole"] + [city for city in selected if city != "North Pole"]

    if optimize_clicked:
        with st.spinner("Optimizing across quantum states..."):
            progress = st.progress(0)
            for pct in range(0, 90, 15):
                progress.progress(pct)

            chosen_df = city_df.loc[selected]
            dist, _ = build_distance_matrix(chosen_df)
            start_dt = dt.datetime.combine(today, start_time)
            start_perf = dt.datetime.utcnow()

            try:
                if algo == "Quantum-Inspired":
                    route = quantum_inspired_tsp(dist, start_dt, strength, chosen_df, speed)
                elif algo == "Classic (Christofides)":
                    route = christofides_route(dist)
                else:
                    route = nearest_neighbor_route(dist)
            except Exception as exc:
                logging.exception("Optimization failed: %s", exc)
                st.error("Optimization failed. Please try a different city set or algorithm.")
                return

            total_distance, avg_risk, arrivals = compute_route_metrics(
                route, chosen_df, dist, start_dt, speed
            )
            elapsed = (dt.datetime.utcnow() - start_perf).total_seconds()
            progress.progress(100)

        st.balloons()
        st.session_state["last_result"] = {
            "cities": selected,
            "route": route,
            "dist": dist,
            "arrivals": arrivals,
            "total_distance": total_distance,
            "avg_risk": avg_risk,
            "elapsed": elapsed,
        }

    if "last_result" in st.session_state:
        result = st.session_state["last_result"]
        chosen_df = city_df.loc[result["cities"]]
        route = result["route"]
        dist = result["dist"]
        arrivals = result["arrivals"]
        total_distance = result["total_distance"]
        avg_risk = result["avg_risk"]
        elapsed = result["elapsed"]
        with right:
            st.subheader("Route Visualization")
            fmap = build_map(chosen_df, route, arrivals)
            st_folium(fmap, width=700, height=520)

            metrics = st.columns(3)
            metrics[0].markdown(
                f"<div class='metric-card'><h3>Total Distance</h3><p>{total_distance:.0f} km</p></div>",
                unsafe_allow_html=True,
            )
            metrics[1].markdown(
                f"<div class='metric-card'><h3>Average Risk</h3><p>{avg_risk:.0%}</p></div>",
                unsafe_allow_html=True,
            )
            metrics[2].markdown(
                f"<div class='metric-card'><h3>Optimization Time</h3><p>{elapsed:.2f} s</p></div>",
                unsafe_allow_html=True,
            )

            st.markdown("### Route Details")
            for idx, stop in enumerate(arrivals):
                leg_distance = ""
                if idx < len(route) - 1:
                    leg_km = dist[route[idx], route[idx + 1]]
                    leg_distance = f" - {leg_km:.0f} km to next"
                st.write(
                    f"{stop['order']:02d}. {stop['city']} - "
                    f"{stop['local_time'].strftime('%H:%M')} local - "
                    f"risk {stop['risk']:.0%}{leg_distance}"
                )

            summary = santa_summary(route, chosen_df, total_distance, avg_risk)
            render_share_button(summary)

            if avg_risk > 0.5:
                st.warning("Santa is departing too early. Many kids are still awake.")
            elif avg_risk > 0.3:
                st.info("Consider delaying departure to reduce awake risk.")
            else:
                st.success("Great timing! Most kids are asleep.")
    else:
        with right:
            st.subheader("Route Visualization")
            st.info("Select cities and click OPTIMIZE ROUTE to see the magic.")

    if christmas_mode and enable_sound:
        st.audio(generate_bell_audio(), format="audio/wav")


if __name__ == "__main__":
    main()
