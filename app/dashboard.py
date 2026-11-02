"""Streamlit dashboard: play one simulated week for two policies side by side.

    streamlit run app/dashboard.py

Choose a scenario, a start date, the load-shedding level and two policies (baselines, the
oracle or trained models found under models*/), change tariffs / export rate / outage
intensity with the sliders, press Run, then play the week hour by hour or jump to the end.
"""
import glob
import os
import sys
import time
import warnings

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
warnings.filterwarnings("ignore")

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from solar_rl import SolarHomeEnv, Simulator, load_config  # noqa: E402
from solar_rl.baselines import default_baselines, make_baseline  # noqa: E402
from solar_rl.config import available_scenarios  # noqa: E402
from solar_rl.evaluation import load_model, run_episode  # noqa: E402

MODE_COLORS = {"grid_charge": "#7b3294", "solar_charge": "#f1a340", "hold": "#999999", "discharge": "#2c7bb6",
               "reserve": "#4daf4a", "setpoint": "#e7298a"}
LEVELS = ["light", "medium", "heavy"]

st.set_page_config(page_title="Solar + battery under load-shedding", layout="wide")


# ----------------------------------------------------------------------------- cached setup
@st.cache_resource(show_spinner="Loading weather and computing solar output ...")
def base_simulator(scenario, year):
    return Simulator(load_config(scenario, {"data": {"years": [year]}}))


def available_years():
    found = sorted(int(os.path.basename(p)[8:12]) for p in glob.glob(os.path.join(ROOT, "data", "karachi_*.csv")))
    return found or [2024]


def available_models():
    out = {}
    for p in sorted(glob.glob(os.path.join(ROOT, "models*", "*", "best_model.zip"))):
        run = os.path.basename(os.path.dirname(p))
        out[f"{run} ({os.path.relpath(os.path.dirname(os.path.dirname(p)), ROOT)})"] = p
    return out


def make_policy(label, models):
    if label in models:
        return load_model(models[label], label.split(" ")[0])
    return make_baseline(label)


def run_both(scenario, year, date, level, seed, policy_labels, offpeak, peak, export, intensity):
    base = base_simulator(scenario, year)
    over = {"data": {"years": [year]}, "outages": {"intensity": float(intensity)},
            "tariff": {"export": {"rate": float(export)}}}
    cfg = base.cfg.copy(over)
    t = cfg.to_dict()["tariff"]
    t["import"]["default_rate"] = float(offpeak)
    for p in t["import"]["periods"]:
        if p["name"] == "peak":
            p["rate"] = float(peak)
    cfg = cfg.copy({"tariff": t})
    sim = base.fresh_copy()
    sim.cfg = cfg                       # physics tables are shared; outage intensity comes from cfg
    env = SolarHomeEnv(cfg, sim=sim)
    start = int(np.searchsorted(sim.index, pd.Timestamp(date)))
    ok = sim.valid_starts
    start = int(ok[np.argmin(np.abs(ok - start))])
    ep = dict(start_index=start, level=level, soc=0.5, seed=int(seed))
    models = available_models()
    results = []
    for label in policy_labels:
        pol = make_policy(label, models)
        stats, tr = run_episode(env, pol, ep, record=True)
        tr["time"] = pd.to_datetime(tr["time"])
        tr["hour_index"] = np.arange(len(tr))
        tr["demand"] = tr["essential"] + tr["other"] + tr["cooling"] + tr["flex"]
        tr["unserved"] = tr["unserved_essential"] + tr["unserved_other"]
        tr["action"] = np.where(tr["mode"] == "reserve",
                                "reserve " + (tr["reserve"].fillna(0) * 100).round().astype(int).astype(str) + "%",
                                tr["mode"])
        results.append((label, stats, tr))
    return results, str(sim.index[start].date()), cfg


# ----------------------------------------------------------------------------- charts
def outage_rects(tr):
    off = tr[~tr["grid_on"].astype(bool)]
    return pd.DataFrame({"start": off["time"], "end": off["time"] + pd.Timedelta(hours=1)})


def layered(tr, upto, y_fields, title, y_title, domain_x, y_scale=None, colors=None):
    shown = tr.iloc[: upto + 1]
    long = shown.melt(id_vars=["time"], value_vars=y_fields, var_name="series", value_name="value")
    enc_color = alt.Color("series:N", legend=alt.Legend(orient="top", title=None),
                          scale=alt.Scale(domain=y_fields, range=colors) if colors else alt.Undefined)
    lines = alt.Chart(long).mark_line(interpolate="step-after").encode(
        x=alt.X("time:T", scale=alt.Scale(domain=domain_x), title=None),
        y=alt.Y("value:Q", title=y_title, scale=y_scale or alt.Undefined), color=enc_color,
        tooltip=["time:T", "series:N", alt.Tooltip("value:Q", format=".2f")])
    rects = alt.Chart(outage_rects(shown)).mark_rect(color="#444444", opacity=0.12).encode(x="start:T", x2="end:T")
    return (rects + lines).properties(height=170, title=title)


def actions_chart(tr, upto, domain_x):
    shown = tr.iloc[: upto + 1].copy()
    shown["end"] = shown["time"] + pd.Timedelta(hours=1)
    shown["flex_run"] = np.where(shown["flex"] > 1e-6, "flexible load runs", None)
    shown["ac_label"] = np.where(shown["ac_mode"] == "precool", "pre-cool", None)
    base = alt.Chart(shown).encode(x=alt.X("time:T", scale=alt.Scale(domain=domain_x), title=None), x2="end:T")
    bars = base.mark_rect().encode(y=alt.value(30), y2=alt.value(5),
                                   color=alt.Color("action:N", legend=alt.Legend(orient="top", title=None)),
                                   tooltip=["time:T", "action:N", "ac_mode:N"])
    flex = base.transform_filter("datum.flex_run != null").mark_rect(color="#1a9641").encode(
        y=alt.value(45), y2=alt.value(35))
    pre = base.transform_filter("datum.ac_label != null").mark_rect(color="#1f78b4").encode(
        y=alt.value(60), y2=alt.value(50))
    return (bars + flex + pre).properties(height=70, title="actions (bar: battery; green: flexible load; blue: pre-cool)")


def totals(tr, upto, replacement_cost):
    s = tr.iloc[: upto + 1]
    return {"energy cost (Rs)": float((s["import_cost"] - s["export_credit"]).sum()),
            "battery wear (Rs)": float(s["wear_cost"].sum()),
            "unserved load (kWh)": float(s["unserved"].sum()),
            "discomfort (°C·h)": float(s["comfort_degree_hours"].sum()),
            "grid import (kWh)": float(s["import_kwh"].sum()),
            "battery SoC (%)": float(s["soc"].iloc[-1] * 100)}


# ----------------------------------------------------------------------------- sidebar
st.sidebar.title("Setup")
scenarios = [s for s in available_scenarios()]
scenario = st.sidebar.selectbox("Scenario", scenarios, index=scenarios.index("default") if "default" in scenarios else 0)
years = available_years()
year = st.sidebar.selectbox("Weather year", years, index=len(years) - 1)
date = st.sidebar.date_input("Start date", value=pd.Timestamp(f"{year}-06-10").date(),
                             min_value=pd.Timestamp(f"{year}-01-01").date(),
                             max_value=pd.Timestamp(f"{year}-12-20").date())
level = st.sidebar.selectbox("Load-shedding level", LEVELS, index=2)
seed = st.sidebar.number_input("Random seed", 0, 10_000, 7)
models = available_models()
choices = default_baselines() + list(models)
pol_a = st.sidebar.selectbox("Policy A", choices, index=choices.index("hand_tuned"))
default_b = list(models)[0] if models else "oracle"
pol_b = st.sidebar.selectbox("Policy B", choices, index=choices.index(default_b))
st.sidebar.markdown("---")
st.sidebar.subheader("What if ...")
base_tariff = load_config(scenario).tariff
peak_rate = next((p.rate for p in base_tariff.import_.periods if p.name == "peak"), base_tariff.import_.default_rate)
offpeak = st.sidebar.slider("Import rate, off-peak (Rs/kWh)", 10.0, 100.0, float(base_tariff.import_.default_rate), 1.0)
peak = st.sidebar.slider("Import rate, peak 18-23h (Rs/kWh)", 10.0, 120.0, float(peak_rate), 1.0)
export = st.sidebar.slider("Export rate (Rs/kWh)", 0.0, 40.0, float(base_tariff.export.rate), 0.5)
intensity = st.sidebar.slider("Unscheduled outage intensity (x)", 0.0, 3.0, 1.0, 0.1)
if not base_tariff.verified:
    st.sidebar.caption(f"Tariff '{base_tariff.name}' uses placeholder values.")
run_clicked = st.sidebar.button("Run", type="primary", use_container_width=True)

key = (scenario, year, str(date), level, int(seed), pol_a, pol_b, offpeak, peak, export, intensity)
if run_clicked or "results" not in st.session_state:
    with st.spinner("Simulating both policies ..."):
        res, start_date, cfg = run_both(scenario, year, date, level, seed, [pol_a, pol_b],
                                        offpeak, peak, export, intensity)
    st.session_state.update(results=res, start_date=start_date, key=key, hour=0, playing=False,
                            replacement=cfg.battery.replacement_cost_rs_per_kwh * cfg.battery.capacity_kwh)
results = st.session_state["results"]
n_hours = len(results[0][2])

# ----------------------------------------------------------------------------- header + controls
st.title("Rooftop solar + battery under load-shedding")
stale = st.session_state.get("key") != key
st.caption(f"{scenario} | week from {st.session_state['start_date']} | load-shedding {level} | grey = grid off"
           + ("  —  settings changed: press Run" if stale else ""))
if "pending_hour" in st.session_state:          # set by the play loop on the previous run
    st.session_state["hour"] = st.session_state.pop("pending_hour")
c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 4])
if c1.button("⏮ Restart"):
    st.session_state.update(hour=0, playing=False)
if c2.button("⏸ Pause" if st.session_state["playing"] else "▶ Play"):
    st.session_state["playing"] = not st.session_state["playing"]
if c3.button("+24 h"):
    st.session_state["hour"] = min(st.session_state["hour"] + 24, n_hours - 1)
if c4.button("⏭ End"):
    st.session_state.update(hour=n_hours - 1, playing=False)
speed = c5.select_slider("Speed (hours per frame)", options=[1, 2, 3, 6], value=2)
hour = st.slider("Hour of the week", 0, n_hours - 1, key="hour")

# ----------------------------------------------------------------------------- side-by-side view
t0 = results[0][2]["time"].iloc[0]
domain_x = [t0.isoformat(), (t0 + pd.Timedelta(hours=n_hours)).isoformat()]
st.markdown(f"**{results[0][2]['time'].iloc[hour]:%a %d %b %H:00}**")
cols = st.columns(len(results))
tot = []
for col, (label, stats, tr) in zip(cols, results):
    with col:
        st.subheader(label)
        t = totals(tr, hour, st.session_state["replacement"])
        tot.append(t)
        m = st.columns(3)
        m[0].metric("Energy cost", f"Rs {t['energy cost (Rs)']:,.0f}")
        m[1].metric("Unserved load", f"{t['unserved load (kWh)']:.2f} kWh")
        m[2].metric("Battery wear", f"Rs {t['battery wear (Rs)']:,.0f}")
        m = st.columns(3)
        m[0].metric("Discomfort", f"{t['discomfort (°C·h)']:.1f} °C·h")
        m[1].metric("Grid import", f"{t['grid import (kWh)']:.1f} kWh")
        m[2].metric("Battery", f"{t['battery SoC (%)']:.0f} %")
        st.altair_chart(layered(tr, hour, ["solar", "demand", "import_kwh", "unserved"], "power",
                                "kW", domain_x, colors=["#f1a340", "#000000", "#d7191c", "#e7298a"]),
                        use_container_width=True)
        tr = tr.assign(soc_pct=tr["soc"] * 100)
        st.altair_chart(layered(tr, hour, ["soc_pct"], "battery state of charge", "%", domain_x,
                                y_scale=alt.Scale(domain=[0, 100]), colors=["#2c7bb6"]), use_container_width=True)
        temp_fields = ["temp_out"] + (["room_temp"] if tr["room_temp"].notna().any() else [])
        st.altair_chart(layered(tr, hour, temp_fields, "temperature", "°C", domain_x,
                                y_scale=alt.Scale(zero=False), colors=["#999999", "#e66101"][: len(temp_fields)]),
                        use_container_width=True)
        st.altair_chart(actions_chart(tr, hour, domain_x), use_container_width=True)

st.markdown("#### Running totals")
table = pd.DataFrame(tot, index=[r[0] for r in results]).T
table["difference (A - B)"] = table.iloc[:, 0] - table.iloc[:, 1]
st.dataframe(table.style.format("{:,.2f}"), use_container_width=True)

if st.session_state["playing"]:
    if hour >= n_hours - 1:
        st.session_state["playing"] = False
    else:
        time.sleep(0.15)
        st.session_state["pending_hour"] = min(hour + speed, n_hours - 1)
        st.rerun()
