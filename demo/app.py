# -*- coding: utf-8 -*-
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from pipeline import (
    load_config, DataStore, RuleEngine, ScoreProvider, StreamSimulator, _read_text_utf8
)

st.set_page_config(page_title="P2P Real-time Compliance & Anomaly Dashboard",
                   page_icon="⚡", layout="wide")

# --------- sidebar: config + controls ---------
cfg_path = st.sidebar.text_input("Config path", "config.yaml")
if "cfg" not in st.session_state or st.session_state.get("cfg_path") != cfg_path:
    st.session_state.cfg = load_config(cfg_path)
    st.session_state.cfg_path = cfg_path
    st.session_state.ds = DataStore(st.session_state.cfg)
    st.session_state.rules = RuleEngine(st.session_state.ds.activity_vocab)
    st.session_state.scorer = ScoreProvider(st.session_state.cfg, st.session_state.ds.features)
    st.session_state.sim = StreamSimulator(st.session_state.cfg,
                                           st.session_state.ds,
                                           st.session_state.scorer,
                                           st.session_state.rules)
    # reset state when config changes
    st.session_state["last_flags_len"] = 0
    st.session_state["last_tick"] = 0.0
    st.session_state["alert_log"] = []  # persistent alerts list

cfg = st.session_state.cfg
sim = st.session_state.sim
ds = st.session_state.ds

st.sidebar.subheader("Streaming")
step_n = st.sidebar.number_input("Events per step", min_value=100, max_value=5000,
                                 value=700, step=100)   # default changed to 700
thr = st.sidebar.slider("Anomaly threshold (β-VAE score)", 0.0, 5.0,
                        float(cfg["runtime"]["anomaly_threshold"]), 0.01)
sim.thr = float(thr)

auto = st.sidebar.toggle("Auto-step")
interval = st.sidebar.number_input("Auto-step interval (sec)", min_value=1, value=15)

colA, colB = st.sidebar.columns(2)
if colA.button("▶ Step"):
    sim.step(int(step_n))
if colB.button("⏭ Step x5"):
    for _ in range(5):
        sim.step(int(step_n))

st.sidebar.markdown("---")
st.sidebar.subheader("Model")
st.sidebar.write(f"Kind: `{cfg['model']['kind']}`")
feat_names = json.loads(_read_text_utf8(Path(cfg["model"]["features_used"])))
st.sidebar.caption(f"Features used (n={len(feat_names)}).")
st.sidebar.code(", ".join(feat_names[:20]) + (" ..." if len(feat_names) > 20 else ""))

# ========= MAIN CONTENT ORDER =========

st.title("⚡ Real-time P2P Compliance & Anomaly Detection (Simulation)")
st.write("_by Aleeza Nadeem_")

with st.expander("👋 Quick guide (click to collapse)", expanded=True):
    st.markdown(
        """
**How to use**
1. In the left sidebar, choose **Events per step** (default 700).
2. Click **▶ Step** to stream events **or** toggle **Auto-step**.
3. New violations/anomalies appear as **alerts** and in **Newest flags**.
4. The **Alert log** keeps a persistent list of alerts (downloadable as CSV).
        """
    )

with st.expander("📜 Compliance rules monitored (click to expand)"):
    st.markdown(
        """
**Rule set**
- **IR_BEFORE_GR** — Invoice Receipt occurs before Goods Receipt.
- **PAY_BEFORE_GR** — Payment occurs before Goods Receipt.
- **MISSING_CI** — Case completed without “Clear Invoice”.
        """
    )

# --------- KPIs ---------
stats = sim.stats()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Events processed", f"{stats['events_processed']:,}")
k2.metric("Active cases (last window)", f"{stats['active_cases']:,}")
k3.metric("Flags in last window", f"{stats['recent_flags']:,}")
k4.metric("Clock", stats["now"].strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(stats["now"]) else "-")

# --------- ALERTS ---------
flags_df = stats["flags_df"].copy()
current_flags_len = len(flags_df) if isinstance(flags_df, pd.DataFrame) else 0
last_seen = st.session_state.get("last_flags_len", 0)

def _append_alert(row: pd.Series):
    entry = {
        "time": str(row.get("time", "")),
        "case_id": str(row.get("case_id", "")),
        "kind": str(row.get("kind", "")),
        "rule": "" if pd.isna(row.get("rule", np.nan)) else str(row.get("rule")),
        "score": (None if pd.isna(row.get("score", np.nan)) else float(row.get("score"))),
    }
    st.session_state["alert_log"].insert(0, entry)
    if len(st.session_state["alert_log"]) > 2000:
        st.session_state["alert_log"] = st.session_state["alert_log"][:2000]

if current_flags_len > last_seen and not flags_df.empty:
    new_rows = flags_df.head(current_flags_len - last_seen)
    for _, r in new_rows.head(20).iterrows():
        if r["kind"] == "ANOMALY" and pd.notna(r.get("score", np.nan)):
            try:
                st.toast(f"🧪 Anomaly on case {r['case_id']} (score={float(r['score']):.3f})", icon="🔥")
            except Exception:
                st.toast(f"🧪 Anomaly on case {r['case_id']}", icon="🔥")
        else:
            rr = r.get("rule", "")
            rr = rr if isinstance(rr, str) and rr else "RULE"
            st.toast(f"⚠️ {rr} on case {r['case_id']}", icon="⚠️")
        _append_alert(r)
st.session_state["last_flags_len"] = current_flags_len

# --------- Flags Table ---------
st.subheader("Newest flags (rules + model)")
if not flags_df.empty:
    enrich_cols = []
    if "amount_eur" in ds.cases.columns: enrich_cols.append("amount_eur")
    if "end_activity" in ds.cases.columns: enrich_cols.append("end_activity")

    if enrich_cols:
        flags_df = flags_df.merge(
            ds.cases[enrich_cols],
            left_on="case_id",
            right_index=True,
            how="left"
        )

    flags_df["time"] = pd.to_datetime(flags_df["time"], utc=True)
    flags_df = flags_df.sort_values("time", ascending=False)

    display_df = flags_df.copy()
    if "score" in display_df.columns:
        display_df.rename(columns={"score": "Anomaly Score"}, inplace=True)
        display_df["Anomaly Score"] = display_df["Anomaly Score"].map(lambda v: "" if pd.isna(v) else f"{float(v):.3f}")

    for col in ["rule", "amount_eur", "end_activity"]:
        if col in display_df.columns:
            display_df[col] = display_df[col].where(display_df[col].notna(), "")

    cols = ["time", "case_id", "kind", "rule"]
    if "Anomaly Score" in display_df.columns: cols.append("Anomaly Score")
    if "amount_eur" in display_df.columns: cols.append("amount_eur")
    if "end_activity" in display_df.columns: cols.append("end_activity")
    display_df = display_df[cols]

    st.dataframe(display_df, height=420, use_container_width=True, hide_index=True)
else:
    st.info("No flags yet — step the stream.")

# --------- Anomaly Insights ---------
st.markdown("### Anomaly insights")

if flags_df.empty:
    st.info("No flags yet — step the stream.")
else:
    adf = flags_df.copy()
    if "score" not in adf.columns or adf["score"].dropna().empty:
        st.warning("No anomaly scores available (showing empty plots).")
        adf["score"] = np.nan

    adf["time"] = pd.to_datetime(adf["time"], utc=True, errors="coerce")

    # Row 1: Flags by type
    st.caption("Flags by type (last 500)")
    last = flags_df.head(500)
    bar = (
        alt.Chart(last)
        .mark_bar()
        .encode(x=alt.X("kind:N", title="Flag type"),
                y=alt.Y("count():Q", title="Count"))
        .properties(height=260)
    )
    st.altair_chart(bar, use_container_width=True)

    st.markdown("---")

    # Row 2: Anomalies per day
    st.caption("Anomalies per day (last 60 days)")
    daily = (
        adf[adf["kind"] == "ANOMALY"]
        .set_index("time").sort_index()
        .resample("1D").size()
        .rename("count").reset_index().tail(60)
    )
    line = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X("time:T", title="Date", axis=alt.Axis(format="%b %d, %Y")),
            y=alt.Y("count:Q", title="Anomalies"),
            tooltip=[alt.Tooltip("time:T", title="Date", format="%Y-%m-%d"),
                     alt.Tooltip("count:Q", title="Anomalies")]
        )
        .properties(height=240)
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

    st.markdown("---")

    # Row 3: Severity mix + End activity
    c1, c2 = st.columns(2)
    with c1:
        st.caption("Severity mix (by anomaly score)")
        if adf["score"].dropna().empty:
            st.info("No scores to show severity distribution.")
        else:
            q50 = float(adf["score"].quantile(0.50))
            q75 = float(adf["score"].quantile(0.75))
            q90 = float(adf["score"].quantile(0.90))
            def bucket(s):
                if s < q50: return "Low"
                if s < q75: return "Medium"
                if s < q90: return "High"
                return "Critical"
            sev = adf.assign(severity=adf["score"].map(bucket))
            donut = (
                alt.Chart(sev)
                .mark_arc(innerRadius=50)
                .encode(
                    theta=alt.Theta("count():Q", title="Cases"),
                    color=alt.Color("severity:N", title="Severity")
                )
                .properties(height=260)
            )
            st.altair_chart(donut, use_container_width=True)

    with c2:
        st.caption("Score distribution by end activity")
        if "end_activity" in adf.columns and adf["end_activity"].notna().any() and not adf["score"].dropna().empty:
            bdf = adf[adf["end_activity"].notna()].copy()
            N = 12
            cats = bdf["end_activity"].value_counts().head(N).index.tolist()
            bdf = bdf[bdf["end_activity"].isin(cats)]
            order = (
                bdf.groupby("end_activity")["score"]
                .median().sort_values(ascending=False).index.tolist()
            )
            box = (
                alt.Chart(bdf)
                .mark_boxplot(outliers=True)
                .encode(
                    x=alt.X("score:Q", title="Anomaly score"),
                    y=alt.Y("end_activity:N", sort=order, title="End activity"),
                )
                .properties(height=max(34 * len(order) + 80, 220))
            )
            st.altair_chart(box, use_container_width=True)
        else:
            st.info("No `end_activity` or anomaly scores available for boxplot.")

st.markdown("---")

# --------- Persistent Alert Log ---------
st.subheader("Alert log (persistent)")
log_list = st.session_state.get("alert_log", [])
if log_list:
    log_df = pd.DataFrame(log_list)
    disp = log_df.copy()
    disp.rename(columns={"score": "Anomaly Score"}, inplace=True)
    disp["Anomaly Score"] = disp["Anomaly Score"].map(lambda v: "" if v is None else f"{v:.3f}")
    st.dataframe(disp, height=260, use_container_width=True, hide_index=True)
else:
    st.info("No alerts captured yet.")
