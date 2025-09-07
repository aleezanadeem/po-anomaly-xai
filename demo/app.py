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
cfg_path = st.sidebar.text_input("Config path", "demo/config.yaml")
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
    st.session_state["last_flags_len"] = 0
    st.session_state["last_tick"] = 0.0
    st.session_state["alert_log"] = []

cfg = st.session_state.cfg
sim = st.session_state.sim
ds = st.session_state.ds

st.sidebar.subheader("Streaming")
step_n = st.sidebar.number_input("Events per step", min_value=100, max_value=5000,
                                 value=700, step=100)
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

# ========= MAIN CONTENT =========

st.title("⚡ Real-time P2P Compliance & Anomaly Detection (Simulation)")
st.write("_by Aleeza Nadeem_")

with st.expander("👋 Quick guide (click to collapse)", expanded=True):
    st.markdown(
        """
**How to use**
1. In the left sidebar, choose **Events per step**.
2. Click **▶ Step** or enable Auto-step.
3. New violations/anomalies show as alerts and also appear in **Newest flags**.
4. The **Alert log** keeps a persistent list of all alerts.
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

# --------- Alerts ---------
flags_df = stats["flags_df"].copy()
current_flags_len = len(flags_df)
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
            st.toast(f"🔥 Anomaly on case {r['case_id']} (score={float(r['score']):.3f})")
        else:
            rr = r.get("rule", "RULE")
            st.toast(f"⚠️ {rr} on case {r['case_id']}")
        _append_alert(r)
st.session_state["last_flags_len"] = current_flags_len

# --------- Flags table ---------
st.subheader("Newest flags (rules + model)")
if not flags_df.empty:
    enrich_cols = [c for c in ["amount_eur", "end_activity"] if c in ds.cases.columns]
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

    cols = ["time", "case_id", "kind", "rule"]
    if "Anomaly Score" in display_df.columns: cols.append("Anomaly Score")
    if "amount_eur" in display_df.columns: cols.append("amount_eur")
    if "end_activity" in display_df.columns: cols.append("end_activity")
    display_df = display_df[cols]

    st.dataframe(display_df, height=420, use_container_width=True, hide_index=True)
else:
    st.info("No flags yet — step the stream.")

# --------- Visual insights ---------
st.markdown("### Anomaly insights")

if not flags_df.empty:
    adf = flags_df.dropna(subset=["score"]).copy()
    if not adf.empty:
        adf["time"] = pd.to_datetime(adf["time"], utc=True)
        adf["score"] = adf["score"].astype(float)

        # Row 1: Flags by type
        st.caption("Flags by type (last 500)")
        last = flags_df.head(500)
        bar = (
            alt.Chart(last)
            .mark_bar()
            .encode(x="kind:N", y="count():Q")
            .properties(height=260)
        )
        st.altair_chart(bar, use_container_width=True)

        st.markdown("---")

        # Row 2: Anomalies per day
        daily = (
            adf[adf["kind"] == "ANOMALY"]
            .set_index("time")
            .resample("1D")
            .size()
            .rename("count")
            .reset_index()
            .tail(60)
        )
        if not daily.empty:
            st.caption("Anomalies per day (last 60 days)")
            line = (
                alt.Chart(daily)
                .mark_line(point=True)
                .encode(x="time:T", y="count:Q", tooltip=["time:T", "count:Q"])
                .properties(height=240)
                .interactive()
            )
            st.altair_chart(line, use_container_width=True)

        st.markdown("---")

        # Row 3: Severity + End activity distribution
        c1, c2 = st.columns(2)
        with c1:
            st.caption("Severity mix (by anomaly score)")
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
                .encode(theta="count():Q", color="severity:N")
                .properties(height=260)
            )
            st.altair_chart(donut, use_container_width=True)

        with c2:
            st.caption("Score distribution by end activity (top categories)")
            if "end_activity" in adf.columns and adf["end_activity"].notna().any():
                N = 20
                adf["end_activity"] = adf["end_activity"].astype(str)
                cats = adf["end_activity"].value_counts().head(N).index.tolist()
                bdf = adf[adf["end_activity"].isin(cats)]
                order = (
                    bdf.groupby("end_activity")["score"]
                    .median()
                    .sort_values(ascending=False)
                    .index.tolist()
                )
                chart_height = max(34 * len(order) + 80, 220)
                box = (
                    alt.Chart(bdf)
                    .mark_boxplot(outliers=True)
                    .encode(
                        x=alt.X("score:Q", title="Anomaly score"),
                        y=alt.Y("end_activity:N", sort=order, title="End activity"),
                        tooltip=["end_activity:N", alt.Tooltip("score:Q", format=".3f")],
                    )
                    .properties(height=chart_height)
                )
                st.altair_chart(box, use_container_width=True)

# --------- Alert log ---------
st.markdown("---")
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

# --------- Auto-step ---------
if auto:
    now = time.time()
    last_tick = st.session_state.get("last_tick", 0.0)
    elapsed = now - last_tick
    if elapsed >= float(interval):
        sim.step(int(step_n))
        st.session_state["last_tick"] = time.time()
        try: st.rerun()
        except Exception: pass
    else:
        time.sleep(max(0.0, float(interval) - elapsed))
        try: st.rerun()
        except Exception: pass

