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
                                 value=int(cfg["runtime"]["stream_step_events"]), step=100)
thr = st.sidebar.slider("Anomaly threshold (β-VAE score)", 0.0, 5.0,
                        float(cfg["runtime"]["anomaly_threshold"]), 0.01)
sim.thr = float(thr)

auto = st.sidebar.toggle("Auto-step")
interval = st.sidebar.number_input("Auto-step interval (sec)", min_value=1, value=15)  # default 15s

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

# 1) Heading
st.title("⚡ Real-time P2P Compliance & Anomaly Detection (Simulation)")
# 2) Byline
st.write("_by Aleeza Nadeem_")

# 3) Quick guide (no checkbox)
with st.expander("👋 Quick guide (click to collapse)", expanded=True):
    st.markdown(
        """
**How to use**
1. In the left sidebar, choose **Events per step**.
2. Click **▶ Step** to stream the next batch of events **or** toggle **Auto-step** to stream automatically every *N* seconds.
3. New violations/anomalies show as **alerts** in the corner and also appear in **Newest flags** below.
4. The **Alert log** keeps a persistent list of all alerts (downloadable as CSV).

**What you’ll see**
- **RULE** flags appear as soon as an order-of-activities violation is detectable.
- **ANOMALY** flags appear when a case completes and its β-VAE score ≥ threshold.
        """
    )

# 4) Compliance rules
with st.expander("📜 Compliance rules monitored (click to expand)"):
    st.markdown(
        """
**Rule set**
- **IR_BEFORE_GR** — *Invoice Receipt occurs before Goods Receipt.* Expected flow is GR → IR.
- **PAY_BEFORE_GR** — *Payment occurs before Goods Receipt.* Payment should not precede GR.
- **MISSING_CI** — *Case completed without “Clear Invoice”.* Indicates incomplete financial clearing.

These rules are evaluated as soon as sufficient evidence appears in the event stream; **MISSING_CI** is assessed only when the case has completed.
        """
    )

# --------- KPIs ---------
stats = sim.stats()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Events processed", f"{stats['events_processed']:,}")
k2.metric("Active cases (last window)", f"{stats['active_cases']:,}")
k3.metric("Flags in last window", f"{stats['recent_flags']:,}")
k4.metric("Clock", stats["now"].strftime("%Y-%m-%d %H:%M:%S UTC") if pd.notna(stats["now"]) else "-")

# --------- ALERTS: toast + persistent log ---------
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

# --------- flags table (rename column to 'Anomaly Score') ---------
st.subheader("Newest flags (rules + model)")
if not flags_df.empty:
    # enrich with amount & end_activity if available
    enrich_cols = []
    if "amount_eur" in ds.cases.columns:
        enrich_cols.append("amount_eur")
    if "end_activity" in ds.cases.columns:
        enrich_cols.append("end_activity")

    if enrich_cols:
        flags_df = flags_df.merge(
            ds.cases[enrich_cols],  # right index = case_id
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

    col_config = {}
    try:
        col_config["end_activity"] = st.column_config.TextColumn(
            "end_activity", help="Case end activity", width="large"
        )
        if "Anomaly Score" in display_df.columns:
            col_config["Anomaly Score"] = st.column_config.TextColumn(
                "Anomaly Score", help="Anomaly score (rounded)", width="small"
            )
    except Exception:
        pass

    st.dataframe(
        display_df,
        height=420,
        use_container_width=True,
        hide_index=True,
        column_config=col_config if col_config else None
    )
else:
    st.info("No flags yet — step the stream.")

# --------- visual insights ---------
st.markdown("### Anomaly insights")

if flags_df.empty or flags_df.dropna(subset=["score"]).empty:
    st.info("No anomaly scores yet — step the stream or enable Auto-step.")
else:
    adf = flags_df.dropna(subset=["score"]).copy()
    adf["time"] = pd.to_datetime(adf["time"], utc=True)
    adf["score"] = adf["score"].astype(float)

    # Row 1: Flags by type (keep)
    st.caption("Flags by type (last 500)")
    last = flags_df.head(500)
    bar = (
        alt.Chart(last)
        .mark_bar()
        .encode(
            x=alt.X("kind:N", title="Flag type"),
            y=alt.Y("count():Q", title="Count")
        )
        .properties(height=260)
    )
    st.altair_chart(bar, use_container_width=True)

    st.markdown("---")

    # Row 2: Anomalies per day trend (show YEAR too)
    st.caption("Anomalies per day (last 60 days)")
    daily = (
        adf[adf["kind"] == "ANOMALY"]
        .set_index("time")
        .sort_index()
        .resample("1D")
        .size()
        .rename("count")
        .reset_index()
        .tail(60)
    )
    line = (
        alt.Chart(daily)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "time:T",
                title="Date",
                axis=alt.Axis(format="%b %d, %Y")   # e.g., Jan 05, 2018
            ),
            y=alt.Y("count:Q", title="Anomalies"),
            tooltip=[alt.Tooltip("time:T", title="Date", format="%Y-%m-%d"),
                     alt.Tooltip("count:Q", title="Anomalies")]
        )
        .properties(height=240)
        .interactive()
    )
    st.altair_chart(line, use_container_width=True)

    st.markdown("---")

    # Row 3: Severity mix + Score by end activity (full labels)
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
            .encode(
                theta=alt.Theta("count():Q", title="Cases"),
                color=alt.Color("severity:N", title="Severity"),
                tooltip=[alt.Tooltip("severity:N"), alt.Tooltip("count():Q", title="Cases")]
            )
            .properties(height=260)
        )
        st.altair_chart(donut, use_container_width=True)

    with c2:
        st.caption("Score distribution by end activity (top categories)")
        if "end_activity" in adf.columns and adf["end_activity"].notna().any():
            # Choose top N categories present in the data being plotted
            N = 12
            adf["end_activity"] = adf["end_activity"].astype(str)
            cats = adf["end_activity"].value_counts().head(N).index.tolist()
            bdf = adf[adf["end_activity"].isin(cats)].copy()

            # Order categories by median score (most severe first)
            order = (
                bdf.groupby("end_activity")["score"]
                .median()
                .sort_values(ascending=False)
                .index.tolist()
            )

            # Dynamic height so every label fits; disable label pruning
            chart_height = max(34 * len(order) + 80, 220)

            box = (
                alt.Chart(bdf)
                .mark_boxplot(outliers=True)
                .encode(
                    x=alt.X("score:Q", title="Anomaly score"),
                    y=alt.Y(
                        "end_activity:N",
                        title="End activity",
                        sort=order,
                        axis=alt.Axis(
                            labelOverlap=False,  # don't drop labels
                            labelLimit=0,        # don't truncate
                            labelAngle=0,        # horizontal labels
                            labelPadding=6
                        ),
                    ),
                    tooltip=[
                        alt.Tooltip("end_activity:N", title="End activity"),
                        alt.Tooltip("score:Q", title="Score", format=".3f")
                    ],
                )
                .properties(height=chart_height)
            )
            st.altair_chart(box, use_container_width=True)

            # (Optional) small helper text so viewers understand if few categories appear
            st.caption(f"Showing {len(order)} end activities present in the current anomaly window.")
        else:
            st.info("No `end_activity` available to segment anomaly scores.")
            
st.markdown("---")

# --------- Persistent Alert Log section ---------
st.subheader("Alert log (persistent)")
log_list = st.session_state.get("alert_log", [])
if log_list:
    log_df = pd.DataFrame(log_list)  # newest first
    disp = log_df.copy()
    disp.rename(columns={"score": "Anomaly Score"}, inplace=True)
    disp["Anomaly Score"] = disp["Anomaly Score"].map(lambda v: "" if v is None else f"{v:.3f}")
    st.dataframe(disp, height=260, use_container_width=True, hide_index=True)
    col1, col2 = st.columns([1,1])
    if col1.button("Clear alert log"):
        st.session_state["alert_log"] = []
        try:
            st.rerun()
        except Exception:
            pass
    if col2.download_button("Download alert log (CSV)",
                            data=log_df.to_csv(index=False).encode("utf-8"),
                            file_name="alert_log.csv",
                            mime="text/csv"):
        pass
else:
    st.info("No alerts captured yet. Alerts shown in the corner are also archived here.")

st.caption(
    "Online **RULE** flags appear as soon as a violation is detectable; "
    "**ANOMALY** flags are raised when a case completes (β-VAE score ≥ threshold). "
    "Use Auto-step for a live feed. Alerts are transient; the Alert log persists them."
)

# --------- Auto-step: periodic tick loop ---------
if auto:
    now = time.time()
    last_tick = st.session_state.get("last_tick", 0.0)
    elapsed = now - last_tick
    if elapsed >= float(interval):
        sim.step(int(step_n))
        st.session_state["last_tick"] = time.time()
        try:
            st.rerun()
        except Exception:
            pass
    else:
        time.sleep(max(0.0, float(interval) - elapsed))
        try:
            st.rerun()
        except Exception:
            pass

