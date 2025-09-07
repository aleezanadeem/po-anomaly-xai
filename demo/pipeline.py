# -*- coding: utf-8 -*-
rom __future__ import annotations
import json, time, math, random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml

# ------------------ helpers ------------------

def _read_text_utf8(path: Path) -> str:
    """Read text as UTF-8 with BOM fallback to avoid Windows cp1252 decode errors."""
    try:
        return path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return path.read_text(encoding="utf-8-sig")

# ------------------ config ------------------

def load_config(path: str | Path) -> dict:
    """Load YAML config with safer path resolution (works on Streamlit Cloud)."""
    p = Path(path)

    # If relative path, try resolving it
    if not p.is_absolute():
        # try relative to CWD (repo root)
        cwd_try = Path.cwd() / p
        if cwd_try.exists():
            p = cwd_try
        else:
            # try relative to this file (demo/pipeline.py)
            here_try = Path(__file__).parent / p
            if here_try.exists():
                p = here_try

    if not p.exists():
        raise FileNotFoundError(
            f"config not found: {p.resolve()} "
            f"(cwd={Path.cwd()}, __file__={__file__})"
        )

    return yaml.safe_load(_read_text_utf8(p))

# ------------------ data store ------------------

class DataStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self._events = None
        self._cases = None
        self._features = None
        self._labels = None
        self._activity_vocab = None

    @property
    def events(self) -> pd.DataFrame:
        if self._events is None:
            df = pd.read_parquet(self.cfg["data"]["events_parquet"])
            # required cols: case_id, activity, timestamp
            need = {"case_id","activity","timestamp"}
            if not need.issubset(df.columns):
                raise KeyError(f"events parquet must contain {need}")
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
            df = df.sort_values(["timestamp", "case_id"]).reset_index(drop=True)
            self._events = df
        return self._events

    @property
    def cases(self) -> pd.DataFrame:
        if self._cases is None:
            c = pd.read_parquet(self.cfg["data"]["cases_parquet"])
            # ensure datetimes
            for col in ["start_ts","end_ts"]:
                if col in c.columns:
                    c[col] = pd.to_datetime(c[col], utc=True, errors="coerce")
            # for convenience
            if "amount_eur" not in c.columns:
                if "amount" in c.columns:
                    c = c.rename(columns={"amount":"amount_eur"})
                else:
                    c["amount_eur"] = np.nan
            self._cases = c.set_index("case_id", drop=False)
        return self._cases

    @property
    def features(self) -> pd.DataFrame:
        if self._features is None:
            f = pd.read_parquet(self.cfg["data"]["features_parquet"]).set_index("case_id")
            self._features = f.sort_index()
        return self._features

    @property
    def labels(self) -> Optional[pd.DataFrame]:
        if self._labels is None:
            p = Path(self.cfg["data"].get("labels_parquet",""))
            if p.exists():
                self._labels = pd.read_parquet(p)
                if "case_id" in self._labels.columns:
                    self._labels = self._labels.set_index("case_id")
        return self._labels

    @property
    def activity_vocab(self) -> Dict[str,str]:
        if self._activity_vocab is None:
            p = Path(self.cfg["encoders"]["activities"])
            if p.exists():
                self._activity_vocab = json.loads(_read_text_utf8(p))
            else:
                self._activity_vocab = {}
        return self._activity_vocab

# ------------------ rules (online) ------------------

class RuleEngine:
    """
    Minimal online rules:
      - IR before GR
      - Payment before GR
      - Missing CI (decided at case end)
    """
    def __init__(self, activity_vocab: Dict[str,str]):
        self.tag = {}  # full activity -> tag in {"GR","IR","CI","PAY","OTHER"}
        for k, v in activity_vocab.items():
            lab = str(v).upper()
            if "GOODS" in lab and "RECEIPT" in lab: self.tag[k] = "GR"
            elif "INVOICE RECEIPT" in lab or "RECORD INVOICE RECEIPT" in lab: self.tag[k] = "IR"
            elif "CLEAR INVOICE" in lab: self.tag[k] = "CI"
            elif "REMOVE PAYMENT BLOCK" in lab or "PAYMENT" in lab: self.tag[k] = "PAY"
        self._state: Dict[str, Dict] = {}

    def update(self, case_id: str, activity: str, ts: pd.Timestamp) -> List[Tuple[str, pd.Timestamp]]:
        st = self._state.setdefault(case_id, {"seen": set(), "first_ts": {}, "flags": []})
        tag = self.tag.get(activity)
        st["seen"].add(tag or "OTHER")
        if tag and tag not in st["first_ts"]:
            st["first_ts"][tag] = ts

        fired = []
        if "IR" in st["seen"] and "GR" not in st["seen"]:
            if not any(code == "IR_BEFORE_GR" for code,_ in st["flags"]):
                fired.append(("IR_BEFORE_GR", ts))
        if "PAY" in st["seen"] and "GR" not in st["seen"]:
            if not any(code == "PAY_BEFORE_GR" for code,_ in st["flags"]):
                fired.append(("PAY_BEFORE_GR", ts))

        st["flags"].extend(fired)
        return fired

    def finalize(self, case_id: str, end_ts: pd.Timestamp) -> List[Tuple[str, pd.Timestamp]]:
        st = self._state.get(case_id, {"seen": set(), "flags": []})
        fired = []
        if "CI" not in st["seen"]:
            fired.append(("MISSING_CI", end_ts))
        st["flags"].extend(fired)
        return fired

    def reset_case(self, case_id: str):
        self._state.pop(case_id, None)

# ------------------ scoring ------------------

def _safe_float_array(x):
    x = np.asarray(x, dtype=np.float32)
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x

class ScoreProvider:
    """Try real BVAE first; otherwise fall back to precomputed scores lookup."""
    def __init__(self, cfg: dict, features_df: pd.DataFrame):
        self.cfg = cfg
        self.features_df = features_df
        self._realtime = None
        self._scores_lookup = None
        self._torch = None
        self.feat_names: List[str] = []

        try:
            import torch
            from poetl.model_bvae_44 import BVAE  # training module

            feat_names = json.loads(_read_text_utf8(Path(cfg["model"]["features_used"])))
            scaler = json.loads(_read_text_utf8(Path(cfg["model"]["scaler"])))
            self.mu = _safe_float_array(scaler["mu"])
            self.sd = _safe_float_array(scaler["sd"])
            self.feat_names = list(feat_names)

            d_in = len(self.feat_names)
            model = BVAE(d_in=d_in, bottleneck=16)
            state = torch.load(cfg["model"]["weights"], map_location="cpu")
            model.load_state_dict(state)
            model.eval()
            self._torch = torch
            self._realtime = model
        except Exception:
            self._realtime = None

    def _load_scores_lookup(self):
        if self._scores_lookup is None:
            scores = []
            for k in ["scores_train","scores_val","scores_test"]:
                p = Path(self.cfg["model"].get(k, ""))
                if p.exists():
                    df = pd.read_csv(p)
                    if {"case_id","score"}.issubset(df.columns):
                        scores.append(df[["case_id","score"]])
            if scores:
                lut = pd.concat(scores, axis=0).drop_duplicates("case_id").set_index("case_id")["score"]
            else:
                lut = pd.Series(dtype=float)
            self._scores_lookup = lut
        return self._scores_lookup

    def score_case(self, case_id: str) -> Optional[float]:
        # 1) realtime BVAE if available
        if self._realtime is not None:
            if case_id not in self.features_df.index:
                return None
            x = self.features_df.loc[[case_id]]
            keep = [c for c in self.feat_names if c in x.columns]
            if not keep:
                return None
            x = x[keep].values.astype(np.float32)
            x = (x - self.mu) / self.sd
            x = _safe_float_array(x)

            try:
                with self._torch.no_grad():
                    inp = self._torch.from_numpy(x)
                    out = self._realtime(inp)  # flexible output handling
                    if isinstance(out, (tuple, list)) and len(out) >= 1:
                        recon = out[0]
                    elif isinstance(out, dict):
                        for k in ("recon", "x_recon", "reconstruction", "x_hat"):
                            if k in out:
                                recon = out[k]; break
                        else:
                            recon = next(iter(out.values()))
                    else:
                        recon = out

                    recon_np = recon.detach().cpu().numpy() if hasattr(recon, "detach") else np.asarray(recon)
                    err = (recon_np - x) ** 2
                    return float(np.mean(err))
            except Exception:
                pass

        # 2) fallback: precomputed score lookup
        lut = self._load_scores_lookup()
        if case_id in lut.index:
            return float(lut.loc[case_id])
        return None

# ------------------ stream simulator ------------------

class StreamSimulator:
    def __init__(self, cfg: dict, ds: DataStore, scorer: ScoreProvider, rule_engine: RuleEngine):
        self.cfg = cfg
        self.ds = ds
        self.scorer = scorer
        self.rules = rule_engine
        self.events = ds.events
        self.cases = ds.cases
        self.i = 0
        self.now = self.events["timestamp"].min()
        self.active_cases: Dict[str, pd.Timestamp] = {}
        self.done_cases: set[str] = set()
        self.flag_log: List[dict] = []
        self.thr = float(cfg["runtime"]["anomaly_threshold"])

    def step(self, n_events: int = 500):
        end = min(self.i + int(n_events), len(self.events))
        chunk = self.events.iloc[self.i:end]
        for _, r in chunk.iterrows():
            cid = str(r["case_id"]); act = str(r["activity"]); ts = pd.to_datetime(r["timestamp"], utc=True)
            self.now = ts
            self.active_cases[cid] = ts
            fired = self.rules.update(cid, act, ts)
            for code, t in fired:
                self.flag_log.append({
                    "time": t.isoformat(),
                    "case_id": cid,
                    "kind": "RULE",
                    "rule": code,
                    "score": np.nan,
                })
            if cid in self.cases.index:
                end_ts = self.cases.loc[cid, "end_ts"]
                if pd.notna(end_ts) and ts >= end_ts and cid not in self.done_cases:
                    post = self.rules.finalize(cid, end_ts)
                    for code, t in post:
                        self.flag_log.append({
                            "time": t.isoformat(),
                            "case_id": cid,
                            "kind": "RULE",
                            "rule": code,
                            "score": np.nan,
                        })
                    s = self.scorer.score_case(cid)
                    if s is not None and s >= self.thr:
                        self.flag_log.append({
                            "time": ts.isoformat(),
                            "case_id": cid,
                            "kind": "ANOMALY",
                            "rule": "",
                            "score": float(s),
                        })
                    self.done_cases.add(cid)
                    self.rules.reset_case(cid)
                    self.active_cases.pop(cid, None)
        self.i = end

    def stats(self) -> dict:
        window = pd.Timedelta(minutes=float(self.cfg["runtime"]["window_minutes"]))
        cutoff = self.now - window
        inflight = {k:v for k,v in self.active_cases.items() if v >= cutoff}
        log_df = pd.DataFrame(self.flag_log) if self.flag_log else pd.DataFrame(columns=["time","case_id","kind","rule","score"])
        if not log_df.empty:
            log_df["time"] = pd.to_datetime(log_df["time"], utc=True)
            recent = log_df[log_df["time"] >= cutoff]
        else:
            recent = log_df
        return {
            "events_processed": int(self.i),
            "active_cases": int(len(inflight)),
            "recent_flags": int(len(recent)),
            "now": self.now,
            "flags_df": log_df.sort_values("time", ascending=False).head(500)
        }

