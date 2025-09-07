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

    def step(self, n_events: int = None):
        """
        Process the next chunk of events.
        - If n_events is None or > remaining, process ALL remaining events.
        """
        if n_events is None:
            end = len(self.events)
        else:
            end = min(self.i + int(n_events), len(self.events))

        if self.i >= len(self.events):
            # nothing left
            return

        chunk = self.events.iloc[self.i:end]

        for _, r in chunk.iterrows():
            cid = str(r["case_id"])
            act = str(r["activity"])
            ts = pd.to_datetime(r["timestamp"], utc=True)
            self.now = ts
            self.active_cases[cid] = ts

            # Apply online rule updates
            fired = self.rules.update(cid, act, ts)
            for code, t in fired:
                self.flag_log.append({
                    "time": t.isoformat(),
                    "case_id": cid,
                    "kind": "RULE",
                    "rule": code,
                    "score": np.nan,
                })

            # If case ends, finalize rules + anomaly score
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
        inflight = {k: v for k, v in self.active_cases.items() if v >= cutoff}
        log_df = pd.DataFrame(self.flag_log) if self.flag_log else pd.DataFrame(
            columns=["time", "case_id", "kind", "rule", "score"]
        )
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
            "flags_df": log_df.sort_values("time", ascending=False).head(500),
        }
