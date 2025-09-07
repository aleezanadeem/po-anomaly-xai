import os
import shutil
from pathlib import Path
import pandas as pd
import yaml

# -------- SETTINGS --------
SRC_DATA = Path("data/processed")
SRC_ARTIFACTS = Path("artifacts")

DEMO_ROOT = Path("demo")
DEMO_DATA = DEMO_ROOT / "data" / "processed"
DEMO_ARTIFACTS = DEMO_ROOT / "artifacts"

# how many rows to keep for demo subset
EVENTS_N = 500
CASES_N = 50
FEATURES_N = 50
LABELS_N = 50

# -------- STEP 1: Create folder structure --------
(DEMO_DATA / "features").mkdir(parents=True, exist_ok=True)
(DEMO_DATA / "labels").mkdir(parents=True, exist_ok=True)
(DEMO_ARTIFACTS / "models" / "bvae").mkdir(parents=True, exist_ok=True)
(DEMO_ARTIFACTS / "encoders").mkdir(parents=True, exist_ok=True)

# -------- STEP 2: Downsample Parquet files --------
def downsample_parquet(src, dst, n):
    if Path(src).exists():
        df = pd.read_parquet(src)
        df_demo = df.head(n)
        df_demo.to_parquet(dst, index=False)
        print(f"✅ {dst} written with {len(df_demo)} rows")
    else:
        print(f"⚠️ {src} not found, skipping")

downsample_parquet(SRC_DATA / "events_clean.parquet",
                   DEMO_DATA / "events_clean.parquet", EVENTS_N)

downsample_parquet(SRC_DATA / "cases_clean.parquet",
                   DEMO_DATA / "cases_clean.parquet", CASES_N)

downsample_parquet(SRC_DATA / "features" / "case_features.parquet",
                   DEMO_DATA / "features" / "case_features.parquet", FEATURES_N)

downsample_parquet(SRC_DATA / "labels" / "compliance_labels.parquet",
                   DEMO_DATA / "labels" / "compliance_labels.parquet", LABELS_N)

# -------- STEP 3: Copy artifacts (small JSON/text files only) --------
files_to_copy = [
    (SRC_ARTIFACTS / "models" / "bvae" / "scaler.json",
     DEMO_ARTIFACTS / "models" / "bvae" / "scaler.json"),
    (SRC_ARTIFACTS / "models" / "bvae" / "features_used.json",
     DEMO_ARTIFACTS / "models" / "bvae" / "features_used.json"),
    (SRC_ARTIFACTS / "models" / "bvae" / "scores_test.csv",
     DEMO_ARTIFACTS / "models" / "bvae" / "scores_test.csv"),
    (SRC_ARTIFACTS / "models" / "bvae" / "scores_val.csv",
     DEMO_ARTIFACTS / "models" / "bvae" / "scores_val.csv"),
    (SRC_ARTIFACTS / "models" / "bvae" / "scores_train.csv",
     DEMO_ARTIFACTS / "models" / "bvae" / "scores_train.csv"),
    (SRC_ARTIFACTS / "encoders" / "activities.json",
     DEMO_ARTIFACTS / "encoders" / "activities.json"),
]

for src, dst in files_to_copy:
    if src.exists():
        shutil.copy(src, dst)
        print(f"✅ Copied {src} → {dst}")
    else:
        print(f"⚠️ {src} not found, skipping")

# -------- STEP 4: Write demo config.yaml --------
demo_cfg = {
    "data": {
        "events_parquet": str(DEMO_DATA / "events_clean.parquet"),
        "cases_parquet": str(DEMO_DATA / "cases_clean.parquet"),
        "features_parquet": str(DEMO_DATA / "features" / "case_features.parquet"),
        "labels_parquet": str(DEMO_DATA / "labels" / "compliance_labels.parquet"),
    },
    "model": {
        "kind": "bvae",
        "weights": str(DEMO_ARTIFACTS / "models" / "bvae" / "bvae.pt"),
        "scaler": str(DEMO_ARTIFACTS / "models" / "bvae" / "scaler.json"),
        "features_used": str(DEMO_ARTIFACTS / "models" / "bvae" / "features_used.json"),
        "scores_test": str(DEMO_ARTIFACTS / "models" / "bvae" / "scores_test.csv"),
        "scores_val": str(DEMO_ARTIFACTS / "models" / "bvae" / "scores_val.csv"),
        "scores_train": str(DEMO_ARTIFACTS / "models" / "bvae" / "scores_train.csv"),
    },
    "encoders": {
        "activities": str(DEMO_ARTIFACTS / "encoders" / "activities.json"),
    },
    "runtime": {
        "anomaly_threshold": 0.85,
        "stream_step_events": 200,
        "window_minutes": 120,
        "show_ground_truth": True,
    },
}

with open(DEMO_ROOT / "config.yaml", "w") as f:
    yaml.dump(demo_cfg, f, sort_keys=False)
    print(f"✅ Demo config.yaml written at {DEMO_ROOT / 'config.yaml'}")

print("\n🎉 Demo folder created successfully!")
