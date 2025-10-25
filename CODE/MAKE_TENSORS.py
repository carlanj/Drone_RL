#!/usr/bin/env python3
"""
Excel -> TensorBoard (3 runs: one for each Algo)
- Tracks Reward and Duration across changing Labels by building a continuous per-algo index.
- You set XLSX_PATH and OUT_LOGDIR below. No CLI needed.

Install:
  pip install pandas tensorboard tensorboardX
Run:
  python xlsx_to_tensorboard_local.py
  tensorboard --logdir ".\\tb_from_xlsx"
"""

# ====== CONFIG ======
XLSX_PATH   = r"aggresive.xlsx"   # <-- your spreadsheet
OUT_LOGDIR  = r"MADE/aggressive/tb_from_xlsx"            # where TB event logs go
LAUNCH_TB   = True                         # auto-launch TB on localhost
TB_HOST     = "127.0.0.1"
TB_PORT     = 6008                         # change if needed
# ====================

import os, re, time, math, webbrowser
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
def sanitize(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)

def main():
    assert os.path.isfile(XLSX_PATH), f"File not found: {XLSX_PATH}"
    os.makedirs(OUT_LOGDIR, exist_ok=True)

    df = pd.read_excel(XLSX_PATH)
    # Normalize column names
    df.columns = [c.strip() for c in df.columns]
    if "Duration (Secs)" in df.columns:
        df = df.rename(columns={"Duration (Secs)": "Duration_Secs"})
    required = ["Algo", "Label", "Episode", "Steps", "Reward", "Duration_Secs"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Sort stable: Algo -> Label -> Episode
    df = df.sort_values(["Algo", "Label", "Episode"], kind="mergesort").reset_index(drop=True)
    # Build continuous per-Algo sequence: 0..N-1 across all labels for that algo
    df["AlgoSeq"] = df.groupby("Algo").cumcount()

    # One TB run per Algo
    algo_runs = {}
    for algo in df["Algo"].unique():
        run_dir = os.path.join(OUT_LOGDIR, sanitize(algo))
        os.makedirs(run_dir, exist_ok=True)
        algo_runs[algo] = SummaryWriter(log_dir=run_dir)

    # Log scalars
    for algo, sub in df.groupby("Algo"):
        w = algo_runs[algo]
        for _, row in sub.iterrows():
            step = int(row["AlgoSeq"])
            if pd.notna(row["Reward"]):
                w.add_scalar("reward", float(row["Reward"]), global_step=step)
            if pd.notna(row["Duration_Secs"]):
                w.add_scalar("duration_secs", float(row["Duration_Secs"]), global_step=step)

    for w in algo_runs.values():
        w.flush(); w.close()

    print("TensorBoard logs written to:", os.path.abspath(OUT_LOGDIR))
    print("Open with: tensorboard --logdir", os.path.abspath(OUT_LOGDIR))

    if LAUNCH_TB:
        try:
            from tensorboard import program
            tb = program.TensorBoard()
            tb.configure(argv=[
                "tensorboard",
                "--logdir", OUT_LOGDIR,
                "--host", TB_HOST,
                "--port", str(TB_PORT),
                "--reload_interval", "3",
            ])
            url = tb.launch()
            print("TensorBoard:", url)
            try: webbrowser.open_new_tab(url)
            except Exception: pass
            print("Press Ctrl+C to stop.")
            while True:
                time.sleep(3600)
        except Exception as e:
            print("Could not auto-launch TensorBoard:", e)

if __name__ == "__main__":
    main()
