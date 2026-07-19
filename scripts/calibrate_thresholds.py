"""
calibrate_thresholds.py — Full confidence-threshold sweep for the transformer.

Scores a stratified sample of data/processed/test.csv with the fine-tuned
transformer, then sweeps DEPT_CONF_THRESHOLD and PRIORITY_CONF_THRESHOLD
independently across their full range, reporting accuracy/coverage/sample
size at every threshold — not just a single "smallest cutoff that clears X%"
answer. Use the full sweep CSV to justify a specific threshold with numbers,
including intermediate values between the current 0.60 and a stricter 0.81.

Only scores Stage 2a (the transformer) — no Groq/RAG calls, no API cost.
"""
import os, sys
import numpy as np
import pandas as pd
import torch

ROOT = os.path.abspath(os.path.dirname(__file__) + "/..")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(ROOT, "secrets.env"))
HF_TOKEN = os.getenv("HF_TOKEN")
assert HF_TOKEN and HF_TOKEN.startswith("hf_"), "HF_TOKEN missing in secrets.env"

from pipeline.stage2a_transformer import load_transformer, transformer_predict

OUT_DIR = os.path.join(ROOT, "outputs", "evaluation")
os.makedirs(OUT_DIR, exist_ok=True)
PER_TICKET_PATH = os.path.join(OUT_DIR, "threshold_calibration_per_ticket.csv")
SWEEP_PATH       = os.path.join(OUT_DIR, "threshold_calibration_sweep.csv")
SUMMARY_PATH     = os.path.join(OUT_DIR, "threshold_calibration_summary.csv")

PER_DEPT_CAP  = 300
MIN_BUCKET_N  = 100          # a threshold's accuracy is only trustworthy above this
TARGET_ACC    = 0.90         # the reliability bar used earlier for "smallest cutoff that clears X%"
SWEEP_STEP    = 0.01
CANDIDATES    = [0.55, 0.60, 0.65, 0.68, 0.70, 0.72, 0.75, 0.78, 0.80, 0.81, 0.83, 0.85, 0.88, 0.90]

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else
    "cpu"
)
print(f"Device: {device}")


def build_sample():
    df = pd.read_csv(os.path.join(ROOT, "data/processed/test.csv"))
    df["raw_text"] = (df["subject"].fillna("") + " " + df["body"].fillna("")).str.strip()
    sample = pd.concat([
        group.sample(min(len(group), PER_DEPT_CAP), random_state=42)
        for _, group in df.groupby("queue")
    ]).sample(frac=1, random_state=42).reset_index(drop=True)
    return sample


def score(sample, model, tokenizer, queue_encoder, priority_encoder):
    if os.path.exists(PER_TICKET_PATH):
        cached = pd.read_csv(PER_TICKET_PATH)
        if len(cached) == len(sample):
            print(f"Loaded cached per-ticket scores ({len(cached)} rows)")
            return cached
    records = []
    from tqdm import tqdm
    for _, row in tqdm(sample.iterrows(), total=len(sample)):
        r = transformer_predict(
            row["raw_text"], model, tokenizer, queue_encoder, priority_encoder, device
        )
        records.append({
            "true_dept": row["queue"], "pred_dept": r["dept"], "dept_conf": r["dept_conf"],
            "dept_correct": int(r["dept"] == row["queue"]),
            "true_priority": row["priority"], "pred_priority": r["priority"], "priority_conf": r["priority_conf"],
            "priority_correct": int(r["priority"] == row["priority"]),
        })
    out = pd.DataFrame(records)
    out.to_csv(PER_TICKET_PATH, index=False)
    print(f"Saved per-ticket scores → {PER_TICKET_PATH}")
    return out


def sweep(df, conf_col, correct_col, signal_name):
    rows = []
    for t in np.arange(0.30, 0.96, SWEEP_STEP):
        t = round(float(t), 2)
        bucket = df[df[conf_col] >= t]
        n = len(bucket)
        acc = bucket[correct_col].mean() if n > 0 else float("nan")
        coverage = n / len(df)
        rows.append({
            "signal": signal_name, "threshold": t, "n": n,
            "coverage_pct": round(coverage * 100, 1),
            "accuracy_pct": round(acc * 100, 1) if n > 0 else None,
            "trustworthy_sample": n >= MIN_BUCKET_N,
        })
    return pd.DataFrame(rows)


def smallest_cutoff_clearing_bar(sweep_df, target_acc=TARGET_ACC, min_n=MIN_BUCKET_N):
    ok = sweep_df[(sweep_df["trustworthy_sample"]) & (sweep_df["accuracy_pct"] >= target_acc * 100)]
    if ok.empty:
        return None
    return ok.sort_values("threshold").iloc[0]


def print_candidates(sweep_df, signal_name):
    print(f"\n{'='*72}\n{signal_name.upper()} — candidate thresholds\n{'='*72}")
    print(f"{'threshold':>9} {'n':>6} {'coverage%':>10} {'accuracy%':>10}  reliable?")
    for c in CANDIDATES:
        row = sweep_df.iloc[(sweep_df["threshold"] - c).abs().argsort()[:1]]
        if row.empty:
            continue
        r = row.iloc[0]
        flag = "OK" if r["trustworthy_sample"] else f"n<{MIN_BUCKET_N}"
        acc = f"{r['accuracy_pct']:.1f}" if pd.notna(r["accuracy_pct"]) else "  n/a"
        print(f"{r['threshold']:>9.2f} {int(r['n']):>6} {r['coverage_pct']:>9.1f}% {acc:>9}%  {flag}")


def main():
    sample = build_sample()
    print(f"Sample size: {len(sample)} tickets ({PER_DEPT_CAP}/department cap)")

    model, tokenizer, queue_encoder, priority_encoder = load_transformer(
        "Nethra19/multitask-ticket-model", HF_TOKEN, device
    )
    scored = score(sample, model, tokenizer, queue_encoder, priority_encoder)

    dept_sweep = sweep(scored, "dept_conf", "dept_correct", "department")
    prio_sweep = sweep(scored, "priority_conf", "priority_correct", "priority")
    full = pd.concat([dept_sweep, prio_sweep], ignore_index=True)
    full.to_csv(SWEEP_PATH, index=False)
    print(f"\nSaved full sweep ({len(full)} rows) → {SWEEP_PATH}")

    print_candidates(dept_sweep, "department")
    print_candidates(prio_sweep, "priority")

    summary_rows = []
    for name, sw, current in [("department", dept_sweep, 0.85), ("priority", prio_sweep, 0.60)]:
        best = smallest_cutoff_clearing_bar(sw)
        cur_row = sw.iloc[(sw["threshold"] - current).abs().argsort()[:1]].iloc[0]
        summary_rows.append({
            "signal": name,
            "current_threshold": current,
            "current_accuracy_pct": cur_row["accuracy_pct"],
            "current_coverage_pct": cur_row["coverage_pct"],
            "current_n": int(cur_row["n"]),
            "smallest_cutoff_clearing_90pct": best["threshold"] if best is not None else None,
            "accuracy_at_that_cutoff_pct": best["accuracy_pct"] if best is not None else None,
            "coverage_at_that_cutoff_pct": best["coverage_pct"] if best is not None else None,
            "n_at_that_cutoff": int(best["n"]) if best is not None else None,
        })
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(SUMMARY_PATH, index=False)
    print(f"\nSaved summary → {SUMMARY_PATH}")
    print(f"\n{summary.to_string(index=False)}")


if __name__ == "__main__":
    main()
