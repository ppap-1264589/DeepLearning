"""Thí nghiệm chống overfit: dropout, weight decay, label smoothing (riêng từng cái và kết hợp).

Notebook chỉ cần:
    !python -m src.run_reg --epochs 15

Năm run, mỗi run chỉ đổi đúng một thứ so với baseline (Adam, lr 1e-3, không scheduler, không sampler):
    base : baseline (dùng lại run của lưới nếu đã có cnn_adam_none_none_e15 cùng checkpoint)
    drop : + dropout 0.5
    wd   : + weight decay 1e-2 (Adam -> AdamW)
    ls   : + label smoothing 0.1
    all  : cả ba

Mỗi run xong sẽ tự chạy src.evaluate (split val) để lấy recall từng lớp yếu.
Run đã xong (log đủ epoch VÀ có checkpoint) được bỏ qua, nên chạy lại sau khi Colab ngắt là an toàn.

    python -m src.run_reg --epochs 15                    # 1 seed
    python -m src.run_reg --epochs 15 --seeds 42 43 44   # nhiều seed, bảng có mean ± std
    python -m src.run_reg --epochs 15 --report-only      # chỉ tổng hợp từ log có sẵn
"""
import argparse
import json
import subprocess
import sys

import numpy as np
import pandas as pd

from src.config import ROOT, SEED
from src.plot_history import plot_compare

BASE = ["--optimizer", "adam", "--lr", "1e-3", "--scheduler", "none", "--balance", "none"]
RUNS = {
    "base": [],
    "drop": ["--dropout", "0.5"],
    "wd":   ["--weight-decay", "1e-2"],
    "ls":   ["--label-smoothing", "0.1"],
    "all":  ["--dropout", "0.5", "--weight-decay", "1e-2", "--label-smoothing", "0.1"],
}
LABEL = {"base": "A: baseline", "drop": "B: dropout 0.5", "wd": "C: AdamW wd 1e-2",
         "ls": "D: label smoothing 0.1", "all": "E: B+C+D"}
WEAK = [23, 30, 0]          # các lớp yếu của baseline (xem confusion matrix)
NOISE = 0.01                # chênh lệch dưới 1% macro-F1 trên val coi là chưa đủ bằng chứng

LOG_DIR = ROOT / "outputs" / "logs"
CKPT_DIR = ROOT / "outputs" / "checkpoints"
REP_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"


def run_name(tag, epochs, seed):
    mid = "" if tag == "base" else f"_{tag}"
    name = f"cnn_adam_none_none{mid}_e{epochs}"       # base trùng tên với run của lưới thí nghiệm
    return name if seed == SEED else f"{name}_s{seed}"


def is_done(name, epochs):
    log = LOG_DIR / f"{name}.json"
    if not log.exists() or not (CKPT_DIR / f"{name}.pt").exists():
        return False
    with open(log, encoding="utf-8") as f:
        return len(json.load(f)["epoch"]) == epochs


def sh(args):
    print("\n>>>", " ".join(args))
    subprocess.run(args, cwd=ROOT, check=True)


def train(name, flags, epochs, seed):
    cmd = [sys.executable, "-m", "src.train", "--epochs", str(epochs), *BASE, *flags, "--name", name]
    if seed != SEED:
        cmd += ["--seed", str(seed)]
    sh(cmd)


def evaluate_once(name):
    if (REP_DIR / f"{name}_val_per_class.csv").exists():
        print(f"[bỏ qua evaluate] {name}")
        return
    sh([sys.executable, "-m", "src.evaluate", "--name", name])


def summarize(name, tag, seed):
    with open(LOG_DIR / f"{name}.json", encoding="utf-8") as f:
        h = json.load(f)
    f1, acc = np.array(h["val_f1"]), np.array(h["val_acc"])
    row = dict(tag=tag, seed=seed, name=name,
               f1_last3=f1[-3:].mean(), acc_last3=acc[-3:].mean(),     # TB 3 epoch cuối: bớt nhiễu 1 epoch
               val_loss=h["val_loss"][-1],
               loss_gap=h["val_loss"][-1] - h["train_loss"][-1],       # val - train (cross-entropy thường)
               acc_gap=h["train_acc"][-1] - acc[-1])
    pc_file = REP_DIR / f"{name}_val_per_class.csv"
    if pc_file.exists():
        pc = pd.read_csv(pc_file).set_index("class")
        for c in WEAK:
            row[f"rec{c}"] = pc.loc[c, "recall"]
    return row


def fmt(m, s, n, scale=100.0, d=2):
    return f"{m * scale:.{d}f} ± {s * scale:.{d}f}" if n > 1 else f"{m * scale:.{d}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=15)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()
    epochs, seeds = args.epochs, args.seeds

    rows = []
    for seed in seeds:
        for tag, flags in RUNS.items():
            name = run_name(tag, epochs, seed)
            if not args.report_only:
                if is_done(name, epochs):
                    print(f"[bỏ qua train] {name} đã xong")
                else:
                    train(name, flags, epochs, seed)
                evaluate_once(name)
            if is_done(name, epochs):
                rows.append(summarize(name, tag, seed))

    df = pd.DataFrame(rows)
    if df.empty or "base" not in set(df.tag):
        sys.exit("Chưa có baseline đủ epoch để so sánh.")

    metrics = ["f1_last3", "acc_last3", "val_loss", "loss_gap", "acc_gap"] + \
              [f"rec{c}" for c in WEAK if f"rec{c}" in df]
    g = df.groupby("tag")
    mean, std, cnt = g[metrics].mean(), g[metrics].std().fillna(0.0), g.size()
    order = [t for t in RUNS if t in mean.index]

    head = ["Run", "n", "Macro-F1 % (TB 3 ep cuối)", "Acc % (TB 3 ep cuối)", "Val loss cuối",
            "Val-train loss", "Train-val acc %"] + [f"Recall lớp {c} %" for c in WEAK if f"rec{c}" in df]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for t in order:
        n = int(cnt[t])
        cells = [LABEL[t], str(n),
                 fmt(mean.loc[t, "f1_last3"], std.loc[t, "f1_last3"], n),
                 fmt(mean.loc[t, "acc_last3"], std.loc[t, "acc_last3"], n),
                 fmt(mean.loc[t, "val_loss"], std.loc[t, "val_loss"], n, 1, 3),
                 fmt(mean.loc[t, "loss_gap"], std.loc[t, "loss_gap"], n, 1, 3),
                 fmt(mean.loc[t, "acc_gap"], std.loc[t, "acc_gap"], n)]
        cells += [fmt(mean.loc[t, f"rec{c}"], std.loc[t, f"rec{c}"], n) for c in WEAK if f"rec{c}" in df]
        lines.append("| " + " | ".join(cells) + " |")
    table = "\n".join(lines)

    notes = ["## So với baseline (macro-F1, TB 3 epoch cuối)"]
    b = mean.loc["base", "f1_last3"]
    for t in order[1:]:
        d = mean.loc[t, "f1_last3"] - b
        verdict = "trong ngưỡng nhiễu, chưa kết luận được" if abs(d) < NOISE else ("tốt hơn" if d > 0 else "kém hơn")
        notes.append(f"- {LABEL[t]}: {d * 100:+.2f}% -> {verdict}")
    notes.append("")
    notes.append("Chú thích: loss ghi log là cross-entropy thường (không smoothing) nên so được giữa các run. "
                 "Val-train loss và Train-val acc càng nhỏ thì overfit càng ít. "
                 "Recall từng lớp lấy từ checkpoint tốt nhất (val acc cao nhất), còn các cột còn lại tính ở cuối huấn luyện.")

    REP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(REP_DIR / "reg_runs.csv", index=False)
    (REP_DIR / "reg_summary.md").write_text(
        "\n".join([f"# Kết quả điều chuẩn hóa ({epochs} epoch, seed {seeds}, tập val)", "", table, ""] + notes),
        encoding="utf-8")

    hist = []
    for t in order:
        with open(LOG_DIR / f"{run_name(t, epochs, seeds[0])}.json", encoding="utf-8") as f:
            hist.append(json.load(f))
    plot_compare(hist, FIG_DIR / "reg_compare.png")

    print("\n" + table + "\n\n" + "\n".join(notes))
    print("\nĐã lưu:", REP_DIR / "reg_summary.md")
    print("Đã lưu:", REP_DIR / "reg_runs.csv")
    print("Đã lưu:", FIG_DIR / "reg_compare.png")


if __name__ == "__main__":
    main()