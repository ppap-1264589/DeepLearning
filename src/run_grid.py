"""Lưới thí nghiệm chọn tham số cho SimpleCNN và bảng so sánh.

Notebook chỉ cần một dòng:
    !python -m src.run_grid --epochs 30 --backup /content/drive/MyDrive/DeepLearning_outputs

Quy trình (mỗi bước in lý do chọn; toàn bộ được ghi vào outputs/reports/grid_summary.md):
    1. Lưới optimizer x scheduler (balance=none), cùng số epoch, mỗi optimizer dùng lr riêng.
    2. Nếu --seeds có nhiều giá trị: chạy thêm seed cho top-K cấu hình để giảm nhiễu.
    3. Chọn cấu hình tốt nhất, chạy thêm balance=sampler để so sánh A/B.
    4. Bảng tổng hợp (CSV + Markdown) và biểu đồ so sánh.

Run nào đã xong (log đủ số epoch) sẽ được bỏ qua, nên chạy lại sau khi Colab ngắt là an toàn.

Ví dụ:
    python -m src.run_grid --dry-run                       # chỉ in các lệnh của bước 1
    python -m src.run_grid --epochs 30                     # 1 seed (~7 run)
    python -m src.run_grid --epochs 30 --seeds 42 43 44    # nhiều seed (cần --seed trong train.py)
    python -m src.run_grid --epochs 30 --report-only       # chỉ tổng hợp từ log có sẵn
"""
import argparse
import json
import shutil
import subprocess
import sys
from itertools import product

import numpy as np
import pandas as pd

from src.config import ROOT, SEED
from src.plot_history import plot_compare

OPTIMIZERS = ["adam", "sgd"]
SCHEDULERS = ["none", "plateau", "cosine"]
LR = {"adam": 1e-3, "sgd": 1e-2}      # mỗi optimizer một lr hợp lý, để so optimizer chứ không so lr
NOISE = 0.01                          # chênh lệch dưới 1% trên val coi là chưa đủ bằng chứng

LOG_DIR = ROOT / "outputs" / "logs"
REP_DIR = ROOT / "outputs" / "reports"
FIG_DIR = ROOT / "outputs" / "figures"
KEY = ["optimizer", "scheduler", "balance", "lr"]


# ---------------------------------------------------------------- chạy huấn luyện
def run_name(opt, sch, bal, epochs, seed):
    name = f"cnn_{opt}_{sch}_{bal}_e{epochs}"
    return name if seed == SEED else f"{name}_s{seed}"


def is_done(name, epochs):
    path = LOG_DIR / f"{name}.json"
    if not path.exists():
        return False
    with open(path, encoding="utf-8") as f:
        return len(json.load(f)["epoch"]) == epochs


def backup(dst):
    if dst:
        shutil.copytree(ROOT / "outputs", dst, dirs_exist_ok=True)
        print(f"  (đã sao lưu outputs -> {dst})")


def run(opt, sch, bal, epochs, seed, dst, execute=True):
    name = run_name(opt, sch, bal, epochs, seed)
    if is_done(name, epochs):
        print(f"[bỏ qua] {name} đã chạy xong")
        return
    cmd = [sys.executable, "-m", "src.train", "--epochs", str(epochs),
           "--optimizer", opt, "--lr", str(LR[opt]), "--scheduler", sch,
           "--balance", bal, "--name", name]
    if seed != SEED:
        cmd += ["--seed", str(seed)]
    print("\n>>>", " ".join(cmd))
    if not execute:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)
    backup(dst)


# ---------------------------------------------------------------- tổng hợp kết quả
def collect(epochs):
    """Đọc mọi log đã chạy xong với đúng số epoch này. Mỗi dòng = 1 run."""
    rows = []
    for f in sorted(LOG_DIR.glob("cnn_*.json")):
        with open(f, encoding="utf-8") as fh:
            h = json.load(fh)
        c = h["config"]
        if c["epochs"] != epochs or len(h["epoch"]) != epochs:
            continue
        acc, f1 = np.array(h["val_acc"]), np.array(h["val_f1"])
        b = int(acc.argmax())          # epoch mà train.py đã lưu checkpoint
        rows.append(dict(
            name=h["name"], optimizer=c["optimizer"], scheduler=c["scheduler"],
            balance=c["balance"], lr=c["lr"], seed=c.get("seed", SEED),
            best_epoch=b + 1, best_acc=acc[b], f1_at_best=f1[b],
            last_acc=acc[-1], last_f1=f1[-1], last_loss=h["val_loss"][-1],
            gap=h["train_acc"][-1] - acc[-1]))
    return pd.DataFrame(rows)


def aggregate(df):
    """Gộp các seed của cùng một cấu hình: trung bình và độ lệch chuẩn."""
    agg = df.groupby(KEY).agg(
        n=("name", "count"), best_epoch=("best_epoch", "mean"),
        acc=("best_acc", "mean"), acc_std=("best_acc", "std"),
        f1=("f1_at_best", "mean"), f1_std=("f1_at_best", "std"),
        last_acc=("last_acc", "mean"), last_f1=("last_f1", "mean"),
        last_loss=("last_loss", "mean"), gap=("gap", "mean")).reset_index()
    return agg.fillna(0.0).sort_values("f1", ascending=False).reset_index(drop=True)


def pct(mean, std, n):
    return f"{mean * 100:.2f} ± {std * 100:.2f}" if n > 1 else f"{mean * 100:.2f}"


def md_table(agg):
    lines = ["| Optimizer | Scheduler | Balance | n | Best ep | Val acc % | Val macro-F1 % "
             "| Acc cuối % | F1 cuối % | Val loss cuối | Gap train-val % |",
             "|" + "---|" * 11]
    for _, r in agg.iterrows():
        lines.append(
            f"| {r.optimizer} | {r.scheduler} | {r.balance} | {int(r.n)} | {r.best_epoch:.0f} "
            f"| {pct(r.acc, r.acc_std, r.n)} | {pct(r.f1, r.f1_std, r.n)} "
            f"| {r.last_acc * 100:.2f} | {r.last_f1 * 100:.2f} | {r.last_loss:.3f} | {r.gap * 100:.2f} |")
    return "\n".join(lines)


def pick(agg, balance, keys=None):
    sel = agg[agg.balance == balance]
    if keys is not None:
        sel = sel[[(o, s) in keys for o, s in zip(sel.optimizer, sel.scheduler)]]
    return sel.sort_values("f1", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------- chương trình chính
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--seeds", type=int, nargs="+", default=[SEED])
    ap.add_argument("--topk", type=int, default=2, help="số cấu hình tốt nhất được chạy thêm seed")
    ap.add_argument("--backup", default=None, help="thư mục sao lưu outputs sau mỗi run (vd. trên Drive)")
    ap.add_argument("--dry-run", action="store_true", help="chỉ in lệnh của bước 1 rồi thoát")
    ap.add_argument("--report-only", action="store_true", help="không huấn luyện, chỉ tổng hợp log có sẵn")
    args = ap.parse_args()

    epochs, seeds, base = args.epochs, args.seeds, args.seeds[0]
    execute = not (args.dry_run or args.report_only)

    if any(s != SEED for s in seeds):
        train_src = (ROOT / "src" / "train.py").read_text(encoding="utf-8")
        if "--seed" not in train_src:
            sys.exit("train.py chưa có tham số --seed. Thêm vào train.py:\n"
                     "  parser.add_argument('--seed', type=int, default=SEED)\n"
                     "  torch.manual_seed(args.seed)   # thay cho torch.manual_seed(SEED)")

    log = []

    def say(msg=""):
        print(msg)
        log.append(msg)

    # --- Bước 1: lưới optimizer x scheduler
    print(f"=== Bước 1: lưới {len(OPTIMIZERS)}x{len(SCHEDULERS)} | {epochs} epoch | seed {base} ===")
    for opt, sch in product(OPTIMIZERS, SCHEDULERS):
        run(opt, sch, "none", epochs, base, args.backup, execute)
    if args.dry_run:
        print("\n(dry-run) Các bước sau phụ thuộc kết quả bước 1 nên không in được.")
        return

    df = collect(epochs)
    if df.empty:
        sys.exit("Chưa có log nào đủ epoch để tổng hợp.")
    grid = pick(aggregate(df), "none")
    say("## Bước 1: lưới optimizer x scheduler (balance=none)")
    say(f"Cùng {epochs} epoch; Adam lr 1e-3, SGD lr 1e-2. Xếp theo val macro-F1 tại epoch có val acc cao nhất "
        f"(chính là checkpoint đã lưu).")
    for i, r in grid.head(3).iterrows():
        say(f"- Hạng {i + 1}: {r.optimizer}+{r.scheduler}: F1 {pct(r.f1, r.f1_std, r.n)}%, acc {pct(r.acc, r.acc_std, r.n)}%")

    # --- Bước 2: thêm seed cho top-K
    top = grid.head(args.topk)
    top_keys = set(zip(top.optimizer, top.scheduler))
    if len(seeds) > 1:
        for _, r in top.iterrows():
            for s in seeds[1:]:
                run(r.optimizer, r.scheduler, "none", epochs, s, args.backup, execute)
        agg = aggregate(collect(epochs))
        cand = pick(agg, "none", top_keys)
        say(f"\n## Bước 2: chạy thêm seed {seeds[1:]} cho top-{args.topk}")
        for i, r in cand.iterrows():
            say(f"- {r.optimizer}+{r.scheduler} (n={int(r.n)}): F1 {pct(r.f1, r.f1_std, r.n)}%")
    else:
        cand = grid.head(args.topk)
        say("\n## Bước 2: bỏ qua (chỉ 1 seed, dùng --seeds 42 43 44 để giảm nhiễu)")

    winner = cand.iloc[0]
    if len(cand) > 1 and winner.f1 - cand.iloc[1].f1 < NOISE:
        second = cand.iloc[1]
        say(f"- Cảnh báo: {winner.optimizer}+{winner.scheduler} hơn {second.optimizer}+{second.scheduler} "
            f"chỉ {(winner.f1 - second.f1) * 100:.2f}% F1, nằm trong ngưỡng nhiễu {NOISE * 100:.0f}% của val. "
            f"Chọn theo thứ hạng nhưng chưa nên coi là khác biệt thật.")
    say(f"=> Chọn: **{winner.optimizer} + {winner.scheduler}** (lr {winner.lr:g})")

    # --- Bước 3: A/B sampler
    print("\n=== Bước 3: so sánh balance=sampler với cấu hình đã chọn ===")
    for s in seeds:
        run(winner.optimizer, winner.scheduler, "sampler", epochs, s, args.backup, execute)
    agg = aggregate(collect(epochs))
    same = agg[(agg.optimizer == winner.optimizer) & (agg.scheduler == winner.scheduler)]
    none_row = same[same.balance == "none"].iloc[0]
    samp = same[same.balance == "sampler"]
    say("\n## Bước 3: WeightedRandomSampler (A/B trên cấu hình đã chọn)")
    if samp.empty:
        say("- Chưa có run sampler.")
        verdict = "none"
    else:
        s_row = samp.iloc[0]
        d_f1, d_acc = s_row.f1 - none_row.f1, s_row.acc - none_row.acc
        say(f"- none   : F1 {pct(none_row.f1, none_row.f1_std, none_row.n)}%, acc {pct(none_row.acc, none_row.acc_std, none_row.n)}%")
        say(f"- sampler: F1 {pct(s_row.f1, s_row.f1_std, s_row.n)}%, acc {pct(s_row.acc, s_row.acc_std, s_row.n)}%")
        say(f"- Chênh lệch (sampler - none): F1 {d_f1 * 100:+.2f}%, acc {d_acc * 100:+.2f}%")
        if abs(d_f1) < NOISE:
            say("=> Chưa đủ bằng chứng sampler tốt hơn hay kém hơn (chênh lệch trong ngưỡng nhiễu). "
                "Kiểm tra thêm recall từng lớp yếu (vd. lớp 23) bằng src.evaluate trước khi kết luận.")
            verdict = "none"
        elif d_f1 > 0:
            say("=> Sampler cải thiện macro-F1 rõ hơn ngưỡng nhiễu: dùng sampler.")
            verdict = "sampler"
        else:
            say("=> Sampler làm macro-F1 kém đi rõ rệt: không dùng.")
            verdict = "none"

    # --- Bước 4: xuất kết quả
    REP_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    agg.to_csv(REP_DIR / "grid_summary.csv", index=False)
    table = md_table(agg)
    md = ["# Kết quả lưới thí nghiệm", "", f"{epochs} epoch, seed {seeds}, đo trên tập val.", "", table, "",
          "Best ep = epoch có val acc cao nhất (checkpoint đã lưu). Acc cuối / F1 cuối = giá trị ở epoch cuối, "
          "so với Best để thấy cấu hình nào ổn định. Gap = train acc - val acc ở epoch cuối.", "",
          "# Quá trình chọn", ""] + log
    (REP_DIR / "grid_summary.md").write_text("\n".join(md), encoding="utf-8")

    hist = []
    for n in df[df.seed == base]["name"]:
        with open(LOG_DIR / f"{n}.json", encoding="utf-8") as f:
            hist.append(json.load(f))
    samp_name = run_name(winner.optimizer, winner.scheduler, "sampler", epochs, base)
    if is_done(samp_name, epochs):
        with open(LOG_DIR / f"{samp_name}.json", encoding="utf-8") as f:
            hist.append(json.load(f))
    plot_compare(hist, FIG_DIR / "grid_compare.png")

    print("\n" + table)
    print("\nĐã lưu:", REP_DIR / "grid_summary.csv")
    print("Đã lưu:", REP_DIR / "grid_summary.md")
    print("Đã lưu:", FIG_DIR / "grid_compare.png")
    best = run_name(winner.optimizer, winner.scheduler, verdict if verdict == "sampler" else "none", epochs, base)
    print(f"\nBước tiếp theo: python -m src.evaluate --name {best}")
    backup(args.backup)


if __name__ == "__main__":
    main()