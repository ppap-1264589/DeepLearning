"""Đánh giá một checkpoint trên val hoặc test.

Ví dụ:
    python -m src.evaluate --name cnn_adam_none_none              # mặc định: val
    python -m src.evaluate --name cnn_adam_none_none --split test  # CHỈ chạy khi đã chốt mô hình

Kết quả:
    outputs/reports/<name>_<split>_per_class.csv   precision/recall/F1 theo từng lớp
    outputs/figures/<name>_<split>_confusion.png   confusion matrix (chuẩn hóa theo hàng)
    outputs/figures/<name>_<split>_errors.png      ảnh bị đoán sai cạnh ảnh meta của lớp thật và lớp bị đoán
"""
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import ROOT, RAW, NUM_CLASSES
from src.dataset import get_dataloaders
from src.models.cnn import SimpleCNN


@torch.no_grad()
def predict(model, loader, device):
    """Trả về (nhãn thật, nhãn dự đoán, độ tự tin của dự đoán) theo đúng thứ tự trong CSV.
    Loader val/test không shuffle nên thứ tự khớp với dataset.paths."""
    model.eval()
    labels, preds, confs = [], [], []
    for x, y in loader:
        probs = torch.softmax(model(x.to(device)), dim=1)
        conf, pred = probs.max(dim=1)
        labels.append(y)
        preds.append(pred.cpu())
        confs.append(conf.cpu())
    return (torch.cat(labels).numpy(), torch.cat(preds).numpy(), torch.cat(confs).numpy())


def per_class_table(labels, preds):
    p, r, f, s = precision_recall_fscore_support(
        labels, preds, labels=list(range(NUM_CLASSES)), zero_division=0)
    return pd.DataFrame({"class": range(NUM_CLASSES), "precision": p,
                         "recall": r, "f1": f, "support": s})


def top_confused_pairs(cm, k=10):
    """k cặp (lớp thật -> lớp bị đoán) có số ảnh nhầm nhiều nhất."""
    off = cm.copy()
    np.fill_diagonal(off, 0)
    idx = np.argsort(off, axis=None)[::-1][:k]
    pairs = []
    for flat in idx:
        t, p = divmod(int(flat), off.shape[1])
        if off[t, p] > 0:
            pairs.append((t, p, int(off[t, p])))
    return pairs


def plot_confusion(cm, title, out_path):
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=float), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xlabel("Lớp dự đoán")
    ax.set_ylabel("Lớp thật")
    ax.set_title(title)
    ticks = list(range(0, NUM_CLASSES, 2))
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.tick_params(labelsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, label="tỉ lệ trong lớp thật")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_raw_image(ds, i):
    """Ảnh gốc (đã crop ROI nếu bật) để hiển thị, chưa normalize."""
    return ds.images[i]


def load_meta():
    """Ảnh biển báo chuẩn (Meta) của từng lớp: {class_id: PIL.Image}.
    Cần Meta.csv (cột Path, ClassId) và thư mục Meta/ trong data/raw."""
    csv = RAW / "Meta.csv"
    if not csv.exists():
        print("Không tìm thấy Meta.csv, bỏ qua ảnh meta.")
        return {}
    df = pd.read_csv(csv)
    meta = {}
    for cid, path in zip(df["ClassId"], df["Path"]):
        p = RAW / path
        if p.exists():
            img = Image.open(p).convert("RGBA")
            bg = Image.new("RGBA", img.size, (255, 255, 255, 255))   # nền trắng thay vì đen
            meta[int(cid)] = Image.alpha_composite(bg, img).convert("RGB")
    return meta


def plot_errors(ds, labels, preds, confs, meta, out_path, n=12, groups_per_row=2):
    """Mỗi ảnh sai gồm 3 ô: ảnh bị đoán sai | meta lớp thật | meta lớp bị đoán.
    Chọn các ảnh sai mà mô hình tự tin nhất."""
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("Không có ảnh nào bị đoán sai.")
        return
    wrong = wrong[np.argsort(confs[wrong])[::-1]][:n]
    rows = int(np.ceil(len(wrong) / groups_per_row))
    cols = groups_per_row * 3
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.9, rows * 2.4))
    axes = np.atleast_2d(axes)
    for ax in axes.ravel():
        ax.axis("off")

    for k, i in enumerate(wrong):
        r, c = divmod(k, groups_per_row)
        a0, a1, a2 = axes[r, c * 3], axes[r, c * 3 + 1], axes[r, c * 3 + 2]
        t, p = int(labels[i]), int(preds[i])
        a0.imshow(load_raw_image(ds, i))
        a0.set_title(f"ảnh sai ({confs[i]:.2f})", fontsize=8)
        if t in meta:
            a1.imshow(meta[t])
        a1.set_title(f"thật: {t}", fontsize=8, color="green")
        if p in meta:
            a2.imshow(meta[p])
        a2.set_title(f"đoán: {p}", fontsize=8, color="red")

    fig.suptitle("Ảnh bị đoán sai | meta lớp thật | meta lớp bị đoán")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="cnn_adam_none_none", help="tên checkpoint trong outputs/checkpoints")
    parser.add_argument("--split", choices=["val", "test"], default="val")
    args = parser.parse_args()

    if args.split == "test":
        print("!! Đang chấm TEST. Chỉ làm bước này khi đã chốt mô hình, và không chỉnh gì thêm sau đó.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = ROOT / "outputs" / "checkpoints" / f"{args.name}.pt"
    model = SimpleCNN().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))

    _, val_loader, test_loader = get_dataloaders(balance="none", splits=(args.split,))
    loader = val_loader if args.split == "val" else test_loader
    ds = loader.dataset

    labels, preds, confs = predict(model, loader, device)

    acc = accuracy_score(labels, preds)
    macro_f1 = f1_score(labels, preds, average="macro", labels=list(range(NUM_CLASSES)), zero_division=0)
    print(f"[{args.name} | {args.split}] {len(labels)} ảnh | "
          f"accuracy {acc:.4f} | macro-F1 {macro_f1:.4f} | số ảnh sai {int((labels != preds).sum())}")

    table = per_class_table(labels, preds)
    print("\n10 lớp có F1 thấp nhất:")
    print(table.sort_values("f1").head(10).to_string(index=False, float_format=lambda v: f"{v:.3f}"))

    cm = confusion_matrix(labels, preds, labels=list(range(NUM_CLASSES)))
    print("\nCác cặp hay nhầm nhất (lớp thật -> lớp bị đoán: số ảnh):")
    for t, p, c in top_confused_pairs(cm):
        print(f"  {t:>2} -> {p:>2}: {c}")

    rep_dir = ROOT / "outputs" / "reports"
    fig_dir = ROOT / "outputs" / "figures"
    rep_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    tag = f"{args.name}_{args.split}"
    table.to_csv(rep_dir / f"{tag}_per_class.csv", index=False)
    plot_confusion(cm, f"Confusion matrix: {tag}", fig_dir / f"{tag}_confusion.png")
    plot_errors(ds, labels, preds, confs, load_meta(), fig_dir / f"{tag}_errors.png")
    print("\nĐã lưu:", rep_dir / f"{tag}_per_class.csv")
    print("Đã lưu:", fig_dir / f"{tag}_confusion.png")
    print("Đã lưu:", fig_dir / f"{tag}_errors.png")


if __name__ == "__main__":
    main()