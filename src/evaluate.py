"""Đánh giá một checkpoint trên val hoặc test.

Ví dụ:
    python -m src.evaluate --name cnn_adam_none_none              # mặc định: val
    python -m src.evaluate --name cnn_adam_none_none --split test  # CHỈ chạy khi đã chốt mô hình

Kết quả:
    outputs/reports/<name>_<split>_per_class.csv   precision/recall/F1 theo từng lớp
    outputs/figures/<name>_<split>_confusion.png   confusion matrix (chuẩn hóa theo hàng)
    outputs/figures/<name>_<split>_errors.png      ảnh bị đoán sai (sai mà tự tin nhất)
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
from src.dataset import get_dataloaders       # nếu bạn đã tách file, đổi thành src.dataloaders
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
    img = Image.open(RAW / ds.paths[i]).convert("RGB")
    if ds.use_roi:
        x1, y1, x2, y2 = ds.rois[i]
        img = img.crop((x1, y1, x2 + 1, y2 + 1))
    return img


def plot_errors(ds, labels, preds, confs, out_path, n=24, cols=6):
    wrong = np.where(labels != preds)[0]
    if len(wrong) == 0:
        print("Không có ảnh nào bị đoán sai.")
        return
    wrong = wrong[np.argsort(confs[wrong])[::-1]][:n]     # sai mà tự tin nhất
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2.3))
    axes = np.atleast_1d(axes).ravel()
    for ax in axes:
        ax.axis("off")
    for ax, i in zip(axes, wrong):
        ax.imshow(load_raw_image(ds, i))
        ax.set_title(f"{labels[i]}→{preds[i]} ({confs[i]:.2f})", fontsize=8)
    fig.suptitle("Ảnh bị đoán sai: thật→dự đoán (độ tự tin)")
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

    _, val_loader, test_loader = get_dataloaders(balance="none")
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
    plot_errors(ds, labels, preds, confs, fig_dir / f"{tag}_errors.png")
    print("\nĐã lưu:", rep_dir / f"{tag}_per_class.csv")
    print("Đã lưu:", fig_dir / f"{tag}_confusion.png")
    print("Đã lưu:", fig_dir / f"{tag}_errors.png")


if __name__ == "__main__":
    main()