"""Vẽ biểu đồ từ file log JSON do train.py ghi ra.

Một lần chạy:   python -m src.plot_history outputs/logs/cnn_adam_none_none.json
So sánh nhiều:  python -m src.plot_history outputs/logs/a.json outputs/logs/b.json
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # không cần cửa sổ, chạy được trên Colab/server
import matplotlib.pyplot as plt


def plot_run(h, out_path):
    """4 biểu đồ cho 1 lần chạy: loss, accuracy, val macro-F1, learning rate."""
    ep = h["epoch"]
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))

    ax[0, 0].plot(ep, h["train_loss"], label="train")
    ax[0, 0].plot(ep, h["val_loss"], label="val")
    ax[0, 0].set_title("Loss")

    ax[0, 1].plot(ep, h["train_acc"], label="train")
    ax[0, 1].plot(ep, h["val_acc"], label="val")
    ax[0, 1].set_title("Accuracy")

    ax[1, 0].plot(ep, h["val_f1"], color="tab:green", label="val")
    ax[1, 0].set_title("Macro-F1 (val)")

    ax[1, 1].plot(ep, h["lr"], color="tab:red")
    ax[1, 1].set_yscale("log")
    ax[1, 1].set_title("Learning rate")

    for a in ax.ravel():
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        a.legend()

    fig.suptitle(h["name"])
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(histories, out_path):
    """Chồng nhiều lần chạy lên nhau (chỉ đường val) để so sánh Adam/SGD, có/không scheduler..."""
    fig, ax = plt.subplots(2, 2, figsize=(11, 7))
    panels = [("val_loss", "Val loss", ax[0, 0]), ("val_acc", "Val accuracy", ax[0, 1]),
              ("val_f1", "Val macro-F1", ax[1, 0]), ("lr", "Learning rate", ax[1, 1])]
    for h in histories:
        for key, title, a in panels:
            a.plot(h["epoch"], h[key], label=h["name"])
            a.set_title(title)
    ax[1, 1].set_yscale("log")
    for a in ax.ravel():
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    ax[0, 0].legend(fontsize=8)
    fig.tight_layout()
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    from src.config import ROOT
    files = sys.argv[1:]
    if not files:
        print(__doc__)
        return
    histories = [json.load(open(f, encoding="utf-8")) for f in files]
    fig_dir = ROOT / "outputs" / "figures"
    if len(histories) == 1:
        out = fig_dir / f"{histories[0]['name']}_curves.png"
        plot_run(histories[0], out)
    else:
        out = fig_dir / "compare.png"
        plot_compare(histories, out)
    print("Đã lưu", out)


if __name__ == "__main__":
    main()