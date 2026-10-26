import argparse
import json
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from src.config import ROOT, SEED
from src.dataset import get_dataloaders       # nếu bạn đã tách file, đổi thành src.dataloaders
from src.models.cnn import SimpleCNN
from src.plot_history import plot_run


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Chạy 1 epoch. Có optimizer => huấn luyện; không có => đánh giá.
    Trả về (loss, accuracy, macro-F1).

    `criterion` (có thể có label smoothing) chỉ dùng để tính gradient. Loss GHI LOG luôn là
    cross-entropy thường, để loss so được giữa các run và giữa train với val."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, n = 0.0, 0
    preds, labels = [], []
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += F.cross_entropy(logits.detach(), y).item() * x.size(0)
            n += x.size(0)
            preds.append(logits.argmax(1).cpu())
            labels.append(y.cpu())

    preds = torch.cat(preds).numpy()
    labels = torch.cat(labels).numpy()
    acc = float((preds == labels).mean())
    macro_f1 = f1_score(labels, preds, average="macro")
    return total_loss / n, acc, macro_f1


def build_optimizer(name, params, lr, weight_decay=0.0):
    if name == "adam":
        if weight_decay > 0:      # Adam + weight decay => dùng AdamW (decay tách khỏi gradient)
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=weight_decay)


def build_scheduler(name, optimizer, epochs):
    if name == "plateau":      # giảm lr khi val loss không cải thiện
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=2)
    if name == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)          # SGD thường dùng ~1e-2
    parser.add_argument("--optimizer", choices=["adam", "sgd"], default="adam")
    parser.add_argument("--scheduler", choices=["none", "plateau", "cosine"], default="none")
    parser.add_argument("--balance", choices=["none", "sampler"], default="none")
    parser.add_argument("--dropout", type=float, default=0.0, help="tỉ lệ dropout giữa 2 lớp Linear (0 = tắt)")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="0 = tắt; Adam sẽ chuyển sang AdamW")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="0 = tắt; thường dùng 0.1")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--name", default=None, help="tên lần chạy (mặc định tự sinh từ cấu hình)")
    args = parser.parse_args()

    tags = []
    if args.dropout:
        tags.append(f"do{args.dropout:g}")
    if args.weight_decay:
        tags.append(f"wd{args.weight_decay:g}")
    if args.label_smoothing:
        tags.append(f"ls{args.label_smoothing:g}")
    name = args.name or "_".join([f"cnn_{args.optimizer}_{args.scheduler}_{args.balance}"] + tags)

    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device, "| run:", name)

    train_loader, val_loader, _ = get_dataloaders(balance=args.balance, splits=("train", "val"))
    model = SimpleCNN(dropout=args.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)   # dùng sampler thì KHÔNG thêm class weight ở đây
    optimizer = build_optimizer(args.optimizer, model.parameters(), args.lr, args.weight_decay)
    scheduler = build_scheduler(args.scheduler, optimizer, args.epochs)

    ckpt_dir = ROOT / "outputs" / "checkpoints"
    log_dir = ROOT / "outputs" / "logs"
    fig_dir = ROOT / "outputs" / "figures"
    for d in (ckpt_dir, log_dir, fig_dir):
        d.mkdir(parents=True, exist_ok=True)

    history = {"name": name, "config": vars(args), "epoch": [], "lr": [],
               "train_loss": [], "train_acc": [], "train_f1": [],
               "val_loss": [], "val_acc": [], "val_f1": []}
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        cur_lr = optimizer.param_groups[0]["lr"]
        tr_loss, tr_acc, tr_f1 = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc, va_f1 = run_epoch(model, val_loader, criterion, device)

        if scheduler is not None:
            scheduler.step(va_loss) if args.scheduler == "plateau" else scheduler.step()

        for k, v in [("epoch", epoch), ("lr", cur_lr),
                     ("train_loss", tr_loss), ("train_acc", tr_acc), ("train_f1", tr_f1),
                     ("val_loss", va_loss), ("val_acc", va_acc), ("val_f1", va_f1)]:
            history[k].append(v)

        print(f"epoch {epoch}/{args.epochs} | lr {cur_lr:.2e} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} f1 {va_f1:.4f} | {time.time()-t0:.0f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt_dir / f"{name}.pt")
            print("  -> lưu model tốt nhất")

        # ghi log mỗi epoch: lỡ dừng giữa chừng vẫn còn dữ liệu để vẽ
        with open(log_dir / f"{name}.json", "w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    print(f"Best val acc: {best_val:.4f}")
    plot_run(history, fig_dir / f"{name}_curves.png")
    print("Đã lưu log:", log_dir / f"{name}.json")
    print("Đã lưu biểu đồ:", fig_dir / f"{name}_curves.png")


if __name__ == "__main__":
    main()