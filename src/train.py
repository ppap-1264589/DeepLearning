import argparse
import time

import torch
import torch.nn as nn

from src.config import ROOT, SEED
from src.dataset import get_dataloaders       # nếu bạn đã tách file, đổi thành src.dataloaders
from src.models.cnn import SimpleCNN


def run_epoch(model, loader, criterion, device, optimizer=None):
    """Chạy 1 epoch. Có optimizer => huấn luyện; không có => đánh giá."""
    training = optimizer is not None
    model.train() if training else model.eval()

    total_loss, correct, n = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)

            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * x.size(0)
            correct += (logits.argmax(1) == y).sum().item()
            n += x.size(0)
    return total_loss / n, correct / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_loader, val_loader, _ = get_dataloaders(balance="none")
    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_dir = ROOT / "outputs" / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, device, optimizer)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, device)
        print(f"epoch {epoch}/{args.epochs} | "
              f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
              f"val loss {va_loss:.4f} acc {va_acc:.4f} | {time.time()-t0:.0f}s")

        if va_acc > best_val:
            best_val = va_acc
            torch.save(model.state_dict(), ckpt_dir / "baseline_cnn.pt")
            print("  -> lưu model tốt nhất")

    print(f"Best val acc: {best_val:.4f}")


if __name__ == "__main__":
    main()