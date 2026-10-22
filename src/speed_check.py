import time
import torch
import torch.nn as nn

from src.dataset import get_dataloaders      # nếu đã tách file: src.dataloaders
from src.models.cnn import SimpleCNN


def main():
    train_loader, _, _ = get_dataloaders(balance="none")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_batches = len(train_loader)

    # (1) Chỉ nạp dữ liệu, không có model
    t0 = time.time()
    for i, _ in enumerate(train_loader):
        if i == 49:
            break
    t_data = time.time() - t0
    print(f"nạp dữ liệu : {t_data:.1f}s / 50 batch -> ước tính {t_data/50*n_batches/60:.1f} phút/epoch")

    # (2) Chỉ chạy model trên 1 batch có sẵn (không nạp dữ liệu)
    x, y = next(iter(train_loader))
    x, y = x.to(device), y.to(device)
    model = SimpleCNN().to(device)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    crit = nn.CrossEntropyLoss()
    t0 = time.time()
    for _ in range(50):
        loss = crit(model(x), y)
        opt.zero_grad(); loss.backward(); opt.step()
    t_model = time.time() - t0

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"chạy model  : {t_model:.1f}s / 50 batch -> ước tính {t_model/50*n_batches/60:.1f} phút/epoch")


if __name__ == "__main__":
    main()