import numpy as np
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import (RAW, SPLITS, BATCH_SIZE, NUM_WORKERS,
                        USE_ROI_CROP, NUM_CLASSES)
from src.transforms import get_train_transform, get_eval_transform


class GTSRBDataset(Dataset):
    def __init__(self, csv_path, transform=None, use_roi=USE_ROI_CROP):
        df = pd.read_csv(csv_path)
        self.paths = df["Path"].tolist()
        self.labels = df["ClassId"].to_numpy()
        self.rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].to_numpy()
        self.transform = transform
        self.use_roi = use_roi

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(RAW / self.paths[i]).convert("RGB")
        if self.use_roi:
            x1, y1, x2, y2 = self.rois[i]
            img = img.crop((x1, y1, x2 + 1, y2 + 1))   # +1 vì X2,Y2 tính cả pixel đó
        if self.transform:
            img = self.transform(img)
        return img, int(self.labels[i])

def get_dataloaders(batch_size=BATCH_SIZE, use_roi=USE_ROI_CROP,
                    balance="sampler", num_workers=NUM_WORKERS):
    """balance: 'sampler' (lấy mẫu cân bằng lớp) hoặc 'none'.
    Nếu dùng 'sampler' thì KHÔNG dùng thêm class weight trong loss (tránh bù hai lần)."""
    train_ds = GTSRBDataset(SPLITS / "train.csv", get_train_transform(), use_roi)
    val_ds = GTSRBDataset(SPLITS / "val.csv", get_eval_transform(), use_roi)
    test_ds = GTSRBDataset(SPLITS / "test.csv", get_eval_transform(), use_roi)

    sampler = None
    if balance == "sampler":
        counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
        sample_w = 1.0 / counts[train_ds.labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(train_ds, batch_size=batch_size, sampler=sampler,
                              shuffle=(sampler is None), num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, num_workers=num_workers)
    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    # Kiểm tra nhanh: in kích thước batch + lưu lưới ảnh sau augmentation
    import matplotlib.pyplot as plt
    from src.config import ROOT

    train_loader, val_loader, test_loader = get_dataloaders()
    x, y = next(iter(train_loader))
    print("batch train:", x.shape, y.shape, "| min/max pixel:", x.min().item(), x.max().item())
    xv, yv = next(iter(val_loader))
    print("batch val  :", xv.shape, yv.shape)
    print("số batch   : train", len(train_loader), "| val", len(val_loader), "| test", len(test_loader))

    fig, axes = plt.subplots(2, 8, figsize=(14, 4))
    for ax, img, lab in zip(axes.ravel(), x[:16], y[:16]):
        ax.imshow((img * 0.5 + 0.5).permute(1, 2, 0).clamp(0, 1))   # bỏ chuẩn hóa để xem
        ax.set_title(str(lab.item()), fontsize=9)
        ax.axis("off")
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "aug_check.png", dpi=150, bbox_inches="tight")
    print("Đã lưu", out / "aug_check.png")
