import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from tqdm import tqdm # Để hiển thị thanh tiến trình khi nạp ảnh

from src.config import (RAW, SPLITS, BATCH_SIZE, NUM_WORKERS,
                        USE_ROI_CROP, NUM_CLASSES, MEAN, STD)
from src.transforms import get_train_transform, get_eval_transform

mean = torch.tensor(MEAN).view(3, 1, 1)
std = torch.tensor(STD).view(3, 1, 1)

class GTSRBDataset(Dataset):
    def __init__(self, csv_path, transform=None, use_roi=USE_ROI_CROP):
        df = pd.read_csv(csv_path)
        paths = df["Path"].tolist()
        labels = df["ClassId"].to_numpy()
        rois = df[["Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2"]].to_numpy()
        
        self.transform = transform
        self.labels = labels
        
        # --- KỸ THUẬT NẠP SẴN VÀO RAM ---
        self.images = []
        print(f"Đang nạp dữ liệu từ {csv_path} vào RAM...")
        
        for i in tqdm(range(len(paths))):
            # 1. Đọc và giải mã ảnh
            img = Image.open(RAW / paths[i]).convert("RGB")
            
            # 2. Cắt ROI luôn từ bây giờ để giải phóng CPU sau này
            if use_roi:
                x1, y1, x2, y2 = rois[i]
                img = img.crop((x1, y1, x2 + 1, y2 + 1))
            
            # 3. Lưu ảnh đã cắt vào danh sách trên RAM
            self.images.append(img)
        print(f"Đã nạp xong {len(self.images)} ảnh vào RAM thành công!")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        # Lấy thẳng ảnh từ RAM ra (Không tốn thời gian đọc ổ đĩa hay Crop nữa!)
        img = self.images[i] 
        
        # Bây giờ CPU chỉ làm duy nhất 1 việc nhẹ nhàng: Augmentation (nếu có)
        if self.transform:
            img = self.transform(img)
            
        return img, int(self.labels[i])


def get_dataloaders(batch_size=BATCH_SIZE, use_roi=USE_ROI_CROP,
                    balance="sampler", num_workers=NUM_WORKERS,
                    splits=("train", "val", "test")):
    """Chỉ nạp các split có trong `splits`; split không nạp trả về None."""
    """balance: 'sampler' (lấy mẫu cân bằng lớp) hoặc 'none'.
    Nếu dùng 'sampler' thì KHÔNG dùng thêm class weight trong loss (tránh bù hai lần)."""
    def make(name, tf):
        if name not in splits:
            return None
        return GTSRBDataset(SPLITS / f"{name}.csv", tf, use_roi)

    train_ds = make("train", get_train_transform())
    val_ds = make("val", get_eval_transform())
    test_ds = make("test", get_eval_transform())

    sampler = None
    if balance == "sampler" and train_ds is not None:
        counts = np.bincount(train_ds.labels, minlength=NUM_CLASSES)
        sample_w = 1.0 / counts[train_ds.labels]
        sampler = WeightedRandomSampler(sample_w, num_samples=len(train_ds), replacement=True)

    def loader(ds, **kw):
        if ds is None:
            return None
        return DataLoader(ds, batch_size=batch_size, num_workers=num_workers,
                          prefetch_factor=8, pin_memory=True, **kw)

    train_loader = loader(train_ds, sampler=sampler, shuffle=(sampler is None)) if train_ds else None
    return train_loader, loader(val_ds), loader(test_ds)


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
        ax.imshow((img * std + mean).permute(1, 2, 0).clamp(0, 1))   # bỏ chuẩn hóa để xem
        ax.set_title(str(lab.item()), fontsize=9)
        ax.axis("off")
    out = ROOT / "outputs" / "figures"
    out.mkdir(parents=True, exist_ok=True)
    plt.savefig(out / "aug_check.png", dpi=150, bbox_inches="tight")
    print("Đã lưu", out / "aug_check.png")
