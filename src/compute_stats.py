import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from src.config import SPLITS, IMG_SIZE, USE_ROI_CROP, NUM_WORKERS
from src.dataset import GTSRBDataset


def main():
    # Giống pipeline train nhưng: KHÔNG augmentation, KHÔNG Normalize
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),                      # đưa pixel về [0, 1]
    ])
    ds = GTSRBDataset(SPLITS / "train.csv", transform=tf, use_roi=USE_ROI_CROP)
    loader = DataLoader(ds, batch_size=1024, num_workers=NUM_WORKERS)

    n = 0
    s = torch.zeros(3, dtype=torch.float64)         # tổng pixel theo kênh
    s2 = torch.zeros(3, dtype=torch.float64)        # tổng bình phương theo kênh
    for x, _ in loader:
        x = x.double()
        n += x.shape[0] * x.shape[2] * x.shape[3]   # số pixel mỗi kênh
        s += x.sum(dim=(0, 2, 3))
        s2 += (x ** 2).sum(dim=(0, 2, 3))

    mean = s / n
    std = (s2 / n - mean ** 2).sqrt()
    print("MEAN =", tuple(round(v, 4) for v in mean.tolist()))
    print("STD  =", tuple(round(v, 4) for v in std.tolist()))


if __name__ == "__main__":
    main()