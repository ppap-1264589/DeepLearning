from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
SEED = 42
NUM_CLASSES = 43

# Chạy compute_stats.py để tính MEAN và STD, sau đó copy vào đây
MEAN = (0.3594, 0.3197, 0.3375)
STD  = (0.2731, 0.2567, 0.2671)

IMG_SIZE = 48          # kích thước resize (median ảnh là 43)
USE_ROI_CROP = True    # crop theo Roi.* trước khi resize
BATCH_SIZE = 128
NUM_WORKERS = 2        # Windows: để 0 cho an toàn, tăng sau khi mọi thứ chạy ổn