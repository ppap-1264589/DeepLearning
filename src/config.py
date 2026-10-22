from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SPLITS = ROOT / "data" / "splits"
SEED = 42
NUM_CLASSES = 43

IMG_SIZE = 48          # kích thước resize (median ảnh là 43)
USE_ROI_CROP = True    # crop theo Roi.* trước khi resize
BATCH_SIZE = 32
NUM_WORKERS = 0        # Windows: để 0 cho an toàn, tăng sau khi mọi thứ chạy ổn