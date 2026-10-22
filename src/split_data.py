import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from src.config import RAW, SPLITS, SEED


def main():
    SPLITS.mkdir(parents=True, exist_ok=True)

    train_full = pd.read_csv(RAW / "Train.csv")
    test = pd.read_csv(RAW / "Test.csv")

    # track_id = "lớp_track" (vd 00020_00000). Chỉ có ở tập train.
    train_full["track_id"] = train_full["Path"].str.extract(r"(\d{5}_\d{5})_\d{5}")[0]
    assert train_full["track_id"].notna().all(), "Có đường dẫn không tách được track_id"

    # 5 fold => lấy 1 fold làm validation = ~20%.
    # Group theo track (1 track không bị tách đôi), stratify theo lớp.
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train_idx, val_idx = next(sgkf.split(
        train_full, y=train_full["ClassId"], groups=train_full["track_id"]
    ))
    train_df = train_full.iloc[train_idx].reset_index(drop=True)
    val_df = train_full.iloc[val_idx].reset_index(drop=True)

    # ---- kiểm tra tính đúng đắn ----
    overlap = set(train_df["track_id"]) & set(val_df["track_id"])
    assert len(overlap) == 0, f"Rò rỉ dữ liệu: {len(overlap)} track nằm ở cả train và val"
    assert train_df["ClassId"].nunique() == 43 and val_df["ClassId"].nunique() == 43, \
        "Có lớp bị mất ở train hoặc val"

    print(f"train: {len(train_df)} ({len(train_df)/len(train_full):.1%}) | "
          f"{train_df['track_id'].nunique()} tracks")
    print(f"val  : {len(val_df)} ({len(val_df)/len(train_full):.1%}) | "
          f"{val_df['track_id'].nunique()} tracks")
    print(f"test : {len(test)}")

    train_df.to_csv(SPLITS / "train.csv", index=False)
    val_df.to_csv(SPLITS / "val.csv", index=False)
    test.to_csv(SPLITS / "test.csv", index=False)
    print("Đã lưu vào", SPLITS)


if __name__ == "__main__":
    main()