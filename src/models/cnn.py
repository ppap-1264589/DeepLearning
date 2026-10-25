import torch
import torch.nn as nn

from src.config import IMG_SIZE, NUM_CLASSES


class SimpleCNN(nn.Module):
    """CNN cơ bản: 3 khối (Conv -> ReLU -> MaxPool) + 2 lớp Linear.
    dropout > 0: tắt ngẫu nhiên một tỉ lệ nơ-ron giữa hai lớp Linear (chỉ khi model.train())."""

    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE, dropout=0.0):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),    # 48 -> 24
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),   # 24 -> 12
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 12 -> 6
        )
        flat_dim = 128 * (img_size // 8) ** 2        # 128 * 6 * 6 = 4608
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, 256), nn.ReLU(),
            nn.Linear(256, num_classes),             # 43 đầu ra (logits, chưa softmax)
        )
        # Dropout không có tham số, và không nằm trong self.classifier, nên tên các tham số
        # (classifier.1, classifier.3) giữ nguyên -> checkpoint cũ vẫn load được.
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.features(x)
        x = self.classifier[:3](x)       # Flatten -> Linear -> ReLU
        x = self.drop(x)                 # dropout: bật khi model.train(), tắt khi model.eval()
        return self.classifier[3](x)


if __name__ == "__main__":
    model = SimpleCNN(dropout=0.5)
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    print("output:", model(x).shape)   # kỳ vọng [4, 43]
    print("số tham số:", sum(p.numel() for p in model.parameters()))
    model.eval()
    print("eval xác định (dropout tắt):", torch.equal(model(x), model(x)))
    model.train()
    print("train ngẫu nhiên (dropout bật):", not torch.equal(model(x), model(x)))