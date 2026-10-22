import torch
import torch.nn as nn

from src.config import IMG_SIZE, NUM_CLASSES


class SimpleCNN(nn.Module):
    """CNN cơ bản: 3 khối (Conv -> ReLU -> MaxPool) + 2 lớp Linear."""

    def __init__(self, num_classes=NUM_CLASSES, img_size=IMG_SIZE):
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

    def forward(self, x):
        return self.classifier(self.features(x))


if __name__ == "__main__":
    model = SimpleCNN()
    x = torch.randn(4, 3, IMG_SIZE, IMG_SIZE)
    print("output:", model(x).shape)   # kỳ vọng [4, 43]
    print("số tham số:", sum(p.numel() for p in model.parameters()))