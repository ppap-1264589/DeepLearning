from torchvision import transforms

from src.config import IMG_SIZE


# Chuẩn hóa ảnh sau khi chuyển thành tensor.
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)


def get_train_transform(img_size=IMG_SIZE):
    """Resize và làm biến đổi ngẫu nhiên nhẹ cho ảnh train."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])


def get_eval_transform(img_size=IMG_SIZE):
    """Validation và test chỉ resize, đổi sang tensor và chuẩn hóa."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
