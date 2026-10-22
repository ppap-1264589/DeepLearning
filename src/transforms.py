from torchvision import transforms

from src.config import IMG_SIZE, MEAN, STD

def get_train_transform(img_size=IMG_SIZE):
    """Resize và làm biến đổi ngẫu nhiên nhẹ cho ảnh train."""
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomAffine(degrees=12, translate=(0.05, 0.05), scale=(0.9, 1.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
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
