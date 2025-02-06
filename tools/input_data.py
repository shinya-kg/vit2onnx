from torchvision import transforms
from PIL import Image
import os
import torch
import glob


# 画像の前処理
transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

image_dir = "./images"


def load_image(image_dir):
    image_paths = glob.glob(os.path.join(image_dir, "*.jpeg")) + glob.glob(
        os.path.join(image_dir, "*.jpg")
    )
    images = []
    for image_path in image_paths:
        image = Image.open(image_path).convert("RGB")
        image = transform(image)
        images.append(image)
    return images


if __name__ == "__main__":
    images = load_image(image_dir)
    print(images[0].shape)
