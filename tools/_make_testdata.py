from PIL import Image
import os
from torchvision import transforms
import glob
import re
from torch.utils.data import DataLoader, Dataset
import random

# 変換用の辞書
translate = {
    "cane": "dog",
    "cavallo": "horse",
    "elefante": "elephant",
    "farfalla": "butterfly",
    "gallina": "chicken",
    "gatto": "cat",
    "mucca": "cow",
    "pecora": "sheep",
    "scoiattolo": "squirrel",
    "ragno": "spider",
}
dic = {
    "cane": 0,
    "cavallo": 1,
    "elefante": 2,
    "farfalla": 3,
    "gallina": 4,
    "gatto": 5,
    "mucca": 6,
    "pecora": 7,
    "scoiattolo": 8,
    "ragno": 9,
}
reversed_dic = {value: key for key, value in dic.items()}


class ImageDataset(Dataset):
    """画像のデータセットを作成"""

    def __init__(self, images, labels):
        super().__init__()
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return self.images[index], self.labels[index]


# 変換関数の定義
def _transform(image):
    # 画像の前処理（リサイズ、テンソル変換、正規化）
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = transform(image)
    return image


def _make_testdata():
    """検証用のテストデータを作成する関数

    Returns:
        test_loader:精度検証、推論速度測定用のテストデータ
    """
    image_dir = "../archive/raw-img"

    # ディレクトリ名を取得
    image_files = [os.path.join(image_dir, i) for i in translate.keys()]

    # 画像のファイルパスを取得
    images = []
    
    # 各カテゴリからランダムに103枚ずつ取得
    for i, _ in enumerate(image_files):
        images.extend(
            random.sample(glob.glob(os.path.join(image_files[2], "*jpeg")), 103)
        )

    # ラベルとともにデータを格納
    labeled_data = []

    for image in images:
        for key, label in dic.items():
            if re.search(key, image):
                labeled_data.append((image, label))

    test_images = [item[0] for item in labeled_data]
    test_labels = [item[1] for item in labeled_data]

    # 画像の読み込み
    test_images = [Image.open(i).convert("RGB") for i in test_images]
    print("画像の読み込み完了")

    # 画像の前処理
    test_images = [_transform(i) for i in test_images]

    print("画像の前処理完了")

    # データセットの作成
    test_dataset = ImageDataset(test_images, test_labels)

    # データローダーの作成
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)

    print("データローダー作成完了")
    print("test_loader:", len(test_loader))

    return test_loader


if __name__ == "__main__":
    _make_testdata()
