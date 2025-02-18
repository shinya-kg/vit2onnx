def _make_dataloader():
    """archive内にある画像データを取得し訓練、検証、テスト用のデータローダーを作成する関数

    Returns:
        tuple: 訓練用、検証用、テスト用のDataLoaderオブジェクトのタプル。
            (train_loader, val_loader, test_loader)
    """
    import numpy as np
    from PIL import Image
    import os
    import torch
    from torchvision import transforms
    import glob
    import re
    from sklearn.model_selection import train_test_split
    from torch.utils.data import DataLoader, Dataset

    class ImageDataset(Dataset):
        """画像のデータセットを作成"""

        def __init__(self, images, labels):
            super().__init__()
            self.images = images
            self.labels = labels

        def __len__(self):
            return len(self.images)

        def __getitem__(self, idx):
            return self.images[idx], self.labels[idx]

    # バッチに分割する関数
    def batch_process(data, batch_size):
        for i in range(0, len(data), batch_size):
            yield data[i : i + batch_size]

    # 変換関数の定義
    def _transform(image):
        # 画像の前処理（リサイズ、テンソル変換、正規化）
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        image = transform(image)
        return image

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
    image_dir = "../archive/raw-img"

    image_files = [os.path.join(image_dir, i) for i in translate.keys()]

    images = []

    for i, _ in enumerate(image_files):
        images.extend(glob.glob(os.path.join(image_files[i], "*jpeg")))

    labeled_data = []

    for image in images:
        for key, label in dic.items():
            if re.search(key, image):
                labeled_data.append((image, label))

    # データをテスト用と訓練用に分割
    train_data, test_data = train_test_split(
        labeled_data, test_size=0.15, random_state=42
    )

    # 訓練データを訓練用と検証用に分割
    train_data, valid_data = train_test_split(
        train_data, test_size=0.15, random_state=42
    )

    train_images = [item[0] for item in train_data]
    train_labels = [item[1] for item in train_data]

    val_images = [item[0] for item in valid_data]
    val_labels = [item[1] for item in valid_data]

    test_images = [item[0] for item in test_data]
    test_labels = [item[1] for item in test_data]

    # 画像の読み込み
    train_images = [Image.open(i).convert("RGB") for i in train_images]
    print("画像の読み込み完了")

    # バッチサイズの定義
    batch_size = 32

    re_train_images = []
    # 0〜5000の範囲でバッチ処理　→メモリ不足でバッチに分けて前処理をする
    for batch in batch_process(train_images[:5000], batch_size):
        re_train_images.extend([_transform(i) for i in batch])

    print("0〜5000の範囲でバッチ処理完了")

    # 5001〜10000の範囲でバッチ処理
    for batch in batch_process(train_images[5000:10000], batch_size):
        re_train_images.extend([_transform(i) for i in batch])

    print("5001〜10000の範囲でバッチ処理完了")

    for batch in batch_process(train_images[10000:15000], batch_size):
        re_train_images.extend([_transform(i) for i in batch])

    print("10001〜15000の範囲でバッチ処理完了")

    for batch in batch_process(train_images[15000:], batch_size):
        re_train_images.extend([_transform(i) for i in batch])

    print("15001〜の範囲でバッチ処理完了")

    val_images = [Image.open(i).convert("RGB") for i in val_images]
    test_images = [Image.open(i).convert("RGB") for i in test_images]

    val_images = [_transform(i) for i in val_images]
    test_images = [_transform(i) for i in test_images]

    print("画像の前処理完了")

    train_dataset = ImageDataset(re_train_images, train_labels)
    val_dataset = ImageDataset(val_images, val_labels)
    test_dataset = ImageDataset(test_images, test_labels)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print("DataLoader作成完了")
    print("train_loader: ", len(train_loader))
    print("val_loader: ", len(val_loader))
    print("test_loader: ", len(test_loader))

    return train_loader, val_loader, test_loader


if __name__ == "__main__":
    _make_dataloader()
