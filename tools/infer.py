import sys

sys.path.append("/Users/kogashinya/work_space/practice/onnx_animal")
from make_dataloader import make_dataloader
import torch
import time
import argparse
import torchvision.models as models
from input_data import load_image
from select_model import load_res, load_vit

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


def main():
    """imagesディレクトリ内の画像を読み込み、モデルに入力し、推論結果を表示する
    使い方
    python infer.py --model res --modelname res18.pth
    python infer.py --model vit --modelname vit_cls.pth
    """
    parser = argparse.ArgumentParser(description="Select model")
    parser.add_argument("--model", type=str, required=True, help="Select model")
    parser.add_argument("--modelname", type=str, required=True, help="Model name")
    args = parser.parse_args()

    image_dir = "./images"
    device = torch.device("cpu")
    images = load_image(image_dir)

    # Ensure all tensors in the images list have the correct dimensions
    images = [img.unsqueeze(0) for img in images]
    # Concatenate the list of tensors into a single tensor
    images_tensor = torch.cat(images, dim=0).to(device)

    # モデルのロード
    if args.model == "res":
        model = load_res(args.modelname)
    elif args.model == "vit":
        model = load_vit(args.modelname)

    model.to(device)

    start_time = time.time()
    output = model(images_tensor)
    end_time = time.time()

    # 推論時間の計測
    inference_time = end_time - start_time
    print(f"推論時間：{inference_time}")

    _, pred = torch.max(output.data, 1)

    pred_list = [p.cpu().numpy().item() for p in pred]
    pred_list = [reversed_dic[p] for p in pred_list]
    pred_list = [translate[p] for p in pred_list]
    print(f"予測結果：{pred_list}")


if __name__ == "__main__":
    main()
