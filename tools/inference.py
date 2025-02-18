import sys

sys.path.append("/Users/kogashinya/work_space/practice/onnx_animal")
from _make_dataloader import _make_dataloader
import torch
import time
import argparse
import torchvision.models as models
import onnx
import onnxruntime as ort
import numpy as np
from _input_data import load_image
from _select_model import load_res, load_vit, load_vit_onnx

dic = {
    "dog": 0,
    "horse": 1,
    "elephant": 2,
    "butterfly": 3,
    "chicken": 4,
    "cat": 5,
    "cow": 6,
    "sheep": 7,
    "squirrel": 8,
    "spider": 9,
}

reversed_dic = {value: key for key, value in dic.items()}


def main():
    """imagesディレクトリ内の画像を読み込み、モデルに入力し、推論結果を表示する
    使い方
    python tools/inference.py --model vit_onnx --modelname vit_cls.onnx
    python tools/inference.py --model vit --modelname vit_cls.pth
    """
    parser = argparse.ArgumentParser(description="Select model")
    parser.add_argument("--model", type=str, required=True, help="Select model")
    parser.add_argument("--modelname", type=str, required=True, help="Model name")
    args = parser.parse_args()

    image_dir = "./images"
    device = torch.device("cpu")
    images = load_image(image_dir)

    # デバッグ用に画像リストの長さを出力
    print(f"読み込んだ画像の数: {len(images)}")

    if not images:
        print("画像が読み込まれませんでした。image_dirを確認してください。")
        return

    # 次元数をそろえる
    images = [img.unsqueeze(0) for img in images]
    # テンソルを合わせる
    images_tensor = torch.cat(images, dim=0).to(device)

    # モデルのロード
    if args.model == "res":
        model = load_res(args.modelname)
        model.to(device)
        start_time = time.time()
        output = model(images_tensor)
        end_time = time.time()
    elif args.model == "vit":
        model = load_vit(args.modelname)
        model.to(device)
        start_time = time.time()
        output = model(images_tensor)
        end_time = time.time()
    elif args.model == "vit_onnx":
        model, file_path = load_vit_onnx(args.modelname)
        ort_session = ort.InferenceSession(file_path)
        ort_inputs = {"input": images_tensor.numpy()}
        start_time = time.time()
        output = ort_session.run(None, ort_inputs)[0]
        end_time = time.time()

    # 推論時間の計測
    inference_time = end_time - start_time
    print(f"推論時間：{inference_time}")

    if args.model in ["res", "vit"]:
        _, pred = torch.max(output.data, 1)
        pred_list = [p.cpu().numpy().item() for p in pred]
    elif args.model == "vit_onnx":
        pred_list = output.argmax(axis=1).tolist()

    pred_list = [reversed_dic[p] for p in pred_list]
    print(f"予測結果：{pred_list}")


if __name__ == "__main__":
    main()
