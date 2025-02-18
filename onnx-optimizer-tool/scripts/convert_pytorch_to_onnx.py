import torch
import torch.nn as nn
import torch.onnx
import argparse
import onnx
import sys

sys.path.append("../tools")
from _input_data import load_image


def convert_pytorch_to_onnx(model_path, output_path, input_shape=None):
    """PytorchモデルをONNX形式へ変換"""
    if model_path.endswith(".pth"):
        state_dict = torch.load(
            model_path, map_location=torch.device("cpu"), weights_only=True
        )
        sys.path.append("./models")
        from model import ModelClass

        model = ModelClass()
        model.load_state_dict(state_dict, strict=False)
    else:
        raise ValueError(f"Unsupported Pytorch model format: {model_path}")

    model.eval()

    # dummy_input を input_shape から作成
    # if input_shape is None:
    #     input_shape = (1, 3, 224, 224)  # デフォルト形状
    # dummy_input = torch.rand(input_shape, dtype=torch.float32)

    # 実際の画像を指定してみる
    image_dir = "../images"
    dummy_input = load_image(image_dir)[0].unsqueeze(0)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )
    print(f"✅ PyTorch モデルを ONNX に変換しました: {output_path}")
    # モデルの検証
    onnx.checker.check_model(output_path)

    print("✅ ONNXモデルの検証が成功しました！")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert PyTorch model to ONNX")
    parser.add_argument(
        "--output", type=str, default="models/model.onnx", help="Output ONNX model path"
    )
    args = parser.parse_args()

    convert_pytorch_to_onnx(args.output)
