import torch
import torchvision.models as models
import onnx
from torchvision.models.vision_transformer import ViT_B_16_Weights
import onnxruntime as ort
import sys
sys.path.append("./models")
from model import ModelClass

def load_vit(file_name: str) -> "ModelClass":
    """指定されたファイル名のViTモデルを読み込み、インスタンスを返す
    
    Args:
        file_name (str): 読み込むモデルのファイル名 (例: "vit_model.pth")
        
    Returns:
        ModelClass: 読み込んだモデルのインスタンス

    """
    model_path = f"./models/{file_name}"
    state_dict = torch.load(
        model_path, map_location=torch.device("cpu"), weights_only=True
    )
    model = ModelClass()
    model.load_state_dict(state_dict, strict=True)
    return model


def load_res(file_name: str) -> "ModelClass":
    """指定されたファイル名のResnetモデルの重みを読み込みインスタンスを返す

    Args:
        file_name (str): 読み込むモデルのファイル名 (例: "resnet_model.pth")

    Returns:
        ModelClass:読み込んだモデルのインスタンス
        
    """
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"./models/{file_name}", weights_only=False))
    return model


def load_vit_onnx(file_name):
    """指定されたファイル名のONNXモデルの重みを読み込みインスタンスを返す

    Args:
        file_name (str): 読み込むモデルのファイル名 (例: "onnx_model.pth")

    Returns:
        ModelClass:読み込んだモデルのインスタンス
        file_path (str): 指定したモデルの相対パス
        
    """
    dir_path = "./models/"
    model = onnx.load(dir_path + file_name)
    file_path = dir_path + file_name
    return model, file_path
