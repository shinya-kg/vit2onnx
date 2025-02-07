import torch
import torchvision.models as models
import onnx
import onnxruntime as ort


def load_vit(file_name):
    model = models.vit_b_16()
    model.heads.head = torch.nn.Linear(model.hidden_dim, 10)
    model.load_state_dict(torch.load(f"./models/{file_name}", weights_only=False))
    return model


def load_res(file_name):
    model = models.resnet18()
    model.fc = torch.nn.Linear(512, 10)
    model.load_state_dict(torch.load(f"./models/{file_name}", weights_only=False))
    return model


def load_vit_onnx(file_name):
    dir_path = "./models/"
    model = onnx.load(dir_path + file_name)
    file_path = dir_path + file_name
    return model, file_path