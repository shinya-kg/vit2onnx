import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models.vision_transformer import ViT_B_16_Weights


class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # モデルのロード
        self.model = models.vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
        # クラス数の変更（10クラス分類）
        num_classes = 10
        self.model.heads.head = torch.nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)