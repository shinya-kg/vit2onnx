import torch
import torch.nn as nn
import torchvision.models as models


class ModelClass(nn.Module):
    def __init__(self):
        super(ModelClass, self).__init__()
        # モデルのロード
        self.model = models.vit_b_16(weights=None)
        # クラス数の変更（10クラス分類）
        num_classes = 10
        self.model.heads.head = torch.nn.Linear(self.model.hidden_dim, num_classes)

    def forward(self, x):
        return self.model(x)