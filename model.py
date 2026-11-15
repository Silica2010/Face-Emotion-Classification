# -*- coding: utf-8 -*-
"""实验二 模型定义：MobileNetV2 迁移 + 分类头，另附双流融合结构"""

import torch
import torch.nn as nn
from torchvision import models


def build_backbone(name: str = 'mobilenet_v2', pretrained: bool = True):
    """返回去掉分类层的特征主干及输出维度"""
    if name == 'mobilenet_v2':
        net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT if pretrained else None)
        feat_dim = net.classifier[1].in_features      # 1280
        net.classifier = nn.Identity()
    elif name == 'resnet18':
        net = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)
        feat_dim = net.fc.in_features
        net.fc = nn.Identity()
    else:
        raise ValueError('不支持的主干网络: %s' % name)
    return net, feat_dim


class FaceEmotionModel(nn.Module):
    """单流：全脸输入"""

    def __init__(self, num_classes: int = 7, backbone: str = 'mobilenet_v2',
                 pretrained: bool = True, dropout: float = 0.2):
        super().__init__()
        self.backbone, feat_dim = build_backbone(backbone, pretrained)
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(feat_dim, num_classes),
        )

    def forward(self, x):
        return self.head(self.backbone(x))

    def freeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self):
        for p in self.backbone.parameters():
            p.requires_grad = True


class TwoStreamEmotion(nn.Module):
    """双流：全脸 + 眼部/口部局部子图，用于验证局部特征是否主导表情判别

    说明：局部子图需在数据层按关键点裁剪后作为第二个输入传入，
    属可选的消融/改进项，主实验仍使用 FaceEmotionModel。
    """

    def __init__(self, num_classes: int = 7, backbone: str = 'mobilenet_v2'):
        super().__init__()
        self.global_stream, feat_dim = build_backbone(backbone, True)
        self.local_stream, feat_dim2 = build_backbone(backbone, True)
        self.fuse = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(feat_dim + feat_dim2, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, face, local=None):
        if local is None:
            local = face
        fg = self.global_stream(face)
        fl = self.local_stream(local)
        return self.fuse(torch.cat([fg, fl], dim=1))


if __name__ == '__main__':
    m = FaceEmotionModel(num_classes=7)
    print('全脸输出:', tuple(m(torch.randn(2, 3, 224, 224)).shape))
    m2 = TwoStreamEmotion(num_classes=7)
    print('双流输出:', tuple(m2(torch.randn(2, 3, 224, 224)).shape))
