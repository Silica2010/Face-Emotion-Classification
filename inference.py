# -*- coding: utf-8 -*-
"""实验二 单图推理：人脸对齐 + 七类表情概率"""

import argparse

import cv2
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from face_align import align_face
from model import FaceEmotionModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image', type=str, required=True)
    ap.add_argument('--weights', type=str, default='best_emotion_model.pth')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--no_align', action='store_true')
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt = torch.load(args.weights, map_location=device)
    label2idx = ckpt['label2idx']
    idx2label = {v: k for k, v in label2idx.items()}
    backbone = ckpt.get('backbone', 'mobilenet_v2')

    model = FaceEmotionModel(len(label2idx), backbone=backbone).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    if args.no_align:
        pic = Image.open(args.image).convert('RGB')
    else:
        img = cv2.imread(args.image)
        if img is None:
            raise FileNotFoundError('无法读取图像: %s' % args.image)
        aligned = align_face(img, out_size=args.img_size)
        if aligned is None:
            print('[warn] 未检测到人脸，回退整图推理')
            pic = Image.open(args.image).convert('RGB')
        else:
            pic = Image.fromarray(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))

    tf = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    x = tf(pic).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        prob = torch.softmax(logits, dim=1)[0].cpu()

    best = int(prob.argmax())
    print('预测表情：%s，置信度：%.3f' % (idx2label[best], prob[best]))
    print('\n各类概率：')
    for i in range(len(idx2label)):
        print('  %-10s %.3f' % (idx2label[i], prob[i]))
    top2 = torch.topk(prob, k=min(2, len(idx2label))).indices.tolist()
    if len(top2) == 2:
        print('\n次高类别：%s（%.3f），若两者差值小于 0.1 建议人工复核'
              % (idx2label[top2[1]], prob[top2[1]]))


if __name__ == '__main__':
    main()
