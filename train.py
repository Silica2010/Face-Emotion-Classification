# -*- coding: utf-8 -*-
"""实验二 训练脚本：人脸对齐 + MobileNetV2 两阶段迁移 + 类别加权与标签平滑"""

import argparse
import json
import os
from collections import Counter

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score, classification_report
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from tqdm import tqdm

from face_align import align_face
from model import FaceEmotionModel

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class EmotionDataset(Dataset):
    """读取 csv：image_path,label；可选先做人脸检测对齐"""

    def __init__(self, csv_path, label_map, transform=None, use_align=True, cache_dir=None):
        self.samples = []
        self.transform = transform
        self.use_align = use_align
        self.cache_dir = cache_dir
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        with open(csv_path, 'r', encoding='utf-8') as f:
            header = f.readline().strip().split(',')
            if header[:2] != ['image_path', 'label']:
                raise ValueError('csv 列名应为 image_path,label，实际为 %s' % header)
            for line in f:
                line = line.strip()
                if not line:
                    continue
                path, label = line.split(',')[:2]
                self.samples.append((path, label_map[label]))

    def __len__(self):
        return len(self.samples)

    def _load(self, path):
        if not self.use_align:
            return Image.open(path).convert('RGB')
        if self.cache_dir:
            cached = os.path.join(self.cache_dir,
                                  os.path.basename(path).rsplit('.', 1)[0] + '_aligned.jpg')
            if os.path.exists(cached):
                return Image.open(cached).convert('RGB')
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError('无法读取图像: %s' % path)
        aligned = align_face(img, out_size=224)
        if aligned is None:            # 检测失败则回退原图，并把情况记录下来
            print('[warn] 未检测到人脸，回退整图: %s' % path)
            return Image.open(path).convert('RGB')
        rgb = cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB)
        pic = Image.fromarray(rgb)
        if self.cache_dir:
            pic.save(cached, quality=95)
        return pic

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = self._load(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def build_transform(img_size, train):
    if train:
        return transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation((-10, 10)),
            transforms.RandomResizedCrop(img_size, scale=(0.85, 1.0)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ])
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def class_weights(dataset, n_cls):
    labels = [s[1] for s in dataset.samples]
    cnt = Counter(labels)
    w = torch.tensor([len(labels) / (n_cls * cnt.get(i, 1)) for i in range(n_cls)],
                     dtype=torch.float)
    return w / w.mean()


@torch.no_grad()
def evaluate(model, loader, device, criterion):
    model.eval()
    loss_sum, n = 0.0, 0
    preds, trues = [], []
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        out = model(imgs)
        loss_sum += criterion(out, labels).item() * imgs.size(0)
        n += imgs.size(0)
        preds += out.argmax(1).cpu().tolist()
        trues += labels.cpu().tolist()
    acc = sum(p == t for p, t in zip(preds, trues)) / n
    macro_f1 = f1_score(trues, preds, average='macro', zero_division=0)
    return {'loss': loss_sum / n, 'acc': acc, 'macro_f1': macro_f1,
            'preds': preds, 'trues': trues}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train_csv', type=str, required=True)
    ap.add_argument('--val_csv', type=str, required=True)
    ap.add_argument('--labels', type=str, default='configs/labels.json')
    ap.add_argument('--backbone', type=str, default='mobilenet_v2')
    ap.add_argument('--img_size', type=int, default=224)
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--epochs_stage1', type=int, default=8)
    ap.add_argument('--epochs_stage2', type=int, default=32)
    ap.add_argument('--lr_stage1', type=float, default=3e-4)
    ap.add_argument('--lr_stage2', type=float, default=1e-5)
    ap.add_argument('--smooth', type=float, default=0.1, help='标签平滑系数')
    ap.add_argument('--no_align', action='store_true', help='关闭人脸检测对齐（消融用）')
    ap.add_argument('--weight_decay', type=float, default=1e-4)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--patience', type=int, default=10)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with open(args.labels, 'r', encoding='utf-8') as f:
        label_map = json.load(f)['label2idx']
    n_cls = len(label_map)

    use_align = not args.no_align
    train_ds = EmotionDataset(args.train_csv, label_map,
                              build_transform(args.img_size, True),
                              use_align, 'cache/train' if use_align else None)
    val_ds = EmotionDataset(args.val_csv, label_map,
                            build_transform(args.img_size, False),
                            use_align, 'cache/val' if use_align else None)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.workers, pin_memory=True)

    model = FaceEmotionModel(n_cls, backbone=args.backbone).to(device)
    w = class_weights(train_ds, n_cls).to(device)
    criterion = nn.CrossEntropyLoss(weight=w, label_smoothing=args.smooth)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

    best = {'macro_f1': -1.0}
    bad_epochs = 0
    for stage, (epochs, lr, freeze) in enumerate([
            (args.epochs_stage1, args.lr_stage1, True),
            (args.epochs_stage2, args.lr_stage2, False)], start=1):
        if freeze:
            model.freeze_backbone()
            params = model.head.parameters()
        else:
            model.unfreeze_backbone()
            params = model.parameters()
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs,
                                                              eta_min=1e-6)

        for ep in range(1, epochs + 1):
            model.train()
            running = 0.0
            pbar = tqdm(train_loader, desc='stage%d ep%d' % (stage, ep), ncols=100)
            for imgs, labels in pbar:
                imgs, labels = imgs.to(device), labels.to(device)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = model(imgs)
                    loss = criterion(out, labels)
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running += loss.item() * imgs.size(0)
                pbar.set_postfix(loss='%.4f' % loss.item())
            scheduler.step()

            m = evaluate(model, val_loader, device, criterion)
            print('[stage%d ep%02d] train_loss=%.4f val_loss=%.4f acc=%.4f macro_f1=%.4f'
                  % (stage, ep, running / len(train_ds), m['loss'], m['acc'], m['macro_f1']))

            if m['macro_f1'] > best['macro_f1']:
                best = {'macro_f1': m['macro_f1'], 'acc': m['acc'], 'stage': stage, 'epoch': ep}
                bad_epochs = 0
                torch.save({'model': model.state_dict(), 'label2idx': label_map,
                            'backbone': args.backbone, 'args': vars(args)},
                           'best_emotion_model.pth')
                print('  -> 保存最优模型 best_emotion_model.pth (macro_f1=%.4f)' % m['macro_f1'])
            else:
                bad_epochs += 1
                if bad_epochs >= args.patience:
                    print('验证指标 %d 轮未提升，提前停止' % args.patience)
                    break

    print('训练结束，最优宏平均 F1 %.4f（准确率 %.4f，stage%d/epoch%d）'
          % (best['macro_f1'], best['acc'], best['stage'], best['epoch']))

    ckpt = torch.load('best_emotion_model.pth', map_location=device)
    model.load_state_dict(ckpt['model'])
    m = evaluate(model, val_loader, device, criterion)
    idx2label = {v: k for k, v in label_map.items()}
    print('\n验证集分类报告：')
    print(classification_report(m['trues'], m['preds'],
                                target_names=[idx2label[i] for i in range(n_cls)],
                                zero_division=0))


if __name__ == '__main__':
    main()
