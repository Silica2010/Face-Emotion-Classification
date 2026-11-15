# -*- coding: utf-8 -*-
"""实验二 人脸检测与对齐

优先使用 MTCNN（facenet-pytorch），未安装时回退到 OpenCV Haar 级联 + 双眼旋转校正。
对齐的作用：去掉背景与头部滚转，使同一表情在不同拍摄角度下呈现一致朝向。
"""

import math

import cv2
import numpy as np

_DETECTOR = None
_MODE = None


def _get_mtcnn():
    global _DETECTOR, _MODE
    if _MODE is None:
        try:
            from facenet_pytorch import MTCNN
            _DETECTOR = MTCNN(keep_all=False, select_largest=True, post_process=False)
            _MODE = 'mtcnn'
        except Exception:
            _MODE = 'haar'
            _DETECTOR = {
                'face': cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'),
                'eye': cv2.CascadeClassifier(
                    cv2.data.haarcascades + 'haarcascade_eye.xml'),
            }
    return _DETECTOR


def _rotate(image, angle, center):
    h, w = image.shape[:2]
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nh = int(h * cos + w * sin)
    nw = int(h * sin + w * cos)
    M[0, 2] += nw / 2 - center[0]
    M[1, 2] += nh / 2 - center[1]
    return cv2.warpAffine(image, M, (nw, nh), flags=cv2.INTER_LINEAR,
                          borderMode=cv2.BORDER_REPLICATE)


def _square_crop(image, box, margin=0.25):
    x, y, w, h = box
    cx, cy = x + w / 2.0, y + h / 2.0
    side = max(w, h) * (1 + margin * 2)
    x0, y0 = int(cx - side / 2), int(cy - side / 2)
    x1, y1 = int(cx + side / 2), int(cy + side / 2)
    H, W = image.shape[:2]
    pad = max(0, -x0) + max(0, -y0) + max(0, x1 - W) + max(0, y1 - H)
    if pad > 0:
        image = cv2.copyMakeBorder(image, max(0, -y0), max(0, y1 - H),
                                   max(0, -x0), max(0, x1 - W),
                                   cv2.BORDER_REPLICATE)
        x0 += max(0, -x0)
        y0 += max(0, -y0)
        x1 += max(0, -x0)
        y1 += max(0, -y0)
    return image[max(0, y0):y1, max(0, x0):x1]


def align_face(image_bgr, out_size=224, return_info=False):
    """输入 BGR 图像，返回对齐裁剪后的 BGR 人脸；检测失败返回 None"""
    det = _get_mtcnn()
    if _MODE == 'mtcnn':
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        boxes, _ = det.detect(rgb)
        if boxes is None or len(boxes) == 0:
            return (None, None) if return_info else None
        x0, y0, x1, y1 = boxes[0]
        side = max(x1 - x0, y1 - y0) * 1.3
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        H, W = image_bgr.shape[:2]
        box = (int(cx - side / 2), int(cy - side / 2), int(side), int(side))
        crop = _square_crop(image_bgr, box, margin=0.0)
        info = {'mode': 'mtcnn', 'box': box}
    else:
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = det['face'].detectMultiScale(gray, 1.1, 5, minSize=(48, 48))
        if len(faces) == 0:
            return (None, None) if return_info else None
        faces = sorted(faces, key=lambda r: r[2] * r[3], reverse=True)
        box = tuple(int(v) for v in faces[0])

        # 用双眼位置做旋转校正
        x, y, w, h = box
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = []
        eyes = det['eye'].detectMultiScale(roi_gray, 1.1, 5, minSize=(10, 10))
        if len(eyes) >= 2:
            eyes = sorted(eyes, key=lambda r: r[0])[:2]
            p1 = (x + eyes[0][0] + eyes[0][2] / 2, y + eyes[0][1] + eyes[0][3] / 2)
            p2 = (x + eyes[1][0] + eyes[1][2] / 2, y + eyes[1][1] + eyes[1][3] / 2)
            angle = math.degrees(math.atan2(p2[1] - p1[1], p2[0] - p1[0]))
            if abs(angle) < 45:
                center = ((p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2)
                image_bgr = _rotate(image_bgr, angle, center)
        crop = _square_crop(image_bgr, box, margin=0.25)
        info = {'mode': 'haar', 'box': box, 'eyes': len(eyes)}

    if crop is None or crop.size == 0:
        return (None, None) if return_info else None
    crop = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_LINEAR)
    return (crop, info) if return_info else crop


if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1:
        img = cv2.imread(sys.argv[1])
        out, info = align_face(img, return_info=True)
        print('检测模式:', info)
        print('输出尺寸:', None if out is None else out.shape)
