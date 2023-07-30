from io import BytesIO
from pathlib import Path
from io import StringIO 
import argparse
import sys
import time

from django.core.files.uploadedfile import InMemoryUploadedFile
from django.core.files.base import ContentFile

from numpy import random
from PIL import Image
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn

from yolov7.models.experimental import attempt_load
from yolov7.utils.datasets import LoadImages
from yolov7.utils.general import check_img_size, non_max_suppression, scale_coords
from yolov7.utils.plots import plot_one_box
from yolov7.utils.torch_utils import select_device, load_classifier, time_synchronized

sys.path.insert(0, '/home/imperial/Documents/projects/mysite/.venv/lib/python3.10/site-packages/yolov7')


def get_yolo_detection(img_path):
    img = detect(img_path, "yolov7.pt", 640)
    frame_jpg = cv2.imencode('.jpg', img[0])[1]
    return ContentFile(frame_jpg)


def detect(
        source, weights, imgsz
    ):

    # Initialize
    device = select_device("cpu")

    # Load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride
    imgsz = check_img_size(imgsz, s=stride)  # check img_size

    dataset = LoadImages(source, img_size=imgsz, stride=stride)

    # Get names and colors
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in names]

    old_img_w = old_img_h = imgsz
    old_img_b = 1

    t0 = time.time()
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # Inference
        t1 = time_synchronized()
        with torch.no_grad():   # Calculating gradients would cause a GPU memory leak
            pred = model(img, augment=False)[0] # Augment->Inferencia aumentada
        t2 = time_synchronized()

        # Apply NMS
        # Object confidence threshold. Min Score
        # IOU threshold for NMS
        # Agnostic -> Detectar objectos que no estan clasificados.
        pred = non_max_suppression(pred, 0.55, 0.45, agnostic=False)
        t3 = time_synchronized()

        imgs = []
        # Process detections
        for i, det in enumerate(pred):  # detections per image
            p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)

            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # Write results
                for *xyxy, conf, cls in reversed(det):
                    label = f'{names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=colors[int(cls)], line_thickness=1)

            imgs.append(im0)

        return imgs
