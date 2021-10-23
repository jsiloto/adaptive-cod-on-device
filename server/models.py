import sys
import time
import torch
import yaml
import torchvision.transforms as transforms
from PIL import Image

sys.path.insert(0, '../common')
import constants
sys.path.insert(0, './yolov5')
from yolov5.models.common import AutoShape, AutoShapeDecoder


def get_models():
    full_model = torch.jit.load('./assets/effd2.ptl')
    full_model = AutoShape(full_model)
    full_model.stride = torch.tensor([8., 16., 32.])
    full_model.names = constants.class_names

    decoder_model = torch.jit.load('./assets/effd2_decoder.ptl')
    decoder_model = AutoShapeDecoder(decoder_model)
    decoder_model.stride = torch.tensor([8., 16., 32.])
    decoder_model.names = full_model.names

    return full_model, decoder_model


def get_yolo_model():
    yolo_model = torch.jit.load('./assets/yolov5s.torchscript.pt')
    yolo_model = AutoShape(yolo_model)
    yolo_model.stride = torch.tensor([8., 16., 32.])
    with open('./yolov5/data/coco.yaml', 'r') as f:
        yolo_model.names = yaml.load(f)['names']

    return yolo_model


def detect(im, model, image_id, w, h):
    im = transforms.ToPILImage()(im).convert("RGB")
    start = time.time()
    ntimes = 1
    for i in range(ntimes):
        results = model(im)  # inference

    end = time.time()
    elapsed = (end - start) / ntimes
    detections = pred2det(results.pred[0], image_id, w, h)
    results.render()  # updates data.imgs with boxes and labels
    return Image.fromarray(results.imgs[0]), elapsed, detections


def pred2det(pred, image_id, w, h):
    detections = []
    for p in pred:
        d = {
            "image_id": image_id,
            "category_id": int(p[5] + 1),
            'bbox': invert_affine(p[0:4].numpy().tolist(), w, h, 768),
            'score': p[4].numpy().tolist(),
        }
        detections.append(d)
    return detections


def affine(bbox, orig_w, orig_h, size=768):
    x1, y1, w, h = bbox

    x2 = x1 + w
    y2 = y1 + h
    #
    # x1 = size * x1 / orig_w
    # x2 = size * x2 / orig_w
    # y1 = size * y1 / orig_h
    # y2 = size * y2 / orig_h
    bbox = [x1, y1, x2, y2]
    return bbox


def invert_affine(bbox, orig_w, orig_h, size=768):
    x1, y1, x2, y2 = bbox

    w = x2 - x1
    h = y2 - y1

    # x1 = orig_w * x1 / size
    # w = orig_w * w / size
    # y1 = orig_h * y1 / size
    # h = orig_h * h / size
    return [x1, y1, w, h]
