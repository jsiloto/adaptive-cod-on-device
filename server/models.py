import sys
import time
import torch
import yaml
import torchvision.transforms as transforms
from PIL import Image
sys.path.insert(0, './yolov5')
from yolov5.models.common import AutoShape


def get_models():
    reference_model = torch.jit.load('yolov5s.torchscript.pt')
    reference_model = AutoShape(reference_model)

    test_model = torch.jit.load('efficientdet.ptl')
    test_model = AutoShape(test_model)


    test_model.stride=torch.tensor([8., 16., 32.])
    with open('./yolov5/data/coco.yaml', 'r') as f:
        test_model.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
                'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
                'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag', 'tie',
                'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove',
                'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut',
                'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '', 'tv',
                'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                'toothbrush']

    reference_model.stride=torch.tensor([8., 16., 32.])
    with open('./yolov5/data/coco.yaml', 'r') as f:
        reference_model.names = yaml.load(f)['names']

    return reference_model, test_model


def detect(im, model, image_id, orig_w=640, orig_h=640, size=640):
    im = transforms.ToPILImage()(im).convert("RGB")
    im = im.resize((size, size), Image.ANTIALIAS)  # resize
    start = time.time()
    ntimes=1
    for i in range(ntimes):
        results = model(im)  # inference

    end = time.time()
    elapsed = (end-start)/ntimes
    detections = []
    for p in results.pred[0]:
        d = {
            "image_id": image_id,
            "category_id": int(p[5]+1),
            'bbox': invert_afine(p[0:4].numpy().tolist(), orig_w, orig_h, size),
            'score': p[4].numpy().tolist(),
        }
        detections.append(d)

    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0]), elapsed, detections


def invert_afine(bbox, orig_w=640, orig_h=640, size=640):
    x1, y1, x2, y2 = bbox

    w = x2-x1
    h = y2-y1

    x1 = orig_w * x1 / size
    w = orig_w * w / size
    y1 = orig_h * y1 / size
    h = orig_h * h / size
    return [x1, y1, w, h]