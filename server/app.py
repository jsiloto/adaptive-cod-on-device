import os
import stat
import numpy as np
from flask import Flask, render_template, request, send_file
# from model import get_model
import torchvision.transforms as transforms
import time

import torch
from PIL import Image
import sys
sys.path.insert(0,'./yolov5')
from utils.general import non_max_suppression
from models.common import AutoShape
from utils.torch_utils import copy_attr

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
endpoint_filename = os.path.join(app.config['UPLOAD_FOLDER'], '<filename>')
input_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
reference_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'reference.png')


# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
# print(model.stride)
# print(model.names)
# print(model.names[50])

import yaml
from yaml.loader import SafeLoader

reference_model = torch.jit.load('yolov5s.torchscript.pt')
reference_model = AutoShape(reference_model)

test_model = torch.jit.load('efficientdet.ptl')
test_model = AutoShape(test_model)

elapsed_test = 0
elapsed_reference = 0

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

def detect(im, model, size=640):
    im = transforms.ToPILImage()(im).convert("RGB")
    im = im.resize((size, size), Image.ANTIALIAS)  # resize
    start = time.time()
    for i in range(4):
        results = model(im)  # inference

    end = time.time()
    elapsed = (end-start)/4
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0]), elapsed

@app.route('/')
def show_index():
    global elapsed_test, elapsed_reference

    a = str(elapsed_test)
    print(a)
    print(elapsed_test)

    return render_template("index.html",
                           input_image=input_filename,
                           output_image=output_filename,
                           reference_image=reference_filename,
                           output_time=str(elapsed_test),
                           reference_time=str(elapsed_reference))

@app.route(endpoint_filename)
def image(filename=None):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filename, mimetype='image/png')

# @app.route('/image', methods=['POST'])
# def image():
#     data = np.fromstring(request.data, dtype=np.uint8).reshape([3, 640, 640])
#     print(data)
#     # data = data.astype(np.float32)/255.0
#     data = np.swapaxes(data, 0, 2)
#     data = np.swapaxes(data, 0, 1)
#     print(data.shape)
#     im = Image.fromarray(data)
#     im.save(input_filename)
#     # with open(jpg_filename, 'wb+') as f:
#     #     f.write(request.data)
#
#     os.chmod(input_filename, 0o777)
#     # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
#     return "Welcome to Flask!"

def save_tensor_image(data):
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    image = Image.fromarray(data)
    image.save(input_filename)

@app.route('/compute', methods=['POST'])
def compute():
    global elapsed_test, elapsed_reference

    from PIL import Image
    data = np.fromstring(request.data, dtype=np.uint8).reshape([3, 640, 640])
    save_tensor_image(data)
    x = torch.tensor(data.astype(np.float32)/255)

    output, elapsed_test = detect(x, test_model)
    print(str(elapsed_test))
    output.save(output_filename)
    os.chmod(output_filename, 0o777)

    reference, elapsed_reference = detect(x, reference_model)
    print(str(elapsed_reference))
    reference.save(reference_filename)
    os.chmod(reference_filename, 0o777)


    return "OK"
