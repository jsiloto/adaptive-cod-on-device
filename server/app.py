import os
import stat
import numpy as np
from flask import Flask, render_template, request, send_file
# from model import get_model
import torchvision.transforms as transforms


import sys
sys.path.insert(0,'./yolov5')
from utils.general import non_max_suppression
from models.common import AutoShape
from utils.torch_utils import copy_attr

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
jpg_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
results_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon_result.png')

import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
print(model.stride)
print(model.names)
print(model.names[50])

import yaml
from yaml.loader import SafeLoader

j = torch.jit.load('yolov5s.torchscript.pt')
jitmodel = AutoShape(j)
# print(j.__dict__)
copy_attr(jitmodel, j, include=('yaml', 'nc', 'hyp', 'names', 'stride'), exclude=())  # copy attributes
jitmodel.stride=torch.tensor([8., 16., 32.])
with open('./yolov5/data/coco.yaml', 'r') as f:
    jitmodel.names = yaml.load(f)['names']



def yolo(im, size=640):
    im = transforms.ToPILImage()(im).convert("RGB")
    # g = (size / max(im.size))  # gain
    im = im.resize((size, size), Image.ANTIALIAS)  # resize
    results = jitmodel(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])

def yoloscript(im, size=640):
    im = transforms.Resize((size, size))(im)
    print(im.shape)
    # print(im)
    results = jitmodel(im)  # inference

    print("################################################")
    # print(results)
    print("################################################")
    return results


@app.route('/')
def show_index():
    return render_template("index.html", jpg_image=jpg_filename, results_image=results_filename)

@app.route(jpg_filename)
def jpg_image():
    return send_file(jpg_filename, mimetype='image/jpg')

@app.route(results_filename)
def png_image():
    return send_file(results_filename, mimetype='image/png')

@app.route('/jpg', methods=['POST'])
def jpg():
    with open(jpg_filename, 'wb+') as f:
        f.write(request.data)

    os.chmod(jpg_filename, 0o777)
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return "Welcome to Flask!"

@app.route('/compute', methods=['POST'])
def compute():
    from PIL import Image
    data = np.fromstring(request.data, dtype=np.int8).reshape([3, 480, 640])
    print(data.shape)
    x = torch.tensor(data.astype(np.float32)/255)
    im = yolo(x)
    im.save(results_filename)
    os.chmod(results_filename, 0o777)
    return "OK"
