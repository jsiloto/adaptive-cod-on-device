import os
import stat
import numpy as np
from flask import Flask, render_template, request, send_file
# from model import get_model
import torchvision.transforms as transforms

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
jpg_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
results_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon_result.png')

import torch
from PIL import Image

# Model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')





def yolo(im, size=640):
    g = (size / max(im.size))  # gain
    im = im.resize((int(x * g) for x in im.size), Image.ANTIALIAS)  # resize

    results = model(im)  # inference
    results.render()  # updates results.imgs with boxes and labels
    return Image.fromarray(results.imgs[0])

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
    im = transforms.ToPILImage()(x).convert("RGB")
    im = yolo(im)
    print(im)
    im.save(results_filename)
    os.chmod(results_filename, 0o777)
    return "OK"
