import os
import json
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, send_file
from models import get_models, detect
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

annFile = '../resource/dataset/coco2017/annotations/instances_val2017.json'
cocoGt = COCO(annFile)

app = Flask(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
endpoint_filename = os.path.join(app.config['UPLOAD_FOLDER'], '<filename>')
input_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
reference_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'reference.png')

reference_model, test_model = get_models()

# Global State
complete_results = []
elapsed_test = 0
elapsed_reference = 0

try:
    with open('data.json', ) as f:
        complete_results = json.load(f)
except:
    pass

print(complete_results)


@app.route('/')
def show_index():
    global elapsed_test, elapsed_reference

    # running evaluation
    cocoDt = cocoGt.loadRes('data.json')
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')

    imgIds = [i['image_id'] for i in complete_results]

    cocoEval.params.imgIds = imgIds

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()

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


def save_tensor_image(data):
    data = np.swapaxes(data, 0, 2)
    data = np.swapaxes(data, 0, 1)
    image = Image.fromarray(data)
    image.save(input_filename)


@app.route('/compute', methods=['POST'])
def compute():
    global elapsed_test, elapsed_reference, complete_results

    data = np.fromstring(request.data, dtype=np.uint8).reshape([3, 640, 640])
    image_id = request.headers['image_id']
    image_id = int(image_id.split('.jpg')[0])
    w = int(request.headers['w'])
    h = int(request.headers['h'])

    print(image_id)
    save_tensor_image(data)
    x = torch.tensor(data.astype(np.float32) / 255)

    output, elapsed_test, detections = detect(x, test_model, image_id, orig_w=w, orig_h=h, size=640)
    print(str(elapsed_test))
    output.save(output_filename)
    os.chmod(output_filename, 0o777)

    reference, elapsed_reference, detections_ref = detect(x, reference_model, image_id, orig_w=w, orig_h=h, size=640)
    print(str(elapsed_reference))
    reference.save(reference_filename)
    os.chmod(reference_filename, 0o777)

    complete_results += detections
    # print(complete_results)
    with open('data.json', 'w') as f:
        json.dump(complete_results, f)

    os.chmod('data.json', 0o777)


    return "OK"
