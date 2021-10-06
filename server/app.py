import os
import json
import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, send_file, make_response
from models import get_models, detect, pred2det
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from yolov5.models.common import AutoShapeDecoder, AutoShapeEncoder, AutoShape
from postprocess import detection2response
from models import invert_afine
import sys
sys.path.insert(0, '../common')
from tensor_utils import dequantize_tensor, QuantizedTensor

annFile = '../resource/dataset/coco2017/annotations/instances_val2017.json'
cocoGt = COCO(annFile)

app = Flask(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
endpoint_filename = os.path.join(app.config['UPLOAD_FOLDER'], '<filename>')
input_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
output_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
reference_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'reference.png')
results_filename = "data.json"

reference_model, test_model = get_models()

# Global State
complete_results = []
elapsed_test = 0
elapsed_reference = 0

# encoder = torch.jit.load('effd2_encoder.ptl')
decoder = torch.jit.load('effd2_decoder.ptl')


decoder = AutoShapeDecoder(decoder)


decoder.stride = torch.tensor([8., 16., 32.])
decoder.names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat',
                 'traffic light', 'fire hydrant', '', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
                 'horse',
                 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', '', 'backpack', 'umbrella', '', '', 'handbag',
                 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
                 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', '', 'wine glass', 'cup',
                 'fork', 'knife',
                 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', '', 'dining table', '', '', 'toilet', '',
                 'tv',
                 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                 'refrigerator', '', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier',
                 'toothbrush']


@app.route('/test', methods=['post'])
def test():
    print(request.headers)
    print(request.data)
    return "OK"

@app.route('/map', methods=['DELETE'])
def clean_result():
    global complete_results
    complete_results = []
    # Handle errors while calling os.remove()
    try:
        os.remove(results_filename)
    except:
        print("Error while deleting file ", results_filename)
        open(results_filename, "w+")
    return "OK"


@app.route('/map', methods=['GET'])
def get_result():
    global complete_results
    cocoDt = cocoGt.loadRes(results_filename)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    imgIds = [i['image_id'] for i in complete_results]
    cocoEval.params.imgIds = imgIds

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return {
        "stats": cocoEval.stats.tolist()
    }


@app.route('/')
def show_index():
    global elapsed_test, elapsed_reference

    try:
    # running evaluation
        cocoDt = cocoGt.loadRes('data.json')
        cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
        imgIds = [i['image_id'] for i in complete_results]
        cocoEval.params.imgIds = imgIds
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()
    except:
        pass

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

    # reference, elapsed_reference, detections_ref = detect(x, reference_model, image_id, orig_w=w, orig_h=h, size=640)
    # print(str(elapsed_reference))
    # reference.save(reference_filename)
    # os.chmod(reference_filename, 0o777)

    complete_results += detections

    with open('data.json', 'w') as f:
        json.dump(complete_results, f)

    os.chmod('data.json', 0o777)

    response = make_response(detection2response(detections), 200)
    response.mimetype = "text/plain"

    return response


@app.route('/split', methods=['POST'])
def split():
    global complete_results
    data = np.fromstring(request.data, dtype=np.uint8)
    alpha = len(data)/(1*48*80*80)
    print("#########################")
    print("Setting Width: {}".format(alpha))
    print("#########################")
    decoder.model.set_width(alpha)
    data = data.reshape([1, int(48*alpha), 80, 80])
    # data = np.fromstring(request.data, dtype=np.uint8).reshape([1, 3, 640, 640])

    image_id = request.headers['image_id']
    print(image_id)
    image_path = os.path.join("../resource/dataset/coco2017/val2017/", image_id)
    image_id = int(image_id.split('.jpg')[0])
    w = int(request.headers['w'])
    h = int(request.headers['h'])
    scale = float(request.headers['scale'])
    zero_point = float(request.headers['zero_point'])

    x = QuantizedTensor(tensor=torch.tensor(data), scale=scale, zero_point=zero_point)
    # print(x)
    x = dequantize_tensor(x)
    results = decoder(x)

    from yolov5.models.common import Detections
    im = Image.open(image_path).resize((640, 640), Image.ANTIALIAS)
    results.imgs = [np.asarray(im)]
    results.render()
    im = Image.fromarray(results.imgs[0])
    im.save(reference_filename)

    detections = pred2det(results.pred[0], image_id, w, h)
    complete_results += detections
    with open('data.json', 'w') as f:
        json.dump(complete_results, f)
    response = make_response(detection2response(detections), 200)
    response.mimetype = "text/plain"

    return response
