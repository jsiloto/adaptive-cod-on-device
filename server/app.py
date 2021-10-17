import json
import math
import os
import sys
from json import encoder
from os.path import exists

import numpy as np
import torch
from PIL import Image
from flask import Flask, render_template, request, send_file, make_response
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from models import get_models, detect, pred2det, affine
from postprocess import detection2response
from request import RequestParser

sys.path.insert(0, './yolov5')
from yolov5.models.common import Detections
import torchvision.transforms as transforms

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

annFile = '../resource/dataset/coco2017/annotations/instances_subval2017.json'
cocoGt = COCO(annFile)

# print(cocoGt.__dict__)

app = Flask(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("temp/")
endpoint_filename = os.path.join(app.config['UPLOAD_FOLDER'], '<filename>')
input_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
split_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'split_output.png')
full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'output.png')
results_filename = os.path.join(app.config['UPLOAD_FOLDER'], "data.json")

full_model, decoder_model = get_models()

# Global State
complete_results = []
elapsed_test = 0
elapsed_reference = 0


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
    if not exists(results_filename):
        return {"stats": []}

    cocoDt = cocoGt.loadRes(results_filename)
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    imgIds = [i['image_id'] for i in complete_results]
    cocoEval.params.imgIds = imgIds

    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    stats = cocoEval.stats.tolist()
    stats = [math.ceil(v * 1000) / 1000 for v in stats]
    return {"stats": stats}


@app.route('/')
def show_index():
    global elapsed_test, elapsed_reference

    try:
        # running evaluation
        cocoDt = cocoGt.loadRes(results_filename)
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
                           split_result=split_filename,
                           full_result=full_filename,
                           output_time=str(elapsed_test),
                           reference_time=str(elapsed_reference))


@app.route(endpoint_filename)
def image(filename=None):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filename, mimetype='image/png')



def save_input_image(request_parser):
    path = request_parser.image_path
    image_id = request_parser.image_id
    image = Image.open(path)
    orig_w, orig_h = image.size

    ids = cocoGt.getAnnIds(imgIds=[image_id])
    id_vals = cocoGt.loadAnns(ids=ids)

    vals = [{'bbox': affine(v['bbox'], orig_w, orig_h, size=768),
             'category_id': v['category_id']} for v in id_vals]

    image = [np.asarray(image)]
    y = [torch.tensor([v['bbox'][:] + [1.0, v['category_id'] - 1] for v in vals])]

    a = Detections(image, y, path, [], decoder_model.names, image[0].shape)
    a.render()
    Image.fromarray(a.imgs[0]).save(input_filename)


def save_base_detection(path):
    image = Image.open(path)
    image.save(full_filename)


def save_split_detection(path, results):
    im = Image.open(path)
    results.imgs = [np.asarray(im)]
    results.render()
    im = Image.fromarray(results.imgs[0])
    im.save(split_filename)


def update_global_results(detections):
    global complete_results
    complete_results += detections
    mode = 'w'
    if not exists(results_filename):
        mode = 'w+'

    with open(results_filename, mode) as f:
        json.dump(complete_results, f)
    os.chmod(results_filename, 0o777)


@app.route('/split', methods=['POST'])
def split():
    request_parser = RequestParser(request)
    save_input_image(request_parser)
    save_base_detection(request_parser.image_path)

    # print("#########################")
    # print("[{}]Setting Width: {}".format(request_parser.image_id, request_parser.alpha))
    decoder_model.model.set_width(request_parser.alpha)
    # print("#########################")

    x = request_parser.dequantized_data
    results = decoder_model(x, request_parser.w, request_parser.h)

    save_split_detection(request_parser.image_path, results)

    detections = pred2det(results.pred[0],
                          request_parser.image_id,
                          request_parser.w,
                          request_parser.h)

    update_global_results(detections)

    response = make_response(detection2response(detections), 200)
    response.mimetype = "text/plain"

    return response
