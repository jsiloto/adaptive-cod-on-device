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

from models import get_decoder, detect, pred2det, affine
from postprocess import detection2response
from request import RequestParser
from results import ResultManager
from images import ImageManager
sys.path.insert(0, './yolov5')
from yolov5.models.common import Detections
import torchvision.transforms as transforms

sys.path.insert(0, '../common')
import constants

encoder.FLOAT_REPR = lambda o: format(o, '.3f')

# print(cocoGt.__dict__)

app = Flask(__name__)

app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("temp/")
endpoint_filename = os.path.join(app.config['UPLOAD_FOLDER'], '<filename>')
ground_truth_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'image.png')
prediction_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'split_output.png')
results_filename = os.path.join(app.config['UPLOAD_FOLDER'], "data.jsonl")

decoder_model = get_decoder()

# Global State
global_results = ResultManager(results_filename)
image_manager = ImageManager(ground_truth_filename, prediction_filename)


@app.route('/test', methods=['post'])
def test():
    print(request.headers)
    print(request.data)
    return "OK"


@app.route('/map', methods=['DELETE'])
def clean_result():
    global_results.reset()
    return "OK"


@app.route('/map', methods=['GET'])
def get_result():
    stats = global_results.get()
    return {"stats": stats}


@app.route('/data/<filename>', methods=['POST'])
def post_data(filename=None):
    data = json.loads(request.data)
    os.makedirs("./temp/data/", exist_ok=True)
    full_path = os.path.join("./temp/data/", filename)
    with open(full_path, "w+") as f:
        json.dump(data, f)
    os.chmod(full_path, constants.safe_mode)
    return "OK"


@app.route('/')
def show_index():
    global elapsed_test, elapsed_reference
    return render_template("index.html",
                           input_image=ground_truth_filename,
                           split_result=prediction_filename,
                           output_time="",
                           reference_time="")


@app.route(endpoint_filename)
def image(filename=None):
    filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    return send_file(filename, mimetype='image/png')

@app.route('/split', methods=['POST'])
def split():
    request_parser = RequestParser(request)


    # print("#########################")
    print("[{}]Setting Width: {}".format(request_parser.image_id, request_parser.alpha))
    decoder_model.model.set_width(request_parser.alpha)
    # print("#########################")

    x = request_parser.dequantized_data
    # print(x)
    results = decoder_model(x.to('cuda'), request_parser.w, request_parser.h)
    # results = decoder_model(x, request_parser.w, request_parser.h)
    detections = pred2det(results.pred[0],
                          request_parser.image_id,
                          request_parser.w,
                          request_parser.h)

    # image_manager.update_ground_truth(request_parser.image_id)
    # image_manager.update_prediction(request_parser.image_id, results)
    global_results.update(detections)

    response = make_response(detection2response(detections), 200)
    response.mimetype = "text/plain"

    return response
