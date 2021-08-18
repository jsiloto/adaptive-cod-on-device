import os
import stat
import numpy as np
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
jpg_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
png_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.png')

@app.route('/')
def show_index():
    return render_template("index.html", jpg_image=jpg_filename, png_image=png_filename)

@app.route(jpg_filename)
def jpg_image():
    return send_file(jpg_filename, mimetype='image/jpg')

@app.route(png_filename)
def png_image():
    return send_file(png_filename, mimetype='image/png')

@app.route('/jpg', methods=['POST'])
def jpg():
    with open(jpg_filename, 'wb+') as f:
        f.write(request.data)

    os.chmod(jpg_filename, 0o777)
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return "Welcome to Flask!"

@app.route('/png', methods=['POST'])
def png():
    from PIL import Image
    data = np.fromstring(request.data, dtype=np.int8).reshape([3, 480, 640])
    print(data.shape)
    return "OK"
    im = Image.fromarray()
    im.save(jpg_filename)
    # with open(png_filename, 'wb+') as f:
    #     f.write(request.data)

    os.chmod(jpg_filename, 0o777)
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return "Welcome to Flask!"

