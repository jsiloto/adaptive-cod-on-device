import os
import stat
from flask import Flask, render_template, request, send_file

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False
app.config['UPLOAD_FOLDER'] = os.path.abspath("./photo/")
full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')

@app.route('/')
def show_index():
    return render_template("index.html", user_image=full_filename)

@app.route(full_filename)
def image():
    return send_file(full_filename, mimetype='image/jpg')

@app.route('/remote', methods=['POST'])
def remote():
    with open(full_filename, 'wb+') as f:
        f.write(request.data)

    os.chmod(full_filename, 0o777)
    # full_filename = os.path.join(app.config['UPLOAD_FOLDER'], 'shovon.jpg')
    return "Welcome to Flask!"
