from flask import Flask

app = Flask(__name__)
app.config['JSONIFY_PRETTYPRINT_REGULAR'] = False


@app.route('/')
def test():
    return "Welcome to Flask!"


@app.route('/remote', methods=['POST'])
def remote():
    return "Welcome to Flask!"
