import main as m
from flask import Flask, request, jsonify
from flask_cors import CORS
import uuid
import os

app = Flask(__name__)
CORS(app)

ALLOWED_EXTENSIONS = ["jpg", "jpeg", "bmp", "png"]


@app.route("/")
def hello():
    return "Hello World!"


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/api/predict', methods=['POST'])
def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify([
            {"message": "No files found"}
        ])

    file = request.files['file']
    if file and allowed_file(file.filename):
        filename = uuid.uuid4().hex + ".png"
        file.save(os.path.join("input", filename))
        results = m.predict(filename)

        return jsonify(results)

    return False


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=80)