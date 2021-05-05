import traceback
from flask import Flask, request, jsonify
from network import LPRnet, IMG_SIZE, infer_shaped_image
import cv2
import numpy as np

app = Flask(__name__)
CHECKPOINT = None


def init_lprnet(checkpoint):
    global CHECKPOINT
    CHECKPOINT = checkpoint


def scan_plate(file):
    filestr = file.read()
    #convert string data to numpy array
    npimg = np.fromstring(filestr, np.uint8)
    # convert numpy array to image
    img = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)

    img_w = IMG_SIZE[0]
    img_h = IMG_SIZE[1]

    # img = cv2.imread(fname)

    img = cv2.resize(img, (img_w, img_h))
    img_batch = np.expand_dims(img, axis=0)
    plates = infer_shaped_image(CHECKPOINT, img_batch)
    return {
        "plates": plates
    }

@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/scan', methods=['POST'])
def upload_file():
    f = request.files['plate']
    r = scan_plate(f)
    print(r)
    return r

"""@app.errorhandler(Exception)
def handle_exception(e):
    return jsonify(stackTrace=traceback.format_exc())

"""
