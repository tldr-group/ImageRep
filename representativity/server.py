from flask import (
    Flask,
    request,
    make_response,
    jsonify,
    send_file,
    Response,
    send_from_directory,
)
import numpy as np
import json
from io import BytesIO, BufferedReader
from tifffile import imread
from PIL import Image
from typing import Callable

URL_WHITELIST = [
    "https://sambasegment.z33.web.core.windows.net",
    "http://www.sambasegment.com",
    "https://www.sambasegment.com",
    "https://sambasegment.azureedge.net",
    "http://localhost:8080",
    "https://localhost:8080",
]

app = Flask(
    __name__,
)
"""
python -m flask --app server run
"""


def add_cors_headers(response):
    request_url = request.headers["Origin"]  # url_root  # headers["Origin"]
    if request_url in URL_WHITELIST:
        response.headers.add("Access-Control-Allow-Origin", request_url)
        response.headers.add("Access-Control-Allow-Headers", "*")
        response.headers.add("Access-Control-Allow-Methods", "*")
    return response


def _build_cors_preflight_response():
    response = make_response()
    response = add_cors_headers(response)
    return response


@app.route("/")
def hello_world():
    """Not used except to check app working."""
    return send_from_directory("", "index.html")


def generic_response(request, fn: Callable):
    """Given a HTTP request and response function, return corsified response."""
    if "OPTIONS" in request.method:
        return _build_cors_preflight_response()
    elif "POST" in request.method:
        try:
            response = fn(request)
            return add_cors_headers(response)
        except Exception as e:
            print(e)
            response = Response(f"{{'msg': '{e}' }}", 400, mimetype="application/json")
            return add_cors_headers(response)
    else:
        response = jsonify(success=False)
        return add_cors_headers(response)


def phase_fraction(request) -> Response:
    """User sends file, parse as array (either via tiffile or PIL-> numpy) and return all phase fractions.
    It's cheap to compute all of them and this way it can be done on first upload in the background and
    minimise delays."""
    user_file = request.files["userFile"]
    user_filename = user_file.filename

    file_object = BufferedReader(user_file)
    # note we assume the .tiff or image is greyscale
    if ".tif" in user_filename:
        arr = imread(file_object)
    else:
        img = Image.open(file_object).convert("L")
        arr = np.asarray(img)

    out_fractions = {}
    for val in np.unique(arr):
        key = int(val)
        out_fractions[key] = np.mean(arr == key)
    print(f"Phase fractions: {out_fractions}")

    obj = {"phase_fractions": out_fractions}
    response = Response(json.dumps(obj), status=200)
    return response


@app.route("/phasefraction", methods=["POST", "GET", "OPTIONS"])
def phase_fraction_app():
    """Init route."""
    response = generic_response(request, phase_fraction)
    return response
