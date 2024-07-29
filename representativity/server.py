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

from representativity.core import make_error_prediction

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


def get_arr_from_file(form_data_file) -> np.ndarray:
    user_filename = form_data_file.filename

    file_object = BufferedReader(form_data_file)
    # note we assume the .tiff or image is greyscale
    if ".tif" in user_filename:
        arr = imread(file_object)
    else:
        img = Image.open(file_object).convert("L")
        arr = np.asarray(img)
    arr = preprocess_arr(arr)
    return arr


def phase_fraction(request) -> Response:
    """User sends file, parse as array (either via tiffile or PIL-> numpy) and return all phase fractions.
    It's cheap to compute all of them and this way it can be done on first upload in the background and
    minimise delays."""
    user_file = request.files["userFile"]
    arr = get_arr_from_file(user_file)

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
    """phase fraction route."""
    response = generic_response(request, phase_fraction)
    return response


def preprocess_arr(arr: np.ndarray) -> np.ndarray:
    if len(arr.shape) == 3 and arr.shape[0] == 1:
        # weird (1, H, W) tiffs
        arr = arr[0, :, :]
    return arr


def representativity(request) -> Response:
    user_file = request.files["userFile"]
    arr = get_arr_from_file(user_file)

    selected_phase = int(request.values["selected_phase"])
    selected_conf: float = float(request.values["selected_conf"]) / 100
    selected_err: float = float(request.values["selected_err"]) / 100

    print(f"Phase: {selected_phase}, Conf: {selected_conf}, Err: {selected_err}")

    binary_img = np.where(arr == selected_phase, 1, 0)

    result = make_error_prediction(
        binary_img, selected_conf, selected_err, model_error=True
    )  # make_error_prediction(binary_img, selected_conf, selected_err)
    # this can get stuck sometimes in the optimisation step (usually cls > 1)
    out = {
        "abs_err": result["abs_err"],
        "percent_err": result["percent_err"] * 100,
        "std_model": result["std_model"],
        "l": result["l"],
        "cls": result["integral_range"],
        "pf_1d": result["pf_1d"],
        "cum_sum_sum": result["cum_sum_sum"],
    }

    response = Response(json.dumps(out), status=200)
    return response


@app.route("/repr", methods=["POST", "GET", "OPTIONS"])
def representativity_app():
    """representativity route."""
    response = generic_response(request, representativity)
    return response
