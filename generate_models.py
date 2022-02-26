from re import X
from string import punctuation
import requests
import urllib.parse
from pathlib import Path
import numpy as np
import json
import zipfile
import pickle

from punctuator import Punctuator
import gdown


CACHE_PATH = Path.home() / "cache"
CACHE_PATH.mkdir(exist_ok=True)
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

PUNCTUATOR_MODELS = [
    {
        "url": "http://ltdata1.informatik.uni-hamburg.de/subtitle2go/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl",
        "tests": [{"input": "hallo ich bin ein testsatz", "expected": "Hallo, ich bin ein testsatz."}]
    },
    # Rehosted from https://drive.google.com/drive/folders/0B7BsN5f2F1fZQnFsbzJ3TWxxMms?resourcekey=0-6yhuY9FOeITBBWWNdyG2aw
    {
        "url": "gdrive://1CZ_Os38LjBwyd-jgDMsfpqiWPB6wwVKA",
        "name": "Demo-EUROPARL-EN.zip",
        "pickle_encoding": "latin-1",
        "tests": [
            {
                "input": "hello this is an example sentence",
                "expected": "Hello, this is an example sentence.",
            }
        ],
    }
]


def download_model(model):
    if model["url"].startswith("gdrive://"):
        return download_gdrive_model(model)
    else:
        return download_http_model(model)


def download_gdrive_model(model):
    url_path = urllib.parse.urlparse(model["url"]).netloc
    output_model_file_path = MODEL_PATH / model["name"]
    input_model_file_path = CACHE_PATH / url_path
    if not input_model_file_path.exists():
        print("Downloading", url_path)
        gdown.download(id=url_path, output=str(input_model_file_path), fuzzy=True)
    return input_model_file_path, output_model_file_path


def download_http_model(model):
    url_path = urllib.parse.urlparse(model["url"]).path
    name = Path(url_path).name
    input_model_file_path = CACHE_PATH / name
    output_name = Path(url_path).with_suffix(".zip").name
    output_model_file_path = MODEL_PATH / output_name
    if not input_model_file_path.exists():
        print("Downloading", model["url"])
        req = requests.get(model["url"])
        with open(input_model_file_path, "wb") as f:
            f.write(req.content)

    return input_model_file_path, output_model_file_path


for model in PUNCTUATOR_MODELS:
    input_model_file_path, output_model_file_path = download_model(model)
    with open(input_model_file_path, "rb") as f:
        if 'pickle_encoding' in model:
            u = pickle._Unpickler(f)
            u.encoding = model['pickle_encoding']
            state = u.load()
        else:
            state = pickle.load(f)

    with zipfile.ZipFile(output_model_file_path, "w") as model_zip:
        for k, v in state.items():
            if (
                (isinstance(v, (list, tuple)))
                and v
                and np.ndarray in [type(x) for x in v]
            ):
                with model_zip.open(f"{k}.npyl", "w") as f:
                    for x in v:
                        np.save(f, x, allow_pickle=False)
            else:
                with model_zip.open(f"{k}.json", "w") as f:
                    f.write(json.dumps(v).encode())

    if "tests" in model and model["tests"]:
        punctuation_model = Punctuator(output_model_file_path)
        for test in model["tests"]:
            actual = punctuation_model.punctuate(test["input"])
            assert (
                actual == test["expected"]
            ), f"'{test['expected']}' expected, got {actual}"
