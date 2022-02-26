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

CACHE_PATH = Path.home() / "cache"
CACHE_PATH.mkdir(exist_ok=True)
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(exist_ok=True)

PUNCTUATOR_MODELS = [
    {
        "url": "http://ltdata1.informatik.uni-hamburg.de/subtitle2go/Model_subs_norm1_filt_5M_tageschau_euparl_h256_lr0.02.pcl",
        "tests": [{"input": "hallo ich bin ein testsatz", "expected": "Hallo, ich bin ein testsatz."}]
    }
]

for model in PUNCTUATOR_MODELS:
    url_path = urllib.parse.urlparse(model['url']).path
    name = Path(url_path).name
    input_model_file_path = CACHE_PATH / name
    output_name = Path(url_path).with_suffix(".zip").name
    output_model_file_path = MODEL_PATH / output_name
    if not input_model_file_path.exists():
        req = requests.get(model['url'])
        with open(input_model_file_path, "wb") as f:
            f.write(req.content)

    with open(input_model_file_path, "rb") as f:
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

    if 'tests' in model and model['tests']:
        punctuation_model = Punctuator(output_model_file_path)
        for test in model['tests']:
            actual = punctuation_model.punctuate(test['input'])
            assert actual == test['expected']