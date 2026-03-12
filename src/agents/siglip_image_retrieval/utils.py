import json


def load_metadata(path):

    with open(path, "r") as f:
        return json.load(f)