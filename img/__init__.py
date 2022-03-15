import os

IMG_PATH = os.path.dirname(__file__)


def get_path(scriptpath, filename):
    scriptname = os.path.basename(scriptpath)[:-3]
    path = os.path.join(IMG_PATH, scriptname)
    if not os.path.exists(path):
        os.mkdir(path)
    return os.path.join(path, filename)