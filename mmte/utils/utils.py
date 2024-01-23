import os
from mmte import lib_path

def get_abs_path(rel):
    return os.path.join(lib_path, rel)