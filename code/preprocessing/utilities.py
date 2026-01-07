import os
import shutil
import glob
from fnmatch import fnmatch
from itertools import islice
import xarray as xr
import datetime
import numpy as np

import libctg_HurricaneTrackDataset.ibtracs as ibtracs
import libctg_WeatherDataset.ncep_fnl as ncep_fnl


def RecurseListDir(root: str, pattern: list[str]):
    f = []
    for p in pattern:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, p):
                    f.append(os.path.join(path, name))
    return f


def IterChunk(arr_range, arr_size) -> list[str]:
    arr_range = iter(arr_range)
    return list(iter(lambda: tuple(islice(arr_range, arr_size)), ()))


def CleanDir(path: str):
    if not os.path.exists(path):
        os.makedirs(path)
        return
    for file_obj in os.listdir(path):
        file_obj_path = os.path.join(path, file_obj)
        if (os.path.isfile(file_obj_path)) or (os.path.islink(file_obj_path)):
            os.unlink(file_obj_path)
        else:
            shutil.rmtree(file_obj_path)