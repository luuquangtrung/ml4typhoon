import os
import shutil
import glob
from fnmatch import fnmatch
import datetime

def RecurseListDir(root: str, pattern: list[str]):
    f = []
    for p in pattern:
        for path, subdirs, files in os.walk(root):
            for name in files:
                if fnmatch(name, p):
                    f.append(os.path.join(path, name))
    return f

def getListDir(input_path, time_to_run, data_type):

    rsl = []
    for time_str in time_to_run:
        current = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')

        year = str(current.year).zfill(4)
        month = str(current.month).zfill(2)
        day = str(current.day).zfill(2)
        hour = str(current.hour).zfill(2)

        if data_type == 0: # FNL
            fpath = input_path + '/' + f'fnl_{year}{month}{day}_{hour}_00.grib2.nc'
        else: # MERRA2
            fpath = input_path + '/' + f'merra2_{year}{month}{day}_{hour}_00.nc' # merra2_20211021_12_00.nc
        if os.path.exists(fpath):
            rsl.append(fpath)

    return rsl


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
            
