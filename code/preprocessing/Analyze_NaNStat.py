import xarray as xr
import os
import sys
import numpy as np
from multiprocessing import Queue, Process, set_start_method
from alive_progress import alive_bar
import time
from tqdm import tqdm

import libctg_WeatherDataset.nasa_merra2 as WeatherDataset
import utilities
from configs.NaNStat import *

def Worker(queue:Queue):
    while (queue.qsize()):
        file, output_csv = queue.get()
        if not file:
            break
        ds = xr.load_dataset(file, engine="netcdf4")
        filename = os.path.basename(file)
        res = []
        for var in ds.data_vars:
            is_mul_level = ds[str(var)].indexes.dims.get("isobaricInhPa", None)
            
            if not is_mul_level:
                name = str(var)
                data = ds[name].values
                nan = np.isnan(data)[np.isnan(data) == True].size
                res.append(f"{filename}, {name}, null, {data.size}, {nan}, {round(float(nan)/data.size, 2)}\n")
            else:
                name = str(var)
                data = ds[name].values
                lev = ds[name].indexes.get("isobaricInhPa")
                for l_i in range(int(lev.size)-1):
                    try:
                        sub_data = ds.sel(isobaricInhPa=lev[l_i])[name].values
                        nan = np.isnan(sub_data)[np.isnan(sub_data) == True].size
                        res.append(f"{filename}, {name}, {round(lev[l_i], 2)}, {sub_data.size}, {nan}, {round(float(nan)/sub_data.size, 2)}\n")
                    except Exception as e:
                        print(e)
        with open(output_csv, "at") as f:
            f.writelines(res)
    exit()

def Main():
    utilities.CleanDir(OUTPUT_DIR)
    queue = Queue()
    for task in TASKS_LIST:
        files_dir, output_name = task
        output_path = os.path.join(OUTPUT_DIR, output_name)
        files_list = utilities.RecurseListDir(files_dir, "*.nc")
        with open(output_path, mode="wt") as f:
            f.write("filename, varname, level, size, nancount, nanratio\n")
        for filepath in files_list:
            queue.put((filepath, output_path))
    ps = []
    total_size = queue.qsize()
    for i in range(WORKERS_COUNT):
        p = Process(
            target=Worker,
            args=(queue,)
        )
        p.start()
        ps.append(p)
    with alive_bar(total=total_size, title="nancount", manual=True, force_tty=PROGESS_ANIMATION) as bar: #Enable animation by force_tty=True
        while any(p.exitcode == None for p in ps):
            bar(float(1-queue.qsize()/float(total_size) - 0.01))
            time.sleep(1)
        bar(1)
    for p in ps:
        p.join()

if __name__=="__main__":
    Main()