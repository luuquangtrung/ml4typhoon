import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import numpy as np
import datetime
import time

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from internals import MERRA2_IBTRACS, FNL_IBTRACS, MultiProcessing

def SequenceArea(WD: WeatherDataset, wd_path:str, output_path:str, nstep:int=2):
    print("SequenceArea")
    w_ds = WD.LoadFromDisk(wd_path)
    # lat_min = np.min(np.asarray(w_ds["latitude"].values))
    # lat_max = np.max(np.asarray(w_ds["latitude"].values))
    # lon_min = np.min(np.asarray(w_ds["longitude"].values))
    # lon_max = np.max(np.asarray(w_ds["longitude"].values))
    lat_min = FindNearest(w_ds["latitude"].values, configs.SequenceArea.MIN_LAT)
    lat_max = FindNearest(w_ds["latitude"].values, configs.SequenceArea.MAX_LAT)
    lon_min = FindNearest(w_ds["longitude"].values, configs.SequenceArea.MIN_LON)
    lon_max = FindNearest(w_ds["longitude"].values, configs.SequenceArea.MAX_LON)
    # Find first area center point at top left
    lat_c = lat_min + int(WD.DIM_LAT/2)*(WD.STEP_LAT)
    lon_c = lon_max - int(WD.DIM_LON/2)*(WD.STEP_LON)
    # Calculate max value of lat at center point
    lat_c_max = lat_max - int(WD.DIM_LAT/2)*(WD.STEP_LAT)
    # Calculate min value of lon at center point
    lon_c_min = lon_min + int(WD.DIM_LON/2)*(WD.STEP_LON)

    # Get date and time from file path
    # example path: "./data/temp/ncep-fnl/fnl_20070725_12_00.grib1.nc"
    filename = wd_path.split("/")[4].split(".")[0] # fnl_20070725_12_00
    tmp_date = filename.split("_")[1] 
    tmp_hour = filename.split("_")[2]
    tmp_minute = filename.split("_")[3]
    tmp_year = tmp_date[0:4]
    tmp_month = tmp_date[4:6]
    tmp_day = tmp_date[6:8]

    if int(tmp_year) < configs.SequenceArea.MIN_YEAR:
        return

    # Get date and time from file name
    date_time = datetime.datetime(int(tmp_year), int(tmp_month), int(tmp_day), int(tmp_hour), int(tmp_minute))
    date_c = date_time.date()
    time_c = date_time.time()

    # Resolve the symlink to get the actual target path
    real_path = os.path.realpath(output_path)

    # Create sub folder
    sub_path = os.path.join(real_path, filename)
    # os.mkdir(sub_path)

    if os.access(real_path, os.W_OK):  # Check write permission
        # Create the subdirectory if permission granted
        os.mkdir(sub_path)
    else:
        print(f"Error: Permission denied to create directory in {real_path}")
        return
    
    if not os.access(sub_path, os.W_OK):  # Check write permission
        print(f"Error: Permission denied to create directory in {sub_path}")
        return


    # Move center point from left to right, top to bottom
    lat_start  = lat_c
    while lat_start <= lat_c_max:
        lon_index = lon_c  # Reset lon_index for each latitude
        while lon_index >= lon_c_min:
            s_ds = WD.GetSample(w_ds, "", lat_c, lon_c, date_c, time_c, negative_type="SEQUENCE_AREA")
            if not s_ds:
                print("not s_ds")
                lon_index -= nstep * WD.STEP_LON
                continue

            save_path = os.path.join(sub_path, f"{filename}_{lat_start}_{lon_index}.nc")
            ret = WD.SaveToDisk(save_path, s_ds)
            if ret != 0: 
                print(f"save to disk error: {ret}")
                return
            lon_index -= nstep * WD.STEP_LON

        lat_start += nstep * WD.STEP_LAT # Move to the next latitude

def Worker(queue:mp.Queue, WD:WeatherDataset, output_path:str, nstep:int=2):
    while (queue.qsize()):
        wds_file = queue.get()
        if not wds_file:
            print("not wds_file")
            break
        SequenceArea(WD, wds_file, output_path, nstep)
        pass
    print("Done.")
    exit()

def SequenceArea_Main(WD: WeatherDataset, PrepDir:str, OutDir:str, nworker:int=2, nstep:int=configs.SequenceArea.STEP_COUNT):
    wds_files = utilities.RecurseListDir(PrepDir, ["*.nc"])
    queue = mp.Queue()
    for f in wds_files:
        queue.put(f)
        pass
    MultiProcessing(Worker, (queue, WD, OutDir, nstep), nworker)
    return

def Fnl():
    print("NCEP-FNL dataset extraction")
    FNL, IBTRACS  = FNL_IBTRACS()
    out_dir = os.path.join(configs.paths.FNL_IBTRACS_OUT, "SequenceArea")
    utilities.CleanDir(out_dir)
    SequenceArea_Main(FNL, configs.paths.FNL_IBTRACS_PREP, out_dir, configs.nworkers.FNL_SEQUENCE_AREA)
    print("NCEP-FNL dataset extraction: Done.")
    return

def Merra2():
    print("NASA-MERRA2 dataset extraction")
    MERRA2, IBTRACS  = MERRA2_IBTRACS()
    out_dir = os.path.join(configs.paths.MERRA2_IBTRACS_OUT, "SequenceArea")
    utilities.CleanDir(out_dir)
    SequenceArea_Main(MERRA2, configs.paths.MERRA2_IBTRACS_PREP, out_dir, configs.nworkers.MERRA2_SEQUENCE_AREA)
    print("NASA-MERRA2 dataset extraction: Done.")
    return

def testExtract():
    print("testExtract started")
    FNL, IBTRACS  = FNL_IBTRACS()
    out_dir = os.path.join(configs.paths.FNL_IBTRACS_OUT, "SequenceArea")
    utilities.CleanDir(out_dir)
    wds_files = utilities.RecurseListDir(configs.paths.FNL_IBTRACS_PREP, ["*.nc"])
    SequenceArea(FNL, "./data/temp/ncep-fnl/fnl_20070725_12_00.grib1.nc", out_dir)
    sq_files = utilities.RecurseListDir("./data/out/ncep-fnl/SequenceArea/fnl_20070725_12_00", ["*.nc"])
    print(len(sq_files))
    print("testExtract finished")

if __name__=="__main__":
    print("Sequence Area")
    Fnl()
    Merra2()
    print("Sequence Area: Done.")
    # start_time = time.time()
    # testExtract()
    # end_time = time.time()

    # runtime = end_time - start_time
    # print(f"Runtime: {runtime:.4f} seconds")
    


