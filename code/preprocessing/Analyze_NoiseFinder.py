import os
import multiprocessing as mp
import pandas as pd
import xarray as xr
import datetime
import math
import numpy as np

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import libctg_WeatherDataset.utilities as WD_utilities
import configs
from internals import MERRA2_IBTRACS, FNL_IBTRACS, MultiProcessing

def Distance2d(A:tuple[float, float], B:tuple[float, float]) -> float:
    return math.sqrt(abs(A[0]-B[0])**2 + abs(A[1]-B[1])**2)

def NoiseAnnotate(flag:bool) -> str:
    if flag is None:
        return "FAR"
    return "IN" if flag else "NEAR"

def CheckNoise(WD: WeatherDataset, HT: HurricaneTrack, htdf: pd.DataFrame, wd:xr.Dataset) -> tuple[bool, list[tuple[str, str, float]]]:
    wd_sid = wd.attrs["SID"]
    wd_lat_grid = wd["latitude"].values
    wd_lon_grid = wd["longitude"].values
    ex_wd_lat_grid = WD_utilities.ExtendAxis(wd_lat_grid, WD.STEP_LAT, deg_count=configs.NoiseFinder.LAT_THRESHOLD)
    ex_wd_lon_grid = WD_utilities.ExtendAxis(wd_lon_grid, WD.STEP_LON, deg_count=configs.NoiseFinder.LON_THRESHOLD)
    wd_lat_max = np.max(wd_lat_grid)
    wd_lat_min = np.min(wd_lat_grid)
    wd_lon_max = np.max(wd_lon_grid)
    wd_lon_min = np.min(wd_lon_grid)
    wd_lat_c = WD_utilities.findMiddle(wd_lat_grid.tolist())
    wd_lon_c = WD_utilities.findMiddle(wd_lon_grid.tolist())
    wd_timestamp = datetime.datetime.strptime(wd.attrs["ISO_TIME"], "%Y-%m-%d %H:%M:%S")
    # max_lat = np.arange(wd_lat_max, wd_lat_max + configs.NoiseFinder.LAT_THRESHOLD * WD.STEP_LAT, WD.STEP_LAT).max()
    # min_lat = np.arange(wd_lat_min - configs.NoiseFinder.LAT_THRESHOLD * WD.STEP_LAT, wd_lat_min, WD.STEP_LAT).min()
    # max_lon = np.arange(wd_lon_max, wd_lon_max + configs.NoiseFinder.LON_THRESHOLD * WD.STEP_LON, WD.STEP_LON).max()
    # min_lon = np.arange(wd_lon_min - configs.NoiseFinder.LON_THRESHOLD * WD.STEP_LON, wd_lon_min, WD.STEP_LON).min()
    # max_lat = np.arange(wd_lat_max, wd_lat_max + configs.NoiseFinder.LAT_THRESHOLD + WD.STEP_LAT, WD.STEP_LAT).max()
    # min_lat = np.arange(wd_lat_min - configs.NoiseFinder.LAT_THRESHOLD, wd_lat_min + WD.STEP_LAT, WD.STEP_LAT).min()
    # max_lon = np.arange(wd_lon_max, wd_lon_max + configs.NoiseFinder.LON_THRESHOLD + WD.STEP_LON, WD.STEP_LON).max()
    # min_lon = np.arange(wd_lon_min - configs.NoiseFinder.LON_THRESHOLD, wd_lon_min + WD.STEP_LON, WD.STEP_LON).min()
    # max_lat = wd_lat_max + configs.NoiseFinder.LAT_THRESHOLD
    # min_lat = wd_lat_min - configs.NoiseFinder.LAT_THRESHOLD
    # max_lon = wd_lon_max + configs.NoiseFinder.LON_THRESHOLD
    # min_lon = wd_lon_min - configs.NoiseFinder.LON_THRESHOLD
    max_lat = ex_wd_lat_grid.max()
    min_lat = ex_wd_lat_grid.min()
    max_lon = ex_wd_lon_grid.max()
    min_lon = ex_wd_lon_grid.min()
    htdf_c = htdf.loc[
        (htdf["ISO_TIME"] == wd_timestamp)
        & (htdf["LAT"] <= max_lat) & (htdf["LAT"] >= min_lat)
        & (htdf["LON"] <= max_lon) & (htdf["LON"] >= min_lon)
    ]
    if htdf_c.empty:
        return NoiseAnnotate(False), None
    isNoise = NoiseAnnotate(False)
    noiseList = []
    for i in htdf_c.index:
        id = htdf_c["SID"][i]
        if (id == wd_sid):
            continue
        lat_c = htdf_c["LAT"][i]
        lat_c = FindNearest(ex_wd_lat_grid, lat_c)
        lon_c = htdf_c["LON"][i]
        lon_c = FindNearest(ex_wd_lon_grid, lon_c)
        dis = Distance2d((lat_c, lon_c), (wd_lat_c, wd_lon_c))
        if (lat_c <= wd_lat_max and lat_c >= wd_lat_min and lon_c <= wd_lon_max and lon_c >= wd_lon_min):
            isNoise = NoiseAnnotate(True)
            noiseList.append((id, NoiseAnnotate(True), dis))
            continue
        noiseList.append((id, NoiseAnnotate(False), dis))
        continue
    return isNoise, noiseList

def Worker(queue:mp.Queue, WD:WeatherDataset, HT:HurricaneTrack, htdf:pd.DataFrame, output_path:str):
    while (queue.qsize()):
        wd_filepath  = queue.get()
        if not wd_filepath:
            break
        wd = WD.LoadFromDisk(wd_filepath)
        isNoise, noiseList = CheckNoise(WD, HT, htdf, wd)
        noiseStr = ""
        if noiseList:
            noiseStr = "-".join(f"{n[0]}@{n[1]}@{n[2]}" for n in noiseList)
            with open(output_path, "at") as f:
                f.write(f"{wd_filepath}, {isNoise}, {noiseStr}\n")
        continue
    print("Done.")
    exit()

def NoiseFinder_Main(WD: WeatherDataset, HT: HurricaneTrack, PrepDir:str, CheckDir:str, OutPath:str, nworker:int=1):
    ht_files = utilities.RecurseListDir(PrepDir, ["FULL_*.csv"])
    htdf = HT.LoadBatch(ht_files)
    wd_files = utilities.RecurseListDir(CheckDir, ["NEGATIVE_*.nc"])
    queue = mp.Queue()
    for f in wd_files:
        queue.put(f)
    MultiProcessing(Worker, (queue, WD, HT, htdf, OutPath), nworker)
    return

def Fnl():
    print("NCEP-FNL NoiseFinder")
    FNL, IBTRACS = FNL_IBTRACS()
    out_path = configs.paths.FNL_IBTRACS_NOISE
    utilities.CleanDir(os.path.dirname(out_path))
    NoiseFinder_Main(FNL, IBTRACS, configs.paths.FNL_IBTRACS_PREP, configs.paths.FNL_IBTRACS_OUT, out_path, configs.nworkers.FNL_NOISE)
    print("NCEP-FNL NoiseFinder: Done.")
    return

def Merra2():
    print("NASA-MERRA2 NoiseFinder")
    MERRA2, IBTRACS = MERRA2_IBTRACS()
    out_path = configs.paths.MERRA2_IBTRACS_NOISE
    utilities.CleanDir(os.path.dirname(out_path))
    NoiseFinder_Main(MERRA2, IBTRACS, configs.paths.MERRA2_IBTRACS_PREP, configs.paths.MERRA2_IBTRACS_OUT, out_path, configs.nworkers.MERRA2_NOISE)
    print("NASA-MERRA2 NoiseFinder: Done.")
    return

def Main():
    print("Noise Finder")
    Fnl()
    Merra2()
    print("Noise Finder: Done")
    return

if __name__=="__main__":
    Main()
