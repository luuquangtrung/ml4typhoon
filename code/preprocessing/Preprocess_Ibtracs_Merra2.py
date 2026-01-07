import os
import multiprocessing as mp
import pandas as pd
import datetime

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from internals import MERRA2_IBTRACS, MultiProcessing

MERRA2, IBTRACS = MERRA2_IBTRACS()

def PreprocessMerra2(file, output_path):
    ds = MERRA2.LoadRaw([file])
    ds = MERRA2.ProcessRaw(ds)
    ds_list = MERRA2.Split(ds)
    for sub_ds in ds_list:
        timestamp = datetime.datetime.fromisoformat(str(sub_ds.attrs["ISO_TIME"]))
        filename = timestamp.strftime("%Y%m%d_%H_%M")
        filepath_to_save = os.path.join(output_path, f"merra2_{filename}.nc")
        MERRA2.SaveToDisk(filepath_to_save, sub_ds)
    return 0

def Worker(queue: mp.Queue, output_path:str):
    while (queue.qsize()):
        file = queue.get()
        if not file:
            break
        PreprocessMerra2(file, output_path)
    print("Done.")
    exit()

def Preprocess_Merra2_Main():
    print("Preprocess Merra2")
    files = utilities.RecurseListDir(configs.paths.MERRA2_RAW, ["*.nc4"])
    queue = mp.Queue()
    for f in files:
        queue.put(f)
    MultiProcessing(Worker, (queue, configs.paths.MERRA2_IBTRACS_PREP), configs.nworkers.MERRA2_PREP)
    print("Preprocess Merra2: Done")
    pass

def Preprocess_Ibtracs_Main():
    print("Preprocess Ibtracs")
    files = utilities.RecurseListDir(configs.paths.IBTRACS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_full = IBTRACS.ProcessRaw(df_raw, filter_first=False)
    df_first = IBTRACS.ProcessRaw(df_raw)
    df_full_path = os.path.join(
        configs.paths.MERRA2_IBTRACS_PREP, "FULL_MERRA2_IBTRACS.csv")
    df_first_path = os.path.join(
        configs.paths.MERRA2_IBTRACS_PREP, "FIRST_MERRA2_IBTRACS.csv")
    IBTRACS.SaveToDisk(df_full_path, df_full)
    IBTRACS.SaveToDisk(df_first_path, df_first)
    print("Preprocess Ibtracs: Done")
    return 0

def Main():
    print("Preprocess IBTRACS-MERRA2")
    utilities.CleanDir(configs.paths.MERRA2_IBTRACS_PREP)
    Preprocess_Ibtracs_Main()
    Preprocess_Merra2_Main()
    print("Preprocess IBTRACS-MERRA2: Done.")


if __name__ == "__main__":
    Main()
