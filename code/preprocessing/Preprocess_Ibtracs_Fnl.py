import os
import multiprocessing as mp
import pandas as pd

from libctg_HurricaneTrackDataset import *
from libctg_WeatherDataset import *
import utilities
import configs
from internals import FNL_IBTRACS, MultiProcessing

FNL, IBTRACS = FNL_IBTRACS()

def Preprocess_Fnl(file:str, output_path:str):
    filename = os.path.basename(file)
    ds = FNL.LoadRaw([file])
    ds = FNL.ProcessRaw(ds)
    save_path = os.path.join(output_path, f"{filename}.nc")
    FNL.SaveToDisk(save_path, ds)

def Worker(queue:mp.Queue, output_path:str):
    while (queue.qsize()):
        file = queue.get()
        if not file:
            break
        Preprocess_Fnl(file, output_path)
    print("Done.")
    exit()

def Preprocess_Fnl_Main():
    print("Preprocess Fnl")
    files = utilities.RecurseListDir(configs.paths.FNL_RAW, ["*.grib1", "*.grib2"])
    queue = mp.Queue()
    for f in files:
        queue.put(f)
    MultiProcessing(Worker, (queue, configs.paths.FNL_IBTRACS_PREP), configs.nworkers.FNL_PREP)
    print("Preprocess Fnl: Done.")
    return 0

def Preprocess_Ibtracs_Main():
    print("Preprocess Ibtracs")
    files = utilities.RecurseListDir(configs.paths.IBTRACS_RAW, ["*.csv"])
    df_raw = IBTRACS.LoadRawCSVs(files)
    df_full = IBTRACS.ProcessRaw(df_raw, filter_first=False)
    df_first = IBTRACS.ProcessRaw(df_raw)
    df_full_path = os.path.join(configs.paths.FNL_IBTRACS_PREP, "FULL_FNL_IBTRACS.csv")
    df_first_path = os.path.join(configs.paths.FNL_IBTRACS_PREP, "FIRST_FNL_IBTRACS.csv")
    IBTRACS.SaveToDisk(df_full_path, df_full)
    IBTRACS.SaveToDisk(df_first_path, df_first)
    print("Preprocess Ibtracs: Done")
    return 0

def Main():
    print("Preprocess IBTRACS-FNL")
    utilities.CleanDir(configs.paths.FNL_IBTRACS_PREP)
    Preprocess_Ibtracs_Main()
    Preprocess_Fnl_Main()
    print("Preprocess IBTRACS-FNL: Done.")

if __name__=="__main__":
    Main()