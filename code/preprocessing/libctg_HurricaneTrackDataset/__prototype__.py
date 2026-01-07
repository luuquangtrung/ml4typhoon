import xarray as xr
import pandas as pd
import datetime

from .__prototype__ import *


class HurricaneTrack:
    def __init__(self):
        pass

    # This function is used to load raw *.csv from data provider
    # then merge them as one panda.dataframe
    def LoadRawCSVs(self, paths: list[str]) -> pd.DataFrame:
        pass

    # This function is used to load the processed csv file
    # that is saved before
    # The saved file has format: SID,ISO_TIME,BASIN,NAME,LAT,LON
    def LoadFromDisk(self, path: str) -> pd.DataFrame:
        dataframe = pd.read_csv(filepath_or_buffer=path, sep=",", date_format="%Y-%m-%d %H:%M:%S",
                                parse_dates=[1], index_col=False, low_memory=False)
        return dataframe

    def LoadBatch(self, files:list[str]) -> pd.DataFrame:
        df_list = [self.LoadFromDisk(f) for f in files]
        df = self.Merge(df_list)
        return df
    
    # This function is used to process the loadded raw dataframe
    def ProcessRaw(self, dataframe: pd.DataFrame, filter_first: bool = True) -> pd.DataFrame:
        pass

    # This function is used to save the processed dataframe
    # return 0 if success, otherwise fail
    # Save format: SID,ISO_TIME,BASIN,NAME,LAT,LON
    def SaveToDisk(self, path: str, dataframe: pd.DataFrame) -> int:
        try:
            dataframe.to_csv(path_or_buf=path, sep=",", index=False)
            return 0
        except:
            return -1

    # This function is used to process the loadded raw dataframe
    def ProcessRaw(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        pass

    # Split
    def Split(self, df:pd.DataFrame, batch_size:int=0) -> list[pd.DataFrame]:
        if not batch_size:
            return df
        else:
            return [df[i:i+batch_size] for i in range(0, df.shape[0], batch_size)]
    
    # Merge
    def Merge(self, df_list: list[pd.DataFrame]) -> pd.DataFrame:
        return pd.concat(df_list)
