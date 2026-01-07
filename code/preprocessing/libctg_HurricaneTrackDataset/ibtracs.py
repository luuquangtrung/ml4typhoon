from .__prototype__ import *

class Ibtracs(HurricaneTrack):
    def __init__(self):
        self.DATETIME_FORMAT = "%Y-%m-%d %H:%M:%S"
        self.TIME_POINTS_HOURS = [0,6,12,18]
        self.TIME_POINTS_MINS = [0]
        self.TIME_POINTS_SECS = [0]
        self.DATETIME_COL_INX = ["ISO_TIME"]
        self.SKIP_ROWS = [1]
        self.SEPERATOR = ","
        self.COLUMNS = ["SID", "ISO_TIME", "LAT", "LON", "BASIN","SUBBASIN"]
        self.FILTERS = [
            ("BASIN", ["WP"])
        ]

    # This function is used to load raw *.csv from data provider
    # then merge them as one panda.dataframe
    def LoadRawCSVs(self, paths:list[str]) -> pd.DataFrame:
        df_list = [
            pd.read_csv(filepath_or_buffer=p, sep=self.SEPERATOR, date_format=self.DATETIME_FORMAT, parse_dates=self.DATETIME_COL_INX, index_col=False, low_memory=False, skiprows=self.SKIP_ROWS)
            for p in paths
        ]
        dataframe = pd.concat(df_list)
        return dataframe

    # This function is used to process the loadded raw dataframe
    def ProcessRaw(self, dataframe:pd.DataFrame, filter_first:bool=True) -> pd.DataFrame:
        temp_df = dataframe
        # Remove time points
        temp_df = temp_df.where(
            temp_df["ISO_TIME"].dt.time.isin([datetime.time(h, m, s) for h in self.TIME_POINTS_HOURS for m in self.TIME_POINTS_MINS for s in self.TIME_POINTS_SECS])
        )
        # Get only the first record
        if filter_first:
            temp_df = temp_df.sort_values(by="ISO_TIME", ascending=False).groupby("SID").tail(1).sort_values(by="SID", ascending=True)
        # Only in West Pacific WP
        for f in self.FILTERS:
            temp_df = temp_df.where(
                temp_df[f[0]].isin(f[1])
            )
        # Get only columns of interest
        if (self.COLUMNS):
            temp_df = temp_df[self.COLUMNS]
        temp_df.dropna(inplace=True)
        return temp_df