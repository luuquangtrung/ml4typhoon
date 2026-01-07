import pandas as pd
import numpy as np

from math import nan
from pathlib import Path
from datetime import datetime, timedelta

from .Metrics import *

class Model_result():

    """
    Module đánh giá map

    ### Khởi tạo lớp:
    - **csv_path** (`Path`): Đường dẫn đến file CSV chứa score của model.
    - **step_forecast** (`int`, mặc định=2): Dự đoán cho bn step tiếp theo
    - **step_resolution** (`timedelta`, mặc định=3 giờ): Mỗi step ứng với bn thời gian

    ### Phương thức:

    #### detection_acc
    Đánh giá khả năng nhận biết bão cho toàn bộ map.
    
    - **score** (`float`, mặc định=0.5): Ngưỡng positive/negative cho từng sample.
    - **count** (`float`, mặc định=0.1): Nếu số ô positive trên map vượt quá `count`, map đó được dự đoán có bão.
    - **dt_first** (`datetime`, mặc định=datetime(2021, 5, 1)): Ngày bắt đầu khoảng thời gian đánh giá.
    - **dt_last** (`datetime`, mặc định=datetime(2021, 12, 1)): Ngày kết thúc khoảng thời gian đánh giá.
    - **out_path** (`Path`, mặc định=None): Đường dẫn lưu file kết quả đánh giá.
    
    #### location_acc
    Đánh giá độ chính xác vị trí dự báo bão.
    
    - **score** (`float`, mặc định=0.5): Ngưỡng phân loại positive/negative cho từng sample.
    - **dt_first** (`datetime`, mặc định=datetime(2021, 5, 1)): Ngày bắt đầu khoảng thời gian đánh giá.
    - **dt_last** (`datetime`, mặc định=datetime(2021, 12, 1)): Ngày kết thúc khoảng thời gian đánh giá.
    - **out_path** (`Path`, mặc định=None): Đường dẫn lưu file kết quả đánh giá.

    """

    def __init__(self,
                 csv_path: Path,
                 step_forecast: int = 2,
                 step_resolution: timedelta = timedelta(hours=3)):

        self.step_forecast = step_forecast
        self.step_resolution = step_resolution
        
        df_first = pd.read_csv('path-to-first-track.csv')
        df_first['ISO_TIME'] = pd.to_datetime(df_first['ISO_TIME'])
        df_first[['LAT', 'LON']] = df_first.loc[:, ['LAT', 'LON']].astype(float)
        df_first['LAT'] = ((df_first['LAT'] + 2.5) // 5 * 5).round(0)
        df_first['LON'] = ((df_first['LON'] - 100 + 2.5) // 5 * 5 + 100).round(0)
        df_first = df_first[df_first['LAT'].between(0, 30, inclusive='both') &
                            df_first['LON'].between(100, 150, inclusive='both')]
        
        df_first = df_first.rename(columns={'LAT': 'Lat',
                                            'LON': 'Lon',
                                            'ISO_TIME': 'Datetime'})
        df_first = df_first[['Datetime', 'Lat', 'Lon']]
        self.df_first = df_first
        
        df_full = pd.read_csv('path-to-full-track.csv')
        df_full['ISO_TIME'] = pd.to_datetime(df_full['ISO_TIME'])
        df_full[['LAT', 'LON']] = df_full.loc[:, ['LAT', 'LON']].astype(float)
        df_full['LAT'] = ((df_full['LAT'] + 2.5) // 5 * 5).round(0)
        df_full['LON'] = ((df_full['LON'] - 100 + 2.5) // 5 * 5 + 100).round(0)
        df_full = df_full[df_full['LAT'].between(0, 30, inclusive='both') &
                          df_full['LON'].between(100, 150, inclusive='both')]
        
        df_full = df_full.rename(columns={'LAT': 'Lat',
                                            'LON': 'Lon',
                                            'ISO_TIME': 'Datetime'})
        df_full = df_full[['Datetime', 'Lat', 'Lon']]
        self.df_full = df_full
        
        df = pd.read_csv(csv_path)
        df['Datetime'] = pd.to_datetime(df['Datetime'])
        df['Lat'] = df['Point'].str.split('_').str[0].astype(float)
        df['Lon'] = df['Point'].str.split('_').str[1].astype(float)
        
        # assign label
        df_first['Datetime'] = df_first['Datetime'] - step_resolution * step_forecast
        df_first['True'] = True
        df = df.merge(df_first, how='left', on=['Datetime', 'Lat', 'Lon'])
        df.loc[df['True'].isna(), 'True'] = False
        df.loc[df['True'], 'Label'] = 1
        df = df.drop(columns='True')
        self.df = df
        df_first['Datetime'] = df_first['Datetime'] + step_resolution * step_forecast
    
    def compute_metric(self, true, pred,
                       out_path: Path = None):
        
        scoreboard = pd.DataFrame()
        
        for metric in [ACC]:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
    
        for metric in [PRS, RCL, F1S]:
            scoreboard.loc['All', metric.__name__] = metric(true, pred)
            
            for _class in np.unique(true):
                scoreboard.loc[str(_class), metric.__name__] = metric(true, pred, label=_class)
        
        scoreboard['Support'] = 0
        scoreboard['Predict'] = 0
        scoreboard['Score'] = 0
        scoreboard.loc['All', 'Support'] = len(true)
        scoreboard.loc['All', 'Predict'] = len(pred)
        for _class in np.unique(true):
            scoreboard.loc[str(_class), 'Support'] = len(true[true == _class])
            scoreboard.loc[str(_class), 'Predict'] = len(true[pred == _class])
        
        if out_path is not None:
            scoreboard.to_excel(out_path)
        
        return scoreboard
    
    def detection_acc(self,
                      score: float = 0.5,
                      count: float = 0.1,
                      dt_first: datetime = datetime(2021, 5, 1),
                      dt_last: datetime = datetime(2021, 12, 1),
                      dt_list: list = None,
                      out_path: Path = None):
    
        df = self.df.copy()
        
        if dt_list is not None:
            dt_arr = []
            for dt_first, dt_last in dt_list:
                dt_iter = dt_first
                while dt_iter <= dt_last:
                    dt_arr.append(dt_iter)
                    dt_iter += self.step_resolution
            dt_arr = np.array(dt_arr)
            df = df[df['Datetime'].isin(dt_arr - self.step_resolution * self.step_forecast)]
            
            
        else:
            df = df[df['Datetime'].between(
                dt_first - self.step_resolution * self.step_forecast, 
                dt_last - self.step_resolution * self.step_forecast, 
                inclusive='both',
            )]
        
        df['Score'] = (df['Score'] > score).astype(int)
        grp = df.groupby(['Datetime'])['Label'].max().reset_index(name='Label')
        df = df.drop(columns='Label')
        grp = grp.merge(df.groupby(['Datetime'])['Score'].mean().reset_index(name='Score'), on=['Datetime'])
        df = grp

        df['Label'] = 0
        df.loc[df['Datetime'].isin((self.df_full['Datetime'] - self.step_resolution * self.step_forecast).unique()), 'Label'] = nan
        df.loc[df['Datetime'].isin((self.df_first['Datetime'] - self.step_resolution * self.step_forecast).unique()), 'Label'] = 1
        
        true = df.loc[~ df['Label'].isna(), 'Label'].astype(int).values
        pred = (df.loc[~ df['Label'].isna(), 'Score'] > count).astype(int).values
        
        return self.compute_metric(true, pred, out_path)
    
    def location_acc(self,
                     score: float = 0.5,
                     dt_first: datetime = datetime(2021, 5, 1),
                     dt_last: datetime = datetime(2021, 12, 1),
                     dt_list: list = None,
                     out_path: Path = None):
        
        df = self.df.copy()
        
        if dt_list is not None:
            dt_arr = []
            for dt_first, dt_last in dt_list:
                dt_iter = dt_first
                while dt_iter <= dt_last:
                    dt_arr.append(dt_iter)
                    dt_iter += self.step_resolution
            dt_arr = np.array(dt_arr)
            df = df[df['Datetime'].isin(dt_arr - self.step_resolution * self.step_forecast)]
            
        else:
            df = df[df['Datetime'].between(
                dt_first - self.step_resolution * self.step_forecast, 
                dt_last - self.step_resolution * self.step_forecast, 
                inclusive='both',
            )]
        
        df['Score'] = (df['Score'] > score).astype(int)
        df = df[df['Datetime'].isin(df[df['Label'] == 1]['Datetime'].values)]

        true = df.loc[~ df['Label'].isna(), 'Label'].astype(int).values
        pred = df.loc[~ df['Label'].isna(), 'Score'].astype(int).values
        
        return self.compute_metric(true, pred, out_path)
        