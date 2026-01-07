import os, glob, shutil
import datetime
import pandas as pd
import numpy as np

def gen_csv(
    slice_windows_path,
    data_type,
    run_opt,
    agg_steps,
    time_to_run,
    save_path,
):

    points = os.listdir(slice_windows_path)
    rsl = []
    for point in points:
        if run_opt == 0:
            for time_str in time_to_run:
                current = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                data_path = getFilePathFromTime(slice_windows_path + '/' + point, current)
                if os.path.exists(data_path):
                    rsl.append([current, point, data_path, 0])
        elif run_opt == 1:
            
            if data_type == 0: #FNL
                time_step_size = 6
            else: # merra
                time_step_size = 3

            for time_str in time_to_run:
                date_time = datetime.datetime.strptime(time_str, '%Y-%m-%d %H:%M:%S')
                for step in range(agg_steps):
                    current = date_time - datetime.timedelta(hours=step * time_step_size)
                    data_path = getFilePathFromTime(slice_windows_path + '/' + point, current)
                    if os.path.exists(data_path):
 
                        rsl.append([date_time, point, data_path, step * -1])

                 
        
    df = pd.DataFrame(rsl, columns = ['datetime', 'point', 'path', 'step'])
    
    # trong trường hợp aggregate, kiểm tra và loại bỏ những datetime không có đủ n step trước đó để aggregate
    if run_opt == 1:
        groups = df.groupby(['datetime', 'point']).count()
        indexes = groups[groups['step'] < agg_steps].index
        df = df[~df.set_index(['datetime', 'point']).index.isin(indexes)]

    df.to_csv(save_path, index=False) 

# hàm con phục vụ hàm gen_csv
def getFilePathFromTime(folder_path, date_time):
    year = str(date_time.year).zfill(4)
    month = str(date_time.month).zfill(2)
    day = str(date_time.day).zfill(2)
    hour = str(date_time.hour).zfill(2)
    data_path = folder_path + '/' + f'{year}{month}{day}_{hour}00.nc'
    return data_path


##########################################################################
from .Model.dataset import BaseDataset
from .Model.model import create_model 
import torch
from torch.utils.data import DataLoader

def model_predict(
        csv_path,
        data_type,
        run_opt,
        agg_steps,
        model_path,
        lead_time,
        prediction_results_path,
    ):

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # load data dựa trên file csv, cần sửa lại code trong phần dataset
    dataset = BaseDataset(
        csv_path, 
        data_type,
        run_opt,
        agg_steps,
    )

    # tạo và load model, cần sửa lại code trong phần model
    model = create_model()
    model.load_state_dict(torch.load(model_path, weights_only=True))

    model.to(device)
    model.eval()

    dataLoader = DataLoader(dataset,
        batch_size = 1,
        num_workers= 4,
        shuffle=False
    )

    # ước tính, lưu vào csv
    rsl = []
    for batch_input in dataLoader:
        date_time, point, input_tensor = batch_input

        input_tensor = input_tensor.to(device)

        pred = model(input_tensor)

        pred = pred.cpu().detach().numpy()

        pred = pred[0][1] # xác suất nhãn 1


        date_time = datetime.datetime.strptime(date_time[0], '%Y-%m-%d %H:%M:%S')
        time_to_predict = date_time + datetime.timedelta(hours = lead_time)

        point = point[0]

        rsl.append([date_time, time_to_predict, point, pred])
    

    df = pd.DataFrame(rsl, columns = ['input_time', 'predict_time', 'location', 'score'])
    
    df.to_csv(prediction_results_path, index=False)


# từ file csv output tạo ra map
def csv2nc(
    data_path, output_folder, 
    lat_min,lat_max,lon_min,lon_max,
    data_type,
    gen_pdf
    ):
    df = pd.read_csv(data_path)

    # tạo map theo từng thời gian
    groups = df.groupby(['input_time', 'predict_time'], as_index=False)
    
    for name, group in groups:
        input_time, predict_time = name

        df2nc(group, predict_time, output_folder, lat_min,lat_max,lon_min,lon_max, data_type, gen_pdf)
        

def df2nc(
    df, timeInfo, output_folder, 
    lat_min,lat_max,lon_min,lon_max,
    data_type,
    gen_pdf,
    ):
    print(timeInfo)
    # xây dựng lưới (grid) output, gán giá trị score vào các vị trí trên lưới
    timeInfo = datetime.datetime.strptime(timeInfo, '%Y-%m-%d %H:%M:%S')
    
    if data_type == 0: #FNL
        step_x = 0.1
        step_y = 0.1
        
    elif data_type == 1: #MERRA
        step_x = 0.0625
        step_y = 0.05

    a_point = df['location'].values[0]
    lat, lon = [float(x) for x in a_point.split('_')]
    

    dist_y = int((lat_max - lat)/step_y)
    dist_x = int((lon - lon_min)/step_x)

    # tìm tọa độ top left của lưới
    top_left_y = lat + dist_y * step_y
    top_left_x = lon - dist_x * step_x

    # tìm số hàng, cột (kích thước) của lưới
    nrows = len(np.arange(top_left_y, lat_min - step_y/2, -step_y))
    ncols = len(np.arange(top_left_x, lon_max + step_x/2, step_x))

    arr = np.zeros((nrows, ncols))

    # gán kết quả từ dataframe model predict lên lưới
    df['lat'] = df['location'].apply(lambda x: float(x.split('_')[0]))
    df['lon'] = df['location'].apply(lambda x: float(x.split('_')[1]))
    df = df[['lat', 'lon', 'score']]

    for rows in df.values:
        lat, lon, score = rows

        i = int((lat - lat_max) / -step_y)
        j = int((lon  - lon_min) / step_x)

        arr[i][j] = score


    # tạo dataframe từ lưới: lat, lon, score    
    rsl = []
    for i in range(nrows):
        for j in range(ncols):
            lat = lat_max - step_y * i
            lon = lon_min + step_x * j

            rsl.append([lat, lon, arr[i][j]])
    
    grid_df = pd.DataFrame(rsl, columns = ['lat', 'lon', 'score'])

    
    # convert dataframe -> xarray dataset
    grid_df.set_index(['lat', 'lon'], inplace=True)
    result_ds = xr.Dataset.from_dataframe(grid_df)

    # save to netcdf
    print(timeInfo)
    # return
    output_file_path = output_folder + '/' + f"{timeInfo.strftime('%Y%m%d_%H')}.nc"
    result_ds.to_netcdf(output_file_path)

    # tạo pdf
    if gen_pdf:
        output_plot_path = output_folder + '/' + f"{timeInfo.strftime('%Y%m%d_%H')}.pdf"
        Plot(result_ds, output_plot_path)


###########HUST plot code
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr

# code tạo pdf
def Plot(ds:xr.Dataset, savepath:str):
    """
    Plot the hurricane prediction data on a map.

    Parameters:
    - ds (xr.Dataset): The dataset containing the hurricane prediction data.
    - savepath (str): The path to save the generated plot.

    Returns:
    None
    """
    
    LAT_ARR = np.asarray(ds["lat"].values)
    LON_ARR = np.asarray(ds["lon"].values)
    data = np.asarray(ds['score'].values)
    # print(LAT_ARR)
    # print(LON_ARR)
    lat_min = np.min(LAT_ARR)
    lat_max = np.max(LAT_ARR)
    lon_min = np.min(LON_ARR)
    lon_max = np.max(LON_ARR)

    plt.figure(figsize=(16,9), dpi=100)
    m = Basemap(
        projection='cyl',llcrnrlat=lat_min,urcrnrlat=lat_max,\
        llcrnrlon=lon_min,urcrnrlon=lon_max,resolution='c'
        )

    # DATA
    cs = m.pcolormesh(LON_ARR, LAT_ARR, data)
    # LEGEND
    cbar = m.colorbar(cs, location='bottom', pad="10%")
    # cbar.set_label("Probability")

    # GRID
    m.drawparallels(np.arange(lat_min, lat_max+10, 5.), labels=[1,0,0,0], fontsize=5)
    m.drawmeridians(np.arange(lon_min, lon_max+10, 5.), labels=[0,0,0,1], fontsize=5)
    
    # BORDER
    m.drawcountries()

    # COASTLINE
    m.drawcoastlines()
    plt.title('Hurricane prediction at {time} using {model} of {predictor}')

    plt.savefig(savepath, bbox_inches='tight')


