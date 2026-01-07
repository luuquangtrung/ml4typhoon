import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import xarray as xr

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
    pass
