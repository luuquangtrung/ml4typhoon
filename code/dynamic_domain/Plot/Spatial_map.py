import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as tkr

from mpl_toolkits.basemap import Basemap
from datetime import datetime
import matplotlib.patheffects as PathEffects
from matplotlib.colors import Normalize
from matplotlib.colors import LinearSegmentedColormap
from string import ascii_lowercase as alc
from scipy.stats import gaussian_kde

data = {}


for i, step in enumerate(range(2, 20, 2)):
    csv_path = f'/N/slate/tnn3/DucHGA/TC2/ModelMerra/Dynamic/ResNet/Output/History_RUS30_agg0_w6/Step_{step}_v2/map_hist.csv'
    csv_path = f'/N/slate/tnn3/DucHGA/TC2/ModelMerra/Dynamic/ResNet/Output/RUS30_agg0_w6_0.6/Step_{step}_v1/map_test.csv'
    csv_path = f'/N/slate/tnn3/DucHGA/TC2/ModelMerra/Dynamic/ResNet/Output/RUS30_agg0_w6_ckft/Step_{step}_v1/map_test.csv'
    
    try:
        df = pd.read_csv(csv_path)
        print(f'Processing: {csv_path}')
    except Exception as e:
        print(f'Error loading {csv_path}: {e}')
        continue
    
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df = df[df['Datetime'].dt.month.between(5, 11, inclusive='both')]
    df['Lat'] = df['Point'].str.split('_').str[0].astype(float)
    df['Lon'] = df['Point'].str.split('_').str[1].astype(float)
    df = df.groupby(['Lat', 'Lon'])['Score'].mean().reset_index(name='Score')
    df = df.sort_values(['Lat', 'Lon'], ascending=[False, True])
    data[step] = df
    
df = pd.read_csv('/N/u/tqluu/BigRed200/@PUBLIC/data_preprocessed/nasa-merra2/FIRST_MERRA2_IBTRACS.csv')
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df[['LAT', 'LON']] = df.loc[:, ['LAT', 'LON']].astype(float)
df['LAT'] = ((df['LAT'] + 2.5) // 5 * 5).round(0)
df['LON'] = ((df['LON'] - 100 + 2.5) // 5 * 5 + 100).round(0)
df = df[df['LAT'].between(0, 30, inclusive='both') &
                    df['LON'].between(100, 150, inclusive='both')]

df = df.rename(columns={'LAT': 'Lat',
                                    'LON': 'Lon',
                                    'ISO_TIME': 'Datetime'})
df = df[['Datetime', 'Lat', 'Lon']]
df = df[df['Datetime'].dt.year.between(2017, 2022, inclusive='both')]
df['Label'] = 1
df = df.groupby(['Lat', 'Lon'])['Label'].sum().reset_index(name='Score')
df_truth = df.copy()

gt = data[2].copy()
gt = gt[['Lat', 'Lon']]
gt = gt.merge(df_truth, on=['Lat', 'Lon'], how='left')
gt['Score'] = gt['Score'].fillna(0)

gt['Score'] = gt['Score'] / gt['Score'].sum()
gt.loc[gt['Score'] == 0, 'Score'] = -float('inf')

data[0] = gt

degree_sign = u'\N{DEGREE SIGN}'

# Define steps

# Create figure with 3x3 subplots
fig, axes = plt.subplots(3, 3, figsize=(24, 13), dpi=200)

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_colors = cmap(np.linspace(minval, maxval, n))
    return LinearSegmentedColormap.from_list(f"trunc({cmap.name},{minval:.2f},{maxval:.2f})", new_colors)
original_cmap = plt.get_cmap('twilight')
cropped_cmap = truncate_colormap(original_cmap, 0.0, 0.6)

for i, step in enumerate(range(0, 18, 2)):
    row, col = divmod(i, 3)
    ax = axes[row, col]
    # if i > 0:
    df = data[step]
    LAT_ARR = np.asarray(df['Lat'].drop_duplicates().values)
    LON_ARR = np.asarray(df['Lon'].drop_duplicates().values)
    arr = np.asarray(df['Score'].values).reshape((df['Lat'].nunique(), df['Lon'].nunique()))
    if i > 0:
        arr[arr < 0.015] = np.nan  # Set values below 0.015 to NaN
    else:
        arr[arr == 0] = np.nan  
    
    # Create Basemap
    lat_min, lat_max = -2.5, 32.5
    lon_min, lon_max = 97.5, 152.5
    map_ax = Basemap(ax=ax, projection='cyl',
                      llcrnrlat=lat_min, urcrnrlat=lat_max,
                      llcrnrlon=lon_min, urcrnrlon=lon_max,
                      suppress_ticks=False,
                      resolution='c')
    map_ax.drawcountries()
    map_ax.drawcoastlines()
    
    ax.set_xticks([])
    ax.set_yticks([])
    
    if col == 0:
        ax.set_yticks([0, 10, 20, 30])  # Hide y ticks
        ax.set_yticklabels([f'{i}{degree_sign}N' for i in [0, 10, 20, 30]], fontsize=25)
    if row == 2:
        ax.set_xticks([100, 110, 120, 130, 140, 150])
        ax.set_xticklabels([f'{i}{degree_sign}E' for i in [100, 110, 120, 130, 140, 150]], fontsize=25, rotation=30)
    
    # Plot values
    if i > 0:
        for idx, row in df.iterrows():
            if row['Lat'] == 30 and row['Lon'] < 115:
                continue
            
            if row['Score'] < 0.015:
                row['Score'] = -float('inf')
    # if row['Score'] > 0.11:
    # # if 0.06 < row['Score'] < 0.09:  # Highlight values in this range
    #     txt = ax.text(row['Lon'], row['Lat'], f'{row["Score"]:.2f}'.lstrip("0"), fontsize=19, ha='center', va='center', color='white')
    # elif row['Score'] > 0.015:
    #     txt = ax.text(row['Lon'], row['Lat'], f'{row["Score"]:.2f}'.lstrip("0"), fontsize=19, ha='center', va='center', color='black')
    # txt.set_path_effects([PathEffects.withStroke(linewidth=0.5, foreground='w')])
    cs = map_ax.pcolormesh(LON_ARR, LAT_ARR, arr, cmap=cropped_cmap, shading='auto')
    if i > 0:
        txt = ax.text(99, 29, f'{alc[step // 2]}) {(step * 3):02d}h', fontsize=30, ha='left', va='center')
    else:
        txt = ax.text(99, 29, 'a) IBTrACS', fontsize=30, ha='left', va='center')
    txt.set_bbox(dict(facecolor='pink', alpha=1, edgecolor='none', boxstyle='round,pad=0.3'))  # Background box for text
    cbar = fig.colorbar(cs, format=tkr.FormatStrFormatter('%.2f'))
    for t in cbar.ax.get_yticklabels():
        t.set_fontsize(30)

# Adjust layout and add colorbar
fig.tight_layout()
# cbar_ax = fig.add_axes([1, 0.15, 0.02, 0.7])
# cbar = fig.colorbar(cs, cax=cbar_ax)
# cbar.set_label('Prediction probability', fontsize=33)
# for t in cbar.ax.get_yticklabels():
#     t.set_fontsize(30)  # Set colorbar tick label size
plt.savefig('/N/slate/tnn3/DucHGA/Plot/Save/Fig6_expert.eps', bbox_inches='tight', format='eps')
plt.savefig('/N/slate/tnn3/DucHGA/Plot/Save/Fig6_enpert.pdf', bbox_inches='tight', format='pdf')

# plt.tight_layout()
plt.show()
