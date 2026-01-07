import pandas as pd
import calendar

lon_range = (100, 150)
lat_range = (0, 30)

grps = {}

for step in range(2, 20, 2):
    csv_path = f'map_predict.csv'
    print(f"Processing: {csv_path}")

    df = pd.read_csv(csv_path)
    df['Datetime'] = pd.to_datetime(df['Filename'], format='merra2_%Y%m%d_%H_00.nc')

    df['Datetime'] = pd.to_datetime(df['Datetime'])
    df['Month'] = df['Datetime'].dt.month

    df['Count'] = (df['Score'] > 0.5).astype(int)
    grp = df.groupby('Month')['Count'].sum().reset_index()
    grps[step] = grp

df = pd.read_csv('ibtracs.csv')
df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'])
df[['LAT', 'LON']] = df.loc[:, ['LAT', 'LON']].astype(float)
# df['LAT'] = ((df['LAT'] + 2.5) // 5 * 5).round(0)
# df['LON'] = ((df['LON'] - 100 + 2.5) // 5 * 5 + 100).round(0)
df = df[df['LAT'].between(0, 30, inclusive='both') &
                    df['LON'].between(100, 150, inclusive='both')]

df = df.rename(columns={'LAT': 'Lat',
                                    'LON': 'Lon',
                                    'ISO_TIME': 'Datetime'})
df = df[['Datetime', 'Lat', 'Lon']]
df = df[df['Datetime'].dt.year.between(1980, 1985, inclusive='both')]
df['Month'] = df['Datetime'].dt.month
df_truth = df.groupby('Month').size().reset_index(name='Count')
df_truth = df_truth.sort_values(by='Month')
df_truth.loc[-1] = [2, 0]
df_truth.index = df_truth.index + 1
df_truth = df_truth.sort_values('Month').reset_index(drop=True)
print(df_truth)

import os
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import calendar

# plt.style.use('ggplot')
plt.figure(figsize=(20, 10), dpi=200)

xticklabels = calendar.month_name[1:]  # Get month names from 1 to 12 (excluding empty string for index 0)

values = df_truth['Count'].values
values = values / values.sum()
plt.plot(range(12), values, marker='o', color='black', linestyle='-', linewidth=5, markersize=8)

# for step in range(8):
    
#     values = df[f'Model step t-{step * 2 + 2}'].values
#     values = values / values.sum()
#     plt.plot(range(12), values, marker='o', linestyle='-', linewidth=3, markersize=7)
    
for step in range(2, 18, 2):
    values = grps[step]['Count'].values
    values = values / values.sum()
    # Vẽ line plot
    plt.plot(range(12), values, marker='o', linestyle='-', linewidth=3, markersize=7)

# Cấu hình biểu đồ
plt.xticks(range(12), labels=xticklabels, fontsize=25, rotation=30)  # Sử dụng xticklabels để hiển thị tên tháng

plt.yticks(fontsize=25)  # Thiết lập kích thước chữ cho trục y
plt.ylabel('Density of Positive prediction\n', fontsize=30)
# plt.title('POS SAMPLE Prediction by Month in Study Area', fontsize=17)
legend_title = ['IBTrACS']
legend_title.extend([f"{(i * 6):02d}h" for i in range(1, 9)])
legend = plt.legend(
    legend_title,
    title='Lead time forecast',
    ncol = 3,
    bbox_to_anchor=(0.02, 1),
    loc='upper left',
    fontsize = 25)  # Thêm chú thích (legend)
legend.get_title().set_fontsize(30)  # Thiết lập kích thước chữ cho tiêu đề của legend
plt.grid(axis='y', zorder=0, alpha=0.7)
plt.gca().spines[['right', 'top']].set_visible(False)
plt.gca().spines[['left', 'bottom']].set_linewidth(1)

# Lưu hình
# plt.savefig(plot_path, bbox_inches='tight')
plt.savefig('Output.pdf', bbox_inches='tight', format='pdf')  # Save the figure to a file

plt.show()
plt.close()

# print(f"Saved combined plot: {plot_path}")
