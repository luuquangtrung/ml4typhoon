import os
import xarray as xr
import numpy as np
import pandas as pd
from datetime import datetime

def process_merra2_stats(merra_folder, output_excel='csv/data_statistics.xlsx'):
    PRESS_VAR = ["H", "OMEGA", "QI", "QL", "QV", "RH", "T", "U", "V"]
    SINGLE_VAR = ['PHIS', 'PS', 'SLP']
    PRESS_LEVELS = [str(p) for p in range(1000, 75, -25)]  # ['1000', '975', ..., '100']
    LEVEL = 21  # 1000 hPa - 100 hPa

    stats_dict = {}

    for var in PRESS_VAR:
        for level in PRESS_LEVELS:
            stats_dict[f"{var}_{level}"] = []

    for var in SINGLE_VAR:
        stats_dict[var] = []

    for filename in os.listdir(merra_folder):
        if filename.endswith(".nc"):
            filepath = os.path.join(merra_folder, filename)
            try:
                ds = xr.open_dataset(filepath)

                for var in PRESS_VAR:
                    if var in ds.variables:
                        data = ds.variables[var].data[:LEVEL]  # (level, lat, lon)
                        data = np.nan_to_num(data)
                        for i, level in enumerate(PRESS_LEVELS):
                            flattened = data[i].flatten()
                            stats_dict[f"{var}_{level}"].extend(flattened.tolist())

                for var in SINGLE_VAR:
                    if var in ds.variables:
                        data = ds.variables[var].data
                        data = np.nan_to_num(data).flatten()
                        stats_dict[var].extend(data.tolist())

                ds.close()
            except Exception as e:
                print(f"error {filename}: {e}")

    rows = []
    for var, values in stats_dict.items():
        values = np.array(values)
        if values.size == 0:
            continue
        mean = np.mean(values)
        std = np.std(values)
        varr = np.var(values)
        rows.append({'Variable': var, 'Mean': mean, 'Std': std, 'Variance': varr})

    df = pd.DataFrame(rows)
    df = df.sort_values(by='Variable')
    df.to_excel(output_excel, index=False)


def create_first_merra2_ibtracs(ibtracs_file, output_file='FIRST_MERRA2_IBTRACS.csv'):
    # Read the input CSV
    df = pd.read_csv(ibtracs_file, parse_dates=['ISO_TIME'], dayfirst=False, infer_datetime_format=True)
    df['ISO_TIME'] = pd.to_datetime(df['ISO_TIME'], errors='coerce')
    df = df.dropna(subset=['ISO_TIME'])

    # Sort by time and keep only the first storm (by SID)
    df_first = df.sort_values(by='ISO_TIME').groupby('SID', as_index=False).first()

    df_first = df_first[['SID', 'ISO_TIME', 'LAT', 'LON', 'BASIN', 'SUBBASIN']]
    df_first['ISO_TIME'] = df_first['ISO_TIME'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_first.to_csv(output_file, index=False)


def build_merra_label_file(ibtracs_file, merra_folder, output_file='merra_full_new.csv'):

    df_ib = pd.read_csv(ibtracs_file, parse_dates=['ISO_TIME']) 
    df_ib['MERRA_TIME'] = df_ib['ISO_TIME'].dt.strftime('%Y%m%d_%H_00')
    

    df_first = df_ib.sort_values(by='ISO_TIME').groupby('SID', as_index=False).first()
    first_times = set(df_first['MERRA_TIME'].tolist())
    
    all_storm_times = set(df_ib['MERRA_TIME'].tolist())

    storm_but_not_first = all_storm_times - first_times

    records = []

    for filename in sorted(os.listdir(merra_folder)):
        if filename.endswith('.nc') and filename.startswith('merra2_'):
            full_path = os.path.join(merra_folder, filename)
            
            parts = filename.replace('.nc', '').split('_')  # ['merra2', '19800101', '00', '00']
            if len(parts) != 4:
                continue
            
            merra_time_str = f"{parts[1]}_{parts[2]}_00"

            year = int(parts[1][:4])
            
            if merra_time_str in storm_but_not_first:
                label = -1
            else:
                label = ''
            
            records.append({
                'Path': full_path,
                'Filename': filename,
                'Year': year,
                'Label': label
            })

    # Save to CSV
    df_out = pd.DataFrame(records)
    df_out.to_csv(output_file, index=False)

def main():
    merra_folder = "/data"
    ibtracs_file = "/csv/IBTRACS.csv"

    process_merra2_stats(merra_folder, output_excel='/csv/data_statistics.xlsx')
    create_first_merra2_ibtracs(ibtracs_file, output_file='/csv/FIRST_MERRA2_IBTRACS.csv')
    build_merra_label_file(ibtracs_file, merra_folder, output_file='/csv/merra_full_new.csv')


if __name__ == '__main__':
    main()
