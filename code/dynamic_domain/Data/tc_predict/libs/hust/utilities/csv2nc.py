import os
import xarray as xr
import pandas as pd
import numpy as np

def PipelineTC_csv2nc(predict_file:str, output_file_path:str) -> xr.Dataset:
    result_df = pd.read_csv(
        filepath_or_buffer=predict_file,
        delimiter=",",
        usecols=["lat","lon","sample_path", "score"]
    )
    if len(result_df) == 0:
        return None

    for ind in result_df.index:
        samplepath = result_df['sample_path'][ind]
        la, lo = (float(i) for i in str(samplepath).split("/")[0].split("_"))
        sco = float(result_df['score'][ind])
        result_df.at[ind, 'score'] = sco
        result_df.at[ind, 'lat'] = la
        result_df.at[ind, 'lon'] = lo

    full_data = []
    for i in np.arange(0, 30, 0.5):
        for j in np.arange(100, 150, 0.625):
            full_data.append([i, j, 0])
    full_df = pd.DataFrame(full_data, columns = ['lat', 'lon', 'score'])
        
    result_df.drop(columns=["sample_path"], inplace=True)

    result_df = pd.concat([result_df, full_df])

    result_df = result_df.sort_values(by = ['lat', 'lon', 'score'], ascending=[True, True, False])

    result_df = result_df.drop_duplicates(subset = ['lat', 'lon'])
    
    result_df.set_index(['lat', 'lon'], inplace=True)
    
    result_ds = xr.Dataset.from_dataframe(result_df)
    
    result_ds.to_netcdf(output_file_path)
    
    return result_ds