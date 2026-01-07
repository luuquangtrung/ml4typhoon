import pandas as pd

from Utils.Map_evaluate import *
print("Map_eval loaded.")
# csv_temp = '/N/slate/tnn3/DucHGA/TC/ModelMerra/Output/DynamicSingleFix/ResNet/Dataset1_{}_v0/map_out.csv'

csv_temp = 

suffix = ''
rus = 4
agg = 0
weight = 0.8
version = 0
cols = ['PRS', 'RCL', 'F1S', 'Predict']
col1 = ['PRS_1', 'RCL_1', 'F1S_1', 'Predict_1']
col0 = ['PRS_0', 'RCL_0', 'F1S_0', 'Predict_0']
t = col1.copy()
t.extend(col0)
scoreboard = pd.DataFrame(columns=t)

for step in range(2, 20, 2):
    print(step)
    csv_path = csv_temp.format(step, version)
    try:
        pd.read_csv(csv_path)
    except:
        continue

    result = Model_result(csv_path, step_forecast=step)
    df = result.location_acc(dt_list=[
        (datetime(2017, 1, 1), datetime(2023, 1, 1)),
        # (datetime(2017, 5, 1), datetime(2017, 12, 1)),
        # (datetime(2018, 5, 1), datetime(2018, 12, 1)),
        # (datetime(2019, 5, 1), datetime(2019, 12, 1)),
        # (datetime(2020, 5, 1), datetime(2020, 12, 1)),
        # (datetime(2021, 5, 1), datetime(2021, 12, 1)),
        # (datetime(2022, 5, 1), datetime(2022, 12, 1)),
    ])
    
    print(df)

    scoreboard.loc[step, col1] = df.loc['1', cols].values
    scoreboard.loc[step, col0] = df.loc['0', cols].values
    
scoreboard.to_excel(f'path')
    