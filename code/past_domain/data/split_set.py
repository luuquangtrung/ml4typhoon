from data.dataset import MerraDataset
import pandas as pd
import time
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
import numpy as np 

def save_dataset_to_csv(dataset, file_name):
    """
    Save the dataset to a CSV file with 'Path' and 'Label' columns.

    Args:
    - dataset: MerraDataset object
    - file_name: Name of the CSV file to save
    """
    data_dict = {
        'Path': dataset.paths,
        'Filename': dataset.data['Filename'],
        'Year': dataset.data['Year'],
        'Label': dataset.labels,
    }
    df = pd.DataFrame(data_dict)
    df.sort_values(by='Filename', inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(file_name, index=False)
    print(f"Dataset saved to {file_name}")

def convert_timestamp_to_filename(timestamp, time_steps_back=0):
    try:
        # Parse the timestamp
        dt = datetime.strptime(timestamp, "%Y-%m-%d %H:%M:%S")
        
        # Calculate the new timestamp by going back the specified number of time steps
        # Each time step is 3 hours
        dt -= timedelta(hours=time_steps_back * 3)
        
        # Format into the desired filename
        filename = f"merra2_{dt.strftime('%Y%m%d_%H_00')}.nc"
        return filename
    except ValueError as e:
        print(f"Error parsing timestamp: {timestamp}. Ensure it is in 'YYYY-MM-DD HH:MM:SS' format.")
        return None

def undersample_data(data, label_column="Label", ratio=10):

    data_minority = data[data[label_column] == 1]
    data_majority = data[data[label_column] == 0]
    
    target_majority_count = len(data_minority) * ratio
    data_majority_sampled = data_majority.sample(n=target_majority_count, random_state=42)
    
    undersampled_data = pd.concat([data_minority, data_majority_sampled], axis=0)
    undersampled_data = undersampled_data.sample(frac=1, random_state=42).reset_index(drop=True)
    
    return undersampled_data

# Split the dataset
def split_and_normalize(csv_file, pos_ind, small_set, norm_type='new', under_sample=True, rus=10): 
    # Load the CSV and filter based on criteria
    data = pd.read_csv(csv_file)

    ibtracs_file = 'csv/FIRST_MERRA2_IBTRACS.csv'
    ibtracs_data = pd.read_csv(ibtracs_file)

    ibtracs_data = ibtracs_data[(ibtracs_data['LAT'] >= 0) &
                    (ibtracs_data['LAT'] <= 30) &
                    (ibtracs_data['LON'] >= 100) &
                    (ibtracs_data['LON'] <= 150)]
    
    ibtracs_filenames = set(
    ibtracs_data['ISO_TIME'].apply(lambda x: convert_timestamp_to_filename(x, time_steps_back=pos_ind))
    )

    data['Label'] = np.where(data['Filename'].isin(ibtracs_filenames), 1, data['Label'])

    # data["Path"] = data["Path"].str.replace("nasa-merra2", "nasa-merra2.old")

    # Filter out the samples near typhoon events if we use under sample
    if under_sample:
        data = data[data['Label'] != -1].reset_index(drop=True)
    
    data.loc[data['Label'] != 1, 'Label'] = 0

    if not small_set:
        train_data = data[data['Year'].between(1986, 2022)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'].between(1980, 1985)]
    else:
        train_data = data[data['Year'].between(2008, 2012)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'] == 2018]

    if under_sample:
        # Undersampling
        train_data = undersample_data(train_data, label_column="Label", ratio=rus)
        val_data = undersample_data(val_data, label_column="Label", ratio=rus)
        test_data = undersample_data(test_data, label_column="Label", ratio=rus)
    
        pos = train_data[train_data["Label"] == 1].sort_values(by=["Year","Filename"])
        neg = train_data[train_data["Label"] == 0].sort_values(by=["Year","Filename"])
        
        train_data = pd.DataFrame()
        for i in range(len(pos)):
            train_data = pd.concat([train_data,pos.iloc[i:i+1,:],neg.iloc[rus*i:rus*(i+1),:]],axis = 0)
    
    # Create dataset objects
    train_dataset = MerraDataset(train_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    val_dataset = MerraDataset(val_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    test_dataset = MerraDataset(test_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)

    save_dataset_to_csv(train_dataset, "csv/train_dataset.csv")
    save_dataset_to_csv(val_dataset, "csv/val_dataset.csv")
    save_dataset_to_csv(test_dataset, "csv/test_dataset.csv")

    return train_dataset, val_dataset, test_dataset


def split_and_normalize_fullmap(csv_file,
                                pos_ind, 
                                small_set, 
                                norm_type='old', 
                                under_sample=True, 
                                rus=10, 
                                strict=False): 
    
    # Load the CSV and filter based on criteria
    data = pd.read_csv(csv_file)

    ibtracs_file = 'csv/FIRST_MERRA2_IBTRACS.csv'
    ibtracs_data = pd.read_csv(ibtracs_file)

    ibtracs_data = ibtracs_data[(ibtracs_data['LAT'] >= 0) &
                    (ibtracs_data['LAT'] <= 30) &
                    (ibtracs_data['LON'] >= 100) &
                    (ibtracs_data['LON'] <= 150)]
    
    ibtracs_filenames = set(
    ibtracs_data['ISO_TIME'].apply(lambda x: convert_timestamp_to_filename(x, time_steps_back=pos_ind))
    )
    
    if strict:
        data['Label'] = np.where(data['Filename'].isin(ibtracs_filenames), 1, data['Label'])

    else:
        data['Label'] = 0
        matching_indices = data.index[data['Filename'].isin(ibtracs_filenames)]
        for idx in matching_indices:
            storm_idx = min(idx + pos_ind, len(data) - 1)
            data.loc[idx:storm_idx, 'Label'] = 1

    data.loc[data['Label'] != 1, 'Label'] = 0

    if not small_set:
        train_data = data[data['Year'].between(1986, 2022)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'].between(1980, 1985)]
    else:
        train_data = data[data['Year'].between(2008, 2012)]
        train_data, val_data = train_test_split(train_data, test_size=0.1, random_state=42)
        test_data = data[data['Year'] == 2018]

    if under_sample:
        # Undersampling
        train_data = undersample_data(train_data, label_column="Label", ratio=rus)
        val_data = undersample_data(val_data, label_column="Label", ratio=rus)
        test_data = undersample_data(test_data, label_column="Label", ratio=rus)
    
    # Create dataset objects
    train_dataset = MerraDataset(train_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    val_dataset = MerraDataset(val_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)
    test_dataset = MerraDataset(test_data, pos_ind=pos_ind, norm_type=norm_type, small_set=small_set)

    save_dataset_to_csv(train_dataset, "csv/train_dataset.csv")
    save_dataset_to_csv(val_dataset, "csv/val_dataset.csv")
    save_dataset_to_csv(test_dataset, "csv/test_dataset.csv")

    return train_dataset, val_dataset, test_dataset
