# ml4typhoon
From Reanalysis to Climatology: Deep Learning Reconstruction of Tropical Cyclogenesis in the Western North Pacific

## 1. Introduction

This code repository provides (*i*) a workflow to generate meteorological observation samples from different climatological datasets; and (*ii*) a deep learning framework to exploit the generated data, based on RestNet-18 architecture.

For the former, two climatological datasets are used:
* MERRA-2 by the National Aeronautics and Space Administration (NASA) (Gelaro et al., 2017) and
* Tropical cyclone information from the International Best Track Archive for Climate Stewardship (IBTrACS) (Knapp et al., 2010).

The standardized datasets will be used for subsequent ML tasks in the latter framework.

This repository has been submitted to Zenodo for official reference: https://doi.org/10.5281/zenodo.17459622.

## 2. Input data
### 2.1. IBTrACS

The IBTrACS dataset provides information on tropical cyclone positions. It is structured on a two-dimensional grid with a longitude range of 0 to +360 degrees (0.001-degree resolution) and a latitude range from -90 to +90 degrees (also 0.001-degree resolution). The data span from the 1980s to December 31, 2022, and are sampled every 3 hours. The IBTrACS dataset is stored in a single CSV file with a total size of approximately 300 MB.

This dataset is included in the zip file (directory: `./data/ibtracs.ALL.list.v04r00.zip`). 

### 2.1. NASA-MERRA2

The NASA-MERRA2 dataset contains 13 meteorological variables. The data are mapped onto a two-dimensional grid: longitude spans from -180 to +180 degrees with a resolution of 0.625 degrees; latitude ranges from -90 to +90 degrees with a resolution of 0.5 degrees. Altitude is defined by 42 pressure levels, ranging from 1000 hPa to 0.1 hPa. The data were collected from January 1, 1980, to December 31, 2022, with a sampling frequency of every 3 hours. Each file contains data for 4 forecast times per day, stored in NetCDF format, and is approximately 2.2–2.3 GB in size. The complete dataset occupies around 18 TB.

The list of MERRA2 files to be downloaded is provided in the file: `mrr2_1980_2023.txt` (directory: ./data/merra2_1980_2023.txt). The following instruction (also available at ./data/merra2_download_instruction.txt) provides the steps to download the NASA-MERRA2 files for this study.

1. First, you need to create an account on EarthData (https://urs.earthdata.nasa.gov/)

2. Install the `aria2` downloader:
```
bash
sudo apt install aria2
```
If you do not have root access, you can use pre-built binaries or compile it from source. 
See the `aria2` homepage (https://aria2.github.io/) for more details.

3. Modify the following script and save it as `download_merra2.sh`:
```
bash
#!/bin/bash
DEST=/path/to/download/folder
LOG=/path/to/a/file
COOKIES=/path/to/a/file
USER=replace_with_your_earthdata_username
PASS=replace_with_your_earthdata_password
LIST=/path/to/mrr2_1980_2023.txt/file
aria2c --http-user="$USER" --http-passwd="$PASS" --save-cookies="$COOKIES" --load-cookies="$COOKIES" --log="$LOG" --log-level="notice" --content-disposition --continue=true --check-integrity=true --dir="$DEST" --input-file="$LIST"
```
Notes:
* `DEST` is the directory where all data will be saved.
* `LIST` is the path to the text file containing the URLs.
* `COOKIES` should point to a writable file (it will be created if it doesn't exist).

## 4. Make the following bash script and run it:
```
bash
chmod +x download_merra2.sh
./download_merra2.sh
```
If everything is set correctly, the download should start immediately.

**Warning:** The full dataset is very large (about 20 terabytes). Make sure you have enough storage.

**Resuming:** If the download is interrupted, you can resume it by running the script again. `aria2c` was invoked with `--continue=true`, so partially downloaded files will continue instead of restarting.

## 3. Preprosessing and Sampling
### 3.1. Prerequisites

The following command is used to install the required libraries (only needs to be executed once):
```
conda create -n hurricane-ml python=3.10 ipykernel autopep8 numpy xesmf pandas xarray pynio netcdf4 matplotlib cfgrib pytables h5netcdf tqdm alive-progress -c conda-forge -y
```

### 3.2. Data Preprocessing
Run on a local machine or use interactive shells: 
```
# Pre-processing
python3 Preprocess_Ibtracs_Fnl.py
python3 Preprocess_Ibtracs_Merra2.py
Using Slurm Jobs
```
In addition to running locally, the above source code can also be executed on a server by submitting Slurm jobs. Sample Slurm job scripts are provided in the ./scripts/slurm_job directory.

### 3.3. Data Sampling

##### Positive Sampling

Each positive sample is a data file containing information within a 33×33 grid centered around the eye of the targeted tropical cyclone at the forecast timestamp when it forms.

##### Negative Sampling Using the Past Domain Strategy (PastDomain):

The PastDomain strategy generates negative samples by using the location of a positive sample and going back in time by n ≥ 1 timestamps. This method results in a more balanced number of negative samples compared to positive ones. Negative sample filenames under this strategy follow the convention: {SID}_{n}.nc4, where n ∈ [1, N] represents the number of time steps prior to the positive sample timestamp.

##### Negative Sampling Using the Dynamic Domain Strategy (DynamicDomain):

The DynamicDomain strategy selects negative samples from regions adjacent to the positive sample location. For each positive sample, eight negative samples are generated, each corresponding to one of the neighboring grid blocks in the 2D space: Northwest (nw), North (n), Northeast (ne), West (w), East (e), Southwest (sw), South (s), and Southeast (se) (as illustrated in Figure 1). Additionally, samples are taken from n ≥ 0 time steps in the past for each of the eight neighboring regions.

The following commands are used to generate samples using the above-mentionned strategies:
```
# Data extraction
python3 Extract_FixedDomain.py
python3 Extract_DynamicDomain.py
python3 Extract_PastDomain.py
```

### 3.4. Data Postprocessing

The following commands are used to analyze the NaN and noise statistics of the preprocessed data:
```
# Post-processing (NaN and noise statistics)
python3 Analyze_NaNStat.py
python3 Analyze_NoiseFinder.py
```

## 5. Evaluation using ResNet-18

In our work "From Reanalysis to Climatology: Deep Learning Reconstruction of Tropical Cyclogenesis in the Western North Pacific," we adapted the original ResNet-18 architecture for our TCG applications. The modified network consists of eight residual blocks, preceded by an initial convolutional layer for input embedding, and followed by a fully connected layer with a softmax activation to predict the probability of storm occurrence, forming a total of 18 layers.

### 5.1. Evaluation using Past Domain Sampling Strategy

For detailed instructions, see past_domain/README.md

##### Libraries installation

To install all required libraries, run the following commands:
```
bash
pip install torch torchvision
pip install numpy pandas matplotlib seaborn scikit-learn tqdm
pip install pytorch-metric-learning
pip install xarray
```

##### Running the Script

To run the scripts above, you need the following inputs:
* A directory path containing MERRA2 files with the format: `merra2_19800101_00_00.nc`
* An `IBTRACS.csv` file containing tropical cyclone information (placed in the `csv` directory)

Execute the `prepare_csv.py` script to generate the necessary input CSV files for training the model:
```
python prepare_csv.py
```
The output will include the following files (located in the `csv` directory):
* `data_statistics.xlsx` – statistical summaries of MERRA2 features, used for data normalization
* `FIRST_MERRA2_IBTRACS.csv` – information on the first occurrence time of each storm
* `merra_full_new.csv` – list of paths to MERRA2 files, with a label:
* `-1` if the file corresponds to a time during a storm (but not the first occurrence) blank if no storm is occurring at that time

After this step, proceed to train and evaluate the model.

To train and evaluate the model, use the following command:
```
bash
python main.py --time t2_rus4_cw3_fe --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 4 --class_weight 3 --small_set
```

##### Command-Line Arguments:
See past_domain/README.md for details.

##### Example Usage

To train with undersampling and specific class weights:
This is using normalised method, learning rate 1e-7, positive sample is time step t-2, using undersampling method - with ratio 1:10 and class weight balanced
```
bash
python main.py --time experiment_1 --norm_type new --lr 1e-7 --pos_ind 2 --under_sample --rus 10 --class_weight 1
```
For a quick test using a small dataset:
This is using normalised method, learning rate 1e-7, positive sample is time step t-2, not using undersampling method, only use small subset of data and class weight balanced
```
bash
python main.py --time test_run --norm_type new --lr 1e-7 --pos_ind 2 --small_set --class_weight 1
```

##### Running the Script Evaluate Trained Model on Full months (eval_fullyear.py)

Besides the standard training and testing routine, this project includes a dedicated script for full year predictions. The file eval_fullyear.py is used as:
```
bash
python eval_fullyear.py --timestep t2_rus4_cw1 --strict --fullmonth --model_path result/model/model.pth
```

##### Command-Line Arguments

See past_domain/README.md for details.

##### Output 
* Results for --fullmonth enabled: ./result_fullmap/all_months
* Results for --fullmonth disabled: ./result_fullmap/storm_months

### 5.2. Dynamic Domain Sampling Strategy

For detailed instructions, see dynamic_domain/README.md

##### Libraries installation

To install all required libraries, run the following commands, either in native python environment or a virtual environment:
```
bash
pip install pandas tqdm numpy torch sklearn open-cv2 xarray pandarallel calendar matplotlib basemap
```

##### Running script

**Prepare data**

Data should be in NetCDF format and organized by a csv file called "data_path", which contain the path of each sample, its metadata (position, step, ...).
Run this command to generate train, val and test data in csv format for a single predicting-step.
```
python Prepare_data.py --path $path --step $step --ratio $ratio --dst $dst
```
where:

$path: the csv file containing the path(s) of every sample
$step: the step forecast, a step stand for 3 hours predict
$ratio: the ratio for under resampling the dataset
$dst: output directory

**Modeling**

Run this command to train model from scratch
```
python Train.py --inp_dir $inp_dir --out_dir $out_dir --weight $weight --map_path $map_path
```
where:
* $inp_dir: the input dataset, including a train set, validate set and a test set.
* $out_dir: the output directory, saving model checkpoint and evaluation results.
* $weight: the class weight assigned for positive sample, if set to 0, the class weight is computed balancedly
* $map_path: the csv file containing the path(s) of every sample for map evaluating. 

**Evaluating**

- Run this command to evaluate model performance on selected area.
```
python Map_eval.py --temp $temp --out $out
```
where:
* $temp: the csv template of map prediction for every forecasting step in range from 2 to 18. This should be a csv file path with the step forecast left in {}.
* $out: the output directory where the scoreboard will be exported

- Run this command to visual the map distribution of mean score in the selected area
```
python Spatial_map.py --temp $temp --out $out
```
where:
* $temp: the csv template of map prediction for every forecasting step in range from 2 to 18. This should be a csv file path with the step forecast left in {}.
* $out: the output directory where the figure will be exported

## How to cite our Work
1. Duc-Trong Le, Tran-Binh Dang, Anh-Duc Hoang Gia, Duc-Hai Nguyen, Minh-Hoa Tien, Quang-Trung Luu, Quang-Lap Luu, Minh-Thanh Nguyen, Truong Ngo, Tai-Hung Nguyen, Thanh T. N. Nguyen, and Chanh Kieu, "Data Processing for TCG-Net: Reconstructing Tropical Cyclogenesis Climatology," https://doi.org/10.5281/zenodo.15640334, 2025.
2. Duc-Trong Le, Tran-Binh Dang, Anh-Duc Hoang Gia, Duc-Hai Nguyen, Minh-Hoa Tien, Quang-Trung Luu, Quang-Lap Luu, Tai-Hung Nguyen, Thanh T. N. Nguyen, and Chanh Kieu, "From Reanalysis to Climatology: Deep Learning Reconstruction of Tropical Cyclogenesis in the Western North Pacific," submitted to Geoscientific Model Development, 2025.

