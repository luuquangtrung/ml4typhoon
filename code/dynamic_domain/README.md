# Typhoon Formation Prediction with Deep Learning
**Dynamic Domain**

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Running the Script Train Model (main.py)](#project-structure)
- [Running the Script Evaluate Trained Model on Full months (eval_fullyear.py)](#project-structure)

## Overview

This project is structured to achieve the following:

- **Data Loading and Preprocessing:**
  - Read Merra2 data saved in netCDF format, organized by csv dataset.
  - Generate sample for final Domain evaluation

- **Model Training:**
  - Adapt the ResNet architecture to work with meteorological data.
  - Train the model to differentiate between typhoon and non-typhoon images.

- **Performance Evaluation:**
  - Log and visualize training metrics (loss, accuracy) to assess model performance.
  - Evaluate final trained models on Study area.

## Libraries installation

To install all required libraries, run the following commands, either in native python environment or a virtual environment:

```bash
pip install pandas tqdm numpy torch sklearn open-cv2 xarray pandarallel calendar matplotlib basemap
```
---

## Project Structure

``` bash
root/ # Root directory of the project 
    │── Data/ # Prepare data for modeling and domain evaluating 
    │── Model/ # Model architecture 
    │── Utils/ # Helper functions and utility scripts (e.g, training, validation, visualizing results, ..)

    │── Map_eval.py # Evaluate on Study area
    │── Prepare_data.py # Prepare data set with step forecast
    │── Spatial_map # Plot to map distrubution
    │── Train.py # Script to run a single step
```

## Running script

Prepare data

- Data should be in NetCDF format and organized by a csv file called "data_path", which contain the path of each sample, its metadata (position, step, ...).

- Run this command to generate train, val and test data in csv format for a single predicting-step.
```
python Prepare_data.py --path $path --step $step --ratio $ratio --dst $dst
```
Where:
  - $path: the csv file containing the path(s) of every sample
  - $step: the step forecast, a step stand for 3 hours predict
  - $ratio: the ratio for under resampling the dataset
  - $dst: output directory

Modeling
- Run this command to train model from scratch
```
python Train.py --inp_dir $inp_dir --out_dir $out_dir --weight $weight --map_path $map_path
```
Where:
  - $inp_dir: the input dataset, including a train set, validate set and a test set.
  - $out_dir: the output directory, saving model checkpoint and evaluation results.
  - $weight: the class weight assigned for positive sample, if set to 0, the class weight is computed balancedly
  - $map_path: the csv file containing the path(s) of every sample for map evaluating. 

Evaluating

- Run this command to evaluate model performance on selected area.
```
python Map_eval.py --temp $temp --out $out
```
Where:
  - $temp: the csv template of map prediction for every forecasting step in range from 2 to 18. This should be a csv file path with the step forecast left in {}.
  - $out: the output directory where the scoreboard will be exported

- Run this command to visual the map distribution of mean score in the selected area
```
python Spatial_map.py --temp $temp --out $out
```
Where:
  - $temp: the csv template of map prediction for every forecasting step in range from 2 to 18. This should be a csv file path with the step forecast left in {}.
  - $out: the output directory where the figure will be exported


