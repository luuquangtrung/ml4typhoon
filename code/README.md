# Source code for paper "Reconstructing Tropical Cyclogenesis Climatology in the Northwestern Pacific Basin using Deep Learning (TCG-Net, V1.0)"

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)

---

## Overview

This project is structured to achieve the following:

- **Data Preprocessing:**
  - FNL & MERRA2 preprocessing

- **Dynamic domain:**
  - Predicting tropical cyclone genesis locations using a ResNet model on MERRA-2 data

- **Past domain:**
  - Predicting the timing of tropical cyclone genesis using a ResNet model on MERRA-2 data


## Project Structure

``` bash
root/ # Root directory of the project 
    │── preprocessing/ #  Source code for preprocessing FNL and MERRA2 preprocessing
    │── dynamic_domain/ # Source code for dynamic domain including: training, testing and map evaluation   
    │── past_domain/ # Source code for past domain including: training, testing and map evaluation 

```
## Run
/past_domain.sh
/dynamic_domain.sh