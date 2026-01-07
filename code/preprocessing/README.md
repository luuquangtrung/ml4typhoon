# libtcg_dataset

This project is used for extracting and sampling weather data related to hurricanes for machine learning.

Documentation: Available [HERE](https://husteduvn-my.sharepoint.com/:f:/g/personal/trung_luuquang_hust_edu_vn/EmEAfSA-cPlMv_vp9KBfChYBZQxK9V0uUDKovhBvjz_kPg?e=rP2V0A)

## Libs install

```bash
conda create -n hurricane-ml python=3.10 ipykernel autopep8 numpy xesmf pandas xarray pynio netcdf4 matplotlib cfgrib pytables h5netcdf tqdm alive-progress -c conda-forge -y
```

## Western Pacific Basin region

Warning center: Japan Meteorological Agency

Area of responsibility:
- LAT: +000 -> +060
- LON: +100 -> +180

Proposed area of interest:
- LAT: -050 -> +070
- LON: +060 -> +220

## Files to run

### Local computer or interactive shells
```bash
# Pre-Processing
python3 Preprocess_Ibtracs_Fnl.py
python3 Preprocess_Ibtracs_Merra2.py
# Data Extraction
python3 Extract_FixedDomain.py
python3 Extract_DynamicDomain.py
python3 Extract_PastDomain.py
# Post-Processing
# > Nothing to run
# Statistics
python3 Analyze_NaNStat.py
python3 Analyze_NoiseFinder.py
```

### Slurm jobs
There are pre-defined scripts for submitting to slurm located in `./scripts/slurm_job`. Remember to define slurm account in the script before submitting using `sbatch` command.

### Naming scheme for each type of dataset:

Each dataset types is contained in a folder corresponding its name. The naming schema is explained below.

1. `FixedDomain`
    - Positive: `{SID}.nc`
    - Negative: `{SID}.nc`
2. `DynamicDomain`
    - Positive: `{SID}.nc`
    - Negative: `{SID}_{quardantId}.nc`
3. `PastDomain`
    - Positive: `{SID}.nc`
    - Negative: `{SID}_{t}.nc`

Where:
- `quardantId` is either (see table below): 
    - `nw` (`north west`)
    - `n` (`north`)
    - `ne` (`north east`)
    - `e` (`east`)
    - `se` (`south east`)
    - `s` (`south`)
    - `sw` (`south west`)
    - `w` (`west`)
- `t` is either `1` (`t-1`), `2` (`t-2`), ..., `n` (`t-n`)<br>with `n` is `STEP_BACK_COUNTS` in `*_config.py`.

| `quadrantId`	    | position     	|              	    |
|--------------	    |------------	|-------------- 	|
| `nw` north-west 	| `n` north    	| `ne` north-east 	|
| `w` west       	| `0` positive 	| `e` east       	|
| `sw` south-west 	| `s` south    	| `se` south-east 	|

| `t`               |                   |                   |                   |
|--------------	    |-------------- 	|-------------- 	|-------------- 	|
| ...               | `2` (t-2)         | `1` (t-1)         | `null` positive   |

### Attributes

Some key attributes in `*.nc` files:
- `SID`: Storm ID.
- `ISO_TIME`: Time of the sample (useful with `PastDomain` dataset).
- `TYPE`: Type of sample.
- `LAT`/`LON`: center, max, min `latitude`/`longitude` value of the sample.

### Data folder structure

```txt
.
├── Analyze_noise2
├── configs/
│   ├── dims.py
│   ├── DynamicDomain.py
│   ├── FixedDomain.py
│   ├── NanStat.py
│   ├── NoiseFinder.py
│   ├── nworkers.py
│   ├── paths.py
│   └── SequenceArea.py
├── data/
│   ├── analyze
│   ├── out/
│   │   ├── nasa-merra2/
│   │   │   ├── POSITIVE/
│   │   │   │   ├── POSITIVE_{SID}.nc
│   │   │   │   └── ...
│   │   │   ├── PastDomain/
│   │   │   │   ├── NEGATIVE_{SID}_{n}_{yyyymmdd}_{hhmm}.nc
│   │   │   │   └── ...
│   │   │   ├── DynamicDomain/
│   │   │   │   ├── NEGATIVE_{SID}_{quad}_{n}.nc
│   │   │   │   └── ...
│   │   │   └── FixedDomain/
│   │   │       ├── NEGATIVE_merra2_{yyyymmdd}_{hh}_{mm}.nc
│   │   │       └── ...
│   │   └── ncep-fnl/
│   │       ├── POSITIVE/
│   │       │   ├── POSITIVE_{SID}.nc
│   │       │   └── ...
│   │       ├── PastDomain/
│   │       │   ├── NEGATIVE_{SID}_{n}_{yyyymmdd}_{hhmm}.nc
│   │       │   └── ...
│   │       ├── DynamicDomain/
│   │       │   ├── NEGATIVE_{SID}_{quad}_{n}.nc
│   │       │   └── ...
│   │       └── FixedDomain/
│   │           ├── NEGATIVE_fnl_{yyyymmdd}_{hh}_{mm}.nc
│   │           └── ...
│   ├── raw/
│   │   ├── ibtracs/
│   │   │   └── ibtracs.ALL.list.v04r00.csv
│   │   ├── nasa-merra2/
│   │   │   ├── *.nc
│   │   │   └── ...
│   │   └── ncep-fnl/
│   │       ├── grib1/
│   │       │   └── {yyyy}/
│   │       │       ├── *.grib1
│   │       │       └── ...
│   │       └── grib2/
│   │           └── {yyyy}/
│   │               ├── *.grib2
│   │               └── ...
│   └── temp/
│       ├── nasa-merra2/
│       │   ├── merra2_{yyyymmdd}_{hh}_{mm}.nc
│       │   └── ...
│       └── ncep-fnl/
│           ├── fnl_{yyyymmdd}_{hh}_{mm}.nc
│           └── ...
├── libtcg_HurricaneTrackDataset
├── libtcg_WeatherDataset
├── scripts/
│   └── slurm/
│       ├── *.sh
│       └── ...
└── ...
```

## Warning

REMEMBER TO EDIT THE CONFIG FILE !!!

Great `WORKER_COUNT` brings big performance but also big resources requirements.


## Sampling strategy

1. Fixed domain: 1 positive sample + 1 fixed location negative sample.
2. Dynamic & Past domain: 1 positive sample + 8 quads in 40 steps in the past.
3. Past domain: 1 positive sample + n sample at the location but in the past (6h, 12h, 18h, ... earlier).
