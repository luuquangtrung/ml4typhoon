#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=tqluu@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=4-0:0:00
#SBATCH --mem=128gb
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=x_fix
#SBATCH -A r00043

######  Module commands #####

module load python/gpu
module load miniconda

######  Job commands go below this line #####
cd ~/workspace/libtcg_Dataset2
conda activate hurricane-ml
python3 Extract_FixedDomain.py