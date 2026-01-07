#!/bin/bash
#####  Constructed by HPC everywhere #####
#SBATCH --mail-user=tqluu@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=64
#SBATCH --time=1-0:0:00
#SBATCH --mem=128gb
#SBATCH --partition=general
#SBATCH --mail-type=FAIL,BEGIN,END
#SBATCH --job-name=cpdata
#SBATCH -A r00043

######  Module commands #####

######  Job commands go below this line #####
echo COPY-DATA
today=`date +"%Y%m%d"`
homedir=~/workspace/libtcg_Dataset2/data
basedir=${homedir}/Dataset2/${today}
echo New data will be saved in: ${basedir}
cd ${homedir}
mkdir ${basedir}
echo Copying data to working space...
cp -r out/* Dataset2/

function compose() {
    echo Composing dataset: $1
    cd ${homedir}
    cd Dataset2/$1
    mkdir POSITIVE
    mv DynamicDomain/POSITIVE* POSITIVE
    rm PastDomain/POSITIVE*
    rm FixedDomain/POSITIVE*
}

compose ncep-fnl
compose nasa-merra2

echo Creating new dataset folder...
cd ${homedir}
cd Dataset2
mv ncep-fnl ${today}
mv nasa-merra2 ${today}

echo Done.
