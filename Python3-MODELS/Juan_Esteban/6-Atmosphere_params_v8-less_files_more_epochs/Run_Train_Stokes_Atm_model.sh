#!/bin/bash

#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks=1  ## number of tasks (analyses) to run
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task
#SBATCH --mem-per-cpu=500M   # memory per CPU core
#SBATCH --partition=normal  ## the partitions to run in (comma seperated)
#SBATCH --time=3-00:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-user=juagudeloo@unal.edu.co

## Load modules

./girg/juagudeloo/env_juanes/bin/activate
pip3 install numpy
pip3 install matplotlib
pip3 install tensorflow
pip3 install pandas

## Insert code, and run your programs here (use 'srun').

for i in {53..60..3}
do
python3 /girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v8/Train_Stokes-Atm_model.py $i
done

python3 /girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v8/Obtain_Stokes-Atm_model.py
deactivate