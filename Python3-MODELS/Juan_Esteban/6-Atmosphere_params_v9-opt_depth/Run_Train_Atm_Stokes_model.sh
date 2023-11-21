#!/bin/bash

#SBATCH --output slurm-%j.out   ## filename of the output; the %j is equal to jobID; default is slurm-[jobID].out
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1  ## the number of threads allocated to each task  
#SBATCH --partition=normal  ## the partitions to run in (comma seperated)
#SBATCH --time=7-00:00:00  ## time for analysis (day-hour:min:sec)
#SBATCH --mail-type=begin        # send email when job begins
#SBATCH --mail-type=end          # send email when job ends
#SBATCH --mail-type=fail         # send email if job fails
#SBATCH --mail-user=juagudeloo@unal.edu.co
#SBATCH --gres=gpu:1

## Load modules

. /girg/juagudeloo/env_juanes/bin/activate

## Insert code, and run your programs here (use 'srun').

python3	/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v9-opt_depth/Begin_Train_Atm-Stokes_model.py 080000
for i in 087000 094000 101000 108000 115000
do
    python3	/girg/juagudeloo/Proyecto_DL-Trials/Python3-MODELS/Juan_Esteban/6-Atmosphere_params_v9-opt_depth/Train_Atm-Stokes_model.py $i
done

deactivate
