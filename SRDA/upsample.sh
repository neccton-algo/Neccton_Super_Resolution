#!/bin/bash -l
#
# Project:
#SBATCH --account=nn2993k --qos=preproc
#
# Job name:
#SBATCH -J "SR_upsample"
#
#SBATCH -N 1
##SBATCH --exclusive
#SBATCH --ntasks=128
## --cpus-per-task=1
#SBATCH --mem-per-cpu=4G
#
# Wall clock limit:
#SBATCH -t 01:00:00
#
#SBATCH -o log/upsample.out
#SBATCH -e log/upsample.err

# set up job environment
set -e # exit on error
set -u # exit on unset variables
ml purge

date=$(cat date.txt)

singularity exec -B /cluster/home/antber:/cluster/home/antber -B /cluster/work/users/antber:/cluster/work/users/antber /cluster/home/antber/container/tfrocm.sif bash -c '
    # Activate the Python environment
    source /cluster/home/antber/sr_env/bin/activate

    # Change to the correct directory
    cd /cluster/work/users/antber/from_Elio

    # Loop and run the Python script with different arguments
    for i in $(seq 1 1); do
       python upsample.py '"$date"' $i $i &
    done

    # Wait for all background processes to finish
    wait
'
