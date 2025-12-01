#!/bin/bash -l
#
# Project:
#SBATCH --account=nn2993k --qos=preproc
#
# Job name:
#SBATCH -J "SR_downsample"
#
#SBATCH -N 1
#SBATCH --exclusive
#SBATCH --ntasks=100
#SBATCH --mem-per-cpu=4G
#
# Wall clock limit:
#SBATCH -t 00:50:00
#
#SBATCH -o log/downsample_BTTP2.out
#SBATCH -e log/downsample_BTTP2.err

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
    for i in $(seq 1 100); do
       python downsample_back_to_TP2.py '"$date"' $i $i &
    done

    # Wait for all background processes to finish
    wait
'

mv /cluster/work/users/antber/TP2a0.10/expt_02.5/downsampled_assimilated_SR_fields/restart* /cluster/work/users/antber/TP2a0.10/expt_02.5/data
mv /cluster/work/users/antber/TP2a0.10/expt_02.5/downsampled_assimilated_SR_fields/cice/* /cluster/work/users/antber/TP2a0.10/expt_02.5/data/cice
