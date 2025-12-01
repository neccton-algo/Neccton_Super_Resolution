#!/bin/bash -l
#
# Project:
##SBATCH --account=nn2993k
#SBATCH --account=nn9481k
#
# Job name:
#SBATCH -J "SR_inference"
#
#SBATCH -N 4
##SBATCH --exclusive
#SBATCH --ntasks=100 --cpus-per-task=5
#
# Wall clock limit:
#SBATCH -t 03:00:00
#
#SBATCH -o log/Full_inference.out
#SBATCH -e log/Full_inference.err

# set up job environment
set -e # exit on error
set -u # exit on unset variables
ml purge

date=$(cat date.txt)

## List of variable to super-resolve
## To be modified in assemble_restart.py as well!!
list_var=("temp" "saln" "dp" "u" "v" "ECO_fla" "ECO_dia" "ECO_ccl" "ECO_flac" "ECO_diac" "ECO_cclc") 
# Enable or not each step
save_forecast=1
save_before_SR=1
upsample=1
inference=1
assemble=1
save_post_SR=1


###### SAVING FORECAST TO BACKUP DIR ######
if [ "$save_forecast" -ne 0 ]; then
 echo "Saving forecast"
 for mem in $(seq 1 25); do
   srun -N 1 -n 1 -r 0 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python save_forecast.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 26 50); do
   srun -N 1 -n 1 -r 1 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python save_forecast.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 51 75); do
   srun -N 1 -n 1 -r 2 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python save_forecast.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 76 100); do
   srun -N 1 -n 1 -r 3 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python save_forecast.py '"$date"' $mem $mem" &
 done
 wait
fi
###### UPSAMPLING ########
if [ "$upsample" -ne 0 ]; then
 echo "Upsampling"
 for mem in $(seq 1 25); do
   srun -N 1 -n 1 -r 0 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python upsample.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 26 50); do
   srun -N 1 -n 1 -r 1 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python upsample.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 51 75); do
   srun -N 1 -n 1 -r 2 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python upsample.py '"$date"' $mem $mem" &
 done
 for mem in $(seq 76 100); do
   srun -N 1 -n 1 -r 3 singularity exec \
      -B /cluster/home/antber:/cluster/home/antber \
      -B /cluster/work/users/antber:/cluster/work/users/antber \
      /cluster/home/antber/container/tfrocm.sif \
      bash -c "source /cluster/home/antber/sr_env/bin/activate && python upsample.py '"$date"' $mem $mem" &
 done
 wait
fi

###### INFERENCE  ########
if [ "$inference" -ne 0 ]; then
echo "Inference"
# Iterate through variables
for var in "${list_var[@]}"; do
    # Node 0: Runs layers 1-13
    for layer in $(seq 1 13); do
        srun -N 1 -n 1 -r 0 singularity exec \
            -B /cluster/home/antber:/cluster/home/antber \
            -B /cluster/work/users/antber:/cluster/work/users/antber \
            /cluster/home/antber/container/tfrocm.sif \
            bash -c "source /cluster/home/antber/sr_env/bin/activate && python inference.py $date $var $layer" &
    done

    # Node 1: Runs layers 14-25
    for layer in $(seq 14 25); do
        srun -N 1 -n 1 -r 1 singularity exec \
            -B /cluster/home/antber:/cluster/home/antber \
            -B /cluster/work/users/antber:/cluster/work/users/antber \
            /cluster/home/antber/container/tfrocm.sif \
            bash -c "source /cluster/home/antber/sr_env/bin/activate && python inference.py $date $var $layer" &
    done

    # Node 2: Runs layers 26-38
    for layer in $(seq 26 38); do
        srun -N 1 -n 1 -r 2 singularity exec \
            -B /cluster/home/antber:/cluster/home/antber \
            -B /cluster/work/users/antber:/cluster/work/users/antber \
            /cluster/home/antber/container/tfrocm.sif \
            bash -c "source /cluster/home/antber/sr_env/bin/activate && python inference.py $date $var $layer" &
    done

    # Node 3: Runs layers 39-50
    for layer in $(seq 39 50); do
        srun -N 1 -n 1 -r 3 singularity exec \
            -B /cluster/home/antber:/cluster/home/antber \
            -B /cluster/work/users/antber:/cluster/work/users/antber \
            /cluster/home/antber/container/tfrocm.sif \
            bash -c "source /cluster/home/antber/sr_env/bin/activate && python inference.py $date $var $layer" &
    done
    wait
done
fi

###### ASSEMBLING  #######
if [ "$assemble" -ne 0 ]; then
echo "Assembling parts"
for mem in $(seq 1 25); do
  srun -N 1 -n 1 -r 0 singularity exec \
     -B /cluster/home/antber:/cluster/home/antber \
     -B /cluster/work/users/antber:/cluster/work/users/antber \
     /cluster/home/antber/container/tfrocm.sif \
     bash -c "source /cluster/home/antber/sr_env/bin/activate && python assemble_restart.py '"$date"' $mem" &
done
for mem in $(seq 26 50); do
  srun -N 1 -n 1 -r 1 singularity exec \
     -B /cluster/home/antber:/cluster/home/antber \
     -B /cluster/work/users/antber:/cluster/work/users/antber \
     /cluster/home/antber/container/tfrocm.sif \
     bash -c "source /cluster/home/antber/sr_env/bin/activate && python assemble_restart.py '"$date"' $mem" &
done
for mem in $(seq 51 75); do
  srun -N 1 -n 1 -r 0 singularity exec \
     -B /cluster/home/antber:/cluster/home/antber \
     -B /cluster/work/users/antber:/cluster/work/users/antber \
     /cluster/home/antber/container/tfrocm.sif \
     bash -c "source /cluster/home/antber/sr_env/bin/activate && python assemble_restart.py '"$date"' $mem" &
done
for mem in $(seq 76 100); do
  srun -N 1 -n 1 -r 1 singularity exec \
     -B /cluster/home/antber:/cluster/home/antber \
     -B /cluster/work/users/antber:/cluster/work/users/antber \
     /cluster/home/antber/container/tfrocm.sif \
     bash -c "source /cluster/home/antber/sr_env/bin/activate && python assemble_restart.py '"$date"' $mem" &
done
wait
fi

## Moving files to right directories
find /cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields -type f -name '*part*' -delete
if [ "$save_before_SR" -ne 0 ]; then
	cp /cluster/work/users/antber/TP2a0.10/expt_02.5/LR_upsampled/restart*mem{001..005}.{a,b}* /cluster/work/users/antber/TP5_SR_Reanalysis/TOBACKUP/SR/LR_upsampled
fi
if [ "$save_post_SR" -ne 0 ]; then
        cp /cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields/restart*mem{001..005}.{a,b}* /cluster/work/users/antber/TP5_SR_Reanalysis/TOBACKUP/SR/post_SR
fi
rm /cluster/work/users/antber/TP2a0.10/expt_02.5/LR_upsampled/restart*
mv /cluster/work/users/antber/TP2a0.10/expt_02.5/SR_fields/restart* /cluster/work/users/antber/TP2a0.10/expt_02.5/data/
mv /cluster/work/users/antber/TP2a0.10/expt_02.5/LR_upsampled/cice/iced* /cluster/work/users/antber/TP2a0.10/expt_02.5/data/cice/

