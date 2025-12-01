#!/bin/bash
#SBATCH --job-name=SR_Topaz_V2
#SBATCH --account=project_465002269
#SBATCH --output=log/training_3V201_layer_pred_%A_layer_target_%a.o
#SBATCH --error=log/training_3V201_layer_pred_%A_layer_target_%a.e
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --partition=small-g
#SBATCH --gpus=1

module load LUMI/23.12 partition/G
# module load rocm

# --- Create unique MIOpen cache directories ---
export MIOPEN_USER_DB_PATH=$(mktemp -d "$SCRATCH/miopen_db_XXXXXX")
export MIOPEN_CUSTOM_CACHE_DIR=$(mktemp -d "$SCRATCH/miopen_cache_XXXXXX")

# --- Set up cleanup trap ---
cleanup() {
    echo "Cleaning up MIOpen cache dirs..."
    rm -rf "$MIOPEN_USER_DB_PATH" "$MIOPEN_CUSTOM_CACHE_DIR"
}
trap cleanup EXIT

var='temp'
layer_pred='1'
layer_target='1'

singularity exec -B"/appl:/appl" \
                 -B"$SCRATCH:$SCRATCH" \
                 /scratch/project_465002269/bernigaud/env/tensorflow_rocm5.5-tf2.11-dev.sif ./python_env.sh training3V201_auto.py $var $layer_pred $layer_target
