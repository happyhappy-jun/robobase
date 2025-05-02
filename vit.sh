#!/bin/bash
#SBATCH --job-name=baseline_vit_base
#SBATCH --gpus=1
#SBATCH --nodes=1
#SBATCH --partition=batch
#SBATCH --output=out/%j_vit_base.out 

DISPLAY_NUM=$((RANDOM % 1000))
echo "Using display number: ${DISPLAY_NUM}"


RUNTIME_DIR="/tmp/runtime-$USER-${DISPLAY_NUM}"
mkdir -p ${RUNTIME_DIR}
chmod 700 ${RUNTIME_DIR}
export XDG_RUNTIME_DIR=${RUNTIME_DIR}

echo "Starting X server with display :${DISPLAY_NUM}"
X :${DISPLAY_NUM} & 
X_PID=$!
sleep 1  # Give X server time to initialize

# Set display for this job
export DISPLAY=:${DISPLAY_NUM}



conda activate robobase
cd /slurm_home/byungjun_alinlab/robobase
HYDRA_FULL_ERROR=1 python3 train.py launch=act_pixel_rlbench method.encoder_model.backbone=vit_base

kill $X_PID
rm -rf ${RUNTIME_DIR}