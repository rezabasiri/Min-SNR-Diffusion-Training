#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=FID
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH -t 01:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

# set -ex

# pip install -r requirements.txt
# pip install -e .

DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"
FID_DIR="/home/rbasiri/Dataset/GAN/train_footFID/train_foot.npz"

GPUS=1

if [ ! -d edm ]; then
    git clone https://github.com/NVlabs/edm.git
fi

export NCCL_DEBUG=WARN

cd edm
torchrun --standalone --nproc_per_node=$GPUS fid.py ref --data=$DATA_DIR --dest=$FID_DIR