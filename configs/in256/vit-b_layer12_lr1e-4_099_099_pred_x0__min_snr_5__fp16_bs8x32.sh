#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=MinSNR
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=12
#SBATCH -t 1-00:00:00
#SBATCH --mem=256G
#SBATCH --gres=gpu:2
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

# set -ex

# pip install -r requirements.txt
# pip install -e .

DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"

GPUS=2
DEPTH=24
DSteps=2000
IMG_SIZE=128
BATCH_PER_GPU=12
MODEL_NAME="vit_xl_patch2_32"
EXP_NAME=vit_layer24_Dsteps2000_128_FOOT_3stRun

MODEL_BLOB="/mnt/external"
if [ ! -d $MODEL_BLOB ]; then
    MODEL_BLOB="/home/rbasiri/Dataset/saved_models/Diffusion"
fi

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/$EXP_NAME"

mkdir -p $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR \
    torchrun --nproc_per_node=$GPUS --master_port=23456 scripts_vit/image_train_vit.py \
    --data_dir $DATA_DIR --image_size $IMG_SIZE --class_cond False --save_interval=10000 --diffusion_steps $DSteps \
    --noise_schedule cosine --rescale_learned_sigmas False \
    --lr 1e-8 --batch_size $BATCH_PER_GPU --log_interval 10000 --beta1 0.99 --beta2 0.99 \
    --exp_name $EXP_NAME --use_fp16 False --weight_decay 0.03 \
    --use_wandb False --model_name ${MODEL_NAME} --depth $DEPTH \
    --predict_xstart True --warmup_steps 0 --use_checkpoint True --lr_anneal_steps 0 \
    --mse_loss_weight_type min_snr_5 --clip_norm -1 \
    --in_chans 3 --drop_label_prob 0 --drop_rate 0.1 --attn_drop_rate 0.05