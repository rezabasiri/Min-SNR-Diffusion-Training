#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=Diffusion
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH -t 1-00:00:00
#SBATCH --mem=128G
#SBATCH --gres=gpu:1
## #SBATCH --gres=gpu:p100:3
#SBATCH -o %x-%j.out

# set -ex

# pip install -r requirements.txt
# pip install -e .

DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"

GPUS=$1
BATCH_PER_GPU=$2
EXP_NAME=vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bs${GPUS}x${BATCH_PER_GPU}

MODEL_BLOB="/mnt/external"
if [ ! -d $MODEL_BLOB ]; then
    MODEL_BLOB="/home/rbasiri/Dataset/saved_models/Diffusion"
fi

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/$EXP_NAME"
# if permission denied
mkdir -p $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR \
    torchrun --nproc_per_node=1 --master_port=23456 scripts_vit/image_train_vit.py \
    --data_dir $DATA_DIR --image_size 32 --class_cond False --diffusion_steps 1000 \
    --noise_schedule cosine --rescale_learned_sigmas False \
    --lr 1e-4 --batch_size 2 --log_interval 10 --beta1 0.99 --beta2 0.99 \
    --exp_name $EXP_NAME --use_fp16 True --weight_decay 0.03 \
    --use_wandb False --model_name vit_base_patch2_32 --depth 12 \
    --predict_xstart True --warmup_steps 0 --lr_anneal_steps 0 \
    --mse_loss_weight_type min_snr_5 --clip_norm -1 \
    --in_chans 4 --drop_label_prob 0.15
