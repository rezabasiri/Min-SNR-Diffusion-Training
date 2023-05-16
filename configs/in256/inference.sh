#!/bin/sh
#SBATCH -p gpu-v100
#SBATCH --job-name=MinSNR
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

if [ ! -d edm ]; then
    git clone https://github.com/NVlabs/edm.git
fi

export NCCL_DEBUG=WARN

GPUS=1
IMG_SIZE=128
BATCH_SIZE=12
NUM_SAMPLES=10
MODEL_NAME="vit_xl_patch2_32"  # #"vit_large_patch4_64" #
DEPTH=24
GUIDANCE_SCALES="0"
STEPS="1000"
PRED_X0=True
NUM_InChanl=3
NameTag='128ema_0.9999_120000'
# --data_dir '/home/rbasiri/Dataset/GAN/train_foot/train/'

CKPT="/home/rbasiri/Dataset/saved_models/Diffusion/exp/guided_diffusion/vit_layer24_Dsteps2000_128_FOOT_3stRun/ema_0.9999_120000.pt"
FID_DIR="/home/rbasiri/Dataset/GAN/train_footFID/train_foot.npz"
MODEL_BLOB="/home/rbasiri/Dataset/saved_models/Diffusion"

MODEL_FLAGS="--class_cond False --image_size $IMG_SIZE --model_name ${MODEL_NAME} --depth $DEPTH --predict_xstart $PRED_X0 \
--schedule_sampler 'uniform' --lr 0.0001 --weight_decay 0.03 --lr_anneal_steps 0 --microbatch -1 --ema_rate '0.9999' \
--log_interval 500 --save_interval 10000 --use_fp16 False --fp16_scale_growth 0.001 --use_wandb False \
--drop_path_rate 0.0 --use_conv_last False --use_shared_rel_pos_bias False --warmup_steps 0 --lr_final 1e-05 --drop_label_prob 0.15 \
--use_rel_pos_bias False --in_chans $NUM_InChanl --num_channels 128 --num_res_blocks 2 --num_heads 4 --num_heads_upsample -1 --attention_resolutions 16,8 --num_head_channels -1 \
--dropout 0.2 --use_checkpoint False --use_scale_shift_norm True --resblock_updown False --use_new_attention_order False --learn_sigma False \
--use_kl False --mse_loss_weight_type 'min_snr_5'"
DIFFUSION_FLAGS="--diffusion_steps 1000 --noise_schedule cosine --rescale_learned_sigmas False --rescale_timesteps False"

# ----------- scale loop ------------- #
for GUIDANCE_SCALE in $GUIDANCE_SCALES
do

for STEP in $STEPS
do

SAMPLE_FLAGS="--batch_size $BATCH_SIZE --num_samples ${NUM_SAMPLES} --steps $STEP --guidance_scale $GUIDANCE_SCALE"

OPENAI_LOGDIR="${MODEL_BLOB}/exp/guided_diffusion/xl_samples${NUM_SAMPLES}_step${STEP}_scale${GUIDANCE_SCALE}_tag${NameTag}"
mkdir -p $OPENAI_LOGDIR
OPENAI_LOGDIR=$OPENAI_LOGDIR torchrun --nproc_per_node=$GPUS --master_port=23456 scripts_vit/sampler_edm.py --model_path $CKPT $MODEL_FLAGS $DIFFUSION_FLAGS $SAMPLE_FLAGS

cd edm
torchrun --standalone --nproc_per_node=$GPUS fid.py calc --images=$OPENAI_LOGDIR --ref=$FID_DIR --num $NUM_SAMPLES 
cd ..

done
done
# ----------- scale loop ------------- #

echo "----> DONE <----"

# -----------------------------------
#          expected output
# -----------------------------------
# Calculating FID...
# 2.0559