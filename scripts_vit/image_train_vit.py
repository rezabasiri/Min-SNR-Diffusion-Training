"""
Train a diffusion model on images.
"""

import os
import argparse
import wandb
import torch.distributed as dist

import torch
import numpy as np

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop


def build_model(**kwargs):
    if kwargs['model_name'] == 'vit_base_patch2_32':
        from guided_diffusion.vision_transformer import vit_base_patch2_32
        _model = vit_base_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch2_32':
        from guided_diffusion.vision_transformer import vit_large_patch2_32
        _model = vit_large_patch2_32(**kwargs)
    elif kwargs['model_name'] == 'vit_large_patch4_64':
        from guided_diffusion.vision_transformer import vit_large_patch4_64
        _model = vit_large_patch4_64(**kwargs)
    elif kwargs['model_name'] == 'vit_xl_patch2_32':
        from guided_diffusion.vision_transformer import vit_xl_patch2_32
        _model = vit_xl_patch2_32(**kwargs)
    else:
        raise NotImplementedError(f'Such model is not supported')
    return _model

# DATA_DIR="/home/rbasiri/Dataset/GAN/train_foot/train/"
# EXP_NAME = "vit-b_layer12_lr1e-4_099_099_pred_x0__min_snr_5__fp16_bsTest"
# MODEL_BLOB="/home/rbasiri/Dataset/saved_models/Diffusion/"
# OPENAI_LOGDIR1=os.path.join(MODEL_BLOB,"exp/guided_diffusion/",EXP_NAME)
# # os.makedirs(OPENAI_LOGDIR1, exist_ok=True)
# # os.chdir(OPENAI_LOGDIR1)

# # @dataclass
# class TrainingConfig:
#     image_size =32  # the generated image resolution
#     nproc_per_node=1
#     master_port=23456
#     data_dir= DATA_DIR
#     class_cond=True
#     diffusion_steps=1000
#     noise_schedule="cosine"
#     rescale_learned_sigmas=False
#     lr=1e-4
#     batch_size=2
#     log_interval=10
#     beta1=0.99
#     beta2=0.99
#     exp_name=EXP_NAME
#     use_fp16=True
#     weight_decay=0.03
#     use_wandb=False
#     model_name='vit_base_patch2_32'
#     depth=12
#     predict_xstart=True
#     warmup_steps=0
#     lr_anneal_steps=0
#     mse_loss_weight_type="min_snr_5"
#     clip_norm=-1
#     in_chans=4
#     drop_label_prob=0.15
# # args = TrainingConfig()

def main():
    args = create_argparser().parse_args()
    # args = TrainingConfig()

    dist_util.setup_dist('pytorch')
    logger.configure()

    seed = 2022 + dist.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)

    # wandb
    USE_WANDB = args.use_wandb
    if int(os.environ["RANK"]) == 0 and USE_WANDB:
        wandb_log_dir = os.path.join(os.environ.get('OPENAI_LOGDIR', 'exp'), 'wandb_logger')
        os.makedirs(wandb_log_dir, exist_ok=True)
        wandb.login(key='6af4861b52ff891552f5edd4839f6717c8a05526')
        wandb.init(project="guided_diffusion_vit", sync_tensorboard=True,
                   name=args.exp_name, id=args.exp_name, dir=wandb_log_dir)

    logger.log("creating model and diffusion...")
    # model = beit_base_patch4_64()
    model = build_model(**vars(args))
    _, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )
    # log the args & model
    logger.log(f"==> args \n{args}")
    logger.log(f"==> Model \n{model}")

    print(f"dev: {dist_util.dev()}\t rank: {dist.get_rank()}\t worldsize: {dist.get_world_size()}")

    model.to(dist_util.dev())
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
        betas=(args.beta1, args.beta2),
        warmup_steps=args.warmup_steps,
        lr_final=args.lr_final,
    ).run_loop()

    if os.environ["RANK"] == 0 and USE_WANDB:
        wandb.finish()


def create_argparser():
    defaults = dict(
        data_dir="",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=1,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
        beta1=0.9,
        beta2=0.999,
        exp_name="debug",
        use_wandb=True,
        model_name='beit_base_patch4_64',
        # vit related settings
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        use_conv_last=False,
        depth=12,
        use_shared_rel_pos_bias=False,
        warmup_steps=0,
        lr_final=1e-5,
        clip_norm=-1.,
        local_rank=0,
        drop_label_prob=0.,
        use_rel_pos_bias=False,
        in_chans=3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
