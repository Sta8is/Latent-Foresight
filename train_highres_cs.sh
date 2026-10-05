#!/usr/bin/env bash
# Latent-Foresight, stage 2: fine-tune on Cityscapes at high resolution (448x896).
# Setup: a single machine with 16 GPUs (effective batch size 8 x 8 = 64).
# Starts from the low-resolution checkpoint of stage 1 (weights only, fresh optimizer).
#
# Usage (from the repository root, with the environment activated):
#   cs_data_path=/path/to/cityscapes/leftImg8bit_sequence meanstd_ckpt=/path/to/dinov2_stats.pth \
#   lowres_ckpt=/path/to/latent_foresight_cs_lowres.ckpt bash train_highres_cs.sh

exp_name="latent_foresight_cs_highres"
cs_data_path="${cs_data_path:-/path/to/cityscapes/leftImg8bit_sequence}"
meanstd_ckpt="${meanstd_ckpt:-/path/to/dinov2_stats.pth}"
# Low-resolution checkpoint from stage 1 (or the released latent_foresight_cs_lowres.ckpt)
lowres_ckpt="${lowres_ckpt:-checkpoints/latent_foresight_cs_lowres.ckpt}"
output_dir="${output_dir:-logs}"

python train.py \
    --num_workers=16 --num_workers_val=4 --num_gpus=8 --precision 16-mixed --eval_freq 5 --batch_size 1 --accum_iter 8 --max_epochs 40 \
    --img_size 448,896 --hidden_dim 1152 --heads 8 --layers 12 --dropout 0.1 --lr_base 1e-5 --optimizer "adamw" --weight_decay 0.2 \
    --eval_mode_during_training --evaluate --single_step_sample_train --seperable_attention --random_horizontal_flip --random_crop --use_fc_bias \
    --dataset "cityscapes" --cs_data_path "${cs_data_path}" --sequence_length 5 --d_layers 2,5,8,11 \
    --meanstd_ckpt "${meanstd_ckpt}" --dst_path "${output_dir}/${exp_name}/" \
    --P_mean -2 --P_std 1.5 --use_recon_losses --detach_future --detach_target \
    --bottleneck_dim 256 --ae_hidden_dim 1536 --ae_layers 2 --ae_attn_heads 6 --ae_norm_type "bn" --use_rope --sync_batch_norm \
    --high_res_adapt --ckpt "${lowres_ckpt}"
