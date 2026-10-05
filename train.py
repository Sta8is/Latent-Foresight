from src.dataloaders.data import CS_VideoData 
from src.dataloaders.data_nu import NS_VideoData
from src.dataloaders.data_covla import COVLA_VideoData
from src.dataloaders.data_kubric import KUBRIC_VideoData
from src.dataloaders.data_nu_cs_covla import MultiSourceVideoData, create_multisource_datamodule 
from src.latent_foresight import Latent_Foresight
import pytorch_lightning as pl
import torch 
import argparse
import os
from pytorch_lightning.strategies import DDPStrategy
import numpy as np


def parse_tuple(x):
    return tuple(map(int, x.split(',')))

def parse_list(x):
    return list(map(int, x.split(',')))


parser = argparse.ArgumentParser()
# Multisource Data Parameters
parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'nuscenes', 'covla', 'kubric', 'multisource'])  # Add multisource option
parser.add_argument('--kubric_data_path', type=str, default=None, help='Path to the Kubric MOVi export, at the {suite}/{resolution} level')
parser.add_argument('--ns_data_path', type=str, default=None, help='Path to NuScenes dataset (required for multisource)')  # Add NuScenes path
parser.add_argument('--cs_data_path', type=str, default=None, help='Path to CityScapes dataset (required for multisource)')  # Add CityScapes path
parser.add_argument('--covla_data_path', type=str, default="/opt/dlami/nvme/CoVLA-Dataset/videos", help='Path to CoVLA dataset (required for multisource)')  # Add CoVLA path
parser.add_argument('--sampling_mode', type=str, default='equal_random', choices=['equal_random', 'random'], help='Sampling mode for multisource dataset')  # Add random sampling option
parser.add_argument('--use_custom_collate', action='store_true', default=False, help='Use custom collate function for multisource')  # Add custom collate option
# Data Parameters
# parser.add_argument('--dataset', type=str, default='cityscapes', choices=['cityscapes', 'nuscenes'])
parser.add_argument('--data_path', type=str, default='/opt/dlami/nvme/cityscapes/leftImg8bit_sequence')
parser.add_argument('--dst_path', type=str, default=None)
parser.add_argument('--img_size', type=parse_tuple, default=(224,448))
parser.add_argument('--num_workers', type=int, default=8)
parser.add_argument('--num_workers_val', type=int, default=None, 
        help='(Optional) number of workers for the validation set dataloader. If None (default) it is the same as num_workers.')
parser.add_argument('--sequence_length', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--random_crop', action='store_true', default=False)
parser.add_argument('--random_horizontal_flip', action='store_true', default=False)
parser.add_argument('--random_time_flip', action='store_true', default=False)
parser.add_argument('--timestep_augm', type=list, default=None, help='Probabilities for each timestep to be selected for augmentation starting from timestep 2 to \length of prob list e.g. [0.1,0.6,0.1,0.1,0.1] for timesteps [2,3,4,5,6]. If None, timestep [2,3,4] are selected with equal probability')
parser.add_argument('--no_timestep_augm', action='store_true', help='If True, no timestep augmentation is used (i.e., the num_frames_skip is always equal to 2 during training.)')
parser.add_argument('--use_fc_bias', action='store_true', help='Use bias for the fc_in and fc_out layers.')
parser.add_argument('--eval_modality', type=str, default=None, choices=[None, 'segm', 'depth', 'surface_normals',"rgb"], help='Modality to be used for evaluation. If None, the input modality is used.')
# Latent Foresight parmeters
parser.add_argument('--x_loss', action='store_true', default=False, help='Use x-loss instead of v-loss')
parser.add_argument('--detach_future', action='store_true', default=False, help='Detach the last frame of the input (i.e., the future frame) from the computational graph to prevent gradients from flowing through it')
parser.add_argument('--detach_target', action='store_true', default=False, help='Detach the target frame from the computational graph to prevent gradients from flowing through it')
parser.add_argument('--use_recon_losses', action='store_true', default=False, help='If True, compute additional reconstruction losses on the input frames (in addition to the main loss on the predicted future frame). This is only applicable if no AE or PCA checkpoint is provided (i.e., when the model is learning its own latent space without being constrained by a pretrained AE or PCA).')
parser.add_argument('--model_gaussian', action='store_true', default=False, help='If True, the model predicts a Gaussian distribution over the latent space and the loss is computed as the negative log-likelihood of the target latent under this distribution. If False, the model predicts a single point in the latent space and the loss is computed as the distance between the predicted and target latents.')
parser.add_argument('--pred_recon_loss_mask', action='store_true', default=False, help='If True, the reconstruction loss is computed only on the masked regions of the input')
parser.add_argument('--kl_weight', type=float, default=1e-6, help='Weight for the KL divergence loss term when model_gaussian is True')
parser.add_argument('--recon_mse_weight', type=float, default=1.0, help='Weight on the MSE term of the reconstruction losses (both the clean-latent and predicted-frame ones). Set to 0 to train with the cosine term alone.')
parser.add_argument('--recon_cos_weight', type=float, default=1.0, help='Weight on the cosine-similarity term of the reconstruction losses (both the clean-latent and predicted-frame ones). Set to 0 to train with the MSE term alone.')
parser.add_argument('--P_mean', type=float, default=-0.8, help='Mean of the distribution for sampling t')
parser.add_argument('--P_std', type=float, default=0.8, help='Standard deviation of the distribution for sampling t')
parser.add_argument('--uniform_t', action='store_true', default=False, help='If True, t is sampled from a uniform distribution')
parser.add_argument('--d_layers', type=parse_list, default=[2,5,8,11])
parser.add_argument('--hidden_dim', type=int, default=768)
parser.add_argument('--heads', type=int, default=8)
parser.add_argument('--layers', type=int, default=16)
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--attn_dropout', type=float, default=0.3)
parser.add_argument('--step', type=int, default=15)
parser.add_argument('--seperable_attention', action='store_true', default=False)
parser.add_argument('--seperable_window_size', type=int, default=1)
parser.add_argument('--train_mask_frames', type=int, default=1)
parser.add_argument('--use_first_last', action='store_true', default=False)
# AE parameters
parser.add_argument('--bottleneck_dim', type=int, default=256)
parser.add_argument('--ae_layers', type=int, default=2)
parser.add_argument('--ae_attn_heads', type=int, default=6)
parser.add_argument('--ae_hidden_dim', type=int, default=1536)
parser.add_argument('--ae_ckpt', type=str, default=None)
parser.add_argument('--meanstd_ckpt', type=str, required=True, help='Path to the precomputed DINOv2 feature statistics (.pth with "mean" and "std")')
parser.add_argument('--ae_norm_type', type=str, default='bn', choices=['bn', 'ln', "None", "gn"])
parser.add_argument('--use_rope', action='store_true')
parser.add_argument('--use_inv_bn', action='store_true', default=False, help='If True, use inverse batch normalization in the AE (i.e., normalize the input to the AE and denormalize the output of the AE). This is only applicable if ae_norm_type is set to "bn".')
parser.add_argument('--sync_batch_norm', action='store_true', default=False, help='Use Sync Batch Norm')
parser.add_argument('--ae_wd', type=float, default=0.15, help='Weight decay for the AE')
# Segm Head Parameters
parser.add_argument('--train_head', action='store_true', default=False)
parser.add_argument('--num_classes', type=int, default=19, choices=[19, 256, 1, 2, 3], help="19 Classes for segmentation, 256(classification) for depth, 3 for Normals")
parser.add_argument('--use_bn', action='store_true', default=False)
parser.add_argument('--use_cls', action='store_true', default=False)
parser.add_argument('--nfeats', type=int, default=256)
parser.add_argument('--dpt_out_channels', type=parse_list, default=[128, 256, 1024, 1024])
parser.add_argument('--head_ckpt', type=str, default=None)
# training parameters
parser.add_argument('--max_epochs', type=int, default=800) 
parser.add_argument('--seed', type=int, default=123)
parser.add_argument('--single_step_sample_train', action='store_true', default=False)
parser.add_argument('--precision', type=str, default='32-true',choices=['16-true','16-mixed','32-true', '32'])
parser.add_argument('--ckpt', type=str, default=None, help='Path of a checkpoint to resume training')
parser.add_argument('--num_gpus', type=int, default=1)
parser.add_argument('--accum_iter', type=int, default=1)
parser.add_argument('--warmup_p', type=float, default=0.0)
parser.add_argument('--lr_base', type=float, default=1e-3)
parser.add_argument('--weight_decay', type=float, default=0)
parser.add_argument('--scheduler', type=str, default="cosine", choices=["cosine", "poly"])
parser.add_argument('--optimizer', type=str, default="adam", choices=["adam", "adamw"])
parser.add_argument('--gclip', type=float, default=1.0)
parser.add_argument('--evaluate', action='store_true', default=False)
parser.add_argument('--eval_last', action='store_true', default=False)
parser.add_argument('--eval_ckpt_only', action='store_true', default=False)
parser.add_argument('--eval_mode_during_training', action='store_true', help='if activated (True) it uses the evaluation mode (i.e., step-by-step prediction and mIoU computation) during the training loop')
parser.add_argument('--eval_freq', type=int, default=1)
parser.add_argument('--use_val_to_train', action='store_true', default=False)
parser.add_argument('--use_train_to_val', action='store_true', default=False)

parser.add_argument('--eval_midterm', action='store_true', default=False)
parser.add_argument('--eval_longterm_ns', action='store_true', default=False)
parser.add_argument('--eval_unroll_steps', type=int, default=None,
                    help='Number of autoregressive rollout steps at eval time, overriding the '
                         'implicit 1 / 3 (--eval_midterm) / 6 (--eval_longterm_ns). The scored '
                         'frame is window_end + (unroll_steps-1)*stride, so on Kubric '
                         '--eval_unroll_steps 8 reproduces VFMF: context [0,2,4,6] -> frame 22.')

parser.add_argument('--compute_frechet_distance', action='store_true', default=False, help='If True, compute Frechet Distance between the predicted and target latents at the end of each epoch. This is only applicable if model_gaussian is True.')
parser.add_argument('--high_res_adapt', action='store_true', default=False)
parser.add_argument('--eps_adamw', type=float, default=1e-8, help='Epsilon for AdamW optimizer (default: 2e-6)')

args = parser.parse_args()


args.eval_mode = args.eval_mode_during_training

if args.eval_unroll_steps is None:
    if args.eval_midterm:
        args.eval_unroll_steps = 8 if args.dataset == 'kubric' else 3
    elif args.eval_longterm_ns:
        args.eval_unroll_steps = 6 # for longer term 9
    else:
        args.eval_unroll_steps = 1
print(f"Eval rollout steps: {args.eval_unroll_steps}")
pl.seed_everything(args.seed, workers=True)


if args.dataset == 'cityscapes':
    args.data_path = args.cs_data_path if args.cs_data_path is not None else args.data_path
    data = CS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
elif args.dataset == 'nuscenes':
    args.data_path = args.ns_data_path if args.ns_data_path is not None else args.data_path
    data = NS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
elif args.dataset == 'covla':
    args.data_path = args.covla_data_path if args.covla_data_path is not None else args.data_path
    data = COVLA_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
elif args.dataset == 'kubric':
    args.data_path = args.kubric_data_path if args.kubric_data_path is not None else args.data_path
    data = KUBRIC_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
elif args.dataset == 'multisource':
    # Handle multisource dataset
    if args.cs_data_path is None:
        args.cs_data_path = args.data_path  # Use data_path as default for CityScapes
    if args.ns_data_path is None:
        raise ValueError("NuScenes data path (--ns_data_path) is required for multisource dataset")
    print(f"Creating multisource dataset with:")
    print(f"  CityScapes path: {args.cs_data_path}")
    print(f"  NuScenes path: {args.ns_data_path}")
    print(f"  CoVLA path: {args.covla_data_path}")
    print(f"  Sampling mode: {args.sampling_mode}")
    print(f"  Custom collate: {args.use_custom_collate}")
    
    data = create_multisource_datamodule(
        arguments=args,
        cs_data_path=args.cs_data_path,
        ns_data_path=args.ns_data_path,
        covla_data_path=args.covla_data_path,
        use_custom_collate=args.use_custom_collate,
        sampling_mode=args.sampling_mode
    )
else:
    raise ValueError('Unknown dataset')

if args.precision == '32':
    args.precision = 32

if 'WORLD_SIZE' in os.environ and 'LOCAL_RANK' in os.environ:
    # Started by torchrun / torch.distributed.run, or a child spawned by Lightning
    # (Lightning's children get LOCAL_RANK/NODE_RANK/WORLD_SIZE but no RANK)
    args.gpu = int(os.environ['LOCAL_RANK'])
    args.world_size = int(os.environ['WORLD_SIZE'])
    args.rank = int(os.environ.get('RANK', int(os.environ.get('NODE_RANK', 0)) * args.num_gpus + args.gpu))
elif int(os.environ.get('SLURM_NTASKS', 1)) > 1:
    # srun with one task per GPU
    args.rank = int(os.environ['SLURM_PROCID'])
    args.world_size = int(os.environ['SLURM_NTASKS'])
    args.gpu = int(os.environ['SLURM_LOCALID'])
else:
    # Plain `python train.py`: Lightning spawns the remaining num_gpus - 1 processes itself
    args.rank = 0
    args.world_size = args.num_gpus
    args.gpu = 0
args.node = args.rank // max(torch.cuda.device_count(), 1)
args.device = torch.device('cuda:' + str(args.gpu))

print(f'rank={args.rank} - world_size={args.world_size} - gpu={args.gpu} - device={args.device}')
args.max_steps = (args.max_epochs * (len(data.train_dataloader()) // (args.world_size * args.accum_iter)))
args.warmup_steps = int(args.warmup_p * args.max_steps)
args.effective_batch_size = args.batch_size * args.world_size * args.accum_iter
args.lr = (args.lr_base * args.effective_batch_size) / 8 # args.lr_base is specified for an effective batch-size of 8
print(f'Effective batch size:{args.effective_batch_size} lr_base={args.lr_base} lr={args.lr} max_epochs={args.max_epochs} - max_steps={args.max_steps}')

print(f'Dataset: {args.dataset}')
if args.dataset == 'multisource':
    print(f'Total samples: {len(data.train_dataloader().dataset)}')
    print(f'CityScapes samples: {data.train_dataloader().dataset.cs_length}')
    print(f'NuScenes samples: {data.train_dataloader().dataset.ns_length}')
    print(f"CoVLA samples: {data.train_dataloader().dataset.covla_length}")
    print(f'Total val samples: {len(data.val_dataloader().dataset)}')
    print(f'CityScapes validation samples: {data.val_dataloader().dataset.cs_length}')
    print(f'NuScenes validation samples: {data.val_dataloader().dataset.ns_length}')
elif args.dataset == 'covla':
    print(f'Total samples: {len(data.train_dataloader().dataset)}')
    print(f'Total val samples: {len(data.val_dataloader().dataset)}')
elif args.dataset == 'cityscapes':
    print(f'Total samples: {len(data.train_dataloader().dataset)}')
    print(f'Total val samples: {len(data.val_dataloader().dataset)}')
elif args.dataset == 'nuscenes':
    print(f'Total samples: {len(data.train_dataloader().dataset)}')
    print(f'Total val samples: {len(data.val_dataloader().dataset)}')
elif args.dataset == 'kubric':
    print(f'Total samples: {len(data.train_dataloader().dataset)}')
    print(f'Total val samples: {len(data.val_dataloader().dataset)}')



if not args.high_res_adapt:
    lf_model = Latent_Foresight(args)
else:
    lf_model = Latent_Foresight.load_from_checkpoint(args.ckpt,args=args,strict=False, map_location='cpu', weights_only=False) # For Finetuning
    

callbacks = []
checkpoint_callback = pl.callbacks.ModelCheckpoint(monitor='val/loss', mode='min', save_last=True, save_top_k=2)
callbacks.append(checkpoint_callback)

if args.dst_path is None:
    args.dst_path = os.getcwd()
if args.max_epochs < args.eval_freq:
    args.eval_freq = 1
num_nodes = int(os.environ.get('SLURM_JOB_NUM_NODES', 1))
trainer = pl.Trainer(
    accelerator='gpu',
    strategy=(DDPStrategy(find_unused_parameters=False) if args.num_gpus > 1 else 'auto'),
    devices=args.num_gpus,
    num_nodes=num_nodes,
    callbacks=callbacks,
    max_epochs=args.max_epochs,
    gradient_clip_val=args.gclip,
    default_root_dir=args.dst_path,
    precision=args.precision,
    log_every_n_steps=5,
    check_val_every_n_epoch=args.eval_freq,
    accumulate_grad_batches=args.accum_iter,
    sync_batchnorm = args.sync_batch_norm)

if not args.eval_ckpt_only:
   if args.ckpt and not args.high_res_adapt:
       trainer.fit(lf_model,data,ckpt_path=args.ckpt, weights_only=False)
   else:
       trainer.fit(lf_model,data)
else:
    args.evaluate = True
    args.eval_last = True
    checkpoint_callback.last_model_path = args.ckpt
    if args.dataset == 'cityscapes':
        data = CS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
    elif args.dataset == 'nuscenes':
        data = NS_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
    elif args.dataset == 'kubric':
        data = KUBRIC_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
    elif args.dataset == "covla":
        data = COVLA_VideoData(arguments=args,subset='train',batch_size=args.batch_size)
    elif args.dataset == 'multisource':
        data = create_multisource_datamodule(
            arguments=args,
            cs_data_path=args.cs_data_path,
            ns_data_path=args.ns_data_path,
            use_custom_collate=args.use_custom_collate,
            random_sampling=args.random_sampling
        )    
    else:
        raise ValueError('Unknown dataset')

# Evaluation
if args.evaluate:
    args.eval_mode = True
    if not args.eval_last:
        print('Loading best model')
        checkpoint_path = checkpoint_callback.best_model_path
    else:
        print('Loading last model')
        checkpoint_path = checkpoint_callback.last_model_path

    print(f'checkpoint_path = {checkpoint_path}')
    
    lf_model = Latent_Foresight.load_from_checkpoint(checkpoint_path, args=args, strict=False, map_location='cpu', weights_only=False)
    
    print('-----------MaskedGIVT.eval_mode = ',lf_model.args.eval_mode)
    lf_model.to(args.device)
    lf_model.eval()

    val_data_loader = data.val_dataloader()
    out_metrics = trainer.validate(model=lf_model, dataloaders=val_data_loader)
    loss = out_metrics[0]['val/mean_loss']
    mse = out_metrics[0]['val/mse']
    cos_sim = out_metrics[0]['val/cos_sim']
    if args.eval_modality=='segm':
        mIoU = out_metrics[0]['val/mIoU']
        MO_mIoU = out_metrics[0]['val/MO_mIoU']
        # movi_a has no semantic classes, so segmentation there is
        # foreground/background: the second number is the foreground IoU, not a
        # movable-object mIoU, and calling it MO_mIoU in results.txt would be
        # misleading when comparing against Cityscapes numbers.
        secondary_name = 'fg_IoU' if args.dataset == 'kubric' else 'MO_mIoU'
        # Only logged for a binary fg/bg task; under Cityscapes' 19-class
        # labelling class 0 is "road", not background.
        bg_IoU = out_metrics[0].get('val/bg_IoU')
        if args.rank==0:
            result_path = os.path.join(trainer.log_dir,'results.txt')
            with open(result_path,'w') as f:
                f.write(f'Mean Loss: {loss}\n')
                f.write(f'MSE: {mse}\n')
                f.write(f'Cosine Similarity: {cos_sim}\n')
                if args.eval_modality=='segm':
                    f.write(f'mIoU: {mIoU}\n')
                    f.write(f'{secondary_name}: {MO_mIoU}\n')
                    if bg_IoU is not None:
                        f.write(f'bg_IoU: {bg_IoU}\n')
            print(f'Results saved in {result_path}')
    elif args.eval_modality == 'depth':
        d1 = out_metrics[0]["d1"]
        d2 = out_metrics[0]["d2"]
        d3 = out_metrics[0]["d3"]
        abs_rel = out_metrics[0]["abs_rel"]
        rmse = out_metrics[0]["rmse"]
        rmse_log = out_metrics[0]["rmse_log"]
        silog = out_metrics[0]["silog"]
        sq_rel = out_metrics[0]["sq_rel"]
        log_10 = out_metrics[0]["log_10"]
        if args.rank == 0:
            # Save d1 to a text file
            result_path = os.path.join(trainer.log_dir, 'results.txt')
            with open(result_path, 'w') as f:
                f.write(f'Mean Loss: {loss}\n')
                f.write(f'MSE: {mse}\n')
                f.write(f'd1: {d1}\n')
                f.write(f'd2: {d2}\n')
                f.write(f'd3: {d3}\n')
                f.write(f'abs_rel: {abs_rel}\n')
                f.write(f'rmse: {rmse}\n')
                f.write(f'rmse_log: {rmse_log}\n')
                f.write(f'sq_rel: {sq_rel}\n')
                f.write(f'log_10: {log_10}\n')
                f.write(f'silog: {silog}\n')
                f.write(f'Cosine Similarity: {cos_sim}\n')
            print(f'Results saved at: {result_path}')
    elif args.eval_modality == 'surface_normals':
        mean_ae = out_metrics[0]["mean_ae"]
        median_ae = out_metrics[0]["median_ae"]
        rmse = out_metrics[0]["rmse"]
        a1 = out_metrics[0]["a1"]
        a2 = out_metrics[0]["a2"]
        a3 = out_metrics[0]["a3"]
        a4 = out_metrics[0]["a4"]
        a5 = out_metrics[0]["a5"]
        if args.rank == 0:
            # Save d1 to a text file
            result_path = os.path.join(trainer.log_dir, 'results.txt')
            with open(result_path, 'w') as f:
                f.write(f'Mean Loss: {loss}\n')
                f.write(f'MSE: {mse}\n')
                f.write(f'mean_ae: {mean_ae}\n')
                f.write(f'median_ae: {median_ae}\n')
                f.write(f'rmse: {rmse}\n')
                f.write(f'a1: {a1}\n')
                f.write(f'a2: {a2}\n')
                f.write(f'a3: {a3}\n')
                f.write(f'a4: {a4}\n')
                f.write(f'a5: {a5}\n')
                f.write(f'Cosine Similarity: {cos_sim}\n')
            print(f'Results saved at: {result_path}')
