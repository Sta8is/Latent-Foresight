import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch import optim
import einops
from time import time
import torchvision.transforms as T
from torchvision.transforms import functional as TF
import numpy as np
from torchmetrics import JaccardIndex
from torchmetrics.regression import MeanSquaredError
from torchmetrics.aggregation import MeanMetric
import math
import sys
from src.modules.dpt import DPTHead
from src.modules.autoencoders import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont
from src.metrics import *
import matplotlib.pyplot as plt
from src.modules.dit import MaskTransformer


# Size of the frame label burned into the debug gifs. The PIL bitmap default is
# ~11px, which is unreadable on a 896x448 frame -- bump this to resize the label.
LABEL_FONT_SIZE = 32


def label_font(size=LABEL_FONT_SIZE):
    try:
        return ImageFont.load_default(size=size)  # scalable default, Pillow >= 10.1
    except TypeError:
        # Older Pillow only has the fixed-size bitmap font.
        return ImageFont.load_default()


def draw_label(img, text):
    """Frame label with a dark outline so it stays legible over sky//bright pixels."""
    ImageDraw.Draw(img).text((12, 12), text, fill=(255, 255, 255), font=label_font(),
                             stroke_width=2, stroke_fill=(0, 0, 0))


CITYSCAPES_PALETTE = np.array([
    (128, 64,128), (244, 35,232), ( 70, 70, 70), (102,102,156), (190,153,153),
    (153,153,153), (250,170, 30), (220,220,  0), (107,142, 35), (152,251,152),
    ( 70,130,180), (220, 20, 60), (255,  0,  0), (  0,  0,142), (  0,  0, 70),
    (  0, 60,100), (  0, 80,100), (  0,  0,230), (119, 11, 32),
], dtype=np.uint8)


def depth_colormap(name='turbo'):
    """256-entry RGB lookup table for colouring an argmax'd depth bin index."""
    cmap = plt.get_cmap(name, 256)
    return (np.array([cmap(i)[:3] for i in range(256)]) * 255).astype(np.uint8)


def segm_palette(num_classes):
    """RGB lookup table for colouring a label map, indexed by train id."""
    if num_classes == len(CITYSCAPES_PALETTE):
        return CITYSCAPES_PALETTE
    # Any other label space (e.g. Kubric's binary fg/bg) just gets evenly
    # spaced colours so the classes stay visually distinguishable.
    cmap = plt.get_cmap('tab20', num_classes)
    return (np.array([cmap(i)[:3] for i in range(num_classes)]) * 255).astype(np.uint8)



class Latent_Foresight(pl.LightningModule):
    def __init__(self,args):
        super(Latent_Foresight, self).__init__()
        self.args = args
        self.sequence_length = args.sequence_length 
        self.batch_size = args.batch_size 
        self.hidden_dim = args.hidden_dim 
        self.heads = args.heads
        self.layers = args.layers 
        self.dropout = args.dropout
        self.img_size  = args.img_size
        self.d_layers = args.d_layers
        self.patch_size = 14 # To be added as arg
        self.d_num_layers = len(self.d_layers) if isinstance(self.d_layers, list) else self.d_layers
        self.shape = (self.sequence_length,self.img_size[0]//(self.patch_size), self.img_size[1]//(self.patch_size)) 
        self.dino_v2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitb14_reg', pretrained=True)
        for param in self.dino_v2.parameters():
            param.requires_grad = False
        self.dino_v2.eval()
        self.feature_dim = self.dino_v2.embed_dim

        self.patch_h = self.img_size[0] // self.patch_size  
        self.patch_w = self.img_size[1] // self.patch_size


        self.ae = TransformerAE(feat_dim=4*768, layers=self.args.ae_layers, heads=self.args.ae_attn_heads, hidden_dim=self.args.ae_hidden_dim, 
                                    bottleneck_dim=self.args.bottleneck_dim, feature_size=(self.patch_h, self.patch_w),
                                    model_gaussian=self.args.model_gaussian, assymetric=False, norm_type=self.args.ae_norm_type, 
                                    use_inv_bn=self.args.use_inv_bn, use_rope=self.args.use_rope)
        self.embedding_dim = self.ae.bottleneck_dim
        
        ''' For pretrained ae - Two stage 
        if self.args.ae_ckpt is not None:
            ae_ckpt = torch.load(self.args.ae_ckpt, map_location='cpu', weights_only=False)
            # Extract only ae weights and remove 'ae.' prefix
            ae_state_dict = {k.replace('ae.', ''): v for k, v in ae_ckpt['state_dict'].items() if k.startswith('ae.')}
            self.ae.load_state_dict(ae_state_dict, strict=False)
            self.ae.eval()
            for param in self.ae.parameters():
                param.requires_grad = False
            self.ae.norm.train()  # allows running_mean/running_var to update
            print(f"AE loaded from {self.args.ae_ckpt} (frozen). "
                    f"ae.norm ({self.ae.norm.__class__.__name__}, affine=False) set to train mode — " 
                    f"running stats will update. training={self.ae.norm.training}")
        '''


        self.mean_stds = torch.load(args.meanstd_ckpt, weights_only=False)
        self.mean_f = nn.Parameter(torch.tensor(self.mean_stds['mean']), requires_grad=False)
        self.std_f = nn.Parameter(torch.tensor(self.mean_stds['std']), requires_grad=False)

        self.maskvit = MaskTransformer(shape=self.shape, embedding_dim=self.embedding_dim, hidden_dim=self.hidden_dim, depth=self.layers,
                                       heads=self.heads, mlp_dim=4*self.hidden_dim, dropout=self.dropout,use_fc_bias=args.use_fc_bias,
                                       seperable_attention=args.seperable_attention,seperable_window_size=args.seperable_window_size,
                                       use_first_last=args.use_first_last)
        self.P_mean = args.P_mean
        self.P_std = args.P_std

        self.embed = nn.Linear(self.embedding_dim, self.hidden_dim, bias=True)
        # nn.init.zeros_(self.embed_sc.bias)
        self.maskvit.fc_in = nn.Identity()
        # Necessary for evaluation
        self.mean_metric = MeanMetric()
        self.mse_metric = MeanSquaredError()
        self.mean_metric_cos = MeanMetric()
        self._fd_pred_feats: list = []
        self._fd_gt_feats: list = []
        if self.args.eval_modality in ["segm", "depth", "surface_normals", "rgb"]:
            self.ignore_index = 255 if self.args.eval_modality == "segm" else 0
            # An RGB head always emits 3 channels, so it does not depend on
            # --num_classes the way the label-space heads do.
            nclass = 3 if self.args.eval_modality == "rgb" else self.args.num_classes
            self.head = DPTHead(nclass=nclass,in_channels=self.feature_dim, features=self.args.nfeats,
                                use_bn=self.args.use_bn, out_channels=self.args.dpt_out_channels, use_clstoken=self.args.use_cls)
            # Load the head checkpoint
            if self.args.head_ckpt is not None:
                state_dict = {}
                for k, v in torch.load(self.args.head_ckpt, weights_only=False)["state_dict"].items():
                    state_dict[k.replace("head.","")] = v
                self.head.load_state_dict(state_dict, strict=False)
                self.head.eval()
                for param in self.head.parameters():
                    param.requires_grad = False
            if self.args.eval_modality == "segm":
                self.iou_metric = JaccardIndex(task="multiclass", num_classes=self.args.num_classes, ignore_index=self.ignore_index, average=None)
            elif self.args.eval_modality == "depth":
                self.ignore_index = 0
                self.d1 = MeanMetric()
                self.d2 = MeanMetric()
                self.d3 = MeanMetric()
                self.abs_rel = MeanMetric()
                self.rmse = MeanMetric()
                self.log_10 = MeanMetric()
                self.rmse_log = MeanMetric()
                self.silog = MeanMetric()
                self.sq_rel = MeanMetric()
            elif self.args.eval_modality == "surface_normals":
                self.mean_ae = MeanMetric()
                self.median_ae = MeanMetric()
                self.rmse = MeanMetric()
                self.a1 = MeanMetric()
                self.a2 = MeanMetric()
                self.a3 = MeanMetric()
                self.a4 = MeanMetric()
                self.a5 = MeanMetric()
        self.batch_crops = []
        self.random_crop = T.RandomCrop(32,64)
        if self.args.dataset == "cityscapes":
            self.size = (1024, 2048)
        elif self.args.dataset == "nuscenes":
            self.size = (900, 1600)
        elif self.args.dataset == "kubric":
            self.size = tuple(self.img_size)
        else:
            self.size = tuple(self.img_size)
        # Cityscapes' movable-object classes start at index 11. That slice is
        # empty for a 2-class (foreground/background) task like Kubric's, where
        # the analogous number is simply the foreground IoU.
        self.segm_secondary_name = "fg_IoU" if self.args.num_classes <= 11 else "MO_mIoU"
        # self.mean_var = MeanMetric()
        self.save_hyperparameters()

    

    def extract_features(self, x, reshape=False):
        bt, c, h, w = x.shape # x.shape [B,T,H,W,C]
        with torch.no_grad():
            x = self.dino_v2.get_intermediate_layers(x,n=self.d_layers, reshape=reshape)
            if self.d_num_layers > 1:
                x = torch.cat(x,dim=-1)
            else:
                x = x[0]
        return x
            

    def preprocess(self, x):
        B, T, C, H, W = x.shape
        # DINOv2 accepts 4 dimensions [B,C,H,W]. 
        # We use flatten at batch and time dim of x.
        x = x.flatten(end_dim=1) # x.shape [B*T,C,H,W]
        x = self.extract_features(x) # [B*T,H*W,C]
        x_norm  = (x - self.mean_f) / self.std_f
        ## Encode
        x, kl = self.ae.encode(x_norm)
        x = einops.rearrange(x, '(b t) (h w) c -> b t h w c',b=B, t=T, h=self.patch_h, w=self.patch_w)
        return x, x_norm, kl 

    def postprocess(self, x, argmax=True):
        B, T, H, W, C = x.shape
        x = einops.rearrange(x, 'b t h w c -> (b t) (h w) c')
        x = self.ae.decode(x)
        x = einops.rearrange(x, '(b t) (h w) c -> b t h w c', b=B, t=T, h=H, w=W)
        # Denormalize
        x = x * self.std_f + self.mean_f
        return x
            

    def get_timesteps_dist(self, step, device):
        # Uniform quantiles of your training distribution
        u = torch.linspace(0, 1, step+1, device=device)
        # Inverse CDF of logistic-normal
        z = torch.erfinv(2 * u - 1) * math.sqrt(2) * self.P_std + self.P_mean
        timesteps = torch.sigmoid(z)
        timesteps, _ = timesteps.sort()
        return timesteps

    
    def sample(self, x, batch_idx, step=15, mask_frames=1):
        B = x.shape[0]
        self.maskvit.eval()
        with torch.no_grad():
            n_start = 0
            x, x_norm, _ = self.preprocess(x)  # [B, T, H, W, C]
            B,T,H,W,C = x.shape
            timesteps = self.get_timesteps_dist(step, x.device)
            # For Context Frames() z = x since t=1
            z_context = x[:, :-mask_frames] # [B, T-1, H, W, C]
            # For Future Frame z starts as pure noise
            noise = torch.randn_like(x[:, -mask_frames:])
            z_future = noise
            # z_future = timesteps[n_start] * x[:, -mask_frames:] + (1 - timesteps[n_start]) * torch.randn_like(x[:, -mask_frames:])
            z = torch.cat([z_context, z_future], dim=1)
            # SC signal for future frames; starts as zeros (null condition at step 0)
            x_sc_future = torch.zeros_like(x[:, -mask_frames:])

            z_futures = []
            for i in range(n_start, step):
                t_scalar = timesteps[i]
                t_next_scalar = timesteps[i + 1]

                t_ctx = torch.ones(B, self.sequence_length - mask_frames, 1, 1, 1, device=x.device)
                t_fut = torch.full((B, mask_frames, 1, 1, 1),t_scalar.item(), device=x.device)
                t = torch.cat([t_ctx, t_fut], dim=1)

                z_emb = self.embed(z)
                x_pred, = self.maskvit(z_emb, timestep=t[:, -1:])
                x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c', sl=self.shape[0], h=self.shape[1], w=self.shape[2])
                # Extract only the future frame prediction
                x_pred_future = x_pred[:, -mask_frames:]
                v_pred_future = (x_pred_future - z_future) / (1 - t_scalar).clamp_min(5e-2)
                x_sc_future = x_pred_future
                # Update future frame with Euler step. Z_context remains fixed as x since it's clean and at t=1.
                dt = t_next_scalar - t_scalar
                z_future = z_future + dt * v_pred_future
                
                
                z[:, -mask_frames:] = z_future
                
                pred_loss = F.mse_loss(x_pred_future, x[:, -mask_frames:]).item()
                

            # Decode latents → DINOv2 feature space [B, T, H, W, C] 
            samples = self.postprocess(z)

            pred_flat  = einops.rearrange(z[:, -1], 'b h w c -> b (h w) c')
            pred_dec   = self.ae.decode(pred_flat)
            x_norm_fut = x_norm.unflatten(0, (B, self.sequence_length))[:, -1]
            loss       = F.mse_loss(pred_dec, x_norm_fut).item()

        return samples, loss
    

    def sample_unroll(self, x, gt_feats, step=15, mask_frames=1, unroll_steps=3,
                      num_samples=None, batch_idx=None):
        B = x.shape[0]
        
        self.maskvit.eval()
        with torch.no_grad():
            x, _, _ = self.preprocess(x)  # [B, T, H, W, C]
            H, W = x.shape[2], x.shape[3]
            timesteps = torch.linspace(0.0, 1.0, step + 1, device=x.device)


            for s in range(unroll_steps):
                # For Context Frames() z = x since t=1
                z_context = x[:, :-mask_frames] # [B, T-1, H, W, C]
                Bn = z_context.shape[0]
                # For Future Frame z starts as pure noise (independent per draw)
                z_future = torch.randn(
                    (Bn, mask_frames) + tuple(x.shape[2:]), device=x.device, dtype=x.dtype
                )
                z = torch.cat([z_context, z_future], dim=1)
                x_sc_future = torch.zeros_like(z_future)

                for i in range(step):
                    t_scalar = timesteps[i]
                    t_next_scalar = timesteps[i + 1]

                    t_ctx = torch.ones(Bn, self.sequence_length - mask_frames, 1, 1, 1, device=x.device)
                    t_fut = torch.full((Bn, mask_frames, 1, 1, 1), t_scalar.item(), device=x.device)
                    t = torch.cat([t_ctx, t_fut], dim=1)
                    z_emb = self.embed(z)

                    
                    x_pred, = self.maskvit(z_emb, timestep=t[:, -1:])
                    
                    x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c', sl=self.shape[0], h=self.shape[1], w=self.shape[2])
                    # Extract only the future frame prediction
                    x_pred_future = x_pred[:, -mask_frames:]
                    x_sc_future = x_pred_future
                    # Derive velocity prediction for the future frame
                    v_pred_future = (x_pred_future - z_future) / (1 - t_scalar).clamp_min(5e-2)
                    # Update future frame with Euler step. Z_context remains fixed as x since it's clean and at t=1.
                    dt = t_next_scalar - t_scalar
                    z_future = z_future + dt * v_pred_future
                    z = torch.cat([z_context, z_future], dim=1)

                # Roll window forward with newly predicted future frame(s).
                # Same layout as before: drop the oldest frame, append the new
                # one twice (the trailing copy is the slot the next step
                # re-noises), rebuilt from x so it stays at batch B after the
                # averaging above.
                x = torch.cat([x[:, mask_frames:-mask_frames], z_future, z_future], dim=1)
            prediction = self.postprocess(x)
            loss = F.mse_loss(
                prediction[:, -1].flatten(end_dim=-2),
                gt_feats.flatten(end_dim=-2),
            )

        return prediction, loss

    def forward_loss(self, pred, target, t):
        """MSE on the predicted future frame (pred/target are the v- or x-space tensors)."""
        target = target[:,-1] # Only compute loss on the predicted future frame
        pred = pred[:,-1] # Only compute loss on the predicted future frame
        if self.args.detach_target:
            target = target.detach()
        loss = (target - pred) ** 2
        return loss.mean()

 

    def forward(self, x, t):
        e = torch.randn_like(x)
        # e = torch.randn_like(x) * self.enc_std
        z = t * x + (1 - t) * e
        if self.args.detach_future:
            z = torch.cat([z[:, :-1], z[:, -1:].detach()], dim=1)
        x_pred, = self.maskvit(self.embed(z), timestep=t[:, -1:])
        x_pred = einops.rearrange(x_pred, 'b (sl h w) c -> b sl h w c',sl=self.shape[0], h=self.shape[1], w=self.shape[2])
        if self.args.x_loss:
            loss = self.forward_loss(pred=x_pred, target=x, t=t)
        else:
            v = (x - z) / (1 - t).clamp_min(5e-2)
            v_pred = (x_pred - z) / (1 - t).clamp_min(5e-2)
            loss = self.forward_loss(pred=v_pred, target=v, t=t)
        return loss, x_pred


    def sample_t(self, n: int, device=None):
        z = torch.randn(n, device=device) * self.P_std + self.P_mean
        return torch.sigmoid(z)

    def sample_t_shifted_uniform(self, n: int, alpha: float, device=None):
        """Sample t in [0, 1] from p(t)=alpha/(alpha+(1-alpha)t)^2 via inverse CDF."""
        if alpha <= 0:
            raise ValueError(f"shifted_uniform_alpha must be > 0, got {alpha}")
        u = torch.rand(n, device=device)
        return (alpha * u) / (1.0 - (1.0 - alpha) * u)


    def training_step_single(self, batch, batch_idx):
        B = batch.shape[0]
        x, x_norm, kl = self.preprocess(batch)
        # 2. Per-channel stats [B*T*H*W, C]
    
        t = self.sample_t(B, device=x.device).view(-1, *([1] * (x.ndim - 1))) # from shape [B] to [B,1,1,1,1]
        
        # For the dim=1 (sequence length) we want t for last frame and t=1 for the rest. Concatenate [B,1,1,1,1] with torch.ones of shape [B,sequence_length-1,1,1,1] along dim=1 to get t of shape [B,sequence_length,1,1,1]
        t_clean = torch.ones(B, self.sequence_length-1, 1, 1, 1).to(x.device)
        # Concatenate t and t_clean along dim=1
        t = torch.cat((t_clean, t), dim=1)
        loss, x_pred = self.forward(x, t)
        self.log("Train/pred_loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        if self.args.use_recon_losses:
            recon = self.ae.decode(einops.rearrange(x, 'b t h w c -> (b t) (h w) c'))
            # Weights default to 1.0, i.e. the original mse + cosine sum. Zeroing
            # one isolates the other, which is what the MSE-only / cosine-only
            # ablations use. Both terms are logged separately either way so their
            # relative contribution stays visible.
            w_mse = self.args.recon_mse_weight
            w_cos = self.args.recon_cos_weight
            mse_loss = F.mse_loss(recon, x_norm)
            cos_sim_loss = 1 - F.cosine_similarity(recon, x_norm, dim=-1).mean()
            recon_loss = (w_mse * mse_loss + w_cos * cos_sim_loss)
            self.log("Train/recon_mse", mse_loss, batch_size=B*self.sequence_length, logger=True, on_step=True, rank_zero_only=True)
            self.log("Train/recon_cos", cos_sim_loss, batch_size=B*self.sequence_length, logger=True, on_step=True, rank_zero_only=True)
            # Reconstruction Loss on predicted
            recon_pred = self.ae.decode(einops.rearrange(x_pred[:, -1], 'b h w c -> b (h w) c'))
            x_norm_re = x_norm.unflatten(dim=0, sizes=(B, self.sequence_length))[:,-1] 
            mse_loss_pred = F.mse_loss(recon_pred, x_norm_re)
            cos_sim_loss_pred = 1 - F.cosine_similarity(recon_pred, x_norm_re, dim=-1).mean()
            recon_loss_pred = (w_mse * mse_loss_pred + w_cos * cos_sim_loss_pred)
            loss += recon_loss_pred
            self.log("Train/recon_loss_pred", recon_loss_pred, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
            self.log("Train/recon_mse_pred", mse_loss_pred, batch_size=B, logger=True, on_step=True, rank_zero_only=True)
            self.log("Train/recon_cos_pred", cos_sim_loss_pred, batch_size=B, logger=True, on_step=True, rank_zero_only=True)
            loss += recon_loss
            self.log("Train/recon_loss", recon_loss, batch_size=B*self.sequence_length, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        if self.args.model_gaussian:
            kl_loss = self.args.kl_weight * kl.mean()
            loss += kl_loss
            self.log("Train/kl_loss", kl_loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        # `self.training` matters: validation_step falls through to training_step when
        # args.eval_mode is False, and the cross-rank reduction below is a collective.
        if getattr(self.args, 'use_sigreg', False) and self.training:
            if self.args.sigreg_tokens > 0 and self.args.sigreg_tokens < x_flat.size(0):
                idx = torch.randperm(x_flat.size(0), device=x_flat.device)[:self.args.sigreg_tokens]
                z_sig = x_flat[idx]
            else:
                z_sig = x_flat
            # A must be bit-identical across ranks for the all-reduce to be valid,
            # so seed from global_step rather than drawing from the (desynced) global RNG.
            gen = torch.Generator(device=z_sig.device)
            gen.manual_seed(int(self.args.seed) * 1000003 + int(self.global_step))
            sync = not self.args.sigreg_local
            stat, stat_max, n_glob = self.sigreg(z_sig, gen=gen, sync=sync)
            sigreg_w = self.args.sigreg_weight
            if self.args.sigreg_warmup_steps > 0:
                sigreg_w *= min(1.0, self.global_step / self.args.sigreg_warmup_steps)
            sigreg_loss = sigreg_w * stat
            loss = loss + sigreg_loss
            if self._sigreg_null is None:
                with torch.no_grad():
                    null_stat, _, _ = self.sigreg(torch.randn_like(z_sig), gen=gen, sync=sync)
                    self._sigreg_null = (null_stat * n_glob).item()
            self.log("SIGReg/loss",      sigreg_loss,      batch_size=B, logger=True, on_step=True, prog_bar=True,  rank_zero_only=True)
            self.log("SIGReg/stat",      stat * n_glob,    batch_size=B, logger=True, on_step=True, prog_bar=False, rank_zero_only=True)
            self.log("SIGReg/stat_max",  stat_max * n_glob, batch_size=B, logger=True, on_step=True, prog_bar=False, rank_zero_only=True)
            self.log("SIGReg/stat_null", self._sigreg_null, batch_size=B, logger=True, on_step=True, prog_bar=False, rank_zero_only=True)
            z32 = z_sig.detach().float()
            z32c = z32 - z32.mean(0)
            self.log("Latent/kurtosis", (z32.pow(4).mean(0) / z32.var(0).clamp_min(1e-6).pow(2) - 3.0).mean(),
                     batch_size=B, logger=True, on_step=True, prog_bar=False, rank_zero_only=True)
            cov = (z32c.T @ z32c) / max(z32c.size(0) - 1, 1)
            self.log("Latent/cov_offdiag_rms", (cov - torch.diag(torch.diag(cov))).pow(2).mean().sqrt(),
                     batch_size=B, logger=True, on_step=True, prog_bar=False, rank_zero_only=True)
        self.log("Train/loss", loss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        # self.log("Train/of_loss", ofloss, batch_size=B, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        lr = self.optimizers().optimizer.param_groups[0]["lr"]
        self.log("Train/lr", lr, logger=True, on_step=True, prog_bar=True, rank_zero_only=True)
        return loss

    def training_step(self, batch, batch_idx):
        # provide only batch of 5 frames
        return self.training_step_single(batch[:, :5], batch_idx)

    def validation_step(self, batch, batch_idx):
        if self.args.eval_mode:
            return self.evaluation_step(batch, batch_idx)
        B = batch[0].shape[0]
        loss = self.training_step(batch, batch_idx)
        self.log('val/loss', loss, prog_bar=True, batch_size=B, logger=True, on_step=True)


    def prepare_segm_gt(self, gt_segm):
        """
        Bring segmentation ground truth into the label space the metric expects.

        Kubric MOVi stores per-object instance ids (0 = background, 1..N per
        object), but movi_a has no semantic classes so the task is
        foreground/background -- the same binary target VFMF's fgbg_seg head
        uses (KubricMultiDataset does `s > 0`). Passing raw instance ids to a
        JaccardIndex built with num_classes=2 would feed it labels of 2..N,
        which are outside its label space. Cityscapes/NuScenes already arrive as
        train ids and pass through untouched.
        """
        if self.args.dataset == "kubric":
            return (gt_segm > 0).long()
        return gt_segm

    def movable_object_miou(self, IoU):
        """Cityscapes movable-object mIoU, or foreground IoU for a 2-class task."""
        return torch.mean(IoU[11:] if IoU.numel() > 11 else IoU[1:])

    def background_iou(self, IoU):
        """
        Background IoU, for a binary foreground/background task only.

        Class 0 is genuinely "background" when there are two classes (Kubric
        fg/bg). Under the 19-class Cityscapes labelling index 0 is "road", so
        reporting it as background would be wrong -- hence None there.
        """
        return IoU[0] if IoU.numel() == 2 else None



    def evaluation_step(self, batch, batch_idx):
        B, sl, C, H, W = batch[0].shape
        if self.args.eval_modality is None or self.args.eval_modality == "rgb":
            data_tensor, gt_img= batch
        elif self.args.eval_modality == "segm":
            data_tensor, gt_img, gt_segm = batch
        elif self.args.eval_modality == "depth":
            data_tensor, gt_img, gt_depth = batch
        elif self.args.eval_modality == "surface_normals":
            data_tensor, gt_img, gt_normals = batch
        gt_feats = self.extract_features(gt_img)
        gt_feats = einops.rearrange(gt_feats, 'b (h w) c -> b h w c',h=H//self.patch_size, w=W//self.patch_size)
        

        unroll_steps = getattr(self.args, 'eval_unroll_steps', None)
        if unroll_steps is None:
            unroll_steps = 3 if self.args.eval_midterm else (6 if self.args.eval_longterm_ns else 1)
        if unroll_steps > 1:
            samples, loss = self.sample_unroll(data_tensor,gt_feats,step=self.args.step, unroll_steps=unroll_steps, batch_idx=batch_idx)
        else:
            samples, loss = self.sample(data_tensor,batch_idx=batch_idx,step=self.args.step)
        # Evaluation
        pred_feats = samples[:,-1]

        gt_feats = gt_feats.unsqueeze(1)
        gt_feats = gt_feats.squeeze(1)
        self.mean_metric.update(loss)
        mean_loss = self.mean_metric.compute()
        self.mse_metric.update(pred_feats.flatten(end_dim=-2), gt_feats.flatten(end_dim=-2).contiguous())
        mse = self.mse_metric.compute()
        self.log('val/mse', mse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        self.mean_metric_cos.update(F.cosine_similarity(pred_feats.flatten(end_dim=-2), gt_feats.flatten(end_dim=-2).contiguous(), dim=-1).mean())
        mean_cos_sim = self.mean_metric_cos.compute()
        self.log('val/mean_cos_sim', mean_cos_sim, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        print(f"Iteration {batch_idx}: Validation MSE: {mse:.4f}, Validation Mean Cosine Similarity: {mean_cos_sim:.4f}")
        # Accumulate spatially mean-pooled frame features for dataset-wise Fréchet distance
        if self.args.compute_frechet_distance:
                self._fd_pred_feats.append(pred_feats.mean(dim=(1, 2)).detach().cpu().float())
                self._fd_gt_feats.append(gt_feats.mean(dim=(1, 2)).detach().cpu().float())
        self.log('val/mean_loss', mean_loss, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
        print(f"Iteration {batch_idx}: Validation Loss: {mean_loss:.4f}")
        if self.args.eval_modality == "segm":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            # pred_feats_list = [pred_feats for i in range(4)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_segm = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_segm = F.interpolate(pred_segm, size=self.size, mode='bicubic', align_corners=False)
            self.iou_metric.update(pred_segm, self.prepare_segm_gt(gt_segm.squeeze(1)))
            IoU = self.iou_metric.compute()
            mIoU = torch.mean(IoU)
            MO_mIoU = self.movable_object_miou(IoU)
            bg_IoU = self.background_iou(IoU)
            self.log('val/mIoU', mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            self.log(f'val/{self.segm_secondary_name}', MO_mIoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
            msg = f"Validation mIoU: {mIoU:.4f}, Validation {self.segm_secondary_name}: {MO_mIoU:.4f}"
            if bg_IoU is not None:
                self.log('val/bg_IoU', bg_IoU, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True)
                msg += f", Validation bg_IoU: {bg_IoU:.4f}"        
        elif self.args.eval_modality == "depth":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_depth = self.head(pred_feats_list,self.patch_h,self.patch_w)
            
            pred_depth = F.interpolate(pred_depth, size=(self.size), mode='bicubic', align_corners=False)
            pred_depth = pred_depth.argmax(dim=1).float()
            update_depth_metrics(pred_depth, gt_depth.squeeze(1), self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            self.log('val/d1', d1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/d2', d2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/d3', d3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/abs_rel', abs_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/log_10', log_10, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/rmse_log', rmse_log, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/silog', silog, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/sq_rel', sq_rel, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
        elif self.args.eval_modality == "surface_normals":
            pred_feats_list = [pred_feats[:,:,:,i*self.feature_dim:(i+1)*self.feature_dim] for i in range(self.d_num_layers)]
            pred_feats_list = [einops.rearrange(x, 'b h w c -> b (h w) c',h=H//self.patch_size, w=W//self.patch_size) for x in pred_feats_list]
            pred_normals = self.head(pred_feats_list,self.patch_h,self.patch_w)
            pred_normals = F.interpolate(pred_normals, size=(self.size), mode='bicubic', align_corners=False)
            update_normal_metrics(pred_normals, gt_normals, self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
            self.log('val/mean_ae', mean_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/median_ae', median_ae, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/rmse', rmse, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/a1', a1, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/a2', a2, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/a3', a3, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/a4', a4, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)
            self.log('val/a5', a5, prog_bar=True, batch_size=1, on_step=True, logger=True, rank_zero_only=True, sync_dist=True)

        self.log('val/loss', loss, prog_bar=True, batch_size=data_tensor.shape[0], sync_dist=True, on_step=False, on_epoch=True, logger=True)
        
        
    def on_validation_epoch_end(self):
        if self.args.eval_mode:
            mean_loss = self.mean_metric.compute()
            
            mse = self.mse_metric.compute()
            print(f"Validation MSE: {mse:.4f}")
            self.log_dict({'val/mse': mse}, prog_bar=True, logger=True)
            cos_sim = self.mean_metric_cos.compute()
            print(f"Validation Mean Cosine Similarity: {cos_sim:.4f}")
            self.log_dict({'val/cos_sim': cos_sim}, prog_bar=True, logger=True)
            print(f"Validation Loss: {mean_loss:.4f}")
            self.log_dict({'val/mean_loss': mean_loss}, prog_bar=True, logger=True)
            self.mean_metric.reset()
            self.mse_metric.reset()
            self.mean_metric_cos.reset()
            if self.args.compute_frechet_distance:
                from torchaudio.functional import frechet_distance
                all_pred = torch.cat(self._fd_pred_feats, dim=0)  # (N, C)
                all_gt = torch.cat(self._fd_gt_feats, dim=0)      # (N, C)
                mu_pred = all_pred.mean(0)
                mu_gt = all_gt.mean(0)
                sigma_pred = torch.cov(all_pred.T)
                sigma_gt = torch.cov(all_gt.T)
                fd = frechet_distance(mu_pred, sigma_pred, mu_gt, sigma_gt)
                print(f"Frechet Distance: {fd:.4f}")
                self.log_dict({'val/frechet_distance': fd}, prog_bar=True, logger=True)
                self._fd_pred_feats.clear()
                self._fd_gt_feats.clear()
            if self.args.eval_modality == "segm":
                IoU = self.iou_metric.compute()
                mIoU = torch.mean(IoU)
                MO_mIoU = self.movable_object_miou(IoU)
                bg_IoU = self.background_iou(IoU)
                print("mIoU = %10f" % (mIoU*100))
                print("%s = %10f" % (self.segm_secondary_name, MO_mIoU*100))
                # Logged under both names: val/MO_mIoU is what train_mgivt.py and
                # existing runs read, val/fg_IoU is the honest name on Kubric.
                metrics = {"val/mIoU": mIoU * 100,
                           "val/MO_mIoU": MO_mIoU * 100,
                           f"val/{self.segm_secondary_name}": MO_mIoU * 100}
                if bg_IoU is not None:
                    print("bg_IoU = %10f" % (bg_IoU*100))
                    metrics["val/bg_IoU"] = bg_IoU * 100
                self.log_dict(metrics, logger=True, prog_bar=True)
                self.iou_metric.reset()
            elif self.args.eval_modality == "depth":
                d1, d2, d3, abs_rel, rmse, log_10, rmse_log, silog, sq_rel = compute_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
                print("d1 =%10f" % (d1), "d2 =%10f" % (d2), "d3 =%10f" % (d3), "abs_rel =%10f" % (abs_rel), "rmse =%10f" % (rmse), "log_10 =%10f" % (log_10), "rmse_log =%10f" % (rmse_log), "silog =%10f" % (silog), "sq_rel =%10f" % (sq_rel))
                self.log_dict({"d1":d1, "d2":d2, "d3":d3, "abs_rel":abs_rel, "rmse":rmse, "log_10":log_10, "rmse_log":rmse_log, "silog":silog, "sq_rel":sq_rel}, logger=True, prog_bar=True)
                reset_depth_metrics(self.d1, self.d2, self.d3, self.abs_rel, self.rmse, self.log_10, self.rmse_log, self.silog, self.sq_rel)
            elif self.args.eval_modality == "surface_normals":
                mean_ae, median_ae, rmse, a1, a2, a3, a4, a5 = compute_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)
                print("mean_ae =%10f" % (mean_ae), "median_ae =%10f" % (median_ae), "rmse =%10f" % (rmse), "a1 =%10f" % (a1), "a2 =%10f" % (a2), "a3 =%10f" % (a3), "a4 =%10f" % (a4), "a5 =%10f" % (a5))
                self.log_dict({"mean_ae":mean_ae, "median_ae":median_ae, "rmse":rmse, "a1":a1, "a2":a2, "a3":a3, "a4":a4, "a5":a5}, logger=True, prog_bar=True)
                reset_normal_metrics(self.mean_ae, self.median_ae, self.rmse, self.a1, self.a2, self.a3, self.a4, self.a5)


    def configure_optimizers(self):
        ae_params = list(self.ae.parameters())
        ae_ids = {id(p) for p in ae_params}
        other_params = [p for p in self.parameters() if p.requires_grad and id(p) not in ae_ids]
        ae_wd = self.args.ae_wd if hasattr(self.args, 'ae_wd') else 0.2
        param_groups = [
        {"params": other_params, "weight_decay": self.args.weight_decay},
        {"params": ae_params,    "weight_decay": ae_wd},
        ]
        print(f"Optimizer groups | other: {len(other_params)} params (wd={self.args.weight_decay}) "f"| ae: {len(ae_params)} params (wd={ae_wd})")
        if self.args.optimizer == "adam":
            optimizer = optim.Adam(param_groups, lr=self.args.lr, betas=(0.9, 0.999))
        elif self.args.optimizer == "adamw":
            optimizer = optim.AdamW(param_groups, lr=self.args.lr, betas=(0.9, 0.95))
        else:
            raise NotImplementedError(f"Optimizer {self.args.optimizer} not implemented")
        if self.args.scheduler == "poly":
            main_lr_scheduler = optim.lr_scheduler.PolynomialLR(optimizer, total_iters=self.args.max_steps-self.args.warmup_steps, power=1.0) 
        elif self.args.scheduler == "cosine":
            main_lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.args.max_steps-self.args.warmup_steps)
        if self.args.warmup_steps > 0:
            warmup_lr_scheduler = optim.lr_scheduler.LinearLR(optimizer, total_iters=self.args.warmup_steps)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup_lr_scheduler, main_lr_scheduler], milestones=[self.args.warmup_steps])
        else:
            scheduler = main_lr_scheduler
        assert hasattr(self.args, 'max_steps') and self.args.max_steps is not None, f"Must set max_steps argument"
        return [optimizer], [dict(scheduler=scheduler, interval='step', frequency=1)]