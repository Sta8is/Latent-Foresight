import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.distributed as dist
import einops
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from typing import Optional, Callable, Tuple
from torch import Tensor
# import parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.modules.dit import apply_rotary_pos_emb
from einops import repeat, rearrange


class SwiGLUFFN(nn.Module):
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: Callable[..., nn.Module] = None,
        drop: float = 0.0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.w12 = nn.Linear(in_features, 2 * hidden_features, bias=bias)
        self.w3 = nn.Linear(hidden_features, out_features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        x12 = self.w12(x)
        x1, x2 = x12.chunk(2, dim=-1)
        hidden = F.silu(x1) * x2
        return self.w3(hidden)

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output

class Attention(nn.Module):
    """
    Attention module of LightningDiT.
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.,
        proj_drop: float = 0.,
        norm_layer: nn.Module = nn.LayerNorm,
        fused_attn: bool = True,
        use_rmsnorm: bool = False,
    ) -> None:
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.fused_attn = fused_attn
        
        if use_rmsnorm:
            norm_layer = RMSNorm
            
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim, linear=True, bias=False) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim, linear=True, bias=False) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        
    def forward(self, x: torch.Tensor, rope=None) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)
        
        if rope is not None:
            q_bshd = q.transpose(1, 2)  # [B, N, H, Dh]
            k_bshd = k.transpose(1, 2)  # [B, N, H, Dh]
            q_bshd = apply_rotary_pos_emb(q_bshd, rope, tensor_format="bshd")
            k_bshd = apply_rotary_pos_emb(k_bshd, rope, tensor_format="bshd")
            q = q_bshd.transpose(1, 2)
            k = k_bshd.transpose(1, 2)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.attn_drop.p if self.training else 0.,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class LightningDiTBlock(nn.Module):
    """
    Ours Lightning DiT Block. We add features including: 
    - ROPE
    - QKNorm 
    - RMSNorm
    - SwiGLU
    - Removed AdaLN
    """
    def __init__(
        self,
        hidden_size,
        num_heads,
        mlp_ratio=4.0,
        use_qknorm=False,
        use_swiglu=False, 
        use_rmsnorm=False,
        **block_kwargs
    ):
        super().__init__()
        
        # Initialize normalization layers
        if not use_rmsnorm:
            self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
            self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        else:
            self.norm1 = RMSNorm(hidden_size, linear=True, bias=False)
            self.norm2 = RMSNorm(hidden_size, linear=True, bias=False)

        # Initialize attention layer
        self.attn = Attention(
            hidden_size,
            num_heads=num_heads,
            qkv_bias=True,
            qk_norm=use_qknorm,
            use_rmsnorm=use_rmsnorm,
            **block_kwargs
        )

        # Initialize MLP layer
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        if use_swiglu:
            # here we did not use SwiGLU from xformers because it is not compatible with torch.compile for now.
            self.mlp = SwiGLUFFN(hidden_size, int(2/3 * mlp_hidden_dim))
        else:
            self.mlp = Mlp(
                in_features=hidden_size,
                hidden_features=mlp_hidden_dim,
                act_layer=approx_gelu,
                drop=0
            )
            
    def forward(self, x, feat_rope=None):
        x = x + self.attn(self.norm1(x), rope=feat_rope)
        x = x + self.mlp(self.norm2(x))
        return x


        
class _AllReduceSum(torch.autograd.Function):
    """SUM all-reduce whose backward re-scales by world_size to cancel DDP's 1/R
    gradient averaging, so the net parameter gradient is that of the global sum."""
    @staticmethod
    def forward(ctx, x):
        y = x.clone()
        dist.all_reduce(y, op=dist.ReduceOp.SUM)
        return y

    @staticmethod
    def backward(ctx, g):
        return g * dist.get_world_size()


def _all_reduce_sum(x):
    if not (dist.is_available() and dist.is_initialized()):
        return x
    return _AllReduceSum.apply(x)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (LeJEPA).

    Epps-Pulley characteristic-function test against N(0,1) on random 1-D
    projections; by Cramer-Wold, driving all projections to N(0,1) drives the
    joint distribution to isotropic Gaussian.

    Deviates from the reference in three ways, all required here:
      - returns the *un-scaled* statistic (no ``* N``) so the loss magnitude is
        invariant to token count and world size; the caller applies a plain weight
      - forces fp32 with autocast disabled: ``err`` is a difference of two O(1)
        quantities of size ~1/N, which fp16 cancellation destroys
      - optional exact cross-rank reduction of the cos/sin sums, which requires
        the caller to pass a rank-identical generator for the projections
    """

    def __init__(self, knots=17, sketches=256):
        super().__init__()
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        w = torch.full((knots,), 2 * dt, dtype=torch.float32)
        w[0] = dt
        w[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t, persistent=False)
        self.register_buffer("phi", window, persistent=False)
        self.register_buffer("weights", w * window, persistent=False)
        self.sketches = sketches

    def forward(self, proj, gen=None, sync=True):
        """proj: [N, D]. Returns (unscaled_statistic, n_global)."""
        proj = proj.float()
        with torch.autocast(proj.device.type, enabled=False):
            A = torch.randn(proj.size(-1), self.sketches, device=proj.device,
                            dtype=torch.float32, generator=gen)
            A = A / A.norm(p=2, dim=0)
            x_t = (proj @ A).unsqueeze(-1) * self.t          # [N, S, K]
            c_sum = x_t.cos().sum(-3)                        # [S, K]
            s_sum = x_t.sin().sum(-3)
            n = torch.tensor(float(proj.size(-2)), device=proj.device, dtype=torch.float32)
            if sync:
                c_sum = _all_reduce_sum(c_sum)
                s_sum = _all_reduce_sum(s_sum)
                n = _all_reduce_sum(n)
            err = (c_sum / n - self.phi).square() + (s_sum / n).square()
            per_sketch = err @ self.weights                  # [S]
            return per_sketch.mean(), per_sketch.max().detach(), n.detach()


class TransformerAE(nn.Module):
    def __init__(self, feat_dim = 4*768, layers=4, heads=6, hidden_dim=768, bottleneck_dim=32,
                 feature_size=(32,64), model_gaussian=False, assymetric=False,
                 norm_type="bn", use_inv_bn=False, use_rope=False,):
        super(TransformerAE, self).__init__()
        self.feat_dim = feat_dim
        self.num_layers = layers
        self.num_heads = heads
        self.hidden_dim = hidden_dim
        self.bottleneck_dim = bottleneck_dim
        self.model_gaussian = model_gaussian
        self.feature_size = feature_size
        self.assymetric = assymetric
        self.norm_type = norm_type
        self.use_inv_bn = use_inv_bn
        self.use_rope = use_rope
        # ---- 2D spatial RoPE ----
        if use_rope:
            per_head_dim = hidden_dim // heads
            assert per_head_dim % 4 == 0, f"head_dim ({per_head_dim}) must be divisible by 4 for 2D RoPE"
            self.rope = RopePosition2DEmb(
                head_dim=per_head_dim,
                len_h=feature_size[0], len_w=feature_size[1], len_t=1,
            )
            self.rope.reset_parameters()
        self.latent_dim = self.bottleneck_dim
        if self.norm_type == "bn":
            self.norm = nn.BatchNorm1d(self.latent_dim, affine=False) 
        elif self.norm_type == "ln":
            self.norm = nn.LayerNorm(self.latent_dim, elementwise_affine=False)
        else:
            self.norm = None

        # ModuleList instead of Sequential — required to pass rope through each block
        self.encoder = nn.ModuleList(self.build_encoder())
        self.decoder = nn.ModuleList(self.build_decoder())

    def build_encoder(self):
        layers = []
        # Initial projection
        layers.append(nn.Linear(self.feat_dim, self.hidden_dim))
        # Transformer blocks
        num_enc_layers = 1 if self.assymetric else self.num_layers
        for _ in range(num_enc_layers):
            layers.append(LightningDiTBlock(self.hidden_dim, self.num_heads, use_qknorm=True, use_swiglu=True, use_rmsnorm=True))
        norm = RMSNorm(self.hidden_dim, linear=True, bias=False)
        layers.append(norm)
        bottleneck_dim = self.bottleneck_dim * 2 if self.model_gaussian else self.bottleneck_dim
        layers.append(nn.Linear(self.hidden_dim, bottleneck_dim))
        
        return layers

    def build_decoder(self):
        layers = []
        layers.append(nn.Linear(self.bottleneck_dim, self.hidden_dim))
        # Transformer blocks
        for _ in range(self.num_layers):
            layers.append(LightningDiTBlock(self.hidden_dim, self.num_heads,
                                                use_qknorm=True, use_swiglu=True, use_rmsnorm=True))
        norm = RMSNorm(self.hidden_dim, linear=True, bias=False)
        layers.append(norm)
        layers.append(nn.Linear(self.hidden_dim, self.feat_dim))
        return layers

    def _get_rope(self, device):
        if not self.use_rope:
            return None
        dummy = torch.zeros(1, 1, self.feature_size[0], self.feature_size[1],
                            self.hidden_dim, device=device)
        return self.rope(dummy)

    def _run_layers(self, x, module_list, rope=None):
        for layer in module_list:
            if isinstance(layer, LightningDiTBlock):
                x = layer(x, feat_rope=rope)
            else:
                x = layer(x)
        return x


    def encode(self, x):
        rope = self._get_rope(x.device)
        if not (self.norm_type in ("bn", "ln")):
            if self.model_gaussian:
                z_mv = self._run_layers(x, self.encoder, rope)
                z = einops.rearrange(z_mv, 'b (h w) c -> b c h w',
                                    h=self.feature_size[0], w=self.feature_size[1])
                posterior = DiagonalGaussianDistribution(z)
                z_out = einops.rearrange(posterior.sample(), 'b c h w -> b (h w) c')
                return z_out, posterior.kl()
            else:
                return self._run_layers(x, self.encoder, rope), None
        else:
            x = self._run_layers(x, self.encoder, rope)
            if self.norm_type == "bn":
                x = einops.rearrange(x, 'b n c -> b c n')
                x = self.norm(x)
                return einops.rearrange(x, 'b c n -> b n c'), None
            elif self.norm_type == "ln":
                x = self.norm(x)
                return x, None
            else:            
                return x, None

    def decode(self, z):
        if self.use_inv_bn:
            z = einops.rearrange(z, 'b n c -> b c n')
            mean = self.norm.running_mean[None, :, None]
            var  = self.norm.running_var[None, :, None]
            z = z * torch.sqrt(var + 1e-6) + mean
            z = einops.rearrange(z, 'b c n -> b n c')
        rope = self._get_rope(z.device)
        return self._run_layers(z, self.decoder, rope)
        
    def forward(self, x):
        z_out, kl = self.encode(x)
        x_recon = self.decode(z_out)
        return x_recon, z_out, kl



class RopePosition2DEmb(nn.Module):
    """
    2D axial RoPE — faithful reduction of VideoRopePosition3DEmb with the temporal
    axis removed. Head dim is split 50/50 across (h, w); no dead temporal band.

    Returns angles (not cos/sin) shaped `(t h w) 1 1 d` with t=1, so it is a drop-in
    for the same apply_rotary_pos_emb(...) used by the 3D version.
    """
    def __init__(
        self,
        *,
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int = 1,                    # accepted for interface parity; must be 1
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        **kwargs,
    ):
        del kwargs
        super().__init__()
        assert len_t == 1, "RopePosition2DEmb is spatial-only; len_t must be 1"
        dim = head_dim
        # 50/50 split; each half must be even so the chunk-2 rotate_half pairing holds
        dim_h = (dim // 2)
        dim_w = dim - dim_h
        assert dim_h % 2 == 0 and dim_w % 2 == 0, \
            f"head_dim split must be even per axis, got dim_h={dim_h}, dim_w={dim_w}"
        assert dim_h == dim_w, \
            "for a square per-axis split keep dim_h == dim_w (head_dim divisible by 4)"

        self.max_h = len_h
        self.max_w = len_w
        self._dim_h = dim_h
        self._dim_w = dim_w

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))

        

        self.register_buffer("seq", torch.arange(max(len_h, len_w), dtype=torch.float),
                             persistent=False)
        self.register_buffer(
            "dim_h_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=False,
        )
        self.register_buffer(
            "dim_w_range",
            torch.arange(0, dim_w, 2)[: (dim_w // 2)].float() / dim_w,
            persistent=False,
        )

    def reset_parameters(self) -> None:
        dev = self.dim_h_range.device
        self.seq = torch.arange(max(self.max_h, self.max_w)).float().to(dev)
        self.dim_h_range = torch.arange(0, self._dim_h, 2)[: (self._dim_h // 2)].float().to(dev) / self._dim_h
        self.dim_w_range = torch.arange(0, self._dim_w, 2)[: (self._dim_w // 2)].float().to(dev) / self._dim_w

    def generate_embeddings(self, B_T_H_W_C: torch.Size,
                            h_ntk_factor=None, w_ntk_factor=None):
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        h_freqs = 1.0 / (h_theta ** self.dim_h_range.float())   # [dim_h//2]
        w_freqs = 1.0 / (w_theta ** self.dim_w_range.float())   # [dim_w//2]

        B, T, H, W, _ = B_T_H_W_C
        assert T == 1, "RopePosition2DEmb expects T=1"
        assert H <= self.max_h and W <= self.max_w, \
            f"(H={H}, W={W}) exceeds (max_h={self.max_h}, max_w={self.max_w})"

        half_emb_h = torch.outer(self.seq[:H], h_freqs)   # [H, dim_h//2]  angles
        half_emb_w = torch.outer(self.seq[:W], w_freqs)   # [W, dim_w//2]  angles

        # broadcast to the H×W grid, then duplicate ([h,w] -> [h,w,h,w]) for chunk-2 rotate_half
        em_H_W_D = torch.cat(
            [
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ] * 2,
            dim=-1,
        )                                                       # [1, H, W, dim]
        return rearrange(em_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0

    def forward(self, x_B_T_H_W_C: torch.Tensor) -> torch.Tensor:
        return self.generate_embeddings(B_T_H_W_C=x_B_T_H_W_C.shape)