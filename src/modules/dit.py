# BERT architecture for the Masked Bidirectional Encoder Transformer
# Updated with: RMSNorm, QK Normalization, and SwiGLU activation
import torch
import einops
from torch import nn
import torch.nn.functional as F
import math
from typing import List, Optional, Tuple, Union
from einops import repeat, rearrange


def modulate(x, shift, scale):
    return x * (1 + scale) + shift

class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5, linear=True, bias=True):
        """ RMSNorm normalization layer
            :param:
                dim    -> int: Dimension of the input
                eps    -> float: Small value for numerical stability
                linear -> bool: Whether to use learnable weight parameter
                bias   -> bool: Whether to use learnable bias parameter
        """
        super().__init__()
        self.eps = eps
        self.linear = linear
        self.add_bias = bias
        if self.linear:
            self.weight = nn.Parameter(torch.ones(dim))
        if self.add_bias:
            self.bias = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        """ Apply RMS normalization """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """ Forward pass through RMSNorm
            :param:
                x -> torch.Tensor: Input tensor
            :return:
                torch.Tensor: Normalized output
        """
        output = self._norm(x.float()).type_as(x)
        if self.linear:
            output = self.weight * output
        if self.add_bias:
            output = output + self.bias
        return output


class PreNorm_Modulate(nn.Module):

    def __init__(self, dim, fn, use_rmsnorm=True):
        """ PreNorm module to apply normalization before a given function
            :param:
                dim         -> int: Dimension of the input
                fn          -> nn.Module: The function to apply after normalization
                use_rmsnorm -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
            """
        super().__init__()
        if use_rmsnorm:
            self.norm = RMSNorm(dim, linear=True, bias=False)
        else:
            self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, shift, scale, **kwargs):
        """ Forward pass through the PreNorm module
            :param:
                x        -> torch.Tensor: Input tensor
                shift    -> torch.Tensor: Shift tensor
                scale    -> torch.Tensor: Scale tensor
                **kwargs -> _ : Additional keyword arguments for the function
            :return
                torch.Tensor: Output of the function applied after normalization
        """
        # return self.fn(self.norm(x), **kwargs)
        return self.fn(modulate(self.norm(x), shift, scale), **kwargs)



class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0., use_swiglu=True, multiple_of=256, bias=True):
        """ Initialize the Multi-Layer Perceptron (MLP) with optional SwiGLU activation.
            :param:
                dim         -> int: Dimension of the input
                hidden_dim  -> int: Dimension of the hidden layer
                dropout     -> float: Dropout rate
                use_swiglu  -> bool: Whether to use SwiGLU activation (True) or standard GELU (False)
                multiple_of -> int: Make hidden dimension a multiple of this value (for SwiGLU)
                bias        -> bool: Whether to use bias in linear layers
        """
        super().__init__()
        self.use_swiglu = use_swiglu
        self.dropout = dropout
        
        if use_swiglu:
            # SwiGLU activation: requires 3 weight matrices
            hidden_dim = int(2 * hidden_dim / 3)
            # Make sure it is a multiple of 'multiple_of' for efficiency
            hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)
            
            self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
            self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
            self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        else:
            # Standard GELU activation
            self.net = nn.Sequential(
                nn.Linear(dim, hidden_dim, bias=bias),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim, bias=bias),
                nn.Dropout(dropout)
            )

    def forward(self, x):
        """ Forward pass through the MLP module.
            :param:
                x -> torch.Tensor: Input tensor
            :return
                torch.Tensor: Output of the function applied after layer
        """
        if self.use_swiglu:
            # SwiGLU: SiLU(W1(x)) * W3(x), then W2
            x = F.silu(self.w1(x)) * self.w3(x)
            if self.dropout > 0. and self.training:
                x = F.dropout(x, self.dropout)
            return self.w2(x)
        else:
            return self.net(x)


class QKNorm(nn.Module):
    def __init__(self, dim: int, use_rmsnorm=True):
        """ QK Normalization layer for normalizing queries and keys
            :param:
                dim -> int: Dimension of the normalized axis (the per-head dim,
                       embed_dim // num_heads). Only affects the LayerNorm variant, since
                       RMSNorm is built here with linear=False/bias=False and has no parameters.
        """
        super().__init__()
        if use_rmsnorm:
            self.query_norm = RMSNorm(dim, linear=False, bias=False)
            self.key_norm = RMSNorm(dim, linear=False, bias=False)
        else:
            self.query_norm = nn.LayerNorm(dim)
            self.key_norm = nn.LayerNorm(dim)

    def forward(self, q, k, v):
        """ Normalize queries and keys
            :param:
                q -> torch.Tensor: Query tensor
                k -> torch.Tensor: Key tensor
                v -> torch.Tensor: Value tensor (used for dtype casting)
            :return:
                Tuple[torch.Tensor, torch.Tensor]: Normalized query and key tensors
        """
        q = self.query_norm(q)
        k = self.key_norm(k)
        return q.to(v), k.to(v)


class Attention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0., use_qk_norm=True, use_rmsnorm=True):
        """ QK normalization is applied over the per-head channel axis (Dh):
            one RMS per (head, token).
        """
        super().__init__()
        self.dim = embed_dim
        self.h = num_heads
        self.use_qk_norm = use_qk_norm
        self.use_rmsnorm = use_rmsnorm
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=True)
        self.o = nn.Linear(embed_dim, embed_dim, bias=True)
        self.drop = nn.Dropout(dropout)
        if use_qk_norm:
            qk_norm_dim = embed_dim // num_heads
            self.qk_norm = QKNorm(qk_norm_dim, use_rmsnorm=use_rmsnorm)

    def forward(self, x, mask=None, rope_freqs: Optional[torch.Tensor] = None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.h, C // self.h).permute(2, 0, 3, 1, 4)  # 3, B, H, N, Dh
        q, k, v = qkv[0], qkv[1], qkv[2]  # each: B, H, N, Dh

        if self.use_qk_norm:
            # Normalize over the per-head channel axis: one RMS per (head, token).
            q, k = self.qk_norm(q, k, v)
            v = v  # v unchanged

        # Apply RoPE to q,k if provided
        if rope_freqs is not None:
            # Expect rope_freqs: [N, 1, 1, Dh]
            q_bshd = q.transpose(1, 2)  # [B, N, H, Dh]
            k_bshd = k.transpose(1, 2)  # [B, N, H, Dh]
            # print(f"Applying RoPE with rope_freqs shape: {rope_freqs.shape}, q shape: {q_bshd.shape}")
            q_bshd = apply_rotary_pos_emb(q_bshd, rope_freqs, tensor_format="bshd")
            k_bshd = apply_rotary_pos_emb(k_bshd, rope_freqs, tensor_format="bshd")
            q = q_bshd.transpose(1, 2)
            k = k_bshd.transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.drop.p if self.training else 0.0, is_causal=False)
        attn_out = attn_out.transpose(1, 2).reshape(B, N, C)
        attn_out = self.o(attn_out)
        return attn_out, None


class TransformerEncoderSeperableAttention(nn.Module):
        def __init__(self, dim, depth, heads, mlp_dim, dropout=0., window_size=1,
                use_rmsnorm=True, use_swiglu=True, use_qk_norm=True):
            """ Initialize the Attention module.
                :param:
                    dim          -> int: number of hidden dimension of attention
                    depth        -> int: number of layer for the transformer
                    heads        -> int: Number of heads
                    mlp_dim      -> int: number of hidden dimension for mlp
                    dropout      -> float: Dropout rate
                    window_size  -> int: Window size for spatial attention
                    use_rmsnorm  -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
                    use_swiglu   -> bool: Whether to use SwiGLU activation in FFN
                    use_qk_norm  -> bool: Whether to use QK normalization in attention
            """
            super().__init__()
            self.window_size = window_size
            self.layers = nn.ModuleList([])
            self.adaLN_mods = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
            # self.adaLN_mods = nn.ModuleList([])
            for _ in range(depth):
                self.layers.append(nn.ModuleList([
                    PreNorm_Modulate(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rmsnorm=use_rmsnorm), use_rmsnorm=use_rmsnorm),
                    PreNorm_Modulate(dim, Attention(dim, heads, dropout=dropout, use_qk_norm=use_qk_norm, use_rmsnorm=use_rmsnorm), use_rmsnorm=use_rmsnorm),
                    PreNorm_Modulate(dim, FeedForward(dim, mlp_dim, dropout=dropout, use_swiglu=use_swiglu), use_rmsnorm=use_rmsnorm)
                ]))
                # mod = nn.Sequential(nn.SiLU(), nn.Linear(dim, 9 * dim, bias=True))
                # self.adaLN_mods.append(mod)

        def forward(self, x, full_shape, rope_freqs: Optional[torch.Tensor] = None, c=None):
            b, t, h, w, d = full_shape
            rope_freqs_spatial, rope_freqs_temporal = rope_freqs
            l_attn = []
            (shift_t,  scale_t,  gate_t,
            shift_s,  scale_s,  gate_s,
            shift_ff, scale_ff, gate_ff) = self.adaLN_mods(c).chunk(9, dim=-1)
            shift_s = repeat(shift_s, 'b 1 d -> (b t) 1 d', t=t)
            scale_s = repeat(scale_s, 'b 1 d -> (b t) 1 d', t=t)
            gate_s = repeat(gate_s, 'b 1 d -> (b t) 1 d', t=t)
            n_repeats = (h // self.window_size) * (w // self.window_size)
            shift_t = repeat(shift_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            scale_t = repeat(scale_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            gate_t = repeat(gate_t, 'b 1 d -> (b n) 1 d', n=n_repeats)
            for (attn_temporal, attn_spatial, ff) in self.layers:
                
                if self.window_size == 1:
                    x = einops.rearrange(x, 'b (t h w) c -> (b h w) t c', b=b, t=t, h=h, w=w)
                    n_repeats = h * w
                else:
                    x = einops.rearrange(x, 'b (t h w) c -> b t h w c', b=b, t=t, h=h, w=w)
                    x = einops.rearrange(
                        x, 'b t (h k1) (w k2) c -> (b h w) (t k1 k2) c',
                        k1=self.window_size, k2=self.window_size)
                    n_repeats = (h // self.window_size) * (w // self.window_size)


                attention_value, attention_weight = attn_temporal(x, rope_freqs=rope_freqs_temporal, shift=shift_t, scale=scale_t)
                x = x + gate_t * attention_value
                l_attn.append(attention_weight)

                if self.window_size == 1:
                    x = einops.rearrange(x, '(b h w) t c -> (b t) (h w) c', b=b, t=t, h=h, w=w)
                else:
                    x = einops.rearrange(
                        x, '(b h w) (t k1 k2) c -> (b t) (h k1) (w k2) c',
                        b=b, t=t, h=h//self.window_size, w=w//self.window_size, k1=self.window_size, k2=self.window_size)
                    x = x.flatten(1,2)

                attention_value, attention_weight = attn_spatial(x, rope_freqs=rope_freqs_spatial, shift=shift_s, scale=scale_s)
                x = x + gate_s * attention_value
                l_attn.append(attention_weight)

                x = einops.rearrange(x, '(b t) (h w) c -> b (t h w) c', b=b, t=t, h=h, w=w)
                x = x + gate_ff * ff(x, shift=shift_ff, scale=scale_ff)
            return x, l_attn

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding


    def forward(self, t):
        # t: (B, T, 1, 1, 1)
        t_flat = einops.rearrange(t, 'b 1 1 1 1 -> (b 1)')
        t_freq = self.timestep_embedding(t_flat, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        t_emb = einops.rearrange(t_emb, '(b 1) d -> b 1 d', b=t.shape[0])
        return t_emb

        

class FinalLayer(nn.Module):
    """
    The final layer of JiT.
    """
    def __init__(self, hidden_size, out_channels, use_bias=True):
        super().__init__()
        self.norm_final = RMSNorm(hidden_size)
        self.linear = nn.Linear(hidden_size, out_channels, bias=use_bias)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=-1)
        shift = repeat(shift, 'b 1 d -> b (1 s) d', s=x.shape[1] // shift.shape[1])
        scale = repeat(scale, 'b 1 d -> b (1 s) d', s=x.shape[1] // scale.shape[1])
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x

class MaskTransformer(nn.Module):
    def __init__(self, shape, img_size=256, embedding_dim=768, hidden_dim=768, depth=24, heads=8, mlp_dim=3072, dropout=0.1, use_fc_bias=False, 
                use_first_last=False, seperable_attention=False, seperable_window_size=1, use_rmsnorm=True, use_swiglu=True, use_qk_norm=True):
        """ Initialize the Transformer model with modern features.
            :param:
                shape                  -> tuple: Shape of input [T, H, W]
                img_size               -> int: Input image size (default: 256)
                embedding_dim          -> int: Embedding dimension
                hidden_dim             -> int: Hidden dimension for the transformer (default: 768)
                depth                  -> int: Depth of the transformer (default: 24)
                heads                  -> int: Number of attention heads (default: 8)
                mlp_dim                -> int: MLP dimension (default: 3072)
                dropout                -> float: Dropout rate (default: 0.1)
                use_fc_bias            -> bool: Whether to use bias in fc layers
                use_first_last         -> bool: Whether to use first/last projection layers
                seperable_attention    -> bool: Whether to use separable space-time attention
                seperable_window_size  -> int: Window size for separable attention
                use_rmsnorm            -> bool: Whether to use RMSNorm (True) or LayerNorm (False)
                use_swiglu             -> bool: Whether to use SwiGLU activation in FFN
                use_qk_norm            -> bool: Whether to use QK normalization in attention
        """
        super().__init__()
        
        # RoPE frequencies generator (per-head dim, must be even)
        per_head_dim = hidden_dim // heads
        assert per_head_dim % 2 == 0, f"Per-head dim must be even for RoPE, got {per_head_dim}"
        self.pos_embd_spat = VideoRopePosition3DEmb(
            head_dim=per_head_dim,
            len_h=32,
            len_w=64,
            len_t=1,
            enable_fps_modulation=False,
        )
        self.pos_embd_spat.reset_parameters()

        self.pos_embd_temp = VideoRopePosition3DEmb(
            head_dim=per_head_dim,
            len_h=1,
            len_w=1,
            len_t=20,
            enable_fps_modulation=False,
        )
        self.pos_embd_temp.reset_parameters()
        

        self.t_embedder = TimestepEmbedder(hidden_size=hidden_dim)
    
        # First layer before the Transformer block
        self.first_layer = nn.Identity() 
        if use_first_last:
            if use_rmsnorm:
                self.first_layer = nn.Sequential(
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )
            else:
                self.first_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                )

        self.seperable_attention = seperable_attention
        if seperable_attention:
            self.transformer = TransformerEncoderSeperableAttention(
                dim=hidden_dim, depth=depth, heads=heads, mlp_dim=mlp_dim, dropout=dropout,
                window_size=seperable_window_size, use_rmsnorm=use_rmsnorm,
                use_swiglu=use_swiglu, use_qk_norm=use_qk_norm)
        else:
            assert "Not implemented for non-separable attention"

        # Last layer after the Transformer block
        self.last_layer = nn.Identity()
        if use_first_last:
            if use_rmsnorm:
                self.last_layer = nn.Sequential(
                    RMSNorm(hidden_dim, linear=True, bias=True),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    RMSNorm(hidden_dim, linear=True, bias=True),
                )
            else:
                self.last_layer = nn.Sequential(
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                    nn.Dropout(p=dropout),
                    nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim, eps=1e-12),
                )

        # Bias for the last linear output
        self.fc_in = nn.Linear(hidden_dim, hidden_dim, bias=use_fc_bias)
        self.fc_in.weight.data.normal_(std=0.02) 

        
        self.fc_out = FinalLayer(hidden_dim, embedding_dim, use_bias=use_fc_bias)

        
        # adaLN-Zero init: zero out the final linear of each adaLN modulator
        nn.init.constant_(self.transformer.adaLN_mods[-1].weight, 0)
        nn.init.constant_(self.transformer.adaLN_mods[-1].bias, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.fc_out.adaLN_modulation[-1].bias, 0)


    def forward(self, vid_token, timestep, return_attn=False, ):
        """ Forward.
            :param:
                vid_token      -> torch.Tensor: video tokens of shape [b, t, h, w, c]
                return_attn    -> Bool: return the attn for visualization
            :return:
                logit:         -> torch.FloatTensor: the predicted output
                attn:          -> list(torch.FloatTensor): list of attention for visualization
        """
        b, t, h, w, c = vid_token.size()


        cond = self.t_embedder(timestep) # shape [B, T, D]
        # Token projection
        x_tokens = self.fc_in(vid_token)

        # Generate RoPE frequencies for q,k
        x_spat = torch.rand(1, 1, 32, 64, c)  # dummy input for spatial RoPE
        x_temp = torch.rand(1, 5, 1, 1, c)   # dummy input for temporal RoPE
        rope_freqs_spat = self.pos_embd_spat(x_spat)  # shape: [(t*h*w), 1, 1, per_head_dim*?], matches apply_rotary_pos_emb
        rope_freqs_temp = self.pos_embd_temp(x_temp)  # shape: [(t*h*w), 1, 1, per_head_dim*?], matches apply_rotary_pos_emb

        rope_freqs = [rope_freqs_spat, rope_freqs_temp]
        # Flatten token stream for transformer
        x = einops.rearrange(x_tokens, 'b t h w c -> b (t h w) c')

        # transformer forward pass with RoPE
        x = self.first_layer(x)
        x, attn = self.transformer(x, full_shape=(b, t, h, w, c), rope_freqs=rope_freqs, c=cond)
        x = self.last_layer(x)

        x_out = self.fc_out(x, cond)
        if return_attn:
            return x_out, attn
        else:
            return x_out,


class VideoRopePosition3DEmb(nn.Module):
    def __init__(
        self,
        *,  # enforce keyword arguments
        head_dim: int,
        len_h: int,
        len_w: int,
        len_t: int,
        base_fps: int = 6,
        h_extrapolation_ratio: float = 1.0,
        w_extrapolation_ratio: float = 1.0,
        t_extrapolation_ratio: float = 1.0,
        enable_fps_modulation: bool = True,
        **kwargs,  # used for compatibility with other positional embeddings; unused in this class
    ):
        del kwargs
        super().__init__()
        self.register_buffer("seq", torch.arange(max(len_h, len_w, len_t), dtype=torch.float))
        self.base_fps = base_fps
        self.max_h = len_h
        self.max_w = len_w
        self.max_t = len_t
        self.enable_fps_modulation = enable_fps_modulation
        dim = head_dim
        dim_h = dim // 6 * 2
        dim_w = dim_h
        dim_t = dim - 2 * dim_h
        assert dim == dim_h + dim_w + dim_t, f"bad dim: {dim} != {dim_h} + {dim_w} + {dim_t}"
        self.register_buffer(
            "dim_spatial_range",
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float() / dim_h,
            persistent=True,
        )
        self.register_buffer(
            "dim_temporal_range",
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float() / dim_t,
            persistent=True,
        )
        self._dim_h = dim_h
        self._dim_t = dim_t

        self.h_ntk_factor = h_extrapolation_ratio ** (dim_h / (dim_h - 2))
        self.w_ntk_factor = w_extrapolation_ratio ** (dim_w / (dim_w - 2))
        self.t_ntk_factor = t_extrapolation_ratio ** (dim_t / (dim_t - 2))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        dim_h = self._dim_h
        dim_t = self._dim_t

        self.seq = torch.arange(max(self.max_h, self.max_w, self.max_t)).float().to(self.dim_spatial_range.device)
        self.dim_spatial_range = (
            torch.arange(0, dim_h, 2)[: (dim_h // 2)].float().to(self.dim_spatial_range.device) / dim_h
        )
        self.dim_temporal_range = (
            torch.arange(0, dim_t, 2)[: (dim_t // 2)].float().to(self.dim_spatial_range.device) / dim_t
        )

    def generate_embeddings(
        self,
        B_T_H_W_C: torch.Size,
        fps: Optional[torch.Tensor] = None,
        h_ntk_factor: Optional[float] = None,
        w_ntk_factor: Optional[float] = None,
        t_ntk_factor: Optional[float] = None,
    ):
        """
        Generate embeddings for the given input size.

        Args:
            B_T_H_W_C (torch.Size): Input tensor size (Batch, Time, Height, Width, Channels).
            fps (Optional[torch.Tensor], optional): Frames per second. Defaults to None.
            h_ntk_factor (Optional[float], optional): Height NTK factor. If None, uses self.h_ntk_factor.
            w_ntk_factor (Optional[float], optional): Width NTK factor. If None, uses self.w_ntk_factor.
            t_ntk_factor (Optional[float], optional): Time NTK factor. If None, uses self.t_ntk_factor.

        Returns:
            Not specified in the original code snippet.
        """
        h_ntk_factor = h_ntk_factor if h_ntk_factor is not None else self.h_ntk_factor
        w_ntk_factor = w_ntk_factor if w_ntk_factor is not None else self.w_ntk_factor
        t_ntk_factor = t_ntk_factor if t_ntk_factor is not None else self.t_ntk_factor

        h_theta = 10000.0 * h_ntk_factor
        w_theta = 10000.0 * w_ntk_factor
        t_theta = 10000.0 * t_ntk_factor

        h_spatial_freqs = 1.0 / (h_theta ** self.dim_spatial_range.float())
        w_spatial_freqs = 1.0 / (w_theta ** self.dim_spatial_range.float())
        temporal_freqs = 1.0 / (t_theta ** self.dim_temporal_range.float())

        B, T, H, W, _ = B_T_H_W_C
        assert H <= self.max_h and W <= self.max_w, (
            f"Input dimensions (H={H}, W={W}) exceed the maximum dimensions (max_h={self.max_h}, max_w={self.max_w})"
        )
        half_emb_h = torch.outer(self.seq[:H], h_spatial_freqs)
        half_emb_w = torch.outer(self.seq[:W], w_spatial_freqs)

        if self.enable_fps_modulation:
            uniform_fps = (fps is None) or (fps.min() == fps.max())
            assert uniform_fps or B == 1 or T == 1, (
                "For video batch, batch size should be 1 for non-uniform fps. For image batch, T should be 1"
            )

            # apply sequence scaling in temporal dimension
            if fps is None:  # image case
                assert T == 1, "T should be 1 for image batch."
                half_emb_t = torch.outer(self.seq[:T], temporal_freqs)
            else:
                half_emb_t = torch.outer(self.seq[:T] / fps[:1] * self.base_fps, temporal_freqs)
        else:
            half_emb_t = torch.outer(self.seq[:T], temporal_freqs)

        em_T_H_W_D = torch.cat(
            [
                repeat(half_emb_t, "t d -> t h w d", h=H, w=W),
                repeat(half_emb_h, "h d -> t h w d", t=T, w=W),
                repeat(half_emb_w, "w d -> t h w d", t=T, h=H),
            ]
            * 2,
            dim=-1,
        )

        return rearrange(em_T_H_W_D, "t h w d -> (t h w) 1 1 d").float()

    @property
    def seq_dim(self):
        return 0
    
    def forward(self, x_B_T_H_W_C: torch.Tensor, fps: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, H, W, C = x_B_T_H_W_C.shape
        emb = self.generate_embeddings(
            B_T_H_W_C=x_B_T_H_W_C.shape,
            fps=fps,
        )
        return emb

def _rotate_half(x: torch.Tensor, interleaved: bool) -> torch.Tensor:
    """Change sign so the last dimension becomes [-odd, +even]

    Args:
        x: torch.Tensor. Input tensor.
        interleaved: bool. Whether to use interleaved rotary position embedding.

    Returns:
        Tensor: Tensor rotated half.
    """
    if not interleaved:
        x1, x2 = torch.chunk(x, 2, dim=-1)
        return torch.cat((-x2, x1), dim=-1)

    # interleaved
    x1 = x[:, :, :, ::2]
    x2 = x[:, :, :, 1::2]
    x_new = torch.stack((-x2, x1), dim=-1)
    return x_new.view(x_new.shape[0], x_new.shape[1], x_new.shape[2], -1)


def _apply_rotary_pos_emb_base(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Base implementation of applying rotary positional embedding tensor to the input tensor.

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]` or `[b, s, h, d]`, on which rotary positional
        embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` or `[s2, b, 1, d2]`
        and dtype 'float', with `s2 >= s` and `d2 <= d`.
    tensor_format : {'sbhd', 'bshd'}, default = 'sbhd'
        Should be `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is of shape
        `[seq, bs, ...]`.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    """
    # [seq, 1, 1, dim] -> [1, seq, 1, dim] or
    # [seq, b, 1, dim] -> [b, seq, 1, dim]
    if tensor_format == "bshd":
        freqs = freqs.transpose(0, 1)

    # cos/sin first then dtype conversion for better precision
    cos_ = torch.cos(freqs).to(t.dtype)
    sin_ = torch.sin(freqs).to(t.dtype)

    rot_dim = freqs.shape[-1]
    # ideally t_pass is empty so rotary pos embedding is applied to all tensor t
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]

    # first part is cosine component
    # second part is sine component, need to change signs with _rotate_half method
    t = (t * cos_) + (_rotate_half(t, interleaved) * sin_)
    return torch.cat((t, t_pass), dim=-1)


def _get_freqs_on_this_cp_rank(
    freqs: torch.Tensor, seqlen: int, cp_size: int, cp_rank: int
) -> torch.Tensor:
    """Get the position embedding on the current context parallel rank.

    Args:
        freqs: torch.Tensor. Positional embedding tensor of shape `[s2, 1, 1, d2]`.
        seqlen: int. Length of the current sequence.
        cp_size: int. Context parallel world size.
        cp_rank: int. Context parallel rank.
    """
    if cp_size > 1:
        cp_seg = seqlen // 2
        full_seqlen = cp_size * seqlen
        return torch.cat(
            [
                freqs[cp_rank * cp_seg : (cp_rank + 1) * cp_seg],
                freqs[full_seqlen - (cp_rank + 1) * cp_seg : full_seqlen - cp_rank * cp_seg],
            ]
        )

    # cp_size == 1
    return freqs[:seqlen]


def apply_rotary_pos_emb(
    t: torch.Tensor,
    freqs: torch.Tensor,
    tensor_format: str = "sbhd",
    start_positions: Union[torch.Tensor, None] = None,
    interleaved: bool = False,
    fused: bool = False,
    cu_seqlens: Union[torch.Tensor, None] = None,
    cp_size: int = 1,
    cp_rank: int = 0,
) -> torch.Tensor:
    """
    Apply rotary positional embedding tensor to the input tensor.

    Support matrix:
    Fused/Unfused:
        Training:
            qkv_formats:            "thd", "bshd", "sbhd"
            context parallel:       yes
            start_positions:        yes
            interleaving:           yes
        Inference:
            qkv_formats:            "thd", "bshd", "sbhd"
            context parallelism:    no
            start_positions:        yes
            interleaving:           yes

    Parameters
    ----------
    t : torch.Tensor
        Input tensor of shape `[s, b, h, d]`, `[b, s, h, d]` or `[t, h, d]`, on which
        rotary positional embedding will be applied.
    freqs : torch.Tensor
        Rotary positional embedding tensor of shape `[s2, 1, 1, d2]` and dtype 'float',
        with `s2 >= s` and `d2 <= d`.
    start_positions : torch.Tensor, default = None.
        Tokens in a sequence `i` should be applied with position encoding offset by
        `start_positions[i]`. If `start_positions=None`, there's no offset.
    tensor_format : {'sbhd', 'bshd', 'thd'}, default = 'sbhd'
        is `bshd` if `t` is of shape `[bs, seq, ...]`, or `sbhd` if `t` is
        of shape `[seq, bs, ...]`. 'thd' is only supported when `fused` is True.
    interleaved : bool, default = False
        Whether to use interleaved rotary position embedding.
    fused : bool, default = False
        Whether to use a fused applying RoPE implementation.
    cu_seqlens : torch.Tensor, default = None.
        Cumulative sum of sequence lengths in a batch for `t`, with shape [b + 1] and
        dtype torch.int32. Only valid when `tensor_format` is 'thd'.
        Should be `cu_seqlens_padded` when cp_size > 1.
    cp_size : int, default = 1.
        Context parallel world size. Only valid when `tensor_format` is 'thd' and `fused` is True.
    cp_rank : int, default = 0.
        Context parallel rank. Only valid when `tensor_format` is 'thd' and `fused` is True.
    """
    assert (
        tensor_format != "thd" or cu_seqlens is not None
    ), "cu_seqlens must not be None when tensor_format is 'thd'."

    # Fused apply rope logic for THD/BSHD/SBHD formats
    if fused:
        return FusedRoPEFunc.apply(
            t, freqs, start_positions, tensor_format, interleaved, cu_seqlens, cp_size, cp_rank
        )

    # Unfused apply rope logic for THD format
    if tensor_format == "thd":
        cu_seqlens = cu_seqlens // cp_size
        seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).tolist()

        # The following code essentially splits the `thd` tensor into corresponding
        # `s1hd` tensors (for each sequence) and applies rotary embedding to
        # those sequences individually.
        # Note that if `start_positions` is not `None`, then for each sequence,
        # the freqs supplied are offset by the corresponding `start_positions` value.
        return torch.cat(
            [
                _apply_rotary_pos_emb_base(
                    x.unsqueeze(1),
                    _get_freqs_on_this_cp_rank(
                        (
                            freqs[start_positions[idx] :] if start_positions is not None else freqs
                        ),  # offset the freqs
                        x.size(0),
                        cp_size,
                        cp_rank,
                    ),
                    interleaved=interleaved,
                )
                for idx, x in enumerate(torch.split(t, seqlens))
            ]
        ).squeeze(1)

    # Unfused apply rope logic for SBHD/BSHD format follows ...

    if tensor_format == "sbhd":
        seqlen = t.size(0)
    elif tensor_format == "bshd":
        seqlen = t.size(1)
    else:
        raise ValueError(f"Unsupported tensor_format: {tensor_format}.")

    if start_positions is not None:
        max_offset = torch.max(start_positions)
        assert (
            max_offset + seqlen * cp_size <= freqs.shape[0]
        ), f"Rotary Embeddings only suppported up to {freqs.shape[0]} sequence length!"

        # Stack staggered rope embeddings along the batch dimension
        freqs = torch.concatenate([freqs[i : i + seqlen * cp_size] for i in start_positions], dim=1)
        # Note that from this point, `freqs` has a shape `(s,b,1,d)`.

    return _apply_rotary_pos_emb_base(
        t,
        _get_freqs_on_this_cp_rank(freqs, seqlen, cp_size, cp_rank),
        tensor_format,
        interleaved=interleaved,
    )