import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.activations import ACT2FN

from scipy import signal

from typing import Optional

# Create a context mask for transformer
def create_context_mask(context_forward, context_backward, max_F) -> torch.LongTensor:
    if context_forward == -1 and context_backward == -1:
        return torch.ones(max_F, max_F).to(torch.int64)
    
    context_forward = context_forward if context_forward >= 0 else max_F
    context_backward = context_backward if context_backward >= 0 else max_F

    mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_forward).to(torch.int64)).transpose(0, 1)
    if context_backward > 0:
        back_mask = (torch.triu(torch.ones(max_F, max_F), diagonal=-context_backward).to(torch.int64))
        mask = mask & back_mask
    return mask

# RoPE
def get_cos_sin(dim, max_F, base=10000, dtype=torch.get_default_dtype(), device=None):
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float().to(device) / dim))
    t = torch.arange(max_F, device=device, dtype=inv_freq.dtype)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    emb = torch.cat((freqs, freqs), dim=-1)
    return emb.cos().to(dtype), emb.sin().to(dtype)

# RoPE
def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), -1)

# RoPE
def apply_rotary_pos_emb(q, k, pos_ids, cos, sin, unsqueeze_dim=1):
    cos = cos[pos_ids].unsqueeze(unsqueeze_dim)
    sin = sin[pos_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

# Smooth spikes and add noise (data augmentation)
class SmoothAndNoise(nn.Module): 
    def __init__(self, config):
        super().__init__()
        self.noise = config.noise
        self.white_noise_sd = config.white_noise_sd
        self.constant_offset_sd = config.constant_offset_sd
        self.smooth = config.smooth_sd is not None
        if self.smooth:
            kernel = torch.from_numpy(signal.windows.gaussian(1 +config.smooth_sd*6, config.smooth_sd))
            kernel = kernel / kernel.sum()
            self.register_buffer("kernel", kernel, persistent=False)
    
    def forward(self, spikes):
        B, T, N = spikes.size()
        if self.smooth:
            spikes = F.conv1d(
                spikes.transpose(-1,-2),
                self.kernel.unsqueeze(0).unsqueeze(0).expand(N, 1, self.kernel.size(0)).to(spikes.dtype), 
                padding="same", groups=N
            ).transpose(-1, -2)

        if self.noise and self.training:
            if self.white_noise_sd is not None:
                spikes += self.white_noise_sd*torch.randn(B, T, N, dtype=spikes.dtype, device=spikes.device)

            if self.constant_offset_sd is not None:
                spikes += self.constant_offset_sd*torch.randn(B, 1, N, dtype=spikes.dtype, device=spikes.device)       
        return spikes

class NeuralMLP(nn.Module):

    def __init__(self, hidden_size, inter_size, act, use_bias, dropout):
        super().__init__()
        self.up_proj    = nn.Linear(hidden_size, inter_size, bias=use_bias)
        self.act        = ACT2FN[act]
        self.down_proj  = nn.Linear(inter_size, hidden_size, bias=use_bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.act(self.up_proj(x))
        return self.dropout(self.down_proj(x))


class NeuralAttention(nn.Module):
    def __init__(self, idx, hidden_size, n_heads, use_bias, dropout, use_rope=False, base=10000., max_F=1024):
        super().__init__()
        
        self.idx = idx
        self.hidden_size = hidden_size
        self.n_heads = n_heads
        assert self.hidden_size % self.n_heads == 0, \
            f"Hidden size ({self.hidden_size}) must be divisible by the number of attention heads ({self.n_heads})"
        self.head_size = self.hidden_size // self.n_heads
        self.use_rope = use_rope

        self.query = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.key = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)
        self.value  = nn.Linear(self.hidden_size, self.hidden_size, bias=use_bias)

        self.attn_dropout = dropout
        self.dropout = nn.Dropout(dropout)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=use_bias)

        if use_rope:
            cos, sin = get_cos_sin(self.head_size, max_F, base=base, dtype=self.query.weight.dtype, device=self.query.weight.device)
            self.register_buffer("cos", cos, persistent=False)
            self.register_buffer("sin", sin, persistent=False)

    def forward(
        self,       
        x:          torch.FloatTensor,                      
        attn_mask:  torch.LongTensor,                      
        timestamp:  Optional[torch.LongTensor] = None,     
    ) -> torch.FloatTensor:                               

        B, T, _  = x.size()   

        assert attn_mask.max() <= 1 and attn_mask.min() >= 0, ["assertion", attn_mask.max(), attn_mask.min()]
        attn_mask = attn_mask.unsqueeze(1).expand(B, self.n_heads, T, T).bool() 
        
        q = self.query(x).reshape(B, T, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        k = self.key(x).reshape(B, T, self.n_heads, self.head_size).permute(0, 2, 1, 3)
        v = self.value(x).reshape(B, T, self.n_heads, self.head_size).permute(0, 2, 1, 3)

        if self.use_rope:
            q, k = apply_rotary_pos_emb(q, k, timestamp, self.cos, self.sin, 1)

        if q.dtype != torch.bfloat16:
            q, k, v = q.to(torch.bfloat16), k.to(torch.bfloat16), v.to(torch.bfloat16)

        out = F.scaled_dot_product_attention(
            q, k, v, 
            attn_mask=attn_mask,
            dropout_p=(self.attn_dropout if self.training else 0.0), 
            is_causal=False
        ) 

        if out.dtype != x.dtype:
            out = out.to(x.dtype)
            
        out = out.permute(0, 2, 1, 3).reshape(B, T, self.hidden_size)

        return self.out_proj(self.dropout(out)) 
    

class NeuralFactorsProjection(nn.Module):
    def __init__(self, hidden_size, config):
        super().__init__()
        self.out_size = config.size if config.active else hidden_size
        self.dropout = nn.Dropout(config.dropout)

        if config.active:
            self.proj = nn.Sequential(
                nn.Linear(hidden_size, config.size, config.bias),
                ACT2FN[config.act]
            )
            if config.fixup_init:
                self.proj[0].weight.data.uniform_(-config.init_range, config.init_range)
                if config.bias:
                    self.proj[0].bias.data.zero_()
        else:
            self.proj = nn.Identity()

    def forward(self, x):
        return self.proj(self.dropout(x))
