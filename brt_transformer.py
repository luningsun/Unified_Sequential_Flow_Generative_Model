from collections import namedtuple
from einops import rearrange, repeat
from inspect import isfunction
import torch
from torch import einsum, nn
import torch.nn.functional as F
from typing import Optional, Tuple
# from x_transformers.x_transformers import (
#     apply_rotary_pos_emb, default, exists, FeedForward, RMSNorm
# )

"""
This may change significantly as I work out how to implement this properly, but until large portions of this are copied from Phil Wang (@lucidrains)
"""
class RMSNorm(nn.Module):
    def __init__(self, dim, eps = 1e-8):
        super().__init__()
        self.scale = dim ** -0.5
        self.eps = eps
        self.g = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = torch.norm(x, dim = -1, keepdim = True) * self.scale
        return x / norm.clamp(min = self.eps) * self.g
def exists(val):
    return val is not None
def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class RotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        use_xpos = False,
        scale_base = 512
    ):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

        if not use_xpos:
            self.register_buffer('scale', None)
            return

        scale = (torch.arange(0, dim, 2) + 0.4 * dim) / (1.4 * dim)

        self.scale_base = scale_base
        self.register_buffer('scale', scale)

    def forward(self, seq_len, device):
        t = torch.arange(seq_len, device = device).type_as(self.inv_freq)
        freqs = torch.einsum('i , j -> i j', t, self.inv_freq)
        freqs = torch.cat((freqs, freqs), dim = -1)

        if not exists(self.scale):
            return freqs, 1.

        power = (torch.arange(seq_len, device = device) - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = torch.cat((scale, scale), dim = -1)

        return freqs, scale

# SeqTensor = TensorType['batch', 'seq_len', 'token_dim']
# StateTensor = TensorType['batch', 'state_len', 'state_dim']

# constants

DEFAULT_DIM_HEAD = 64
MIN_DIM_HEAD = 32

Intermediates = namedtuple('Intermediates', [
    'pre_softmax_attn',
    'post_softmax_attn'
])

LayerIntermediates = namedtuple('Intermediates', [
    'hiddens',
    'attn_intermediates'
])


def cast_tuple(val, num=1):
    return val if isinstance(val, tuple) else ((val,) * num)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        inv_freq = 1. / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, max_seq_len, *, device, offset=0):
        seq = torch.arange(max_seq_len, device=device) + offset
        freqs = einsum('i , j -> i j', seq.type_as(self.inv_freq), self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return rearrange(emb, 'n d -> 1 1 n d')


def rotate_half(x):
    x = rearrange(x, '... (j d) -> ... j d', j=2)
    x1, x2 = x.unbind(dim=-2)
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(t, freqs):
    seq_len, rot_dim = t.shape[-2], freqs.shape[-1]
    t, t_pass = t[..., :rot_dim], t[..., rot_dim:]
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t, t_pass), dim=-1)


class FeedForward(nn.Module):
    def __init__(
        self,
        dim,
        dim_out = None,
        mult = 4,
        glu = False,
        swish = False,
        relu_squared = False,
        post_act_ln = False,
        dropout = 0.,
        no_bias = False,
        zero_init_output = False
    ):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)

        if relu_squared:
            activation = ReluSquared()
        elif swish:
            activation = nn.SiLU()
        else:
            activation = nn.GELU()

        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim, bias = not no_bias),
            activation)

        self.ff = nn.Sequential(
            project_in,
            nn.LayerNorm(inner_dim) if post_act_ln else nn.Identity(),
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out, bias = not no_bias)
        )


    def forward(self, x):
        return self.ff(x)
class RecurrentStateGate(nn.Module):
    """Poor man's LSTM
    """

    def __init__(self, dim: int):
        super().__init__()

        self.main_proj = nn.Linear(dim, dim, bias=True)
        self.input_proj = nn.Linear(dim, dim, bias=True)
        self.forget_proj = nn.Linear(dim, dim, bias=True)

    def forward(self, x: SeqTensor, state: StateTensor) -> StateTensor:
        z = torch.tanh(self.main_proj(x))
        i = torch.sigmoid(self.input_proj(x) - 1)
        f = torch.sigmoid(self.forget_proj(x) + 1)
        return torch.mul(state, f) + torch.mul(z, i)


class Attention(nn.Module):
    """Shamelessly copied from github.com/lucidrains/RETRO-pytorch
    """

    def __init__(
            self,
            dim,
            *,
            dim_head=64,
            heads=8,
            causal=False,
            dropout=0.,
            null_kv=False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        self.causal = causal
        inner_dim = dim_head * heads

        self.norm = RMSNorm(dim)
        self.dropout = nn.Dropout(dropout)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)

        # allowing for attending to nothing (null function)
        # and to save attention from breaking if all retrieved chunks are padded out
        self.null_kv = nn.Parameter(torch.randn(2, inner_dim)) if null_kv else None

    def forward(self, x, mask=None, context=None, pos_emb=None):
        b, device, h, scale = x.shape[0], x.device, self.heads, self.scale

        x = self.norm(x)
        kv_input = default(context, x)

        q = self.to_q(x)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)

        # split heads
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=h), (q, k, v))

        # scale
        q = q * scale

        # apply relative positional encoding (rotary embeddings)
        if exists(pos_emb):
            q_pos_emb, k_pos_emb = cast_tuple(pos_emb, num=2)
            q = apply_rotary_pos_emb(q, q_pos_emb)
            k = apply_rotary_pos_emb(k, k_pos_emb)

        # add null key / values
        if exists(self.null_kv):
            nk, nv = self.null_kv.unbind(dim=0)
            nk, nv = map(lambda t: repeat(t, '(h d) -> b h 1 d', b=b, h=h), (nk, nv))
            k = torch.cat((nk, k), dim=-2)
            v = torch.cat((nv, v), dim=-2)

        # derive query key similarities
        sim = einsum('b h i d, b h j d -> b h i j', q, k)

        # masking
        mask_value = -torch.finfo(sim.dtype).max

        if exists(mask):
            if exists(self.null_kv):
                mask = F.pad(mask, (1, 0), value=True)

            mask = rearrange(mask, 'b j -> b 1 1 j')
            sim = sim.masked_fill(~mask, mask_value)

        if self.causal:
            i, j = sim.shape[-2:]
            causal_mask = torch.ones(i, j, device=device, dtype=torch.bool).triu(j - i + 1)
            sim = sim.masked_fill(causal_mask, mask_value)

        # attention
        attn = sim.softmax(dim=-1)

        attn = self.dropout(attn)

        # aggregate
        out = einsum('b h i j, b h j d -> b h i d', attn, v)

        # merge heads
        out = rearrange(out, 'b h n d -> b n (h d)')

        # combine heads linear out
        return self.to_out(out)



class BlockRecurrentAttention(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_state: int,
            dim_head: int = DEFAULT_DIM_HEAD,
            state_len: int = 512,
            heads: int = 8,
            **kwargs
    ):
        super().__init__()
        self.scale = dim_head ** -0.5

        attn_kwargs = {}

        self.dim = dim
        self.dim_state = dim_state

        self.heads = heads
        self.causal = True
        self.state_len = state_len
        rotary_emb_dim = max(dim_head // 2, MIN_DIM_HEAD)
        self.rotary_pos_emb = RotaryEmbedding(rotary_emb_dim)

        self.input_self_attn = Attention(dim, heads=heads, causal=True, **attn_kwargs)
        self.state_self_attn = Attention(dim_state, heads=heads, causal=False, **attn_kwargs)

        self.input_state_cross_attn = Attention(dim, heads=heads, causal=False, **attn_kwargs)
        self.state_input_cross_attn = Attention(dim_state, heads=heads, causal=False, **attn_kwargs)

        self.proj_gate = RecurrentStateGate(dim)
        self.ff_gate = RecurrentStateGate(dim)

        self.input_proj = nn.Linear(dim + dim_state, dim, bias=False)
        self.state_proj = nn.Linear(dim + dim_state, dim, bias=False)

        self.input_ff = FeedForward(dim)
        self.state_ff = FeedForward(dim_state)


    def forward(
            self,
            x,
            state = None,
            mask=None,
            state_mask=None
    ):
        batch, seq_len, device = x.shape[0], x.shape[-2], x.device
        if not exists(state):
            state = torch.zeros((batch, self.state_len, self.dim_state), device=device)
        self_attn_pos_emb = self.rotary_pos_emb(seq_len, device=device)
        state_pos_emb = self.rotary_pos_emb(self.state_len, device=device)
        input_attn = self.input_self_attn(x, mask=mask, pos_emb=self_attn_pos_emb)
        state_attn = self.state_self_attn(state, mask=state_mask, pos_emb=state_pos_emb)

        # TODO: This is different from how it is implemented in the paper, because the Keys and Values aren't shared
        # between the cross attention and self-attention. I'll implement that later, this is faster for now.
        input_as_q_cross_attn = self.input_state_cross_attn(x, context=state, mask=mask)
        state_as_q_cross_attn = self.state_input_cross_attn(state, context=x, mask=state_mask)

        projected_input = self.input_proj(torch.concat((input_as_q_cross_attn, input_attn), dim=2))
        projected_state = self.state_proj(torch.concat((state_as_q_cross_attn, state_attn), dim=2))

        input_residual = projected_input + x
        state_residual = self.proj_gate(projected_state, state)

        output = self.input_ff(input_residual) + input_residual
        next_state = self.ff_gate(self.state_ff(state_residual), state_residual)

        return output, next_state



