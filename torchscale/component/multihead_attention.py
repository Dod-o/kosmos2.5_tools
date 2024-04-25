# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math

import triton
import torch
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat

try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm

from .multiway_network import MultiwayWrapper
from .xpos_relative_position import XPOS

# if (torch.cuda.get_device_capability()[0] > 7) and False:
#     from .flash_attention_triton import flash_attn_func
#     from .flash_attention_triton_sparse import flash_attn_sparse_func
# else:
#     from .flash_attention_xformers import flash_attn_func

from .model_parallel import ModelParallelLinear
# from fairseq.model_parallel.megatron.mpu import gather_from_model_parallel_region, scatter_to_model_parallel_region

try:
    from flash_attn import flash_attn_func
    has_flash_attn2 = True
    print('Using flash_attn')
except ImportError:
    has_flash_attn2 = False
    print('Failed to import flash_attn')

def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """torch.repeat_interleave(x, dim=1, repeats=n_rep)"""
    bs, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    return (
        x[:, :, None, :, :]
        .expand(bs, n_kv_heads, n_rep, slen, head_dim)
        .reshape(bs, n_kv_heads * n_rep, slen, head_dim)
    )

class MultiheadAttention(nn.Module):
    def __init__(
            self,
            args,
            embed_dim,
            num_heads,
            dropout=0.0,
            self_attention=False,
            encoder_decoder_attention=False,
            subln=False,
            shift_win=False,
            qk_norm=False,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scaling = self.head_dim ** -0.5
        self.scale_length = args.scale_length

        self.self_attention = self_attention
        self.sparse_ratio = args.sparse_ratio
        self.window_sparse_ratio = args.window_sparse_ratio
        self.block_size = args.block_size
        self.large_block_size = args.large_block_size
        self.large_sparse_ratio = args.large_sparse_ratio
        self.subblock_size = args.subblock_size
        self.sparse_qkv = args.sparse_qkv
        self.window_size = args.window_size
        self.head_stride = (self.num_heads - 1) // max(self.sparse_ratio, 1) + 1
        self.encoder_decoder_attention = encoder_decoder_attention
        self.flash_attention = args.flash_attention
        self.image_length = 1024
        self.image_start_pos = 2
        self.shift_win = shift_win
        self.vqk_thresh = 0
        self.qk_norm = qk_norm

        assert self.self_attention ^ self.encoder_decoder_attention
        # if self.flash_attention:
        #     assert self.self_attention

        self.k_proj = MultiwayWrapper(args,
                                      ModelParallelLinear(args, embed_dim, embed_dim, bias=True, parallel_mode='column',
                                                          init_method=qkv_init_method))
        self.v_proj = MultiwayWrapper(args,
                                      ModelParallelLinear(args, embed_dim, embed_dim, bias=True, parallel_mode='column',
                                                          init_method=qkv_init_method))
        self.q_proj = MultiwayWrapper(args,
                                      ModelParallelLinear(args, embed_dim, embed_dim, bias=True, parallel_mode='column',
                                                          init_method=qkv_init_method))
        self.out_proj = MultiwayWrapper(
            args,
            ModelParallelLinear(args, embed_dim, embed_dim, bias=True, parallel_mode='row', init_method=out_init_method)
        )
        self.ln_embed_dim = self.embed_dim // args.group_norm_size
        assert self.embed_dim % args.model_parallel_size == 0
        assert (self.embed_dim // args.model_parallel_size) % self.ln_embed_dim == 0
        self.inner_attn_ln = (
            MultiwayWrapper(args, LayerNorm(self.ln_embed_dim, eps=args.layernorm_eps, elementwise_affine=False))
            if subln and self.self_attention
            else None
        )
        self.dropout_module = torch.nn.Dropout(dropout)
        self.xpos = (
            XPOS(self.head_dim, args.xpos_scale_base)
            if args.xpos_rel_pos and self.self_attention
            else None
        )
        self.num_heads = self.num_heads // args.model_parallel_size
        if args.model_parallel_size == 1:
            self.reset_parameters()

        # if self.block_size > 0 and self.subblock_size > 0:
        #     self.layout = self.create_sparse_layout(
        #         args.tokens_per_sample,
        #         self.block_size,
        #         self.subblock_size,
        #         self.sparse_ratio,
        #     )

        # if self.qk_norm:
        #     self.q_norm = MultiwayWrapper(args, LayerNorm(self.head_dim))
        #     self.k_norm = MultiwayWrapper(args, LayerNorm(self.head_dim))

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1 / math.sqrt(2))
        nn.init.xavier_uniform_(self.out_proj.weight)
        nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(
            self,
            query,
            key,
            value,
            incremental_state=None,
            key_padding_mask=None,
            attn_mask=None,
            rel_pos=None,
            is_causal=False,
            # sope_rel_pos=None,
    ):
        bsz, tgt_len, embed_dim = query.size()
        src_len = tgt_len
        assert embed_dim == self.embed_dim, f"query dim {embed_dim} != {self.embed_dim}"

        key_bsz, src_len, _ = key.size()
        assert key_bsz == bsz, f"{query.size(), key.size()}"
        assert value is not None
        assert bsz, src_len == value.shape[:2]

        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)

        q = q.view(bsz, tgt_len, self.num_heads, self.head_dim)
        k = k.view(bsz, src_len, self.num_heads, self.head_dim)
        v = v.view(bsz, src_len, self.num_heads, self.head_dim)
        offset = src_len - tgt_len

        if incremental_state is not None:
            if "prev_key" in incremental_state:
                prev_key = incremental_state["prev_key"]
                prev_value = incremental_state["prev_value"]
                k = torch.cat([prev_key, k], dim=1)
                v = torch.cat([prev_value, v], dim=1)
            incremental_state["prev_key"] = k
            incremental_state["prev_value"] = v
            src_len = k.size(1)

        # if has_flash_attn2:
        attn = flash_attn_func(q, k, v, causal=is_causal)
        attn_weights = None
        # else:
        #     q = q.transpose(1, 2)
        #     k = k.transpose(1, 2)
        #     v = v.transpose(1, 2)
        #
        #     k = repeat_kv(k, self.n_rep)
        #     v = repeat_kv(v, self.n_rep)
        #
        #     q *= self.scaling
        #     attn_weights = torch.matmul(q, k.transpose(-1, -2))
        #     if attn_mask is None:
        #         attn_mask = torch.triu(
        #             torch.zeros([tgt_len, src_len])
        #                 .float()
        #                 .fill_(float("-inf"))
        #                 .type_as(attn_weights),
        #                 1 + offset,
        #         )
        #     attn_weights = torch.nan_to_num(attn_weights)
        #     attn_weights += attn_mask
        #
        #     attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).type_as(
        #         attn_weights
        #     )
        #
        #     attn = torch.matmul(attn_weights, v)
        #     attn = attn.view(bsz, self.num_heads, tgt_len,
        #                      self.head_dim).transpose(1, 2)

        attn = attn.reshape(bsz, tgt_len, self.head_dim * self.num_heads)

        attn = self.out_proj(attn)
        return attn, attn_weights



def padding_to_multiple_of(n, mult):
    remainder = n % mult
    if remainder == 0:
        return 0
    return mult - remainder


def qkv_init_method(tensor, **kwargs):
    nn.init.xavier_uniform_(tensor, gain=1 / math.sqrt(2))


def out_init_method(tensor, **kwargs):
    nn.init.xavier_uniform_(tensor)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')\


def duplicate_interleave(m):
    """
    A simple version of `torch.repeat_interleave` for duplicating a matrix while interleaving the copy.
    """
    dim0 = m.shape[0]
    m = m.view(-1, 1)  # flatten the matrix
    m = m.repeat(1, 2)  # repeat all elements into the 2nd dimension
    m = m.view(dim0, -1)  # reshape into a matrix, interleaving the copy
    return m


def apply_rotary_pos_emb(x, sin, cos, scale=1):
    sin, cos = map(lambda t: duplicate_interleave(t * scale), (sin, cos))
    # einsum notation for lambda t: repeat(t[offset:x.shape[1]+offset,:], "n d -> () n () (d j)", j=2)
    return (x * cos) + (rotate_every_two(x) * sin)
