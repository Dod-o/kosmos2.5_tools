# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
try:
    from apex.normalization import FusedLayerNorm as LayerNorm
except ModuleNotFoundError:
    from torch.nn import LayerNorm
from .model_parallel import ModelParallelLinear
from einops import rearrange, repeat
# from fairseq.model_parallel.megatron.mpu import gather_from_model_parallel_region, scatter_to_model_parallel_region


class set_torch_seed(object):
    def __init__(self, seed):
        assert isinstance(seed, int)
        self.rng_state = self.get_rng_state()

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)

    def get_rng_state(self):
        state = {"torch_rng_state": torch.get_rng_state()}
        if torch.cuda.is_available():
            state["cuda_rng_state"] = torch.cuda.get_rng_state()
        return state

    def set_rng_state(self, state):
        torch.set_rng_state(state["torch_rng_state"])
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(state["cuda_rng_state"])

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.set_rng_state(self.rng_state)


def make_experts(args, embed_dim, expert_ffn_dim):
    world_size = (
        1
        if not torch.distributed.is_initialized()
        else torch.distributed.get_world_size()
    )
    expert_list = []
    ddp_rank = args.ddp_rank
    start_seed = torch.randint(1000000, (1,)).item()
    # at least as many experts than gpus
    if args.moe_expert_count >= world_size:
        assert (
            args.moe_expert_count % world_size == 0
        ), f"{args.moe_expert_count}, {world_size}"
        local_moe_expert_count = args.moe_expert_count // world_size
        for i in range(local_moe_expert_count):
            with set_torch_seed(start_seed + ddp_rank * local_moe_expert_count + i):
                expert_list.append(
                    FeedForwardNetwork(
                        args,
                        embed_dim,
                        expert_ffn_dim,
                        args.activation_fn,
                        args.dropout,
                        args.activation_dropout,
                        args.layernorm_eps,
                        args.subln,
                        args.sparse_ratio,
                    )
                )
    else:
        assert (
            world_size % args.moe_expert_count == 0
        ), f"{world_size}, {args.moe_expert_count}"

        with set_torch_seed(start_seed + ddp_rank % args.moe_expert_count):
            expert_list.append(
                FeedForwardNetwork(
                    args,
                    embed_dim,
                    expert_ffn_dim,
                    args.activation_fn,
                    args.dropout,
                    args.activation_dropout,
                    args.layernorm_eps,
                    args.subln,
                    args.sparse_ratio,
                )
            )
    experts = nn.ModuleList(expert_list)
    return experts


def get_activation_fn(activation):
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu
    else:
        raise NotImplementedError


class CausalConv(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 kernel_size: int
                 ) -> None:
        super().__init__()
        self.conv = nn.Conv1d(
            embed_dim,
            embed_dim,
            kernel_size,
            padding=kernel_size-1,
            groups=embed_dim
        )

    def forward(self,x):
        """
            x: b l d
        """
        x = rearrange(x, 'b l d -> b d l')
        seq_len = x.shape[-1]
        x = self.conv(x)[..., :seq_len]
        x = rearrange(x, 'b d l -> b l d')
        return x 

def ffn_init_method(tensor, **kwargs):
    nn.init.kaiming_uniform_(tensor, a=math.sqrt(5))

class FeedForwardNetwork(nn.Module):
    def __init__(
        self,
        args,
        embed_dim,
        ffn_dim,
        activation_fn,
        dropout,
        activation_dropout,
        layernorm_eps,
        subln=False,
        sparse_ratio=1,
    ):
        super().__init__()
        self.args = args
        self.embed_dim = embed_dim
        self.activation_fn = get_activation_fn(activation=str(activation_fn))
        self.activation_dropout_module = torch.nn.Dropout(activation_dropout)
        self.dropout_module = torch.nn.Dropout(dropout)
        self.ln_embed_dim = ffn_dim // args.group_norm_size
        assert ffn_dim % args.model_parallel_size == 0
        assert (ffn_dim // args.model_parallel_size) % self.ln_embed_dim == 0
        if sparse_ratio > 1 and getattr(args, 'gating_conv', False):
            ffn_dim = (ffn_dim * 2 // 3 + 7) // 8 * 8 // args.model_parallel_size * args.model_parallel_size
            self.ln_embed_dim = ffn_dim // args.group_norm_size
            self.fc1 = ModelParallelLinear(args, self.embed_dim, ffn_dim * 2, parallel_mode='column', init_method=ffn_init_method)
            self.fc2 = ModelParallelLinear(args, ffn_dim, self.embed_dim, parallel_mode='row', init_method=ffn_init_method)
            self.ffn_layernorm = LayerNorm(self.ln_embed_dim, eps=layernorm_eps, elementwise_affine=True) if subln else None
            self.conv = CausalConv(ffn_dim, sparse_ratio)
        else:
            self.fc1 = ModelParallelLinear(args, self.embed_dim, ffn_dim, parallel_mode='column', init_method=ffn_init_method)
            self.fc2 = ModelParallelLinear(args, ffn_dim, self.embed_dim, parallel_mode='row', init_method=ffn_init_method)
            self.ffn_layernorm = LayerNorm(self.ln_embed_dim, eps=layernorm_eps, elementwise_affine=True) if subln else None
            self.conv = None
        self.ffn_dim = ffn_dim
        if args.model_parallel_size == 1:
            self.reset_parameters()

    def reset_parameters(self):
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        if self.ffn_layernorm is not None:
            self.ffn_layernorm.reset_parameters()

    def forward(self, x):
        x = self.fc1(x)

        if self.conv is None:
            x = self.activation_fn(x.float()).type_as(x)
            x = self.activation_dropout_module(x)
        else:
            residual, x = x.chunk(2,dim=-1)
            x = self.conv(x) * F.silu(residual)

        if self.ffn_layernorm is not None:
            # if self.args.model_parallel_size > 1:
            #     x = x.contiguous()
            #     x = gather_from_model_parallel_region(x)
            if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                x = rearrange(x, 'b l (n d) -> b l n d', d=self.ln_embed_dim)
            x = self.ffn_layernorm(x)
            if (self.ffn_dim // self.args.model_parallel_size) > self.ln_embed_dim:
                x = rearrange(x, 'b l n d -> b l (n d)')
            # if self.args.model_parallel_size > 1:
            #     x = scatter_to_model_parallel_region(x)

        x = self.fc2(x)
        x = self.dropout_module(x)
        return x