# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import torch
from einops import rearrange, repeat
from .flash_attention_triton import _flash_attn_forward as flash_attn_forward_dense
from .flash_attention_triton import _flash_attn_backward as flash_attn_backward_dense
from .flash_attention_triton_sparse import _flash_attn_forward as flash_attn_forward_sparse
from .flash_attention_triton_sparse import _flash_attn_backward as flash_attn_backward_sparse


class FlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def sparsify(x, sparse_ratio):
        x = rearrange(x, 'b (l r1) (r2 h) d -> b l h d r1 r2', r1=sparse_ratio, r2=sparse_ratio)
        x = torch.diagonal(x, offset=0, dim1=4, dim2=5)
        x = rearrange(x, 'b l h d r -> b l (r h) d')
        return x

    @staticmethod
    def forward(ctx, q, k, v, bias=None, causal=False, softmax_scale=None, sparse_ratio=1, block_size=128):
        """
            q: (batch_size, seqlen_q, nheads, headdim)
            k, v: (batch_size, seqlen_k, nheads, headdim)
            bias: optional, shape broadcastible to (batch, nheads, seqlen_q, seqlen_k).
                For example, ALiBi mask for causal would have shape (1, nheads, 1, seqlen_k).
                ALiBi mask for non-causal would have shape (1, nheads, seqlen_q, seqlen_k)
        """
        # Make sure that the last dimension is contiguous
        q, k, v = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, v]]

        with torch.no_grad():
            global_q = q[:, ::sparse_ratio].contiguous()
            global_k = FlashAttnFunc.sparsify(k, sparse_ratio).contiguous()
            global_v = FlashAttnFunc.sparsify(v, sparse_ratio).contiguous()

            local_q = rearrange(q, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()
            local_k = rearrange(k, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()
            local_v = rearrange(v, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()

            o_global, global_lse, ctx.softmax_scale = flash_attn_forward_sparse(
                global_q, global_k, global_v, bias=bias, causal=causal, softmax_scale=softmax_scale, sparse_ratio=sparse_ratio,
            )

            o_local, local_lse, _ = flash_attn_forward_dense(
                local_q, local_k, local_v, bias=bias, causal=causal, softmax_scale=softmax_scale,
            )

            global_o = rearrange(o_global, 'b g h d -> b 1 g h d')
            global_lse = rearrange(global_lse, 'b h g -> b 1 g h')
            local_o = rearrange(o_local, '(b n) g h d -> b n g h d', b=q.size(0))
            local_lse = rearrange(local_lse, '(b n) h g -> b n g h', b=q.size(0))

            global_lse, local_lse = torch.exp(global_lse), torch.exp(local_lse)
            o = (global_o.float() * global_lse + local_o.float() * local_lse) / (global_lse + local_lse)
            o = rearrange(o.type_as(q), 'b n g h d -> b (n g) h d')
            lse = rearrange(torch.log(lse), 'b n g h -> b h (n g)')

        ctx.save_for_backward(q, k, v, o_global, o_local, lse, bias)
        ctx.causal = causal
        ctx.sparse_ratio = sparse_ratio
        ctx.block_size = block_size
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, o_global, o_local, lse, bias = ctx.saved_tensors
        sparse_ratio, block_size = ctx.sparse_ratio, ctx.block_size
        assert not ctx.needs_input_grad[3], 'FlashAttention does not support bias gradient yet'
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(q)
            dk = torch.empty_like(k)
            dv = torch.empty_like(v)

            global_q = q[:, ::sparse_ratio].contiguous()
            global_k = FlashAttnFunc.sparsify(k, sparse_ratio).contiguous()
            global_v = FlashAttnFunc.sparsify(v, sparse_ratio).contiguous()

            local_q = rearrange(q, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()
            local_k = rearrange(k, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()
            local_v = rearrange(v, 'b (n g) h d -> (b n) g h d', g=block_size).contiguous()

            do_global = rearrange(do, 'b (n g) h d -> b n g h d', g=block_size).sum(1)
            dq_global = torch.empty_like(global_q)
            dk_global = torch.empty_like(global_k)
            dv_global = torch.empty_like(global_v)

            flash_attn_backward_sparse(do_global, global_q, global_k, global_v, o_global, lse, dq_global, dk_global, dv_global,
                                 bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale, sparse_ratio=ctx.sparse_ratio)
            
            do_local = rearrange(do, 'b (n g) h d -> (b n) g h d', g=block_size)
            dq_local = torch.empty_like(local_q)
            dk_local = torch.empty_like(local_k)
            dv_local = torch.empty_like(local_v)

            flash_attn_backward_dense(do_local, local_q, local_k, local_v, o_local, lse, dq_local, dk_local, dv_local,
                                 bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)

            # _flash_attn_backward(do, q, k, v, o, lse, dq, dk, dv,
            #                      bias=bias, causal=ctx.causal, softmax_scale=ctx.softmax_scale)
        return dq, dk, dv, None, None, None


flash_attn_func = FlashAttnFunc.apply