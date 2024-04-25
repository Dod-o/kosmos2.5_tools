# Copyright (c) 2022 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import copy

import torch
import torch.nn as nn
try:
    from fairseq.model_parallel.megatron.mpu import (
        ColumnParallelLinear,
        RowParallelLinear,
    )
except:
    pass


def ModelParallelLinear(args, input_dim, output_dim, bias=True, parallel_mode='row', init_method=nn.init.xavier_uniform_):
    if args.model_parallel_size <= 1:
        return nn.Linear(input_dim, output_dim, bias)
    elif parallel_mode == 'row':
        return RowParallelLinear(input_dim, output_dim, bias, input_is_parallel=True, init_method=init_method)
    else:
        assert parallel_mode == 'column'
        return ColumnParallelLinear(input_dim, output_dim, bias, gather_output=False, init_method=init_method)
