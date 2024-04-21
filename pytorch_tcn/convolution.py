import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Tuple
from typing import Union
from typing import Optional
from numpy.typing import ArrayLike
from collections.abc import Iterable




class TemporalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            dilation = 1,
            groups = 1,
            bias = True,
            buffer = None,
            causal = True,
            lookahead = 0,
            **kwargs,
            ):
        
        self.pad_len = (kernel_size - 1) * dilation
        self.causal = causal

        if causal:
            padding = 0
            lookahead = 0
        else:
            padding = self.pad_len // 2

        
        super(TemporalConv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = padding,
            dilation = dilation,
            groups = groups,
            bias = bias,
            **kwargs,
            )
        
        # TODO: lookahead can only be possible for non-causal convolutions
        if lookahead > self.pad_len//2:
            warnings.warn(
                f"""
                Lookahead {lookahead} is greater than half of the kernel size
                {self.pad_len}. Setting lookahead to {self.pad_len//2}.
                """
                )
            lookahead = self.pad_len//2
        self.lookahead = lookahead

        self.buffer_len = self.pad_len - self.lookahead
        self.left_pad = self.buffer_len
        self.right_pad = self.lookahead

        if (lookahead is not None) and (self.pad_len % 2 != 0):
            self.right_pad += 1

        
        if causal:
            if buffer is None:
                buffer = torch.zeros(
                    1,
                    in_channels,
                    self.pad_len,
                    )
            
            self.register_buffer(
                'buffer',
                buffer,
                )
        
        return
    
    def _forward(self, x):
        p = nn.ConstantPad1d(
            ( self.left_pad, self.right_pad ),
            0.0,
            )
        x = p(x)
        x = super().forward(x)
        return x

    def forward(
            self,
            x,
            inference=False,
            ):
        if inference:
            x = self.inference(x)
        else:
            x = self._forward(x)
        return x
    
    def inference(self, x):
        if x.shape[0] != 1:
            raise ValueError(
                f"""
                Streaming inference of CausalConv1D layer only supports
                a batch size of 1, but batch size is {x.shape[0]}.
                """
                )

        x = torch.cat(
            (self.buffer, x),
            -1,
            )

        self.buffer = x[:, :, -self.buffer_len: ]
        x = super().forward(x)
        return x
    
    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[2] != self.pad_len:
            raise ValueError(
                f"""
                Buffer shape {self.buffer.shape} does not match the expected
                shape (1, {self.in_channels}, {self.pad_len}).
                """
                )
        return