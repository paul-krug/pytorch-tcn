
"""
This implementation is based on the following repository:
https://github.com/jik876/hifi-gan

The original code is licensed under the MIT License:

MIT License

Copyright (c) 2020 Jungil Kong

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
import torch.nn.functional as F
import torch.nn as nn

from pytorch_tcn import BaseTCN
from pytorch_tcn import TemporalConv1d as Conv1d
from pytorch_tcn import TemporalConvTranspose1d as ConvTranspose1d
from torch.nn.utils.parametrizations import weight_norm

from typing import Any, Mapping


LRELU_SLOPE = 0.1


class ResBlock( BaseTCN ):
    def __init__(
            self,
            channels,
            kernel_size,
            dilation,
            resblock_type,
            causal,
            ):
        super(ResBlock, self).__init__()

        self.resblock_type = resblock_type
        if resblock_type not in [1,2]:
            raise ValueError()
        
        self.convs1 = nn.ModuleList(
            [
            weight_norm(
                Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=d,
                    causal=causal,
                    )
                )
            for d in dilation
            ]
        )
        #self.convs1.apply( super(ResBlock).init_weights )

        if resblock_type == 2:
            self.convs2 = nn.ModuleList([
                weight_norm(
                    Conv1d(
                        in_channels=channels,
                        out_channels=channels,
                        kernel_size=kernel_size,
                        stride=1,
                        dilation=1,
                        causal=causal,
                        )
                    )
                for _ in dilation
            ])
            #self.convs2.apply( super(ResBlock).init_weights )

        self.init_weights()
        return
    
    def forward(
            self,
            x,
            inference = False,
            ):
        if self.resblock_type == 1:
            for c in self.convs:
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c(
                    x=xt,
                    inference=inference,
                    )
                x = xt + x
        else:
            for c1, c2 in zip(self.convs1, self.convs2):
                xt = F.leaky_relu(x, LRELU_SLOPE)
                xt = c1(
                    x=xt,
                    inference=inference,
                    )
                xt = F.leaky_relu(xt, LRELU_SLOPE)
                xt = c2(
                    x=xt,
                    inference=inference,
                    )
                x = xt + x
        return x

  
class HifiGenerator( BaseTCN ):
    def __init__(
            self,
            in_channels,
            out_channels,
            pre_conv_kernel_size,
            post_conv_kernel_size,
            upsample_initial_channel,
            upsample_rates,
            upsample_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            resblock_type=2,
            embedding_dim=0,
            causal=False,
            ):
        super(HifiGenerator, self).__init__()

        self.upsample_initial_channel = upsample_initial_channel
        self.upsample_kernel_sizes = upsample_kernel_sizes
        self.upsample_rates = upsample_rates
        self.resblock_kernel_sizes = resblock_kernel_sizes
        self.resblock_dilation_sizes = resblock_dilation_sizes

        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)
        
        self.conv_pre = weight_norm(
            Conv1d(
                in_channels=in_channels,
                out_channels=upsample_initial_channel,
                kernel_size=pre_conv_kernel_size,
                stride=1,
                causal=causal,
                )
        )

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(
                weight_norm(
                    ConvTranspose1d(
                        in_channels = upsample_initial_channel // (2 ** i),
                        out_channels = upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size = k,
                        stride = u,
                        causal=causal,
                    )
                )
            )

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel // (2 ** (i + 1))
            for j, (k, d) in enumerate(
                zip(resblock_kernel_sizes, resblock_dilation_sizes)
            ):
                self.resblocks.append(
                    ResBlock(
                        channels=ch,
                        kernel_size=k,
                        dilation=d,
                        resblock_type=resblock_type,
                        causal=causal,
                        )
                    )

        self.recp_numkernels = torch.tensor(
            1.0 / self.num_kernels,
            dtype=torch.float32,
            )
    
        self.conv_post = weight_norm(
            Conv1d(
                in_channels=ch,
                out_channels=out_channels,
                kernel_size=post_conv_kernel_size,
                stride=1,
                causal=causal,
                )
            )
        
        if embedding_dim != 0:
            self.cond = nn.Conv1d(
                embedding_dim,
                upsample_initial_channel,
                1,
            )
        
        self.init_weights()
        self.reset_buffers()
        return

    def forward(
            self,
            x,
            embedding = None,
            inference = False,
            ):
        
        x = self.conv_pre(
            x=x,
            inference=inference,
            )
        if embedding is not None:
            x = x + self.cond(
                embedding.unsqueeze(2)
                )
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, LRELU_SLOPE)
            x = self.ups[i](
                x=x,
                inference=inference,
                )
            xs = self.resblocks[i * self.num_kernels](
                x,
                inference=inference,
                )
            for j in range(1, self.num_kernels):
                xs = xs + self.resblocks[i * self.num_kernels + j](
                    x=x,
                    inference=inference,
                    )
            x = xs * self.recp_numkernels

        x = F.leaky_relu(x)
        x = self.conv_post(
            x=x,
            inference=inference,
            )
        x = torch.tanh(x)

        return x