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
            kernel_size,#=3,
            dilation,#=(1, 3, 5),
            resblock_type,
            causal,
            lookahead,
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
                    lookahead=lookahead,
                    )
                )
            for d in dilation
            ]
        )
        self.convs1.apply( super(ResBlock).init_weights )

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
                        lookahead=lookahead,
                        )
                    )
                for _ in dilation
            ])
            self.convs2.apply( super(ResBlock).init_weights )
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
            in_channels,#80
            out_channels,#1
            kernel_size,#7
            upsample_initial_channel,
            upsample_rates,
            upsample_kernel_sizes,
            resblock_kernel_sizes,
            resblock_dilation_sizes,
            resblock_type=2,
            causal=False,
            lookahead=0,
            ):
        super(HifiGenerator, self).__init__()

        #resblock = CausalResBlock2 if resblock_type == 2 else CausalResBlock1

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
                kernel_size=kernel_size,
                stride=1,
                causal=causal,
                lookahead=lookahead,
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
                        lookahead=lookahead,
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
                        lookahead=lookahead,
                        )
                    )

        self.recp_numkernels = torch.tensor(1.0 / self.num_kernels, dtype=torch.float32)
    
        self.conv_post = weight_norm(
            Conv1d(
                in_channels=ch,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=1,
                causal=causal,
                lookahead=lookahead,
                )
            )
        
        self.init_weights()
        self.reset_buffers()
        return

    def forward(
            self,
            x,
            inference = False,
            ):
        x = self.conv_pre(
            x=x,
            inference=inference,
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