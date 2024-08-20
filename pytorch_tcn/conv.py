import warnings
import torch
import torch.nn as nn
import math


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
            **kwargs,
            ):
        
        self.pad_len = (kernel_size - 1) * dilation
        self.causal = causal
        
        super(TemporalConv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0,
            dilation = dilation,
            groups = groups,
            bias = bias,
            **kwargs,
            )
        
        if causal:
            # Padding is only on the left side
            self.left_pad = self.pad_len
            self.right_pad = 0
        else:
            # Padding is on both sides
            self.left_pad = self.pad_len // 2
            self.right_pad = self.pad_len - self.left_pad
        
        # Buffer is used for streaming inference
        if buffer is None:
            buffer = torch.zeros(
                1,
                in_channels,
                self.pad_len,
                )
        
        # Register buffer as a persistent buffer which is available as self.buffer
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
            in_buffer=None,
            ):
        if inference:
            if in_buffer is None:
                x, self.buffer = self.inference(x, self.buffer)
                return x
            else:
                return self.inference(x, in_buffer)
        else:
            return self._forward(x)
    
    def inference(self, x, in_buffer):
        if x.shape[0] != 1:
            raise ValueError(
                f"""
                Streaming inference of CausalConv1D layer only supports
                a batch size of 1, but batch size is {x.shape[0]}.
                """
                )

        x = torch.cat(
            (in_buffer, x),
            -1,
            )

        out_buffer = x[:, :, -self.pad_len: ]
        x = super().forward(x)
        return x, out_buffer
    
    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[2] != self.pad_len:
            raise ValueError(
                f"""
                Buffer shape {self.buffer.shape} does not match the expected
                shape (1, {self.in_channels}, {self.pad_len}).
                """
                )


class TemporalConvTranspose1d(nn.ConvTranspose1d):
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
            **kwargs,
            ):
        
        super(TemporalConvTranspose1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0,
            output_padding = 0,
            dilation = dilation,
            groups = groups,
            bias = bias,
            **kwargs,
            )
        
        self.causal = causal                      
        self.pad_len = (math.ceil(kernel_size/stride) - 1)
        self.upsampling_factor = stride

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
    
    def _causal_forward(self, x):
        p = nn.ReplicationPad1d(
            (self.pad_len, 0),
            )
        x = p(x)
        x = super().forward(x)
        x = x[:, :, self.upsampling_factor : -self.upsampling_factor]
        return x
    
    def forward(
            self,
            x,
            inference=False,
            in_buffer=None,
            ):
        if inference:
            if in_buffer is None:
                x, self.buffer = self.inference(x, self.buffer)
                return x
            else:
                return self.inference(x, in_buffer)
        else:
            return self._forward(x)
    
    def _forward(
            self,
            x,
            ):
        if self.causal:
            x = self._causal_forward(x)
        else:
            x = super().forward(x)
        return x
    
    def inference(self, x, in_buffer):
        x = torch.cat(
            (in_buffer, x),
            -1,
            )
        out_buffer = x[:, :, -self.pad_len:]
        x = super().forward(x)
        x = x[:, :, self.upsampling_factor : -self.upsampling_factor]
        return x, out_buffer
    
    def reset_buffer(self):
        self.buffer.zero_()

