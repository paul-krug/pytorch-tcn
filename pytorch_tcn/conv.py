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
            padding = 0,
            dilation = 1,
            groups = 1,
            bias = True,
            buffer = None,
            causal = True,
            lookahead = 0,
            **kwargs,
            ):

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                """
                )

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

        if not self.causal:
            raise ValueError(
                """
                Streaming inference is only supported for causal convolutions.
                """
                )

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
            padding = 0,
            groups = 1,
            dilation = 1,
            bias = True,
            buffer = None,
            causal = True,
            lookahead = 0,
            **kwargs,
            ):

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                """
                )

        # This implementation only supports kernel_size == 2 * stride with power of 2 strides
        if kernel_size != 2 * stride or not math.log2(stride).is_integer() or stride < 2:
            raise ValueError(
                f"""
                This implementation only supports kernel_size == 2 * stride with power of 2 strides and stride >= 2.
                """
                )

        if padding != (kernel_size-stride)//2:
            raise ValueError(
                f"""
                TemporalConvTranspose1d only supports padding=(kernel_size-stride)//2.
                """
                )

        self.causal = causal                      
        self.upsampling_factor = stride
        self.buffer_size = (kernel_size // stride) - 1

        if self.causal:
            self.pad_len = 0
        else:
            self.pad_len = (kernel_size-stride)//2

        super(TemporalConvTranspose1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.pad_len,
            output_padding = 0,
            groups = groups,
            bias = bias,
            **kwargs,
            )

        if buffer is None:
            buffer = torch.zeros(
                1,
                in_channels,
                self.buffer_size,
                )
        self.register_buffer(
            'buffer',
            buffer,
            )
        
        return

    def _forward(self, x):
        if self.causal:
            p = nn.ConstantPad1d(
                (self.buffer_size, 0),
                0.0,
                )
            x = p(x)

        x = super().forward(x)

        if self.causal:
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
    
    def inference(self, x, in_buffer):
        if not self.causal:
            raise ValueError(
                """
                Streaming inference is only supported for causal convolutions.
                """
                )
        
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
        out_buffer = x[:, :, -self.buffer_size:]
        x = super().forward(x)
        x = x[:, :, self.upsampling_factor : -self.upsampling_factor]
        return x, out_buffer
    
    def reset_buffer(self):
        self.buffer.zero_()

