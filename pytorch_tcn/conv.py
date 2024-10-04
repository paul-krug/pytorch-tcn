import os
import warnings
import torch
import torch.nn as nn
import math

# Padding modes
PADDING_MODES = [
    'zeros',
    'reflect',
    'replicate',
    'circular',
]

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
            padding_mode='zeros',
            device=None,
            dtype=None,
            buffer = None,
            causal = True,
            lookahead = 0,
            ):
        super(TemporalConv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = 0, # Padding is reimplemented in this class
            dilation = dilation,
            groups = groups,
            bias = bias,
            padding_mode='zeros', # Padding is reimplemented in this class
            device=device,
            dtype=dtype,
            )
        
        # Padding is computed internally
        if padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    If you want to suppress this error in order to use the layer as drop-in replacement
                    for nn.Conv1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0'
                    (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                The parameter will be removed in a future version.
                """
                )


        self.pad_len = (kernel_size - 1) * dilation
        self.causal = causal
        
        if causal:
            # Padding is only on the left side
            self.left_pad = self.pad_len
            self.right_pad = 0
        else:
            # Padding is on both sides
            self.left_pad = self.pad_len // 2
            self.right_pad = self.pad_len - self.left_pad
        
        if padding_mode == 'zeros':
            self.padder = nn.ConstantPad1d(
                ( self.left_pad, self.right_pad ),
                0.0,
                )
        elif padding_mode == 'reflect':
            self.padder = nn.ReflectionPad1d(
                ( self.left_pad, self.right_pad ),
                )
        elif padding_mode == 'replicate':
            self.padder = nn.ReplicationPad1d(
                ( self.left_pad, self.right_pad ),
                )
        elif padding_mode == 'circular':
            self.padder = nn.CircularPad1d(
                ( self.left_pad, self.right_pad ),
                )
        else:
            raise ValueError(
                f"""
                padding_mode must be one of {PADDING_MODES},
                but got {padding_mode}.
                """
                )

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
        x = self.padder(x)
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
            output_padding = 0,
            groups = 1,
            bias = True,
            dilation = 1,
            padding_mode = 'zeros',
            device=None,
            dtype=None,
            buffer = None,
            causal = True,
            lookahead = 0,
            ):
        
        # Padding is computed internally
        if padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    The value of 'padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'padding' must be 0 for TemporalConv1d, because the correct amount
                    of padding is calculated automatically based on the kernel size and dilation.
                    If you want to suppress this error in order to use the layer as drop-in replacement
                    for nn.Conv1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0'
                    (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )
            
        # dilation rate should be 1
        if dilation != 1:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'dilation' must be 1 for TemporalConvTranspose1d, other values are
                    not supported. The value of 'dilation' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'dilation' must be 1 for TemporalConvTranspose1d, other values are
                    not supported. If you want to suppress this error in order to use the layer as drop-in
                    replacement for nn.ConvTranspose1d, set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN'
                    to '0' (will reduce error to a warning) or '1' (will suppress the error/warning entirely).
                    """
                    )
            
        # output_padding should be 0
        if output_padding != 0:
            if os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '0':
                warnings.warn(
                    """
                    The value of arg 'output_padding' must be 0 for TemporalConvTranspose1d, because the correct
                    amount of padding is calculated automatically based on the kernel size and stride. The value
                    of 'output_padding' will be ignored.
                    """
                    )
            elif os.environ.get( 'PYTORCH_TCN_ALLOW_DROP_IN', 'Not set' ) == '1':
                pass
            else:
                raise ValueError(
                    """
                    The value of arg 'output_padding' must be 0 for TemporalConvTranspose1d, because the correct
                    amount of padding is calculated automatically based on the kernel size and stride. If you want
                    to suppress this error in order to use the layer as drop-in replacement for nn.ConvTranspose1d,
                    set the environment variable 'PYTORCH_TCN_ALLOW_DROP_IN' to '0' (will reduce error to a warning)
                    or '1' (will suppress the error/warning entirely).
                    """
                    )

        # Lookahead is only kept for legacy reasons, ensure it is zero
        if lookahead != 0:
            raise ValueError(
                """
                The lookahead parameter is deprecated and must be set to 0.
                The parameter will be removed in a future version.
                """
                )

        # This implementation only supports kernel_size == 2 * stride with power of 2 strides
        if kernel_size != 2 * stride or not math.log2(stride).is_integer() or stride < 2:
            raise ValueError(
                f"""
                This implementation only supports kernel_size == 2 * stride with power of 2 strides and stride >= 2.
                """
                )


        self.causal = causal                      
        self.upsampling_factor = stride
        self.buffer_size = (kernel_size // stride) - 1

        if self.causal:
            self.pad_left = self.buffer_size
            self.pad_right = 0
            self.implicit_padding = 0
        else:
            self.pad_left = 0
            self.pad_right = 0
            self.implicit_padding = (kernel_size-stride)//2

        super(TemporalConvTranspose1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.implicit_padding, # Padding is reimplemented in this class
            output_padding = 0, # Output padding is not supported
            groups = groups,
            bias = bias,
            dilation = 1, # Dilation is not supported
            padding_mode = 'zeros', # Padding mode is reimplemented in this class
            device=device,
            dtype=dtype,
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

        if padding_mode == 'zeros':
            self.padder = nn.ConstantPad1d(
                    (self.pad_left, self.pad_right),
                    0.0,
                    )
        elif padding_mode == 'reflect':
            self.padder = nn.ReflectionPad1d(
                    (self.pad_left, self.pad_right),
                    )
        elif padding_mode == 'replicate':
            self.padder = nn.ReplicationPad1d(
                    (self.pad_left, self.pad_right),
                    )
        elif padding_mode == 'circular':
            self.padder = nn.CircularPad1d(
                    (self.pad_left, self.pad_right),
                    )
        else:
            raise ValueError(
                f"""
                padding_mode must be one of {PADDING_MODES},
                but got {padding_mode}.
                """
                )

        return

    def _forward(self, x):
        x = self.padder(x)
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

