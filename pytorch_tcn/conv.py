import os
import warnings
import torch
import torch.nn as nn
import math

from .pad import TemporalPad1d
from .buffer import BufferIO

from typing import Optional
from typing import Union
from typing import List
    


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
        
        self.padder = TemporalPad1d(
            padding = self.pad_len,
            in_channels = in_channels,
            buffer = buffer,
            padding_mode = padding_mode,
            causal = causal,
            )
        
        return
    
    # In pytorch-tcn >= 1.2.2, buffer is moved to TemporalPad1d
    # We keep the property for backwards compatibility, e.g. in
    # case one wants to load old model weights.
    @property
    def buffer(self):
        return self.padder.buffer
    
    @buffer.setter
    def buffer(self, value):
        self.padder.buffer = value
        return

    def forward(
            self,
            x: torch.Tensor,
            inference: bool = False,
            in_buffer: torch.Tensor = None,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if in_buffer is not None:
            raise ValueError(
                """
                The argument 'in_buffer' was removed in pytorch-tcn >= 1.2.2.
                Instead, you should pass the input buffer as a BufferIO object
                to the argument 'buffer_io'.
                """
                )
        x = self.padder(x, inference=inference, buffer_io=buffer_io)
        x = super().forward(x)
        return x
    
    def inference(self, *args, **kwargs):
        raise NotImplementedError(
            """
            The function "inference" was removed in pytorch-tcn >= 1.2.2.
            Instead, you should use the modules forward function with the
            argument "inference=True" enabled.
            """
            )
        return
    
    def reset_buffer(self):
        self.padder.reset_buffer()
        return


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

        # This implementation only supports kernel_size == 2 * stride
        if kernel_size != 2 * stride:
            raise ValueError(
                f"""
                This implementation of TemporalConvTranspose1d only
                supports kernel_size == 2 * stride, but got 
                kernel_size = {kernel_size} and stride = {stride}.
                """
                )


        self.causal = causal                      
        self.upsampling_factor = stride
        self.buffer_size = (kernel_size // stride) - 1

        if self.causal:
            #self.pad_left = self.buffer_size
            #self.pad_right = 0
            self.implicit_padding = 0
        else:
            #self.pad_left = 0
            #self.pad_right = 0
            self.implicit_padding = (kernel_size-stride)//2

        super(TemporalConvTranspose1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.implicit_padding, # Padding is patially reimplemented in this class
            output_padding = 0, # Output padding is not supported
            groups = groups,
            bias = bias,
            dilation = 1, # Dilation is not supported
            padding_mode = 'zeros', # Padding mode is reimplemented in this class
            device=device,
            dtype=dtype,
            )
        
        self.padder = TemporalPad1d(
            padding = self.buffer_size,
            in_channels = in_channels,
            padding_mode = padding_mode,
            causal = causal,
            )

        # Deprecated in pytorch-tcn >= 1.2.2
        # Keep for backwards compatibility to load old model weights
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
        
    def forward(
            self,
            x: torch.Tensor,
            inference: bool = False,
            in_buffer: torch.Tensor = None,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if in_buffer is not None:
            raise ValueError(
                """
                The argument 'in_buffer' was removed in pytorch-tcn >= 1.2.2.
                Instead, you should pass the input buffer as a BufferIO object
                to the argument 'buffer_io'.
                """
                )
        if self.causal:
            x = self.padder(x, inference=inference, buffer_io=buffer_io)
            x = super().forward(x)
            x = x[:, :, self.upsampling_factor : -self.upsampling_factor]
        else:
            x = super().forward(x)
            # if stride is odd, remove last element due to padding
            if self.upsampling_factor % 2 == 1:
                x = x[..., :-1]
        return x
    
    def inference(self, *args, **kwargs):
        raise NotImplementedError(
            """
            The function "inference" was removed in pytorch-tcn >= 1.2.2.
            Instead, you should use the modules forward function with the
            argument "inference=True" enabled.
            """
            )
        return
    
    def reset_buffer(self):
        self.padder.reset_buffer()

