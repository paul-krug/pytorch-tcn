import os
import warnings
import torch
import torch.nn as nn
import math

from .buffer import BufferIO

from typing import Optional
from typing import Union
from typing import List

# Padding modes
PADDING_MODES = [
    'zeros',
    'reflect',
    'replicate',
    'circular',
]

class TemporalPad1d(nn.Module):
    def __init__(
            self,
            padding: int,
            in_channels: int,
            buffer: Optional[ Union[ float, torch.Tensor ] ] = None,
            padding_mode: str = 'zeros',
            causal: bool = False,
            ):
        super(TemporalPad1d, self).__init__()

        if not isinstance(padding, int):
            raise ValueError(
                f"""
                padding must be an integer, but got {type(padding)}.
                padding must not be a tuple, because the TemporalPadding
                will automatically determine the amount of left and right
                padding based on the causal flag.
                """
                )

        self.pad_len = padding
        self.causal = causal

        if causal:
            # Padding is only on the left side
            self.left_padding = self.pad_len
            self.right_padding = 0
        else:
            # Padding is on both sides
            self.left_padding = self.pad_len // 2
            self.right_padding = self.pad_len - self.left_padding
        
        if padding_mode == 'zeros':
            self.pad = nn.ConstantPad1d(
                (self.left_padding, self.right_padding),
                0.0,
                )
        elif padding_mode == 'reflect':
            self.pad = nn.ReflectionPad1d(
                (self.left_padding, self.right_padding),
                )
        elif padding_mode == 'replicate':
            self.pad = nn.ReplicationPad1d(
                (self.left_padding, self.right_padding),
                )
        elif padding_mode == 'circular':
            self.pad = nn.CircularPad1d(
                (self.left_padding, self.right_padding),
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
            if in_channels is None:
                buffer = torch.zeros(
                    1,
                    self.pad_len,
                    )
            else:
                buffer = torch.zeros(
                    1,
                    in_channels,
                    self.pad_len,
                    )
        elif isinstance(buffer, (int, float)):
            if in_channels is None:
                buffer = torch.full(
                    size = (1, self.pad_len),
                    fill_value = buffer,
                    )
            else:
                buffer = torch.full(
                    size = (1, in_channels, self.pad_len),
                    fill_value = buffer,
                    )
        elif not isinstance(buffer, torch.Tensor):
            raise ValueError(
                f"""
                The argument 'buffer' must be None or of type float,
                int, or torch.Tensor, but got {type(buffer)}.
                """
                )
        
        # Register buffer as a persistent buffer which is available as self.buffer
        self.register_buffer(
            'buffer',
            buffer,
            )
        
        return
    
    def pad_inference(
            self,
            x: torch.Tensor,
            buffer_io: Optional[ BufferIO ] = None,
            ):

        if not self.causal:
            raise ValueError(
                """
                Streaming inference is only supported for causal convolutions.
                """
                )

        batch_size = x.size(0)

        def _align_buffer(buf: torch.Tensor, name: str) -> torch.Tensor:
            # buf is None  -> handled by caller
            if buf is None:
                return None
            if buf.size(0) == batch_size:
                return buf
            if buf.size(0) == 1:
                # replicate the single history for every element in the batch
                return buf.repeat(batch_size, 1, 1)
            raise ValueError(
                f"Batch mismatch between input (N={batch_size}) and "
                f"{name} (N={buf.size(0)}).  Either supply a buffer of the "
                f"same batch size or call .reset_buffer(batch_size=N)."
            )
                
        
        if buffer_io is None:
            in_buffer = _align_buffer(self.buffer, 'internal buffer')
        else:
            in_buffer = buffer_io.next_in_buffer()
            if in_buffer is None:            # first iteration, fall back
                in_buffer = self.buffer
            in_buffer = _align_buffer(in_buffer, 'in_buffer from BufferIO')
            buffer_io.append_internal_buffer(in_buffer)
        
        # pad the current input with the previous history
        x = torch.cat((in_buffer, x), dim=-1)
        
        # remember the most recent history for the *next* call
        out_buffer = x[..., -self.pad_len:]
        if buffer_io is None:
            self.buffer = out_buffer
        else:
            buffer_io.append_out_buffer(out_buffer)

        return x
    
    def forward(
            self,
            x: torch.Tensor,
            inference: bool = False,
            buffer_io: Optional[ BufferIO ] = None,
            ):
        if inference:
            x = self.pad_inference(x, buffer_io=buffer_io)
        else:
            x = self.pad(x)
        return x
    
    def reset_buffer(self, batch_size: int = 1) -> None:
            """
            Reset the streaming buffer to zeros.
        
            Parameters
            ----------
            batch_size : int, default 1
                Number of parallel streams that will be processed in the next
                call(s).  If this differs from the current buffer’s batch
                dimension, the buffer is re‑allocated accordingly.
            """
            if self.buffer.size(0) != batch_size:
                self.buffer = torch.zeros(
                    batch_size,
                    self.buffer.size(1),      # channels
                    self.pad_len,
                    device=self.buffer.device,
                    dtype=self.buffer.dtype,
                )
            else:
                self.buffer.zero_()
            return
