import os
import warnings
from typing import Optional

from .buffer import InternalBuffer
import torch
import torch.nn as nn

class TemporalGRU(nn.GRU, InternalBuffer):
    """
    A custom RNN layer that supports streaming inference with internal buffering,
    following a similar paradigm to TemporalConv1d.

    The TemporalGRU behaves exactly like a standard nn.RNN during training.
    When inference is enabled (inference=True), it uses an internal hidden-state
    buffer (or one provided via a BufferIO object) to maintain continuity across
    sequential calls (e.g. for streaming applications). The updated hidden state
    is stored internally (or via the provided buffer_io) so that subsequent calls
    will use the previous state.
    
    Args:
        input_size (int): The number of expected features in the input.
        hidden_size (int): The number of features in the hidden state.
        num_layers (int, optional): Number of recurrent layers. Default: 1.
        bias (bool, optional): If False, then the layer does not use bias weights.
                               Default: True.
        batch_first (bool, optional): If True, then the input and output tensors are
                                      provided as (batch, seq, feature). Default: False.
        dropout (float, optional): If non-zero, introduces a Dropout layer on the outputs
                                   of each RNN layer except the last layer.
        bidirectional (bool, optional): If True, becomes a bidirectional RNN.
        buffer (torch.Tensor, optional): Initial hidden state buffer. Typically, this is
                                         left as None so that the RNN will use a zero
                                         initial state.
        **kwargs: Additional keyword arguments passed to nn.RNN.
    """
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        bias: bool = True,
        batch_first: bool = False,
        dropout: float = 0.0,
        bidirectional: bool = False,
        buffer: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        super(TemporalGRU, self).__init__(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
            **kwargs,
        )
        # Hidden state buffer for streaming inference.
        # Its shape should be (num_layers * num_directions, batch, hidden_size).
        # Buffer is used for streaming inference
        if buffer is None:
            self._buffer = torch.zeros(
                num_layers * (2 if bidirectional else 1),
                1,  
                hidden_size,)
            
        elif isinstance(buffer, (int, float)):
            self._buffer = torch.full(
                size = (num_layers * (2 if bidirectional else 1), 1, hidden_size),
                fill_value = buffer,
                )
        elif not isinstance(buffer, torch.Tensor):
            raise ValueError(
                f"""
                The argument 'buffer' must be None or of type float,
                int, or torch.Tensor, but got {type(buffer)}.
                """
                )

        self._buffer = buffer

    @property
    def buffer(self) -> torch.Tensor:
        """
        Returns the hidden state buffer.
        """
        return self._buffer
    
    @buffer.setter
    def buffer(self, value: torch.Tensor) -> None:
        """
        Sets the hidden state buffer.
        """
        self._buffer = value

    def forward(
        self,
        x: torch.Tensor,
        inference: bool = False,
        in_buffer: torch.Tensor = None,
        buffer_io: Optional["BufferIO"] = None,
    ) -> torch.Tensor:
        """
        Forward pass of the TemporalGRU.

        Args:
            x (torch.Tensor): Input tensor.
            inference (bool, optional): If True, streaming inference is enabled
                                        and an internal hidden state buffer is used.
                                        Default: False.
            in_buffer (torch.Tensor): This argument has been deprecated.
                                      Use buffer_io instead.
            buffer_io (Optional[BufferIO]): An object that holds a 'buffer' attribute.
                                            This can be used to externally manage the hidden state.

        Returns:
            torch.Tensor: The output sequence of the RNN.
        """
        if in_buffer is not None:
            raise ValueError(
                """
                The argument 'in_buffer' was removed.
                Instead, you should pass the hidden state buffer as a BufferIO object
                to the argument 'buffer_io'.
                """
            )

        if not inference:
            # In training mode, perform a standard forward pass.
            output, _ = super(TemporalGRU, self).forward(x)
            return output
        else:
            # In inference (streaming) mode, use the stored hidden state buffer.
            if buffer_io is None:
                h0 = self._buffer
            else:
                h0 = buffer_io.next_in_buffer()
                if h0 is None:
                    h0 = self._buffer
                    buffer_io.append_internal_buffer(h0)

            # If no buffer is present, default to None so that nn.RNN uses zero initial state.
            if h0 is None:
                output, hidden = super(TemporalGRU, self).forward(x)
            else:
                output, hidden = super(TemporalGRU, self).forward(x, h0)

            # Update the buffer with the new hidden state.
            if buffer_io is None:
                self._buffer = hidden
            else:
                buffer_io.append_out_buffer( hidden )
                

            return output

    def reset_buffer(self) -> None:
        """
        Resets the internal hidden state buffer to None.
        """
        self._buffer = None