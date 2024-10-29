
import torch

from typing import Optional
from typing import Union
from typing import List
from collections.abc import Iterable


class BufferIO():
    def __init__(
            self,
            in_buffers: Optional[ Iterable ] = None,
            ):
        if in_buffers is not None:
            self.in_buffers_length = len( in_buffers )
            self.in_buffers = iter( in_buffers )
        else:
            self.in_buffers_length = None
            self.in_buffers = None
        
        self.out_buffers = []
        return
    
    def __iter__(self):
        return self
    
    def __next__(self):
        if self.in_buffers is not None:
            return next( self.in_buffers )
        else:
            return None
        
    def append_out_buffer(
            self,
            x: torch.Tensor,
            ):
        self.out_buffers.append(x)
        return
        
    def next_in_buffer(
            self,
            ):
        return self.__next__()
        
    def step(self):
        if len( self.out_buffers ) != self.in_buffers_length:
            raise ValueError(
                """
                Number of out buffers does not match number of in buffers.
                """
                )
        self.in_buffers = iter( self.out_buffers )
        self.out_buffers = []
        return