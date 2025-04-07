
import torch
import torch.nn as nn
from collections.abc import Iterable

from .conditional_layer import Additive, Concatenative, FiLM

from typing import Union, List

MODES = {
    'add': Additive,
    'concat': Concatenative,
    'film': FiLM,
    }

class Condition(nn.Module):
    def __init__(
            self,
            dim: int,
            dim_cond: Union[int, List[int]],
            mode: str,
            ):
        super().__init__()
        self.mode = mode
        self.dim = dim
        if not isinstance( dim_cond, Iterable ):
            dim_cond = [ dim_cond ]
        self.dim_cond = dim_cond
        self.condition = MODES[ mode ](
            dim = self.dim,
            dim_cond = sum( self.dim_cond ),
            )
        return
    
    def forward(
            self,
            x: torch.Tensor,
            cond: Union[ torch.Tensor, List[ torch.Tensor ] ],
            ):
        # Expects x of shape ( batch, dim, time )
        # and cond of shape ( batch, dim_cond ) or ( batch, dim_cond, time )
        # or a list of such tensors

        if not isinstance( cond, Iterable ):
            cond = [ cond ]

        e = []
        for c, expected_shape in zip( cond, self.dim_cond ):
            if c.shape[1] != expected_shape:
                raise ValueError(
                    f"""
                    Condition shape {c.shape} passed to 'forward' does
                    not match the expected shape {expected_shape} provided
                    as input to argument 'dim_cond'.
                    """
                    )
            if len( c.shape ) == 2:
                # unsqueeze time dimension of e and repeat it to match x
                e.append( c.unsqueeze(2).repeat(1, 1, x.shape[2]) )
            elif len( c.shape ) == 3:
                # check if time dimension of c matches x
                if c.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"""
                        Condition time dimension {c.shape[2]} does not
                        match the input time dimension {x.shape[2]}.
                        """
                        )
                e.append( c )
            else:
                raise ValueError(
                    f"""
                    Condition tensor must have shape ( batch, dim_cond ) or
                    ( batch, dim_cond, time ).
                    """
                    )
            
        e = torch.cat( e, dim = 1 )
        x = self.condition( x, e )
        return x