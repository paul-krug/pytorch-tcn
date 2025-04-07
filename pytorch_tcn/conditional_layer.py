
import torch
import torch.nn as nn

from typing import Union, List



class Additive(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.cond = nn.Conv1d(
                dim_cond,
                dim,
                1,
            )
        return
    
    def forward(self, x, cond):
        x = x + self.cond( cond )
        return x
    
class Concatenative(nn.Module):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.cond = nn.Conv1d(
                dim + dim_cond,
                dim,
                1,
            )
        return
    
    def forward(self, x, cond):
        x = torch.cat([x, cond], dim = 1)
        x = self.cond(x)
        return x

class FiLM( nn.Module ):
    def __init__(self, dim, dim_cond):
        super().__init__()
        self.to_cond = nn.Conv1d(dim_cond, dim * 2, 1)
        return

    def forward(self, x, cond):
        gamma, beta = self.to_cond(cond).chunk(2, dim = 1)
        return x * gamma + beta