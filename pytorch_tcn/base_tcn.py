import numpy as np
import warnings
import torch
import torch.nn as nn
import torch.nn.functional as F
# Try to import new weight_norm from torch.nn.utils.parametrizations
# But also keep the deprecated version for compatibility
try:
    from torch.nn.utils.parametrizations import weight_norm
except ImportError:
    from torch.nn.utils import weight_norm
    warnings.warn(
        """
        The deprecated weight_norm from torch.nn.utils.weight_norm was imported.
        Update your PyTorch version to get rid of this warning.
        """
        )

from typing import Tuple
from typing import Union
from typing import Optional
from numpy.typing import ArrayLike
from collections.abc import Iterable

from .convolution import TemporalConv1d


activation_fn = dict(
    relu=nn.ReLU,
    tanh=nn.Tanh,
    leaky_relu=nn.LeakyReLU,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    gelu=nn.GELU,
    selu=nn.SELU,
    softmax=nn.Softmax,
    log_softmax=nn.LogSoftmax,
)

kernel_init_fn = dict(
    xavier_uniform=nn.init.xavier_uniform_,
    xavier_normal=nn.init.xavier_normal_,
    kaiming_uniform=nn.init.kaiming_uniform_,
    kaiming_normal=nn.init.kaiming_normal_,
    normal=nn.init.normal_,
    uniform=nn.init.uniform_,
)

def _check_activation_arg(
        activation,
        arg_name,
        ):
    if activation is None and arg_name == 'output_activation':
        return
    if isinstance( activation, str ):
        if activation not in activation_fn.keys():
            raise ValueError(
                f"""
                If argument '{arg_name}' is a string, it must be one of:
                {activation_fn.keys()}. However, you may also pass any
                torch.nn.Module object as the 'activation' argument.
                """
                )
    elif not isinstance( activation, nn.Module ):
        raise ValueError(
            f"""
            The argument '{arg_name}' must either be a valid string or
            a torch.nn.Module object, but {activation} was passed,
            which is of type {type(activation)}.
            """
            )
    return

def _check_generic_input_arg(
        arg,
        arg_name,
        allowed_values,
        ):
    if arg not in allowed_values:
        raise ValueError(
            f"""
            Argument '{arg_name}' must be one of: {allowed_values},
            but {arg} was passed.
            """
            )
    return

def get_kernel_init_fn(
        name: str,
        activation: str,
        ) -> Tuple[ nn.Module, dict ]:
    if isinstance( activation, nn.Module ):
        return kernel_init_fn[ name ], dict()
    # TODO: this means no gain is used for custom activation functions
        
    if name not in kernel_init_fn.keys():
        raise ValueError(
            f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
            )
    if name in [ 'xavier_uniform', 'xavier_normal' ]:
        if activation in [ 'gelu', 'elu', 'softmax', 'log_softmax' ]:
            warnings.warn(
                f"""
                Argument 'kernel_initializer' {name}
                is not compatible with activation {activation} in the
                sense that the gain is not calculated automatically.
                Here, a gain of sqrt(2) (like in ReLu) is used.
                This might lead to suboptimal results.
                """
                )
            gain = np.sqrt( 2 )
        else:
            gain = nn.init.calculate_gain( activation )
        kernel_init_kw = dict( gain=gain )
    elif name in [ 'kaiming_uniform', 'kaiming_normal' ]:
        if activation in [ 'gelu', 'elu', 'softmax', 'log_softmax' ]:
            raise ValueError(
                f"""
                Argument 'kernel_initializer' {name}
                is not compatible with activation {activation}.
                It is recommended to use 'relu' or 'leaky_relu'.
                """
                )
        else:
            nonlinearity = activation
        kernel_init_kw = dict( nonlinearity=nonlinearity )
    else:
        kernel_init_kw = dict()
    
    return kernel_init_fn[ name ], kernel_init_kw

   



class BaseTCN(nn.Module):
    def __init__(
            self,
            ):
        super(BaseTCN, self).__init__()
        return
    
    def inference(
            self,
            x,
            embeddings=None,
            ):
        x = self.forward(
            x,
            embeddings=embeddings,
            inference=True,
            )
        return x
    
    def init_weights(self):
        
        def _init_weights(m):
            if isinstance(m, nn.Conv1d, nn.ConvTranspose1d):
                m.weight.data.normal_(0.0, 0.01)

        self.apply(_init_weights)

        return
    
    def reset_buffers(self):
        def _reset_buffer(x):
            if isinstance(x, TemporalConv1d):
                if hasattr(x, 'reset_buffer'):
                    x.reset_buffer()
        self.apply(_reset_buffer)
        return