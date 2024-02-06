import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import numpy as np
import warnings

from typing import Tuple
from typing import Union
from typing import Optional
from numpy.typing import ArrayLike



activation_fn = dict(
    relu=nn.ReLU,
    tanh=nn.Tanh,
    leaky_relu=nn.LeakyReLU,
    sigmoid=nn.Sigmoid,
    elu=nn.ELU,
    gelu=nn.GELU,
    selu=nn.SELU,
)

kernel_init_fn = dict(
    xavier_uniform=nn.init.xavier_uniform_,
    xavier_normal=nn.init.xavier_normal_,
    kaiming_uniform=nn.init.kaiming_uniform_,
    kaiming_normal=nn.init.kaiming_normal_,
    normal=nn.init.normal_,
    uniform=nn.init.uniform_,
)



def get_kernel_init_fn(
        name: str,
        activation: str,
        ) -> Tuple[ nn.Module, dict ]:
    if name not in kernel_init_fn.keys():
        raise ValueError(
            f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
            )
    if name in [ 'xavier_uniform', 'xavier_normal' ]:
        if activation in [ 'gelu', 'elu' ]:
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
        if activation in [ 'gelu', 'elu' ]:
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



class TemporalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            dilation=1,
            groups=1,
            bias=True,
            causal=True,
            ):
        
        padding = (kernel_size-1) * dilation# if causal else 0

        super(TemporalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            )
        
        self.causal = causal
        return
    
    def forward(self, input):
        if self.causal:
            x = F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
                )
            # Chomp the output to have left padding only (causal padding)
            x = x[:, :, :-self.padding[0]].contiguous()
        else:
            # Implementation of 'same'-type padding (non-causal padding)
    
            # Check if padding has odd length
            # If so, pad the input one more on the right side
            if (self.padding[0] % 2 != 0):
                input = F.pad(input, [0, 1])

            x = F.conv1d(
                input,
                self.weight,
                self.bias,
                self.stride,
                padding=self.padding[0]//2,
                dilation=self.dilation,
                groups=self.groups,
                )

        return x



class TemporalBlock(nn.Module):
    def __init__(
            self,
            n_inputs,
            n_outputs,
            kernel_size,
            stride,
            dilation,
            dropout,
            causal,
            use_norm,
            activation,
            kerner_initializer,
            ):
        super(TemporalBlock, self).__init__()
        self.use_norm = use_norm
        self.activation_name = activation
        self.kernel_initializer = kerner_initializer

        self.conv1 = TemporalConv1d(
            in_channels=n_inputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            )

        self.conv2 = TemporalConv1d(
            in_channels=n_outputs,
            out_channels=n_outputs,
            kernel_size=kernel_size,
            stride=stride,
            dilation=dilation,
            causal=causal,
            )
        
        if use_norm == 'batch_norm':
            self.norm1 = nn.BatchNorm1d(n_outputs)
            self.norm2 = nn.BatchNorm1d(n_outputs)
        elif use_norm == 'layer_norm':
            self.norm1 = nn.LayerNorm(n_outputs)
            self.norm2 = nn.LayerNorm(n_outputs)
        elif use_norm == 'weight_norm':
            self.norm1 = None
            self.norm2 = None
            self.conv1 = weight_norm(self.conv1)
            self.conv2 = weight_norm(self.conv2)
        elif use_norm is None:
            self.norm1 = None
            self.norm2 = None

        self.activation1 = activation_fn[ self.activation_name ]()
        self.activation2 = activation_fn[ self.activation_name ]()
        self.activation_final = activation_fn[ self.activation_name ]()

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        
        self.init_weights()
        return

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation_name,
            )
        initialize(
            self.conv1.weight,
            **kwargs
            )
        initialize(
            self.conv2.weight,
            **kwargs
            )

        if self.downsample is not None:
            initialize(
                self.downsample.weight,
                **kwargs
                )
        return
    
    def apply_norm(
            self,
            norm_fn,
            x,
        ):
        if self.use_norm == 'batch_norm':
            x = norm_fn(x)
        elif self.use_norm == 'layer_norm':
            x = norm_fn( x.transpose(1, 2) )
            x = x.transpose(1, 2)
        return x

    def forward(self, x):
        out = self.conv1(x)
        out = self.apply_norm( self.norm1, out )
        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.apply_norm( self.norm2, out )
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.activation_final(out + res), out



class TCN(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_channels: ArrayLike,
            kernel_size: int = 4,
            dilations: Optional[ ArrayLike ] = None,
            dilation_reset: Optional[ int ] = None,
            dropout: float = 0.1,
            causal: bool = True,
            use_norm: str = 'weight_norm',
            activation: str = 'relu',
            kernel_initializer: str = 'xavier_uniform',
            use_skip_connections: bool = False,
            input_shape: str = 'NCL',
            ):
        super(TCN, self).__init__()
        if dilations is not None and len(dilations) != len(num_channels):
            raise ValueError("Length of dilations must match length of num_channels")
        
        self.allowed_norm_values = ['batch_norm', 'layer_norm', 'weight_norm', None]
        if use_norm not in self.allowed_norm_values:
            raise ValueError(
                f"Argument 'use_norm' must be one of: {self.allowed_norm_values}"
                )
        
        if activation not in activation_fn.keys():
            raise ValueError(
                f"Argument 'activation' must be one of: {activation_fn.keys()}"
                )
        
        if kernel_initializer not in kernel_init_fn.keys():
            raise ValueError(
                f"Argument 'kernel_initializer' must be one of: {kernel_init_fn.keys()}"
                )
        
        self.allowed_input_shapes = ['NCL', 'NLC']
        if input_shape not in self.allowed_input_shapes:
            raise ValueError(
                f"Argument 'input_shape' must be one of: {self.allowed_input_shapes}"
                )

        if dilations is None:
            if dilation_reset is None:
                dilations = [ 2 ** i for i in range( len( num_channels ) ) ]
            else:
                # Calculate after which layers to reset
                dilation_reset = int( np.log2( dilation_reset * 2 ) )
                dilations = [
                    2 ** (i % dilation_reset)
                    for i in range( len( num_channels ) )
                    ]
            
        self.dilations = dilations
        self.activation_name = activation
        self.kernel_initializer = kernel_initializer
        self.use_skip_connections = use_skip_connections
        self.input_shape = input_shape

        if use_skip_connections:
            self.downsample_skip_connection = nn.ModuleList()
            for i in range( len( num_channels ) ):
                # Downsample layer output dim to network output dim if needed
                if num_channels[i] != num_channels[-1]:
                    self.downsample_skip_connection.append(
                        nn.Conv1d( num_channels[i], num_channels[-1], 1 )
                        )
                else:
                    self.downsample_skip_connection.append( None )
            self.init_skip_connection_weights()
            self.activation_out = activation_fn[ self.activation_name ]()
        else:
            self.downsample_skip_connection = None
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = self.dilations[i]

            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [
                TemporalBlock(
                    n_inputs=in_channels,
                    n_outputs=out_channels,
                    kernel_size=kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    dropout=dropout,
                    causal=causal,
                    use_norm=use_norm,
                    activation=activation,
                    kerner_initializer=self.kernel_initializer,
                    )
                ]

        self.network = nn.ModuleList(layers)
        return
    
    def init_skip_connection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation_name,
            )
        for layer in self.downsample_skip_connection:
            if layer is not None:
                initialize(
                    layer.weight,
                    **kwargs
                    )
        return

    def forward(self, x):
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        if self.use_skip_connections:
            skip_connections = []
            # Adding skip connections from each layer to the output
            # Excluding the last layer, as it would not skip trainable weights
            for index, layer in enumerate( self.network ):
                x, skip_out = layer(x)
                if self.downsample_skip_connection[ index ] is not None:
                    skip_out = self.downsample_skip_connection[ index ]( skip_out )
                if index < len( self.network ) - 1:
                    skip_connections.append( skip_out )
            skip_connections.append( x )
            x = torch.stack( skip_connections, dim=0 ).sum( dim=0 )
            x = self.activation_out( x )
        else:
            for layer in self.network:
                x, _ = layer(x)
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        return x