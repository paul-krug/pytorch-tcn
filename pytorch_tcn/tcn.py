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
from collections.abc import Iterable


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
            a torch.nn.Module object, but type {type(activation)} was passed.
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
            Argument '{arg_name}' must be one of: {allowed_values}
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



class CausalConv1d(nn.Conv1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride = 1,
            dilation = 1,
            groups = 1,
            bias = True,
            buffer = None,
            lookahead = 0,
            **kwargs,
            ):
        
        super(CausalConv1d, self).__init__(
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
        
        self.pad_len = (kernel_size - 1) * dilation
        if lookahead > self.pad_len//2:
            lookahead = self.pad_len//2
        self.lookahead = lookahead

        self.buffer_len = self.pad_len - self.lookahead
        #print( 'pad len:', self.pad_len )
        #print( 'lookahead:', self.lookahead )
        #print( 'buffer len:', self.buffer_len )
        
        if buffer is None:
            buffer = torch.zeros(
                1,
                in_channels,
                self.pad_len,
                )
            
        self.register_buffer(
            'buffer',
            buffer,
            )
        
        return
    
    def _forward(self, x):
        p = nn.ConstantPad1d(
            ( self.buffer_len, self.lookahead ),
            0.0,
            )
        x = p(x)
        x = super().forward(x)
        return x

    def forward(
            self,
            x,
            inference=False,
            ):
        if inference:
            x = self._inference(x)
        else:
            x = self._forward(x)
        return x
    
    def inference(self, x):
        if x.shape[0] != 1:
            raise ValueError(
                f"""
                Streaming inference of CausalConv1D layer only supports
                a batch size of 1, but batch size is {x.shape[0]}.
                """
                )
        if x.shape[2] < self.lookahead + 1:
            raise ValueError(
                f"""
                Input time dimension {x.shape[2]} is too short for causal
                inference with lookahead {self.lookahead}. You must pass at
                least lookhead + 1 time steps ({self.lookahead + 1}).
                """
                )
        x = torch.cat(
            (self.buffer, x),
            -1,
            )
        if self.lookahead > 0:
            self.buffer = x[:, :, -(self.pad_len+self.lookahead) : -self.lookahead ]
        else:
            self.buffer = x[:, :, -self.buffer_len: ]
        x = super().forward(x)
        return x
    
    def reset_buffer(self):
        self.buffer.zero_()
        if self.buffer.shape[2] != self.pad_len:
            raise ValueError(
                f"""
                Buffer shape {self.buffer.shape} does not match the expected
                shape (1, {self.in_channels}, {self.pad_len}).
                """
                )
        return



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
            **kwargs,
            ):
        
        self.pad_len = (kernel_size-1) * dilation

        super(TemporalConv1d, self).__init__(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = self.pad_len // 2,
            dilation = dilation,
            groups = groups,
            bias = bias,
            **kwargs,
            )
        
        return
    
    def forward(self, x):
        # Implementation of 'same'-type padding (non-causal padding)
    
        # Check if pad_len is an odd value
        # If so, pad the input one more on the right side
        if (self.pad_len % 2 != 0):
            x = F.pad(x, [0, 1])

        x = super(TemporalConv1d, self).forward(x)

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
            embedding_shapes,
            embedding_mode,
            use_gate,
            lookahead,
            ):
        super(TemporalBlock, self).__init__()
        self.use_norm = use_norm
        self.activation = activation
        self.kernel_initializer = kerner_initializer
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.lookahead = lookahead

        if self.use_gate:
            conv1d_n_outputs = 2 * n_outputs
        else:
            conv1d_n_outputs = n_outputs

        if self.causal:
            self.conv1 = CausalConv1d(
                in_channels=n_inputs,
                out_channels=conv1d_n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                lookahead=self.lookahead,
                )

            self.conv2 = CausalConv1d(
                in_channels=n_outputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                lookahead=self.lookahead,
                )

        else:
            self.conv1 = TemporalConv1d(
                in_channels=n_inputs,
                out_channels=conv1d_n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                )

            self.conv2 = TemporalConv1d(
                in_channels=n_outputs,
                out_channels=n_outputs,
                kernel_size=kernel_size,
                stride=stride,
                dilation=dilation,
                )
        
        if use_norm == 'batch_norm':
            if self.use_gate:
                self.norm1 = nn.BatchNorm1d(2 * n_outputs)
            else:
                self.norm1 = nn.BatchNorm1d(n_outputs)
            self.norm2 = nn.BatchNorm1d(n_outputs)
        elif use_norm == 'layer_norm':
            if self.use_gate:
                self.norm1 = nn.LayerNorm(2 * n_outputs)
            else:
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

        if isinstance( self.activation, str ):
            self.activation1 = activation_fn[ self.activation ]()
            self.activation2 = activation_fn[ self.activation ]()
            self.activation_final = activation_fn[ self.activation ]()
        else:
            self.activation1 = self.activation()
            self.activation2 = self.activation()
            self.activation_final = self.activation()

        if self.use_gate:
            self.activation1 = nn.GLU(dim=1)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, padding=0) if n_inputs != n_outputs else None

        if self.embedding_shapes is not None:
            if self.use_gate:
                embedding_layer_n_outputs = 2 * n_outputs
            else:
                embedding_layer_n_outputs = n_outputs

            self.embedding_projection_1 = nn.Conv1d(
                in_channels = sum( [ shape[0] for shape in self.embedding_shapes ] ),
                out_channels = embedding_layer_n_outputs,
                kernel_size = 1,
                )
            
            self.embedding_projection_2 = nn.Conv1d(
                in_channels = 2 * embedding_layer_n_outputs,
                out_channels = embedding_layer_n_outputs,
                kernel_size = 1,
                )
        
        self.init_weights()
        return

    def init_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
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
    
    def apply_embeddings(
            self,
            x,
            embeddings,
            ):
        
        if not isinstance( embeddings, list ):
            embeddings = [ embeddings ]

        e = []
        for embedding, expected_shape in zip( embeddings, self.embedding_shapes ):
            if embedding.shape[1] != expected_shape[0]:
                raise ValueError(
                    f"""
                    Embedding shape {embedding.shape} passed to 'forward' does not 
                    match the expected shape {expected_shape} provided as input to
                    argument 'embedding_shapes'.
                    """
                    )
            if len( embedding.shape ) == 2:
                # unsqueeze time dimension of e and repeat it to match x
                e.append( embedding.unsqueeze(2).repeat(1, 1, x.shape[2]) )
            elif len( embedding.shape ) == 3:
                # check if time dimension of embedding matches x
                if embedding.shape[2] != x.shape[2]:
                    raise ValueError(
                        f"""
                        Embedding time dimension {embedding.shape[2]} does not
                        match the input time dimension {x.shape[2]}
                        """
                        )
                e.append( embedding )
        e = torch.cat( e, dim=1 )
        e = self.embedding_projection_1( e )
        #print('shapes:', e.shape, x.shape)
        if self.embedding_mode == 'concat':
            x = self.embedding_projection_2(
                torch.cat( [ x, e ], dim=1 )
                )
        elif self.embedding_mode == 'add':
            x = x + e

        return x
    
    def forward(
            self,
            x,
            embeddings,
            inference,
            ):
        out = self.conv1(x, inference=inference)
        out = self.apply_norm( self.norm1, out )

        if embeddings is not None:
            out = self.apply_embeddings( out, embeddings )

        out = self.activation1(out)
        out = self.dropout1(out)

        out = self.conv2(out, inference=inference)
        out = self.apply_norm( self.norm2, out )
        out = self.activation2(out)
        out = self.dropout2(out)

        res = x if self.downsample is None else self.downsample(x)
        return self.activation_final(out + res), out
    
    def inference(
            self,
            x,
            embeddings,
            ):
        if not self.causal:
            raise ValueError(
                """
                This streaming inference mode is made for blockwise causal
                processing and thus, is only supported for causal networks.
                However, you selected a non-causal network.
                """
                )
        x, out = self.forward(x, embeddings, inference=True)
        return x, out



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
            embedding_shapes: Optional[ ArrayLike ] = None,
            embedding_mode: str = 'add',
            use_gate: bool = False,
            lookahead: int = 0,
            output_projection: Optional[ int ] = None,
            output_activation: Optional[ str ] = None,
            ):
        super(TCN, self).__init__()

        if dilations is not None and len(dilations) != len(num_channels):
            raise ValueError("Length of dilations must match length of num_channels")
        
        self.allowed_norm_values = ['batch_norm', 'layer_norm', 'weight_norm', None]
        self.allowed_input_shapes = ['NCL', 'NLC']

        _check_generic_input_arg( causal, 'causal', [True, False] )
        _check_generic_input_arg( use_norm, 'use_norm', self.allowed_norm_values )
        _check_activation_arg(activation, 'activation')
        _check_generic_input_arg( kernel_init_fn, 'kernel_initializer', kernel_init_fn.keys() )
        _check_generic_input_arg( use_skip_connections, 'use_skip_connections', [True, False] )
        _check_generic_input_arg( input_shape, 'input_shape', self.allowed_input_shapes )
        _check_generic_input_arg( embedding_mode, 'embedding_mode', ['add', 'concat'] )
        _check_generic_input_arg( use_gate, 'use_gate', [True, False] )
        _check_activation_arg(output_activation, 'output_activation')

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
        self.activation = activation
        self.kernel_initializer = kernel_initializer
        self.use_skip_connections = use_skip_connections
        self.input_shape = input_shape
        self.embedding_shapes = embedding_shapes
        self.embedding_mode = embedding_mode
        self.use_gate = use_gate
        self.causal = causal
        self.lookahead = lookahead
        self.output_projection = output_projection
        self.output_activation = output_activation

        if embedding_shapes is not None:
            if isinstance(embedding_shapes, Iterable):
                for shape in embedding_shapes:
                    if not isinstance( shape, tuple ):
                        try:
                            shape = tuple( shape )
                        except Exception as e:
                            raise ValueError(
                                f"""
                                Each shape in argument 'embedding_shapes' must be an Iterable of tuples.
                                Tried to convert {shape} to tuple, but failed with error: {e}
                                """
                                )
                    if len( shape ) not in [ 1, 2 ]:
                        raise ValueError(
                            f"""
                            Tuples in argument 'embedding_shapes' must be of length 1 or 2.
                            One-dimensional tuples are interpreted as (embedding_dim,) and
                            two-dimensional tuples as (embedding_dim, time_steps).
                            """
                            )
            else:
                raise ValueError(
                    f"""
                    Argument 'embedding_shapes' must be an Iterable of tuples,
                    but is {type(embedding_shapes)}.
                    """
                    )
            

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
            if isinstance( self.activation, str ):
                self.activation_skip_out = activation_fn[ self.activation ]()
            else:
                self.activation_skip_out = self.activation()
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
                    embedding_shapes=self.embedding_shapes,
                    embedding_mode=self.embedding_mode,
                    use_gate=self.use_gate,
                    lookahead=self.lookahead,
                    )
                ]

        self.network = nn.ModuleList(layers)

        if self.output_projection is not None:
            self.projection_out = nn.Conv1d(
                in_channels=num_channels[-1],
                out_channels=self.output_projection,
                kernel_size=1,
                )
        else:
            self.projection_out = None

        if self.output_activation is not None:
            if isinstance( self.output_activation, str ):
                self.activation_out = activation_fn[ self.output_activation ]()
            else:
                self.activation_out = self.output_activation()
        else:
            self.activation_out = None #nn.Identity()

        if self.causal:
            self.reset_buffers()
        return
    
    def init_skip_connection_weights(self):
        initialize, kwargs = get_kernel_init_fn(
            name=self.kernel_initializer,
            activation=self.activation,
            )
        for layer in self.downsample_skip_connection:
            if layer is not None:
                initialize(
                    layer.weight,
                    **kwargs
                    )
        return

    def forward(
            self,
            x,
            embeddings=None,
            inference=False,
            ):
        if inference and not self.causal:
            raise ValueError(
                """
                This streaming inference mode is made for blockwise causal
                processing and thus, is only supported for causal networks.
                However, you selected a non-causal network.
                """
                )
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        if self.use_skip_connections:
            skip_connections = []
            # Adding skip connections from each layer to the output
            # Excluding the last layer, as it would not skip trainable weights
            for index, layer in enumerate( self.network ):
                x, skip_out = layer(
                    x,
                    embeddings=embeddings,
                    inference=inference,
                    )
                if self.downsample_skip_connection[ index ] is not None:
                    skip_out = self.downsample_skip_connection[ index ]( skip_out )
                if index < len( self.network ) - 1:
                    skip_connections.append( skip_out )
            skip_connections.append( x )
            x = torch.stack( skip_connections, dim=0 ).sum( dim=0 )
            x = self.activation_skip_out( x )
        else:
            for layer in self.network:
                #print( 'TCN, embeddings:', embeddings.shape )
                x, _ = layer(
                    x,
                    embeddings=embeddings,
                    inference=inference,
                    )
        if self.projection_out is not None:
            x = self.projection_out( x )
        if self.activation_out is not None:
            x = self.activation_out( x )
        if self.input_shape == 'NLC':
            x = x.transpose(1, 2)
        return x
    
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
    
    def reset_buffers(self):
        def _reset_buffer(x):
            if isinstance(x, CausalConv1d):
                x.reset_buffer()
        self.apply(_reset_buffer)
        return