# PyTorch-TCN
<p align="center">
<b>Streamable (Real-Time) Temporal Convolutional Networks in PyTorch</b>
</p>

This python package provides a flexible and comprehensive implementation of temporal convolutional neural networks (TCN) in PyTorch analogous to the popular tensorflow/keras package [keras-tcn](https://github.com/philipperemy/keras-tcn). Like keras-tcn, the implementation of pytorch-tcn is based on the TCN architecture presented by [Bai et al.](https://arxiv.org/abs/1803.01271), while also including some features of the original [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) architecture (e.g. skip connections) and the option for automatic reset of dilation sizes to allow training of very deep TCN structures.

<b>Additionally</b>, this package offers a streaming inference option for causal networks (with and without lookahead). This allows to process data in small blocks instead of the whole sequence, which is essential for real-time applications. See section [Streaming Inference](#streaming-inference) for more details.

<p align="center">
  <img src="https://raw.githubusercontent.com/paul-krug/pytorch-tcn/main/misc/tcn_images.jpg">
  <b>Dilated causal (left) and non-causal convolutions (right).</b><br><br>
</p>

## Installation

```bash
pip install pytorch-tcn
```

## How to use the TCN class

```python
from pytorch_tcn import TCN

model = TCN(
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
)
# Continue to train/use model for your task
```

### Input and Output shapes

The TCN expects input tensors of shape (*N, C<sub>in</sub>, L*), where *N, C<sub>in</sub>, L* denote  the batch size, number of input channels and the sequence length, respectively. This corresponds to the input shape that is expected by 1D convolution in PyTorch. If you prefer the more common convention for time series data (*N, L, C<sub>in</sub>*) you can change the expected input shape via the 'input_shape' parameter, see below for details.
The order of output dimensions will be the same as for the input tensors.

### Parameters and how to choose meaningful values

- `num_inputs`: The number of input channels, should be equal to the feature dimension of your data.
- `num_channels`: A list or array that contains the number of feature channels in each residual block of the network.
- `kernel_size`: The size of the convolution kernel used by the convolutional layers. Good starting points may be 2-8. If the prediction task requires large context sizes, larger kernel size values may be appropriate.
- `dilations`: If None, the dilation sizes will be calculated via 2^(1...n) for the residual blocks 1 to n. This is the standard way to do it. However, if you need a custom list of dilation sizes for whatever reason you could pass such a list or array to the argument.
- `dilation_reset`: For deep TCNs the dilation size should be reset periodically, otherwise it grows exponentially and the corresponding padding becomes so large that memory overflow occurs (see [Van den Oord et al.](https://arxiv.org/pdf/1609.03499.pdf)). E.g. 'dilation_reset=16' would reset the dilation size once it reaches a value of 16, so the dilation sizes would look like this: [ 1, 2, 4, 8, 16, 1, 2, 4, ...].
- `dropout`: Is a float value between 0 and 1 that indicates the amount of inputs which are randomly set to zero during training. Usually, 0.1 is a good starting point.
- `causal`: If 'True', the dilated convolutions will be causal, which means that future information is ignored in the prediction task. This is important for real-time predictions. If set to 'False', future context will be considered for predictions.
- `use_norm`: Can be 'weight_norm', 'batch_norm', 'layer_norm' or 'None'. Uses the respective normalization within the resiudal blocks. The default is weight normalization as done in the original paper by [Bai et al.](https://arxiv.org/abs/1803.01271) Whether the other types of normalization work better in your task is difficult to say in advance so it should be tested on case by case basis. If 'None', no normalization is performed.
- `activation`: Activation function to use throughout the network. Defaults to 'relu', similar to the original paper.
- `kernel_initializer`: The function used for initializing the networks weights. Currently, can be 'uniform', 'normal', 'kaiming_uniform', 'kaiming_normal', 'xavier_uniform' or 'xavier_normal'. Kaiming and xavier initialization are also known as He and Glorot initialization, respectively. While [Bai et al.](https://arxiv.org/abs/1803.01271) originally use normal initialization, this sometimes leads to divergent behaviour and usually 'xavier_uniform' is a very good starting point, so it is used as the default here.
- `use_skip_connections`: If 'True', skip connections will be present from the output of each residual block (before the sum with the resiual, similar to WaveNet) to the end of the network, where all the connections are summed. The sum then passes another activation function. If the output of a residual block has a feature dimension different from the feature dimension of the last residual block, the respective skip connection will use a 1x1 convolution for downsampling the feature dimension. This procedure is similar to the way resiudal connections around each residual block are handled. Skip connections usually help to train deeper netowrks efficiently. However, the parameter defaults to 'False', because skip connections were not used in the original paper by [Bai et al.](https://arxiv.org/abs/1803.01271)
- `ìnput_shape`: Defaults to 'NCL', which means input tensors are expected to have the shape (batch_size, feature_channels, time_steps). This corresponds to the input shape that is expected by 1D convolutions in PyTorch. However, a common convention for timeseries data is the shape (batch_size, time_steps, feature_channels). If you want to use this convention, set the parameter to 'NLC'.
- `embedding_shapes`: Accepts an Iterable that contains tuples or types that can be converted to tuples. The tuples should contain the number of embedding dimensions. Embedding can either be 1D, e.g., lets say you train a TCN to generate speech samples and you want to condition the audio generation on a speaker embedding of shape (256,). Then you would pass [(256,)] to the argument. The TCN forward function will then accept tensors of shape (batch_size, 256,) as the argument 'embedding'. The embeddings will be automatically broadcasted to the length of the input sequence and added to the input tensor right before the first activation function in each temporal block. Hence, 1D embedding shapes will lead to a global conditioning of the TCN. For local conditioning, an 'embedding_shapes' argument should be 2D including 'None' as its second dimension (time_steps). It may look like this: [(32,None)]. Then the forward function would accept tensors of shape (batch_size, 32, time_steps). If 'embedding_shapes' is set to None, no embeddings will be used.
- `embedding_mode`: Valid modes are 'add' and 'concat'. If 'add', the embeddings will be added to the input tensor before the first activation function in each temporal block. If 'concat', the embeddings will be concatenated to the input tensor along the feature dimension and then projected to the expected dimension via a 1x1 convolution. The default is 'add'.
- `use_gate`: If 'True', a gated linear unit (see [Dauphin et al.](https://arxiv.org/abs/1612.08083)) will be used as the first activation function in each temporal block. If 'False', the activation function will be the one specified by the 'activation' parameter. Gated units may be used as activation functions to feed in embeddings (see above). This may or may not lead to better results than the regular activation, but it is likely to increase the computational costs. The default is 'False'.
- `lookahead`: If not 0, causal TCNs will use a lookahead on future time frames to increase the modelling accuracy. The lookahead parameter specifies the number of future time steps that will be processed influence the prediction for a specific time step. Default is 0. Will be ignored for non-causal networks which already have the maximum lookahead possible.
- `output_projection`: If not None, the output of the TCN will be projected to the specified dimension via a 1x1 convolution. This may be useful if the output of the TCN is supposed to be of a different dimension than the input or if the last activation should be linear. If None, no projection will be performed. The default is 'None'.
- `output_activation`: If not None, the output of the TCN will be passed through the specified activation function. This maybe useful to etablish a classification head via softmax etc. If None, no activation will be performed. The default is 'None'.

## Streaming Inference

For kernel sizes > 1, a TCN will always use zero padding to ensure that the output has the same number of time steps as the input. This leads to problems during blockwise processing: E.g. let [ X<sub>1</sub>, X<sub>2</sub>, X<sub>3</sub>, X<sub>4</sub> ] be an input sequence. With a kernel size of 3 and a dilation rate of 1, the padding length of the first convolutional layer would be 2. Hence, its input would look like this [ 0, 0, X<sub>1</sub>, X<sub>2</sub>, X<sub>3</sub>, X<sub>4</sub> ] (for a causal network). If the same sequence is divided into two chunks [ X<sub>1</sub>, X<sub>2</sub> ] and [ X<sub>3</sub>, X<sub>4</sub> ], the effective input would look like this [ 0, 0, X<sub>1</sub>, X<sub>2</sub>] + [ 0, 0, X<sub>3</sub>, X<sub>4</sub> ]. These discontinuities in the receptive field of the TCN will lead to different (and very likely degraded) outputs for the same input sequence divided into smaller chunks.

To avoid this issue, a buffer is implemented that stores the networks input history. The history is then used as padding for the next processing step. This way you will obtain the same results as if the whole sequence was processed at once.

For streaming inference the batch size must be 1.

### How to use the streaming option

```python
from pytorch_tcn import TCN

tcn = TCN(
    num_inputs,
    num_channels,
    causal=True,
)

# Important: reset the buffer before processing a new sequence
tcn.reset_buffers()

# blockwise processing
# block should be of shape:
# (1, block_size, num_inputs)
for block in blocks:
    out = tcn.inference(block)

# or alternatively

for block in blocks:
    out = tcn(block, inference=True)
```

### Lookahead

Streaming inference does only make sense for causal networks. However, one may want to use a lookahead on future time frames to increase the modelling accuracy. This can be achieved by setting the lookahead parameter to a value greater than 0. The lookahead parameter specifies the number of future time steps that will be processed in addition to the input block.

Note that lookahead will introduce <b>additional latency</b> in real-time applications.

```python

tcn = TCN(
    num_inputs,
    num_channels,
    causal=True,
    lookahead=1,
)

tcn.reset_buffers()

block # shape: (1, block_size + lookahead, num_inputs)
output = tcn.inference(block)

# output will be of shape:
# (1, block_size, num_outputs)
```