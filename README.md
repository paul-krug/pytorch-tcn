# PyTorch-TCN

Tested with torch 2.0.x (Sep 21, 2023).

This python package provides a flexible and comprehensive implementation of temporal convolutional neural networks (TCN) in PyTorch analogous to the popular tensorflow/keras package [keras-tcn](https://github.com/philipperemy/keras-tcn). Like keras-tcn, the implementation of pytorch-tcn is based on the TCN architecture presented by [Bai et al.](https://arxiv.org/abs/1803.01271), while also including some features of the original [WaveNet](https://arxiv.org/pdf/1609.03499.pdf) architecture (e.g. skip connections). Unlike keras tcn, pytorch-tcn allows skip connections even in the case of non-equal input and output feature dimensions. Furthermore, additional functions, such as the option for automatic reset of dilation sizes for very deep TCN structures, are built in.

## Installation

```bash
pip install pytorch-tcn
pip install pytorch-tcn --no-dependencies  # without the dependencies if you already have PyTorch/Numpy.
```

## How to use the TCN class

```python
model = TCN(
    num_inputs: int,
    num_channels: ArrayLike,
    kernel_size: int = 4,
    dilations: Optional[ ArrayLike ] = None,
    dilaton_reset: Optional[ int ] = None,
    dropout: float = 0.1,
    causal: bool = True,
    use_norm: str = 'weight_norm',
    activation: str = 'relu',
    kernel_initializer: str = 'xavier_uniform',
    use_skip_connections: bool = True,
)
# Continue to train/use model for your task
```

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
- `use_skip_connections`: If 'True', skip connections will be present from the output of each residual block (before the sum with the resiual, similar to WaveNet) to the end of the network, where all the connections are summed. The sum then passes another activation function. If the output of a residual block has a feature dimension different from the feature dimension of the last residual block, the respective skip connection will use a 1x1 convolution for downsampling the feature dimension. This procedure is similar to the way resiudal connections around each residual block are handled. Skip connections usually help to train deeper netowrks efficiently, so they should be used unless you experience a drop in performance.
