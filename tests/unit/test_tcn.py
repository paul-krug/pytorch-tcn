import unittest
import numpy as np
import torch
import pytorch_tcn
from pytorch_tcn import TCN

import inspect
import itertools

def generate_combinations(test_args):
    combinations = []

    for x in test_args:
        kwargs = x['kwargs']
        # kwargs contains a list of values for each key
        # Get all possibe combinations of the values:
        keys = kwargs.keys()
        values = kwargs.values()

        for value_combination in itertools.product(*values):
            combination_dict = dict(zip(keys, value_combination))
            combinations.append(
                dict(
                    kwargs = combination_dict,
                    expected_error = x['expected_error'],
                    )
                )

    return combinations

def get_optional_parameters(cls, method_name):
    sig = inspect.signature(getattr(cls, method_name))
    return [name for name, param in sig.parameters.items() if param.default != inspect.Parameter.empty]

class TestTCN(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.available_activations = pytorch_tcn.tcn.activation_fn.keys()
        self.available_norms = TCN(10,[10]).allowed_norm_values
        self.available_initializers = pytorch_tcn.tcn.kernel_init_fn.keys()

        self.num_inputs = 20
        self.num_channels = [
            32, 64, 64, 128,
            32, 64, 64, 128,
            ]
        
        self.batch_size = 10
        self.time_steps = 196
        
        self.test_args = [
            dict(
                kwargs = dict(
                    kernel_size = [3],
                    causal = [True, False],
                    lookahead = [ 1, 4 ],
                    output_projection = [None, 128],
                    output_activation = [None, 'relu'],
                    ),
                expected_error = None,
            ),
            # Test different kernel sizes
            dict(
                kwargs = dict( kernel_size = [3, 5, 7] ),
                expected_error = None,
            ),
            # Test valid dilation rates
            dict(
                kwargs = dict(
                    dilations = [
                        [1, 2, 3, 4, 1, 2, 3, 4],
                        [[1, 2], [2, 4], [3, 6], [4, 8], [1, 2], [2, 4], [3, 6], [4, 8]],
                        [[1, 2], [2, 4], [3, 6], [4, 8], 1, 2, 3, 4],
                        [1, 2, 3, 4, [1, 2], [2, 4], [3, 6], [4, 8]],
                        None,
                    ],
                ),
                expected_error = None,
            ),
            # Test invalid dilation rates
            dict(
                kwargs = dict( dilations = [ [1, 2, 3] ] ),
                expected_error = ValueError,
            ),
            # Test valid dilation reset values
            dict(
                kwargs = dict( dilation_reset = [4, None] ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( dropout = [0.0, 0.5], ),
                expected_error = None,
            ),
            dict(
                kwargs = dict(
                    causal = [True, False],
                    lookahead = [ 0, 1, 4 ],
                    ),
                expected_error = None,
            ),
            dict(
                kwargs = dict(
                    use_norm = self.available_norms,
                    use_gate = [True, False],
                    ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( use_norm = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict(
                    activation = self.available_activations,
                    use_gate = [True, False],
                    ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( activation = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict( kernel_initializer = self.available_initializers ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( kernel_initializer = [ 'invalid' ] ),
                expected_error = ValueError,
            ),
            dict(
                kwargs = dict( use_skip_connections = [True, False], ),
                expected_error = None,
            ),
            dict(
                kwargs = dict( input_shape = ['NCL', 'NLC'] ),
                expected_error = None,
            ),
            # Test valid embedding shapes
            dict(
                kwargs = dict(
                    embedding_shapes = [
                        [ (10,), ],
                        [ (10,), (128,), ],
                        [ (1, None,), ],
                        [ (32,), (12, None,), ],
                        None,
                    ],
                    embedding_mode = [ 'concat', 'add' ],
                    use_gate = [True, False],
                ),
                expected_error = None,
            ),
            # Test invalid embedding shapes
            dict(
                kwargs = dict(
                    embedding_shapes = [
                        [ (10, 32, 64), ],
                        [ (10, self.time_steps + 32),],
                    ],
                    embedding_mode = [ 'concat', 'add' ],
                    use_gate = [True, False],
                ),
                expected_error = ValueError,
            ),
            # Test different values for force_residual_conv
            dict(
                kwargs = dict( force_residual_conv = [True, False] ),
                expected_error = None,
            ),
        ]

        self.combinations = generate_combinations(self.test_args)

        return

    def test_tcn(self, **kwargs):

        tcn = TCN(
            num_inputs = self.num_inputs,
            num_channels = self.num_channels,
            **kwargs,
        )
        first_layer = next( iter( tcn.children() ) )
        #print( 'tcn first conv layer buffer shape: ', first_layer[0].conv1.buffer.shape)
        #stopü

        x = torch.randn(
            self.batch_size,
            self.num_inputs,
            self.time_steps,
            )
        expected_shape = (
            self.batch_size,
            self.num_channels[-1],
            self.time_steps,
            )
        x_inference = torch.randn(
            1,
            self.num_inputs,
            self.time_steps,
            )
        expected_shape_inference = (
            1,
            self.num_channels[-1],
            self.time_steps - tcn.lookahead,
            )

        time_dimension = -1
        # check if 'input_shape' is 'NCL'
        if 'input_shape' in kwargs and kwargs['input_shape'] == 'NLC':
            time_dimension = 1
            x = x.permute(0, 2, 1)
            expected_shape = (
                self.batch_size,
                self.time_steps,
                self.num_channels[-1],
                )
            x_inference = x_inference.permute(0, 2, 1)
            expected_shape_inference = (
                1,
                self.time_steps - tcn.lookahead,
                self.num_channels[-1],
                )

        if 'embedding_shapes' in kwargs and kwargs['embedding_shapes'] is not None:
            embeddings = []
            embeddings_inference = []
            for shape in kwargs[ 'embedding_shapes' ]:
                #shape_inference = shape
                if None in shape:
                    # replace None with self.time_steps
                    shape = list(shape)
                    shape[ shape.index(None) ] = self.time_steps# - tcn.lookahead
                    shape = tuple(shape)

                    #shape_inference = list(shape_inference)
                    #shape_inference[ shape_inference.index(None) ] = 1
                    #shape_inference = tuple(shape_inference)


                embeddings.append(
                    torch.randn(
                        self.batch_size,
                        *shape,
                        )
                    )
                embeddings_inference.append(
                    torch.randn(
                        1,
                        *shape,
                        )
                    )
        else:
            embeddings = None
            embeddings_inference = None

        y = tcn(x, embeddings = embeddings)
        
        self.assertEqual( y.shape, expected_shape )

        # Testing the streaming inference mode for causal models
        if tcn.causal:
            tcn.eval()

            tcn.reset_buffers()
            #print( 'tcn first conv layer buffer shape, CAUSAL Inf 1: ', first_layer[0].conv1.buffer.shape)
            with torch.no_grad():
                #y_forward = tcn(x_inference, embeddings = embeddings)
                #print( 'x_inference shape: ', x_inference.shape)
                #print( 'expected_shape_inference: ', expected_shape_inference)
                y_inference = tcn.inference(
                    x_inference,
                    embeddings = embeddings_inference,
                    )
                #print( 'y_inference shape: ', y_inference.shape)
                #stop
                
            self.assertEqual( y_inference.shape, expected_shape_inference )

            # piecewise inference:
            tcn.reset_buffers()
            #print( 'tcn first conv layer buffer shape, CAUSAL Inf 2: ', first_layer[0].conv1.buffer.shape)
            y_inference_frames = []
            #index:index+hop_length
            block_size = 1 + tcn.lookahead
            for i in range( 0, self.time_steps-tcn.lookahead ):
                # pick frame from time dimension
                frame = x_inference.narrow(
                    dim = time_dimension,
                    start = i,
                    length = block_size,
                    )
                #print( 'frame shape: ', frame.shape)
                #stop

                if embeddings_inference is None:
                    embeddings_frame = None
                else:
                    embeddings_frame = []
                    for emb in embeddings_inference:
                        if len(emb.shape) == 2:
                            embeddings_frame.append( emb )
                        elif len(emb.shape) == 3:
                            embeddings_frame.append(
                                emb.narrow(
                                    dim = time_dimension,
                                    start = i,
                                    length = block_size,
                                    )
                                )
                        else:
                            raise ValueError('Invalid shape for embeddings')

                with torch.no_grad():
                    #print( 'frame shape: ', frame.shape)
                    y_inference_frames.append(
                        tcn.inference(
                            frame,
                            embeddings = embeddings_frame,
                            )
                    )
            y_inference_frames = torch.cat( y_inference_frames, dim = time_dimension )
            self.assertEqual( y_inference_frames.shape, expected_shape_inference )
            #stop

            ## piecewise inference without buffer
            #tcn.reset_buffers()
            #y_forward_frames = []
            #for i in range(0, self.time_steps):
            #    # pick frame from time dimension
            #    frame = x_inference.select(time_dimension, i).unsqueeze(time_dimension)
            #    y_forward_frames.append(
            #        tcn( frame, embeddings = embeddings )
            #    )
            #y_forward_frames = torch.cat( y_forward_frames, dim = -1 )
            #self.assertEqual( y_forward_frames.shape, expected_shape_inference )

            # Verify Output
            #y_forward_frames = y_forward_frames.detach().cpu().numpy().squeeze()[0]
            #y_inference_frames = y_inference_frames.detach().cpu().numpy().squeeze()[0]
            #y_inference = y_inference.detach().cpu().numpy().squeeze()[0]
            #y_forward = y_forward.detach().cpu().numpy().squeeze()[0]
            #print( 'y shape: ', y_forward.shape )
            #print( 'y_inference shape: ', y_inference.shape )
            #print( 'np med1: ', np.median( abs(y_inference_frames - y_forward) ) )
            #print( 'np med2: ', np.median( abs(y_forward_frames - y_forward) ) )
            #import matplotlib.pyplot as plt
            #plt.plot(y_inference_frames)
            #plt.plot(y_forward)
            #plt.show()

        return
    
    def test_tcn_grid_search(self):

        # Test all valid combinations
        for test_dict in self.combinations:
            kwargs = test_dict['kwargs']
            print( 'Testing kwargs: ', kwargs )
            if test_dict['expected_error'] is None:
                self.test_tcn( **kwargs )
            else:
                with self.assertRaises(test_dict['expected_error']):
                    self.test_tcn( **kwargs )

        return
    
    def test_if_all_args_get_tested(self):
        # Get kwargs of TCN class
        tcn_optional_parameters = get_optional_parameters(TCN, '__init__')
        print( 'Test if allvariable names of tcn get tested: ', tcn_optional_parameters )
        found_params = { x: False for x in tcn_optional_parameters }

        # check that all tcn_kwargs are there as keys in test_args
        for kwarg in tcn_optional_parameters:
            for x in self.test_args:
                kwargs = x['kwargs']
                if kwarg in kwargs.keys():
                    found_params[kwarg] = True
                    break
        print( 'Params that get tested: ', found_params )
        all_params_found = all( found_params.values() )
        self.assertTrue(all_params_found)
        return
   

if __name__ == '__main__':
    unittest.main()