import unittest
import torch
import pytorch_tcn
from pytorch_tcn import TCN


class TestTCN(unittest.TestCase):

    def __init__(self, methodName: str = "runTest") -> None:
        super().__init__(methodName)

        self.num_inputs = 20
        self.num_channels = [
            32, 64, 64, 128,
            32, 64, 64, 128,
            ]

        return

    def test_tcn(self, **kwargs):
        tcn = TCN(
            num_inputs = self.num_inputs,
            num_channels = self.num_channels,
            **kwargs,
        )

        time_steps = 196
        x = torch.randn( 10, self.num_inputs, time_steps )
        y = tcn(x)
        
        self.assertEqual( y.shape, (10, self.num_channels[-1], time_steps) )
        return
    
    def test_kernel_size(self):
        self.test_tcn( kernel_size = 7 )
        return
    
    def test_dilations(self):
        # dilations list len != len(num_channels)
        with self.assertRaises(ValueError):
            self.test_tcn( dilations = [1, 2, 3, 4] )

        # dilations list len == len(num_channels)
        self.test_tcn( dilations = [1, 2, 3, 4, 1, 2, 3, 4] )
        return
    
    def test_dropout(self):
        self.test_tcn( dropout = 0.5 )
        return
    
    def test_causal(self):
        self.test_tcn( causal = True )
        return
    
    def test_non_causal(self):
        self.test_tcn( causal = False )
        return
    
    def test_norms(self):
        available_norms = TCN(10,[10]).allowed_norm_values
        for norm in available_norms:
            print( 'Testing norm:', norm )
            self.test_tcn( use_norm = norm )

        with self.assertRaises(ValueError):
            self.test_tcn( use_norm = 'invalid' )
        return
    
    def test_activations(self):
        available_activations = pytorch_tcn.tcn.activation_fn.keys()
        for activation in available_activations:
            self.test_tcn( activation = activation )

        with self.assertRaises(ValueError):
            self.test_tcn( activation = 'invalid' )
        return
    
    def test_kernel_initializers(self):
        available_initializers = pytorch_tcn.tcn.kernel_init_fn.keys()
        for initializer in available_initializers:
            self.test_tcn( kernel_initializer = initializer )

        with self.assertRaises(ValueError):
            self.test_tcn( kernel_initializer = 'invalid' )
        return

    def test_skip_connections(self):
        self.test_tcn( use_skip_connections = True )
        self.test_tcn( use_skip_connections = False )
        return
    
    def test_input_shape(self):
        self.test_tcn( input_shape = 'NCL' )

        # Test NLC
        tcn = TCN(
            num_inputs = self.num_inputs,
            num_channels = self.num_channels,
            input_shape = 'NLC'
        )

        time_steps = 196
        x = torch.randn( 10, time_steps, self.num_inputs, )
        y = tcn(x)
        
        self.assertEqual( y.shape, (10, time_steps, self.num_channels[-1]) )

        with self.assertRaises(ValueError):
            self.test_tcn( input_shape = 'invalid' )
        return

    


        

if __name__ == '__main__':
    unittest.main()