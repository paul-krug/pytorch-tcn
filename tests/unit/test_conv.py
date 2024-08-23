import torch
import unittest

import torch.nn as nn
from pytorch_tcn.conv import TemporalConv1d, TemporalConvTranspose1d
import tempfile
import os
import onnx
import onnxruntime as ort

class ConvolutionTest(unittest.TestCase):
    def test_causal_internal_buffer_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride value for the convolution
        dilation = 1  # Dilation value for the convolution
        conv_layer = TemporalConv1d(in_channels, out_channels, kernel_size, stride, ((kernel_size - 1) * dilation) // 2, dilation, causal=True)

        self.assertEqual(conv_layer.buffer.shape, (1, in_channels, kernel_size - 1))  # Example assertion for buffer shape

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            # Perform your assertions or checks here
            self.assertEqual(output_tensor.shape, (1, 16, 32))  # Example assertion for output tensor shape

            # Compare with running inference in a loop, with internal buffers
            conv_layer.reset_buffer()
            for t in range(0,input_tensor.shape[2]-1):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice = conv_layer.forward(input_slice, inference=True)
                reference_output_slice = output_tensor[:,:,t:t+1]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))



    def test_causal_external_buffer_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride value for the convolution
        dilation = 1  # Dilation value for the convolution
        conv_layer = TemporalConv1d(in_channels, out_channels, kernel_size, stride, ((kernel_size - 1) * dilation) // 2, dilation, causal=True)

        self.assertEqual(conv_layer.buffer.shape, (1, in_channels, kernel_size - 1))  # Example assertion for buffer shape

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            # Perform your assertions or checks here
            self.assertEqual(output_tensor.shape, (1, 16, 32))  # Example assertion for output tensor shape
            in_buffer = torch.zeros(1, in_channels, kernel_size - 1)

            # Compare with running inference in a loop with external buffers
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice, out_buffer = conv_layer.inference(input_slice, in_buffer=in_buffer)
                reference_output_slice = output_tensor[:,:,t:t+1]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))

                in_buffer = out_buffer

    def test_causal_onnx_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride value for the convolution
        dilation = 1  # Dilation value for the convolution
        conv_layer = TemporalConv1d(in_channels, out_channels, kernel_size, stride, ((kernel_size - 1) * dilation) // 2, dilation, causal=True)

        class ConvModel(nn.Module):
            def __init__(self, conv_layer):
                super(ConvModel, self).__init__()
                self.conv_layer = conv_layer

            def forward(self, x, in_buffer):
                with torch.no_grad():
                    return self.conv_layer.inference(x, in_buffer)

        model = ConvModel(conv_layer)
        in_buffer = torch.zeros(1, in_channels, kernel_size - 1)
        input_slice = input_tensor[:, :, 0:1]

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_model_name = os.path.join(temp_dir, "test_conv_model.onnx")
            torch.onnx.export(model, (input_slice, in_buffer), onnx_model_name, 
                              input_names=['in_x', 'in_buffer'], output_names=['out_x', 'out_buffer'], 
                              opset_version=9, export_params=True)


            ort_session = ort.InferenceSession(onnx_model_name)

            for t in range(0,input_tensor.shape[2]-1):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice, out_buffer = model(input_slice, in_buffer)
                onnx_outputs = ort_session.run(None, {'in_x': input_slice.numpy(), 'in_buffer': in_buffer.numpy()})

                assert(torch.allclose(output_slice, torch.tensor(onnx_outputs[0]), atol=1e-5))
                assert(torch.allclose(out_buffer, torch.tensor(onnx_outputs[1]), atol=1e-5))
        

    def test_noncausal_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride value for the convolution
        dilation = 1  # Dilation value for the convolution
        conv_layer = TemporalConv1d(in_channels, out_channels, kernel_size, stride, dilation, causal=False)

        # Ensure that the results are the same as using a standard Conv1d layer
        standard_conv_layer = nn.Conv1d(in_channels, out_channels, kernel_size, stride=stride, dilation=dilation, padding=(kernel_size-1)//2)

        # Copy the weights and biases from the standard layer to the TCN layer
        conv_layer.weight = standard_conv_layer.weight
        conv_layer.bias = standard_conv_layer.bias

        with torch.no_grad():            
            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            reference_output_tensor = standard_conv_layer(input_tensor)

            assert(torch.allclose(output_tensor, reference_output_tensor, atol=1e-5))


    def test_transpose_causal_internal_buffer_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 8  # Size of the convolutional kernel
        stride = 4  # Stride value for the convolution
        conv_layer = TemporalConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-stride)//2, causal=True)

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor, inference=True)

            # Compare with running inference in a loop, with internal buffers
            conv_layer.reset_buffer()
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice = conv_layer(input_slice, inference=True)
                reference_output_slice = output_tensor[:,:,t*stride:(t+1)*stride]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))

    def test_transpose_causal_external_buffer_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        out_channels = 16
        kernel_size = 8
        stride = 4
        conv_layer = TemporalConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-stride)//2, causal=True)

        with torch.no_grad():
                
                # Apply the convolutional layer to the input tensor
                output_tensor = conv_layer(input_tensor, inference=True)
    
                in_buffer = torch.zeros(1, in_channels, conv_layer.buffer_size)
    
                # Compare with running inference in a loop with external buffers
                for t in range(0,input_tensor.shape[2]):
                    input_slice = input_tensor[:, :, t:t+1]
                    
                    output_slice, out_buffer = conv_layer.inference(input_slice, in_buffer=in_buffer)
                    reference_output_slice = output_tensor[:,:,t*stride:(t+1)*stride]
    
                    assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))
    
                    in_buffer = out_buffer
          
    def test_transpose_causal_onnx_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        out_channels = 16
        kernel_size = 8
        stride = 4
        conv_layer = TemporalConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-stride)//2, causal=True)

        class ConvModel(nn.Module):
            def __init__(self, conv_layer):
                super(ConvModel, self).__init__()
                self.conv_layer = conv_layer

            def forward(self, x, in_buffer):
                with torch.no_grad():
                    return self.conv_layer.inference(x, in_buffer)
                
        model = ConvModel(conv_layer)
        in_buffer = torch.zeros(1, in_channels, conv_layer.buffer_size)
        input_slice = input_tensor[:, :, 0:1]

        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_model_name = os.path.join(temp_dir, "test_conv_model.onnx")
            torch.onnx.export(model, (input_slice, in_buffer), onnx_model_name, 
                              input_names=['in_x', 'in_buffer'], output_names=['out_x', 'out_buffer'], 
                              opset_version=9, export_params=True)


            ort_session = ort.InferenceSession(onnx_model_name)

            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice, out_buffer = model(input_slice, in_buffer)
                onnx_outputs = ort_session.run(None, {'in_x': input_slice.numpy(), 'in_buffer': in_buffer.numpy()})

                assert(torch.allclose(output_slice, torch.tensor(onnx_outputs[0]), atol=1e-5))
                assert(torch.allclose(out_buffer, torch.tensor(onnx_outputs[1]), atol=1e-5))

    def test_transpose_noncausal_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        out_channels = 16
        kernel_size = 4
        stride = 2
        conv_layer = TemporalConvTranspose1d(in_channels, out_channels, kernel_size, stride, padding=(kernel_size-stride)//2, causal=False)

        # Ensure that the results are the same as using a standard ConvTranspose1d layer
        standard_conv_layer = nn.ConvTranspose1d(in_channels, out_channels, kernel_size, stride=stride, padding=(kernel_size-stride)//2)

        # Copy the weights and biases from the standard layer to the TCN layer
        conv_layer.weight = standard_conv_layer.weight
        conv_layer.bias = standard_conv_layer.bias

        with torch.no_grad():            
            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            reference_output_tensor = standard_conv_layer(input_tensor)

            assert(torch.allclose(output_tensor, reference_output_tensor, atol=1e-5))


if __name__ == '__main__':
    unittest.main()