
import os
import tempfile
import torch
import torch.nn as nn
import unittest

from pytorch_tcn.conv import TemporalConv1d, TemporalConvTranspose1d
from pytorch_tcn.buffer import BufferIO


class ConvModel(nn.Module):
            def __init__(
                    self,
                    layer_1,
                    layer_2,
                    ):
                super(ConvModel, self).__init__()
                self.conv1 = layer_1
                self.conv2 = layer_2
                return

            def forward(self, x, in_buffers):
                buffer_io = BufferIO( in_buffers = in_buffers )
                with torch.no_grad():
                    x = self.conv1(
                        x=x,
                        inference=True,
                        buffer_io=buffer_io,
                    )
                    x = self.conv2(
                        x=x,
                        inference=True,
                        buffer_io=buffer_io,
                    )
                out_buffers = buffer_io.out_buffers
                return x, out_buffers

class ConvolutionTest(unittest.TestCase):
        
    def test_conv1d_streaming_with_internal_buffer(self):
        # Example input tensor shape: (batch_size, channels, time_steps)
        batch_size = 1
        in_channels = 3
        time_steps = 32
        input_tensor = torch.randn(batch_size, in_channels, time_steps)

        # Define the convolutional layer
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        conv_layer = TemporalConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = 1,
            dilation = 1,
            causal=True,
            )
        
        # Test the buffer shape
        self.assertEqual(
            conv_layer.buffer.shape,
            (1, in_channels, kernel_size - 1),
            )

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            # Test the output tensor shape
            self.assertEqual(
                output_tensor.shape,
                (batch_size, out_channels, time_steps),
                )

            # Compare with running inference in a loop, with internal buffers
            conv_layer.reset_buffer()
            for t in range(0,input_tensor.shape[2]-1):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice = conv_layer(
                    input_slice,
                    inference=True,
                    )
                reference_output_slice = output_tensor[:,:,t:t+1]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))

        return



    def test_conv1d_streaming_with_external_buffer(self):
        # Example input tensor shape: (batch_size, channels, time_steps)
        batch_size = 1
        in_channels = 3
        time_steps = 32
        input_tensor = torch.randn(batch_size, in_channels, time_steps)

        # Define the convolutional layer
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        conv_layer = TemporalConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = 1,
            dilation = 1,
            causal=True,
            )
        
        # Test the buffer shape
        self.assertEqual(
            conv_layer.buffer.shape,
            (1, in_channels, kernel_size - 1),
            )

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            # Test the output tensor shape
            self.assertEqual(
                output_tensor.shape,
                (batch_size, out_channels, time_steps),
                )

            # Define the BufferIO object
            in_buffer = torch.zeros(1, in_channels, kernel_size - 1)
            buffer_io = BufferIO( in_buffers = [in_buffer] )

            # Compare with running inference in a loop with external buffers
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice = conv_layer(
                    input_slice,
                    inference=True,
                    buffer_io=buffer_io,
                    )
                reference_output_slice = output_tensor[:,:,t:t+1]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))

                buffer_io.step()
        return

    def test_conv1d_streaming_with_onnx(self):
        import onnxruntime as ort

        # Example input tensor shape: (batch_size, channels, time_steps)
        batch_size = 1
        in_channels = 3
        time_steps = 32
        input_tensor = torch.randn(batch_size, in_channels, time_steps)

        # Define the convolutional layer
        hidden_channels = 4
        out_channels = 2  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        conv_layer_1 = TemporalConv1d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = kernel_size,
            stride = 1,
            dilation = 1,
            causal=True,
            )
        conv_layer_2 = TemporalConv1d(
            in_channels = hidden_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = 1,
            dilation = 1,
            causal=True,
            )

        # Define model
        model = ConvModel(
            layer_1=conv_layer_1,
            layer_2=conv_layer_2,
            )
        
        # Define the BufferIO object
        in_buffer_1 = torch.zeros(1, in_channels, kernel_size - 1)
        in_buffer_2 = torch.zeros(1, hidden_channels, kernel_size - 1)
        in_buffers = [in_buffer_1, in_buffer_2]

        # Reference output (full tensor)
        out_tensor, _ = model(input_tensor, in_buffers)

        # Test the streaming inference with ONNX
        input_slice = input_tensor[:, :, 0:1]

        onnx_stream = []
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_model_name = os.path.join(temp_dir, "test_conv_model.onnx")
            torch.onnx.export(
                model=model,
                args=(input_slice, in_buffers),
                f=onnx_model_name, 
                input_names=['in_x', 'in_buffer_1', 'in_buffer_2'],
                output_names=['out_x', 'out_buffer_1', 'out_buffer_2'],
                opset_version=9,
                export_params=True,
                )

            # ONNX streaming output
            ort_session = ort.InferenceSession(onnx_model_name)
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[..., t:t+1]
                
                # Reference output (streamed)
                out_tensor_stream, out_buffers_stream = model(
                    input_slice,
                    in_buffers,
                    )
                onnx_outputs = ort_session.run(
                    None,
                    {
                        'in_x': input_slice.numpy(),
                        'in_buffer_1': in_buffers[0].numpy(),
                        'in_buffer_2': in_buffers[1].numpy(),
                        }
                    )
                onnx_output_slice = torch.tensor(onnx_outputs[0])
                onnx_out_buffers = [ torch.tensor(b) for b in onnx_outputs[1:] ]
                onnx_stream.append(onnx_output_slice)

                # Test that onnx matches reference (streamed inference)
                #   - model output:
                assert(torch.allclose(out_tensor_stream, onnx_output_slice, atol=1e-5))
                #   - buffer output:
                for out_b, onnx_out_b in zip(out_buffers_stream, onnx_out_buffers):
                    assert(torch.allclose(out_b, onnx_out_b, atol=1e-5))

                # Move buffers
                in_buffers = onnx_out_buffers

        # Test that onnx matches reference (full inference)
        assert(torch.allclose(out_tensor, torch.cat(onnx_stream, dim=2), atol=1e-5))
            
        return
        

    def test_noncausal_convolution(self):
        # Example input tensor shape: (batch_size, channels, time_steps)
        batch_size = 1
        in_channels = 3
        time_steps = 32
        input_tensor = torch.randn(batch_size, in_channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        kernel_size = 3  # Size of the convolutional kernel
        stride = 1  # Stride value for the convolution
        dilation = 1  # Dilation value for the convolution
        conv_layer = TemporalConv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            causal=False,
            )

        # Ensure that the results are the same as using a standard Conv1d layer
        standard_conv_layer = nn.Conv1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            dilation = dilation,
            padding = ((kernel_size - 1) * dilation)//2,
            )

        # Copy the weights and biases from the standard layer to the TCN layer
        conv_layer.weight = standard_conv_layer.weight
        conv_layer.bias = standard_conv_layer.bias

        with torch.no_grad():            
            # Apply the convolutional layer to the input tensor
            output_tensor = conv_layer(input_tensor)

            reference_output_tensor = standard_conv_layer(input_tensor)

            assert(torch.allclose(output_tensor, reference_output_tensor, atol=1e-5))


    def test_convtranspose1d_streaming_with_internal_buffers(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)  # Example input tensor shape: (batch_size, channels, time_steps)

        # Define your convolutional layer
        in_channels = 3  # Number of input channels
        out_channels = 16  # Number of output channels
        stride = 4  # Stride value for the convolution
        kernel_size = 2 * stride

        conv_t = TemporalConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            causal=True,
            )

        with torch.no_grad():

            # Apply the convolutional layer to the input tensor
            output_tensor = conv_t(input_tensor)

            # Compare with running inference in a loop, with internal buffers
            conv_t.reset_buffer()
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[:, :, t:t+1]
                
                output_slice = conv_t(input_slice, inference=True)
                reference_output_slice = output_tensor[:,:,t*stride:(t+1)*stride]

                assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))

    def test_convtranspose1d_streaming_with_external_buffers(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        out_channels = 16
        kernel_size = 8
        stride = 4
        conv_t = TemporalConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            causal=True,
            )

        with torch.no_grad():
                
                # Apply the convolutional layer to the input tensor
                output_tensor = conv_t(input_tensor)
    
                in_buffer = torch.zeros(1, in_channels, conv_t.buffer_size)
                buffer_io = BufferIO( in_buffers = [in_buffer] )
    
                # Compare with running inference in a loop with external buffers
                for t in range(0,input_tensor.shape[2]):
                    input_slice = input_tensor[:, :, t:t+1]
                    
                    output_slice = conv_t(
                        input_slice,
                        inference=True,
                        buffer_io=buffer_io,
                        )
                    reference_output_slice = output_tensor[:,:,t*stride:(t+1)*stride]
    
                    assert(torch.allclose(output_slice, reference_output_slice, atol=1e-5))
    
                    buffer_io.step()

    def test_convtranspose1d_streaming_with_onnx(self):
        import onnxruntime as ort

        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        hidden_channels = 4
        out_channels = 16
        kernel_size = 8
        stride = 4
        conv_layer_1 = TemporalConvTranspose1d(
            in_channels = in_channels,
            out_channels = hidden_channels,
            kernel_size = kernel_size,
            stride = stride,
            causal=True,
            )
        conv_layer_2 = TemporalConvTranspose1d(
            in_channels = hidden_channels,
            out_channels = out_channels,
            kernel_size = 8,
            stride = 4,
            causal=True,
            )
                
        # Define model
        model = ConvModel(
            layer_1=conv_layer_1,
            layer_2=conv_layer_2,
            )
        
        # Define the BufferIO object
        in_buffer_1 = torch.zeros(1, in_channels, conv_layer_1.buffer_size)
        in_buffer_2 = torch.zeros(1, hidden_channels, conv_layer_2.buffer_size)
        in_buffers = [in_buffer_1, in_buffer_2]

        # Reference output (full tensor)
        out_tensor, _ = model(input_tensor, in_buffers)

        # Test the streaming inference with ONNX
        input_slice = input_tensor[:, :, 0:1]

        onnx_stream = []
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_model_name = os.path.join(temp_dir, "test_conv_model.onnx")
            torch.onnx.export(
                model=model,
                args=(input_slice, in_buffers),
                f=onnx_model_name, 
                input_names=['in_x', 'in_buffer_1', 'in_buffer_2'],
                output_names=['out_x', 'out_buffer_1', 'out_buffer_2'],
                opset_version=9,
                export_params=True,
                )

            # ONNX streaming output
            ort_session = ort.InferenceSession(onnx_model_name)
            for t in range(0,input_tensor.shape[2]):
                input_slice = input_tensor[..., t:t+1]
                
                # Reference output (streamed)
                out_tensor_stream, out_buffers_stream = model(
                    input_slice,
                    in_buffers,
                    )
                onnx_outputs = ort_session.run(
                    None,
                    {
                        'in_x': input_slice.numpy(),
                        'in_buffer_1': in_buffers[0].numpy(),
                        'in_buffer_2': in_buffers[1].numpy(),
                        }
                    )
                onnx_output_slice = torch.tensor(onnx_outputs[0])
                onnx_out_buffers = [ torch.tensor(b) for b in onnx_outputs[1:] ]
                onnx_stream.append(onnx_output_slice)

                # Test that onnx matches reference (streamed inference)
                #   - model output:
                assert(torch.allclose(out_tensor_stream, onnx_output_slice, atol=1e-5))
                #   - buffer output:
                for out_b, onnx_out_b in zip(out_buffers_stream, onnx_out_buffers):
                    assert(torch.allclose(out_b, onnx_out_b, atol=1e-5))

                # Move buffers
                in_buffers = onnx_out_buffers


        # Test that onnx matches reference (full inference)
        assert(torch.allclose(out_tensor, torch.cat(onnx_stream, dim=2), atol=1e-5))
            
        return

    def test_transpose_noncausal_convolution(self):
        # Define your input tensor
        input_tensor = torch.randn(1, 3, 32)

        # Define your convolutional layer
        in_channels = 3
        out_channels = 16
        kernel_size = 4
        stride = 2
        conv_layer = TemporalConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            causal=False,
            )

        # Ensure that the results are the same as using a standard ConvTranspose1d layer
        standard_conv_layer = nn.ConvTranspose1d(
            in_channels = in_channels,
            out_channels = out_channels,
            kernel_size = kernel_size,
            stride = stride,
            padding = (kernel_size-stride)//2,
            )

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