import os
import tempfile
import torch
import torch.nn as nn
import unittest

from pytorch_tcn.buffer import BufferIO
# Replace the import below with the appropriate path to your TemporalGRU implementation.
from pytorch_tcn.rnn import TemporalGRU


class RNNModel(nn.Module):
    """
    A simple model composed of two TemporalGRU layers. It expects an external buffer list,
    which is wrapped in a BufferIO object. This is analogous to the ConvModel for the temporal convs.
    """
    def __init__(self, layer_1, layer_2):
        super(RNNModel, self).__init__()
        self.rnn1 = layer_1
        self.rnn2 = layer_2

    def forward(self, x, in_buffers):
        buffer_io = BufferIO(in_buffers=in_buffers)
        with torch.no_grad():
            x = self.rnn1(x, inference=True, buffer_io=buffer_io)
            x = self.rnn2(x, inference=True, buffer_io=buffer_io)
        out_buffers = buffer_io.out_buffers
        return x, out_buffers


class TemporalGRUTest(unittest.TestCase):

    def test_rnn_streaming_with_internal_buffer(self):
        # Use batch_first=True so that x has shape (batch, seq, features)
        batch_size = 1
        seq_len = 32
        input_size = 3
        hidden_size = 16
        num_layers = 1

        input_tensor = torch.randn(batch_size, seq_len, input_size)

        # Define the TemporalGRU layer.
        rnn_layer = TemporalGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        # For testing internal buffering, we initialize the buffer explicitly.
        initial_buffer = torch.zeros(num_layers, batch_size, hidden_size)
        rnn_layer.buffer = initial_buffer.clone()

        # Create a standard RNN (with identical weights) to compute a full-sequence reference.
        standard_rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        standard_rnn.load_state_dict(rnn_layer.state_dict())
        full_output, _ = standard_rnn(input_tensor)

        # Now run streaming inference one time-step at a time.
        rnn_layer.reset_buffer()
        rnn_layer.buffer = initial_buffer.clone()
        streamed_outputs = []
        for t in range(seq_len):
            input_slice = input_tensor[:, t:t+1, :]  # shape: (batch, 1, input_size)
            output_slice = rnn_layer(input_slice, inference=True)
            streamed_outputs.append(output_slice)
        # Concatenate the per-time-step outputs along the time dimension.
        streamed_output = torch.cat(streamed_outputs, dim=1)

        self.assertEqual(streamed_output.shape, full_output.shape)
        self.assertTrue(torch.allclose(streamed_output, full_output, atol=1e-5))

        # Verify that reset_buffer() clears the internal hidden state.
        rnn_layer.reset_buffer()
        self.assertIsNone(rnn_layer.buffer)

    def test_rnn_streaming_with_external_buffer(self):
        batch_size = 1
        seq_len = 32
        input_size = 3
        hidden_size = 16
        num_layers = 1

        input_tensor = torch.randn(batch_size, seq_len, input_size)

        rnn_layer = TemporalGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )

        # Build a reference RNN with the same parameters/weights.
        standard_rnn = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        standard_rnn.load_state_dict(rnn_layer.state_dict())
        full_output, _ = standard_rnn(input_tensor)

        # Create an external buffer (hidden state) with the correct shape.
        initial_buffer = torch.zeros(num_layers, batch_size, hidden_size)
        buffer_io = BufferIO(in_buffers=[initial_buffer.clone()])

        # Run streaming inference one time-step at a time using external buffers.
        rnn_layer.reset_buffer()  # Ensure internal buffer is not used.
        streamed_outputs = []
        for t in range(seq_len):
            input_slice = input_tensor[:, t:t+1, :]
            output_slice = rnn_layer(input_slice, inference=True, buffer_io=buffer_io)
            streamed_outputs.append(output_slice)
            buffer_io.step()  # Advance the buffer pointer.
        streamed_output = torch.cat(streamed_outputs, dim=1)

        self.assertEqual(streamed_output.shape, full_output.shape)
        self.assertTrue(torch.allclose(streamed_output, full_output, atol=1e-5))

    def test_rnn_deprecated_in_buffer(self):
        # Ensure that using the deprecated "in_buffer" argument raises an error.
        input_tensor = torch.randn(1, 1, 3)
        rnn_layer = TemporalGRU(input_size=3, hidden_size=4, batch_first=True)
        with self.assertRaises(ValueError):
            rnn_layer(input_tensor, inference=True, in_buffer=torch.zeros(1, 1, 4))

    def test_rnn_streaming_with_onnx(self):
        try:
            import onnxruntime as ort
        except ImportError:
            self.skipTest("onnxruntime not available")

        batch_size = 1
        seq_len = 32
        input_size = 3
        hidden_size = 16
        num_layers = 1

        # Build a two-layer model using TemporalGRU.
        rnn_layer_1 = TemporalGRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        rnn_layer_2 = TemporalGRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        model = RNNModel(
            layer_1=rnn_layer_1,
            layer_2=rnn_layer_2,
        )
        # Create external buffers for each layer.
        initial_buffer_1 = torch.zeros(num_layers, batch_size, hidden_size)
        initial_buffer_2 = torch.zeros(num_layers, batch_size, hidden_size)
        in_buffers = [initial_buffer_1.clone(), initial_buffer_2.clone()]

        # Run full inference on the entire sequence as reference.
        input_tensor = torch.randn(batch_size, seq_len, input_size)
        full_output, _ = model(input_tensor, in_buffers)

        # Export the model to ONNX using a single time-step.
        input_slice = input_tensor[:, :1, :]
        with tempfile.TemporaryDirectory() as temp_dir:
            onnx_model_name = os.path.join(temp_dir, "test_rnn_model.onnx")
            torch.onnx.export(
                model=model,
                args=(input_slice, in_buffers),
                f=onnx_model_name,
                input_names=['in_x', 'in_buffer_1', 'in_buffer_2'],
                output_names=['out_x', 'out_buffer_1', 'out_buffer_2'],
                opset_version=9,
                export_params=True,
            )

            ort_session = ort.InferenceSession(onnx_model_name)
            onnx_stream = []
            # Reset the buffers for streaming inference.
            for t in range(seq_len):
                input_slice = input_tensor[:, t:t+1, :]
                # Run the model (reference streaming inference).
                ref_output, out_buffers = model(input_slice, in_buffers)

                # Run the ONNX model.
                ort_inputs = {
                    'in_x': input_slice.numpy(),
                    'in_buffer_1': in_buffers[0].numpy(),
                    'in_buffer_2': in_buffers[1].numpy(),
                }
                onnx_outputs = ort_session.run(None, ort_inputs)
                onnx_output_slice = torch.tensor(onnx_outputs[0])
                onnx_out_buffers = [torch.tensor(b) for b in onnx_outputs[1:]]

                onnx_stream.append(onnx_output_slice)
                # Compare the model output and buffers with the ONNX outputs.
                self.assertTrue(torch.allclose(ref_output, onnx_output_slice, atol=1e-5))
                for ref_buf, onnx_buf in zip(out_buffers, onnx_out_buffers):
                    self.assertTrue(torch.allclose(ref_buf, onnx_buf, atol=1e-5))
                # Update the buffers for the next step.
                in_buffers = onnx_out_buffers

            streamed_output = torch.cat(onnx_stream, dim=1)
            self.assertTrue(torch.allclose(full_output, streamed_output, atol=1e-5))


if __name__ == '__main__':
    unittest.main()