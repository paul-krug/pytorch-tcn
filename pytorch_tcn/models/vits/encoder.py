
import torch
from pytorch_tcn import BaseTCN
from pytorch_tcn import TemporalConv1d as Conv1d

class Encoder( BaseTCN ):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_channels,
        kernel_size,
        dilation_rate,
        n_layers,
        embedding_dim=0,
        causal=False,
        ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels
        self.kernel_size = kernel_size
        self.dilation_rate = dilation_rate
        self.n_layers = n_layers
        self.embedding_dim = embedding_dim
        self.causal = causal

        self.pre = Conv1d(
            in_channels=in_channels,
            out_channels=hidden_channels,
            kernel_size=1,
            causal=causal
            )
        self.enc = WN(
            hidden_channels,
            kernel_size,
            dilation_rate,
            n_layers,
            embedding_dim=embedding_dim,
            causal=causal,
            )
        self.proj = Conv1d(
            in_channels=hidden_channels,
            out_channels=out_channels * 2,
            kernel_size=1,
            causal=causal
            )

    def forward(self, x, x_lengths, g=None):
        x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.pre(x) * x_mask
        x = self.enc(x, x_mask, g=g)
        stats = self.proj(x) * x_mask
        m, logs = torch.split(stats, self.out_channels, dim=1)
        z = (m + torch.randn_like(m) * torch.exp(logs)) * x_mask
        return z, m, logs, x_mask