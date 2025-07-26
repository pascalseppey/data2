import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv

class GNNEncoder(torch.nn.Module):
    """Simple GAT-based encoder."""
    def __init__(self, in_channels: int, hidden_channels: int = 64, out_channels: int = 64, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=0.6)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=0.6)

    def forward(self, x, edge_index):
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return x

class LinkPredictor(torch.nn.Module):
    """Predicts the existence of a link between two nodes."""
    def __init__(self, in_channels: int):
        super().__init__()
        self.lin = torch.nn.Linear(2 * in_channels, 1)

    def forward(self, z_src, z_dst):
        x = torch.cat([z_src, z_dst], dim=-1)
        x = self.lin(x)
        return torch.sigmoid(x)

class SimpleDecoder(torch.nn.Module):
    """A minimal GRU-based decoder for Lobe 2."""
    def __init__(self, vocab_size: int, embed_dim: int = 32, hidden_dim: int = 64):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        self.gru = torch.nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.out = torch.nn.Linear(hidden_dim, vocab_size)

    def forward(self, input_ids, hidden=None):
        emb = self.embed(input_ids)
        output, hidden = self.gru(emb, hidden)
        logits = self.out(output)
        return logits, hidden
