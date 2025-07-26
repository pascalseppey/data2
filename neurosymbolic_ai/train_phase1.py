"""Train a simple GAT on the knowledge graph for link prediction (Phase 1).
This example uses PyTorch Geometric's built-in utilities and runs for a few epochs.
"""

import pickle

import torch
from torch_geometric.data import Data
from torch_geometric.utils import train_test_split_edges

from .models import GNNEncoder, LinkPredictor

GRAPH_PATH = "data/knowledge_graph.gpickle"
VECTORS_PATH = "data/lexical_vectors.pkl"
EPOCHS = 20




if __name__ == "__main__":
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)

    nodes = list(G.nodes())
    idx = {n: i for i, n in enumerate(nodes)}
    edge_index = torch.tensor([[idx[u], idx[v]] for u, v in G.edges()]).t().contiguous()

    # load 32-bit lexical vectors
    with open(VECTORS_PATH, "rb") as f:
        vectors = pickle.load(f)

    x = torch.zeros((len(nodes), 32), dtype=torch.float)
    for lemma, bits in vectors.items():
        if lemma in idx:
            x[idx[lemma]] = torch.tensor([int(b) for b in bits], dtype=torch.float)

    data = Data(x=x, edge_index=edge_index)
    data = train_test_split_edges(data)

    encoder = GNNEncoder(in_channels=32, hidden_channels=64, out_channels=64)
    predictor = LinkPredictor(in_channels=64)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(predictor.parameters()), lr=0.01)
    loss_fn = torch.nn.BCELoss()

    for epoch in range(1, EPOCHS + 1):
        encoder.train()
        predictor.train()
        optimizer.zero_grad()
        z = encoder(data.x, data.train_pos_edge_index)
        pos_out = predictor(z[data.train_pos_edge_index[0]], z[data.train_pos_edge_index[1]])
        neg_out = predictor(z[data.train_neg_edge_index[0]], z[data.train_neg_edge_index[1]])
        pos_loss = loss_fn(pos_out.squeeze(), torch.ones_like(pos_out.squeeze()))
        neg_loss = loss_fn(neg_out.squeeze(), torch.zeros_like(neg_out.squeeze()))
        loss = pos_loss + neg_loss
        loss.backward()
        optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}/{EPOCHS} Loss {loss.item():.4f}")

