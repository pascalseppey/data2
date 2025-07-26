"""Phase 2 training: map thought vectors to sentences with a simple decoder."""

import pickle

import torch
from torch.utils.data import DataLoader

from .models import GNNEncoder, SimpleDecoder
from .persona import PersonaManager

GRAPH_PATH = "data/knowledge_graph.gpickle"
VECTORS_PATH = "data/lexical_vectors.pkl"

# Tiny toy corpus for supervised training
CORPUS = [
    ("Le chien mange.", ["le", "chien", "manger", "."]),
    ("Le roi parle.", ["le", "roi", "parler", "."]),
]


class ToyDataset(torch.utils.data.Dataset):
    def __init__(self, corpus, lemma_map):
        self.samples = []
        for sent, tokens in corpus:
            ids = [lemma_map.get(tok, 0) for tok in tokens]
            self.samples.append(torch.tensor(ids, dtype=torch.long))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


if __name__ == "__main__":
    # Load resources
    with open(GRAPH_PATH, "rb") as f:
        G = pickle.load(f)
    with open(VECTORS_PATH, "rb") as f:
        vectors = pickle.load(f)

    lemma_map = {lemma: i for i, lemma in enumerate(vectors.keys())}
    dataset = ToyDataset(CORPUS, lemma_map)
    loader = DataLoader(dataset, batch_size=1, shuffle=True)

    encoder = GNNEncoder(in_channels=32, hidden_channels=64, out_channels=64)
    decoder = SimpleDecoder(vocab_size=len(lemma_map))
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    loss_fn = torch.nn.CrossEntropyLoss()

    persona = PersonaManager()

    for epoch in range(3):
        for seq in loader:
            optimizer.zero_grad()
            # Use encoder to produce a thought vector (here simplified as mean of embeddings)
            x = torch.stack([torch.tensor([int(b) for b in vectors.get(tok, "0"*32)], dtype=torch.float) for tok in vectors.keys()])
            thought = encoder(x, torch.tensor(list(G.edges())).t().contiguous())
            # decode
            logits, _ = decoder(seq[:, :-1])
            loss = loss_fn(logits.reshape(-1, logits.size(-1)), seq[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch+1}, loss {loss.item():.4f}")

