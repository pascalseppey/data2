"""Phase 3 training: simple reinforcement learning loop.

This toy script demonstrates how the three lobes could be trained with a
reinforcement objective. The environment rewards responses that match a small
set of reference answers. It uses a REINFORCE-style update on the decoder.
"""

import random
import pickle

import torch
from torch.nn import functional as F
from torch.distributions.categorical import Categorical

from .models import GNNEncoder, SimpleDecoder
from .persona import PersonaManager

GRAPH_PATH = "data/knowledge_graph.gpickle"
VECTORS_PATH = "data/lexical_vectors.pkl"

# Tiny conversation corpus: (user -> expected agent response)
CONV = [
    ("Bonjour", "Bonjour !"),
    ("Qui es-tu ?", "Je suis un assistant."),
]

MAX_LEN = 5
EPOCHS = 5


def load_resources():
    with open(GRAPH_PATH, "rb") as f:
        graph = pickle.load(f)
    with open(VECTORS_PATH, "rb") as f:
        vectors = pickle.load(f)
    lemma_map = {l: i for i, l in enumerate(vectors.keys())}
    return graph, vectors, lemma_map


def encode_input(text, lemma_map):
    # simple whitespace tokenizer using lemmas directly
    tokens = text.lower().split()
    ids = [lemma_map.get(t, 0) for t in tokens]
    return torch.tensor(ids, dtype=torch.long).unsqueeze(0)


def generate(decoder, start_id, max_len, temperature=1.0):
    generated = [start_id]
    hidden = None
    for _ in range(max_len):
        inp = torch.tensor([[generated[-1]]])
        logits, hidden = decoder(inp, hidden)
        logits = logits[:, -1, :] / temperature
        dist = Categorical(logits=logits)
        next_id = dist.sample()
        generated.append(next_id.item())
        if next_id.item() == start_id:
            break
    return generated[1:], hidden


if __name__ == "__main__":
    graph, vectors, lemma_map = load_resources()
    vocab_size = len(lemma_map)

    encoder = GNNEncoder(in_channels=32, hidden_channels=32, out_channels=32)
    decoder = SimpleDecoder(vocab_size=vocab_size)
    optim = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.01)
    persona = PersonaManager()

    start_id = 0  # assume index 0 exists

    for epoch in range(EPOCHS):
        total_reward = 0.0
        for user, target in CONV:
            optim.zero_grad()
            inp = encode_input(user, lemma_map)
            # dummy thought vector from encoder
            thought = encoder(torch.zeros((len(lemma_map), 32)), torch.tensor(list(graph.edges())).t().contiguous())
            generated_ids, _ = generate(decoder, start_id, MAX_LEN)
            gen_text = " ".join(list(lemma_map.keys())[i] for i in generated_ids)
            reward = 1.0 if gen_text.strip() == target.lower() else 0.0
            logprobs = torch.tensor(0.0)
            # recompute log probability for simplicity
            hidden = None
            for idx in generated_ids:
                inp_step = torch.tensor([[start_id]]) if hidden is None else torch.tensor([[idx]])
                logits, hidden = decoder(inp_step, hidden)
                logprob = F.log_softmax(logits[:, -1, :], dim=-1)[0, idx]
                logprobs += logprob
            loss = -logprobs * reward
            loss.backward()
            optim.step()
            total_reward += reward
        print(f"Epoch {epoch+1}: reward {total_reward:.2f}")
