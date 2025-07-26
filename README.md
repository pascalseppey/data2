# Neuro-Symbolic AI Example

This repository contains a minimal prototype implementing concepts from the
"Guide d'Implémentation pour une IA Réflective Neuro‑Symbolique Francophone".
It demonstrates how 32‑bit lexical vectors, a small knowledge graph and three
interacting modules can be combined.

## Quick start

1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate
   pip install -r neurosymbolic_ai/requirements.txt
   python -m spacy download fr_dep_news_trf
   ```
2. Generate lexical vectors and build the knowledge graph:
   ```bash
   python neurosymbolic_ai/generate_vectors.py
   python neurosymbolic_ai/build_knowledge_graph.py
   ```
3. Train the GNN and run the interactive demo:
   ```bash
   python neurosymbolic_ai/train_phase1.py
   python neurosymbolic_ai/interactive_demo.py
   ```

The demo lets you chat in French. Type `history` at any time to review the
conversation.
