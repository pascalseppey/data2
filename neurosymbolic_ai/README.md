# Neuro-Symbolic French AI Prototype

This directory provides a small prototype inspired by the guide for a reflective French neuro‑symbolic AI.  It contains scripts to:

1. Generate 32‑bit lexical vectors from Lexique383 and FEEL. The script now
   assigns simple subclass bits and stores a mapping from surface forms to
   their lemmas.
2. Build a small knowledge graph using these vectors. Forms are connected to
   their lemmas with `FORM_OF` edges while synonym, hypernym and affective
   relations illustrate the multi‑relation structure.
3. Train a small Graph Attention Network on link prediction.
4. Interact with a tiny demo agent combining the three lobes. The demo can now
   list synonyms and hypernyms found in the graph and shows the conversation
   history with the `history` command.

The scripts remain intentionally simplified but now mirror more closely the structure proposed in the specification.  They should compile without external data but require the listed dependencies to run.

## Requirements
Install dependencies in a virtual environment:
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python -m spacy download fr_dep_news_trf
```

## Usage
Generate the vectors:
```bash
python generate_vectors.py
```
Build the graph:
```bash
python build_knowledge_graph.py
```
Train the GNN (phase 1):
```bash
python train_phase1.py
```
Train the encoder and a small decoder on sentences (phase 2):
```bash
python train_phase2.py
```
Experiment with reinforcement learning (phase 3):
```bash
python train_phase3.py
```
Run the interactive demo:
```bash
python interactive_demo.py
```

The persona manager (Lobe 3) modulates the demo's responses using simple
affective bits.  Type `history` during the demo to display the conversation so
far.  The knowledge graph links words sharing the same valence to illustrate
affective clusters.
