"""Interactive demo combining the three lobes.

This small script loads the generated resources and lets a user chat with a
minimal agent.  It demonstrates how the persona manager (Lobe 3) can modulate
responses while the knowledge graph provides simple semantic lookups.
"""

import pickle
import os

import spacy

from .persona import PersonaManager, PersonaProfile

LEXICAL_DB = "data/lexical_vectors.pkl"
GRAPH_PATH = "data/knowledge_graph.gpickle"


def load_resources():
    with open(LEXICAL_DB, "rb") as f:
        vectors = pickle.load(f)
    try:
        with open(GRAPH_PATH, "rb") as f:
            graph = pickle.load(f)
    except FileNotFoundError:
        graph = None
    return vectors, graph


def find_synonyms(graph, lemma):
    """Return synonyms of a lemma from the graph."""
    if graph is None or not graph.has_node(lemma):
        return []
    return [v for u, v, d in graph.edges(lemma, data=True) if d.get("type") == "SYNONYM"]


def find_hypernyms(graph, lemma):
    """Return direct hypernyms of a lemma."""
    if graph is None or not graph.has_node(lemma):
        return []
    return [v for u, v, d in graph.edges(lemma, data=True) if d.get("type") == "HYPERNYM"]


def main():
    if not os.path.exists(LEXICAL_DB):
        print("Please run generate_vectors.py and build_knowledge_graph.py first.")
        return

    vectors, graph = load_resources()
    nlp = spacy.load("fr_dep_news_trf")
    persona = PersonaManager(PersonaProfile())

    print("Type 'quit' to exit.")
    while True:
        user = input("Vous: ")
        if user.strip().lower() in {"quit", "exit"}:
            break
        if user.strip().lower() == "history":
            for h in persona.state.history:
                print(h)
            continue
        persona.remember("USER:" + user)
        doc = nlp(user)
        syns = []
        hypers = []
        for tok in doc:
            syns.extend(find_synonyms(graph, tok.lemma_))
            hypers.extend(find_hypernyms(graph, tok.lemma_))
        parts = []
        if syns:
            parts.append(f"Synonymes: {', '.join(syns)}")
        if hypers:
            parts.append(f"Hyperonymes: {', '.join(hypers)}")
        resp = " ".join(parts) if parts else "Je prends note."
        style = persona.get_style_bits()
        prefix = {
            "10": "!",
            "00": "...",
            "01": "",
        }.get(style["valence"], "")
        final = f"{prefix} {resp}"
        print("Agent:", final)
        persona.remember("AGENT:" + final)


if __name__ == "__main__":
    main()
