"""Build a small knowledge graph from the generated vectors.

The graph contains lemma nodes connected by dependency edges extracted from a
tiny corpus.  A handful of synonym and hypernym relations are added to
illustrate the multi‑relation structure described in the guide.
"""

import pickle
import os
from collections import Counter, defaultdict
from math import log

import pandas as pd
import spacy
import networkx as nx

LEXICAL_DB = "data/lexical_vectors.pkl"
LEMMA_MAP = "data/lemma_id_map.pkl"
FORM_LEMMA = "data/form_to_lemma.pkl"
CORPUS = "data/corpus.txt"
GRAPH_PATH = "data/knowledge_graph.gpickle"

# Very small lexical resources for demonstration
SYNONYMS = {
    "roi": ["monarque"],
    "chien": ["canidé"],
}

HYPERNYMS = {
    "chien": "animal",
    "roi": "personne",
}


def add_affective_edges(graph, vectors):
    """Add SHARES_POLARITY edges based on valence bits."""
    groups = defaultdict(list)
    for lemma, bits in vectors.items():
        valence = bits[24:26]  # bits 24-25
        groups[valence].append(lemma)

    for lemmas in groups.values():
        for i, src in enumerate(lemmas):
            for dst in lemmas[i + 1 :]:
                graph.add_edge(src, dst, type="AFFECT")
                graph.add_edge(dst, src, type="AFFECT")


def add_form_edges(graph, form_map):
    """Link surface forms to their lemma."""
    for form, lemma in form_map.items():
        if form != lemma:
            graph.add_node(form, type="form")
            graph.add_edge(form, lemma, type="FORM_OF")


def add_cooccurrence_edges(graph, corpus_path, lemma_map, nlp, window=2):
    """Add CO_OCCURS_WITH edges weighted by PMI."""
    if not os.path.exists(corpus_path):
        return
    counts = Counter()
    totals = Counter()
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            doc = nlp(line.strip())
            lemmas = [t.lemma_ for t in doc if t.is_alpha]
            for i, lem in enumerate(lemmas):
                totals[lem] += 1
                for j in range(max(0, i - window), min(len(lemmas), i + window + 1)):
                    if i == j:
                        continue
                    pair = tuple(sorted((lem, lemmas[j])))
                    counts[pair] += 1
    N = sum(totals.values())
    for (a, b), c_ab in counts.items():
        if a in lemma_map and b in lemma_map and c_ab >= 1:
            pmi = log((c_ab * N) / (totals[a] * totals[b] + 1e-9))
            graph.add_edge(a, b, type="COOCCUR", weight=pmi)
            graph.add_edge(b, a, type="COOCCUR", weight=pmi)


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(CORPUS):
        with open(CORPUS, "w", encoding="utf-8") as f:
            f.write("Le chien mange la nourriture.\n")
            f.write("Un roi puissant gouverne le peuple.\n")

    with open(LEXICAL_DB, "rb") as f:
        vectors = pickle.load(f)

    with open(LEMMA_MAP, "rb") as f:
        lemma_map = pickle.load(f)
    try:
        with open(FORM_LEMMA, "rb") as f:
            form_map = pickle.load(f)
    except FileNotFoundError:
        form_map = {}

    nlp = spacy.load("fr_dep_news_trf")
    G = nx.MultiDiGraph()

    # add lexical nodes
    for lemma in lemma_map.keys():
        G.add_node(lemma, type="lemma")
    add_form_edges(G, form_map)

    # synonym edges (undirected)
    for lem, syns in SYNONYMS.items():
        for syn in syns:
            if lem in lemma_map and syn in lemma_map:
                G.add_edge(lem, syn, type="SYNONYM")
                G.add_edge(syn, lem, type="SYNONYM")

    # hypernym edges
    for hypo, hyper in HYPERNYMS.items():
        if hypo in lemma_map and hyper in lemma_map:
            G.add_edge(hyper, hypo, type="HYPERNYM")

    # simple dependency edges from corpus
    dep_counts = Counter()
    with open(CORPUS, "r", encoding="utf-8") as f:
        for line in f:
            doc = nlp(line.strip())
            for token in doc:
                if token.head is not token:
                    dep_counts[(token.head.lemma_, token.lemma_, token.dep_)] += 1

    for (src, dst, dep), count in dep_counts.items():
        if src in lemma_map and dst in lemma_map:
            G.add_edge(src, dst, type="DEP", dep=dep, weight=count)

    # co-occurrence edges with PMI weights
    add_cooccurrence_edges(G, CORPUS, lemma_map, nlp)

    # affective clusters based on valence bits
    add_affective_edges(G, vectors)

    with open(GRAPH_PATH, "wb") as f:
        pickle.dump(G, f)
    print(f"Graph saved with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
