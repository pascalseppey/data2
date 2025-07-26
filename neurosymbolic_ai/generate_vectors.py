"""Generate 32‑bit lexical vectors from Lexique383 and FEEL.

This script follows the layout described in the implementation guide.  It
remains intentionally compact but it tries to respect the proposed 32‑bit
structure:

* **Bits 0‑7**   – grammatical class and optional subclass (spaCy POS tags).
* **Bits 8‑15**  – morpho‑syntactic features (gender/number or verb inflection).
* **Bits 16‑23** – lemma identifier (12 bits, partially reused from subclass).
* **Bits 24‑31** – affective/pragmatic flags derived from the FEEL lexicon.

The resulting dictionaries are saved in ``data/lexical_vectors.pkl`` and
``data/lemma_id_map.pkl`` for later use by the knowledge graph builder.
"""

import os
import pickle
from typing import Dict, Tuple

import pandas as pd
import spacy
from pylexique import Lexique383

LEXICAL_DB = "data/lexical_vectors.pkl"
LEMMA_MAP = "data/lemma_id_map.pkl"
FORM_LEMMA = "data/form_to_lemma.pkl"
FEEL_CSV = "http://advanse.lirmm.fr/ressources/FEEL.csv"

# Basic encoding tables for CLASSE bits (0-3)
CLASS_CODES: Dict[str, str] = {
    "NOM": "0001",
    "ADJ": "0011",
    "VER": "0100",
    "ADV": "0101",
    "DET": "0010",
    "PRO": "1000",
    "PRE": "0110",
    "CON": "0111",
    "INT": "1001",
}


def encode_class(cgram: str) -> str:
    """Return 4 bits representing the broad grammatical class."""
    for key in CLASS_CODES:
        if cgram.startswith(key):
            return CLASS_CODES[key]
    return "1111"  # reserved/unknown


def subclass_bits(item, spacy_token) -> str:
    """Return 4 bits for the subclass field using simple heuristics."""
    cgram = item.cgram
    if cgram.startswith("VER"):
        # auxiliary vs main verb using spaCy tag
        if spacy_token.tag_ == "AUX" or "AUX" in cgram:
            return "0110"  # Auxiliaire
        return "0001"  # Transitif direct (placeholder)
    if cgram.startswith("NOM"):
        return "0001" if "prop" not in cgram.lower() else "0010"
    if cgram.startswith("ADJ"):
        return "0001"
    if cgram.startswith("DET"):
        return "0001"
    return "0000"


def flexion_bits(item, spacy_token) -> str:
    """Return 8 bits describing inflection information using spaCy features."""
    cgram = item.cgram
    if spacy_token is None:
        spacy_token = type("Dummy", (), {"morph": {}})()
    if cgram.startswith(("NOM", "ADJ")):
        # Gender + number, remaining bits reserved
        gender = spacy_token.morph.get("Gender") if hasattr(spacy_token, "morph") else []
        number = spacy_token.morph.get("Number") if hasattr(spacy_token, "morph") else []
        genre = "1" if (gender and gender[0] == "Fem") else "0"
        nombre = "1" if (number and number[0] == "Plur") else "0"
        return genre + nombre + "000000"

    if cgram.startswith("VER") or cgram.startswith("AUX"):
        mood = spacy_token.morph.get("Mood")
        tense = spacy_token.morph.get("Tense")
        person = spacy_token.morph.get("Person")

        mode_map = {
            "Ind": "000",
            "Sub": "001",
            "Cnd": "010",
            "Imp": "011",
            "Inf": "101",
            "Par": "110",
            "Ger": "111",
        }
        tense_map = {
            "Pres": "000",
            "Imp": "001",
            "Fut": "010",
            "Past": "011",
            "Pqp": "100",
        }

        mode_bits = mode_map.get(mood[0] if mood else "Ind", "000")
        tense_bits = tense_map.get(tense[0] if tense else "Pres", "000")
        if person:
            p = int(person[0])
            person_bits = format((p - 1) % 3, "02b")
        else:
            person_bits = "00"

        return mode_bits + tense_bits + person_bits

    return "00000000"


def load_feel():
    """Load FEEL sentiment lexicon if available."""
    try:
        df = pd.read_csv(FEEL_CSV, sep=";")
        return {row["word"]: row["polarity"] for _, row in df.iterrows()}
    except Exception:
        # When running offline we simply return an empty dictionary.
        return {}


def affective_bits(lemma: str, feel):
    """Return the 8 affective bits using a tiny valence mapping."""
    pol = feel.get(lemma, "neutral")
    if pol == "positive":
        valence = "10"
    elif pol == "negative":
        valence = "00"
    else:
        valence = "01"
    tension = "01"  # neutral
    relation = "11"  # engaged/open
    intention = "00"  # declarative
    return valence + tension + relation + intention


if __name__ == "__main__":
    nlp = spacy.load("fr_dep_news_trf")
    lexique = Lexique383()
    feel = load_feel()

    vectors: Dict[str, str] = {}
    lemma_map: Dict[str, int] = {}
    form_to_lemma: Dict[str, str] = {}

    for i, (ortho, info) in enumerate(list(lexique.lexique.items())[:500]):
        items = info if isinstance(info, list) else [info]
        for item in items:
            cls = encode_class(item.cgram)
            needs_sub = item.cgram.startswith(
                ("NOM", "VER", "ADJ", "DET", "ADV")
            )
            doc = nlp(item.ortho)
            tok = doc[0] if doc else None
            subclass = subclass_bits(item, tok) if needs_sub else "0000"
            flex = flexion_bits(item, tok)
            lemma_id = lemma_map.setdefault(item.lemme, len(lemma_map))
            lemma_bits_full = format(lemma_id, "012b")
            if not needs_sub:
                subclass = lemma_bits_full[:4]
            lemma_bits = lemma_bits_full[-8:]
            affect = affective_bits(item.lemme, feel)
            bits = cls + subclass + flex + lemma_bits + affect
            if len(bits) == 32:
                vectors[item.ortho] = bits
                form_to_lemma[item.ortho] = item.lemme
    os.makedirs("data", exist_ok=True)
    with open(LEXICAL_DB, "wb") as f:
        pickle.dump(vectors, f)
    with open(LEMMA_MAP, "wb") as f:
        pickle.dump(lemma_map, f)
    with open(FORM_LEMMA, "wb") as f:
        pickle.dump(form_to_lemma, f)
    print(f"Stored {len(vectors)} vectors")

