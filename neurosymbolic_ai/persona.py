"""Simple persona manager for Lobe 3.

This minimal module stores a persona profile with affective
preferences and conversational memory.  It exposes methods to
update the context and to provide stylistic hints to the decoder.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class PersonaProfile:
    name: str = "assistant"
    valence: str = "positive"
    tension: str = "neutral"
    relation: str = "engaged"
    intention: str = "declarative"

@dataclass
class ConversationState:
    history: List[str] = field(default_factory=list)

class PersonaManager:
    def __init__(self, profile: PersonaProfile | None = None):
        self.profile = profile or PersonaProfile()
        self.state = ConversationState()

    def remember(self, utterance: str) -> None:
        """Store a line of dialogue."""
        self.state.history.append(utterance)
        if len(self.state.history) > 20:
            self.state.history.pop(0)

    def get_style_bits(self) -> Dict[str, str]:
        """Return affective bits influenced by the persona."""
        valence_bits = {
            "positive": "10",
            "negative": "00",
            "neutral": "01",
        }.get(self.profile.valence, "01")
        tension_bits = {
            "high": "10",
            "low": "00",
            "neutral": "01",
        }.get(self.profile.tension, "01")
        relation_bits = {
            "engaged": "11",
            "assertive": "10",
            "submissive": "01",
            "distant": "00",
        }.get(self.profile.relation, "11")
        intention_bits = {
            "declarative": "00",
            "interrogative": "01",
            "imperative": "10",
            "exclamative": "11",
        }.get(self.profile.intention, "00")
        return {
            "valence": valence_bits,
            "tension": tension_bits,
            "relation": relation_bits,
            "intention": intention_bits,
        }
