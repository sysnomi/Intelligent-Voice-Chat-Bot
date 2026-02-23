"""
brain/state.py — LangGraph ConversationState TypedDict.

This is the single source of truth for a voice session's memory.
Every node in the LangGraph reads from and writes to this state.
"""
from typing import TypedDict, Annotated
from operator import add


class EntityModel(TypedDict):
    name: str
    type: str          # Person | Place | Date | Organization | Product | Other
    confidence: float


class RelationshipModel(TypedDict):
    subject: str
    relation: str      # e.g. works_at, wants, is_traveling_to
    object: str


class ConversationState(TypedDict):
    session_id: str

    # Full turn-by-turn history fed into the LLM
    conversation_history: Annotated[list[dict], add]

    # Current user utterance (populated after STT)
    current_transcript: str

    # Accumulated extraction results (merged across turns)
    extracted_entities: list[EntityModel]
    relationships: list[RelationshipModel]
    sentiment: str           # Positive | Negative | Neutral | Frustrated | Excited
    intent: str              # booking | inquiry | complaint | greeting | other
    missing_info: list[str]  # e.g. ["destination", "date"]

    # What the AI will say / is saying
    response_audio_text: str

    # Clarification state
    trigger_clarification: bool
    pending_clarification: str | None

    # Session metadata
    turn_count: int
    is_interrupted: bool     # True if user spoke while TTS was playing
