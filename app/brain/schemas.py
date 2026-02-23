"""
brain/schemas.py — Pydantic models for LLM structured output.

The LLM is forced (via .with_structured_output) to return an LLMResponse.
This ensures 100% predictable JSON from every provider.
"""
from pydantic import BaseModel, Field


class Entity(BaseModel):
    name: str = Field(description="The extracted entity value, e.g. 'John', 'London', '2024-03-15'")
    type: str = Field(description="Category: Person | Place | Date | Organization | Product | Other")
    confidence: float = Field(ge=0.0, le=1.0, description="Extraction confidence 0-1")


class Relationship(BaseModel):
    subject: str = Field(description="The subject of the relationship")
    relation: str = Field(description="The relationship verb, e.g. works_at, wants, is_traveling_to")
    object: str = Field(description="The object of the relationship")


class LLMResponse(BaseModel):
    """
    Structured output schema enforced on every LLM call.
    All fields are required — the LLM must populate them.
    """
    entities: list[Entity] = Field(
        default_factory=list,
        description="All named entities found in the user's message"
    )
    relationships: list[Relationship] = Field(
        default_factory=list,
        description="Semantic relationships between extracted entities"
    )
    sentiment: str = Field(
        description="Overall tone: Positive | Negative | Neutral | Frustrated | Excited"
    )
    intent: str = Field(
        description="Primary user intent: booking | inquiry | complaint | greeting | other"
    )
    missing_info: list[str] = Field(
        default_factory=list,
        description="List of entity types that are needed but not yet provided"
    )
    response_audio_text: str = Field(
        description="The exact text the AI assistant should speak aloud in response"
    )
    trigger_clarification: bool = Field(
        description="True if missing_info is non-empty and a follow-up question is needed"
    )
    clarification_question: str | None = Field(
        default=None,
        description="The specific follow-up question to ask the user (only when trigger_clarification=True)"
    )
