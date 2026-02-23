"""
tests/test_brain.py — Tests for LangGraph brain: schemas, state, and graph routing.
"""
import pytest
from app.brain.schemas import LLMResponse, Entity, Relationship
from app.brain.state import ConversationState
from app.brain.prompts import format_extraction_prompt


class TestLLMResponseSchema:
    def test_valid_extraction(self):
        response = LLMResponse(
            entities=[Entity(name="London", type="Place", confidence=0.97)],
            relationships=[Relationship(subject="User", relation="wants_to_travel_to", object="London")],
            sentiment="Positive",
            intent="booking",
            missing_info=["travel_date"],
            response_audio_text="That sounds great! When would you like to travel?",
            trigger_clarification=True,
            clarification_question="When would you like to travel to London?",
        )
        assert response.trigger_clarification is True
        assert response.entities[0].name == "London"
        assert 0 <= response.entities[0].confidence <= 1.0

    def test_confidence_bounds(self):
        with pytest.raises(Exception):
            Entity(name="Test", type="Person", confidence=1.5)  # Out of range

    def test_no_clarification(self):
        response = LLMResponse(
            entities=[],
            relationships=[],
            sentiment="Neutral",
            intent="greeting",
            missing_info=[],
            response_audio_text="Hello! How can I help you today?",
            trigger_clarification=False,
            clarification_question=None,
        )
        assert response.trigger_clarification is False
        assert response.clarification_question is None


class TestPrompts:
    def test_format_with_history(self):
        history = [
            {"role": "user", "content": "I want to book a flight"},
            {"role": "assistant", "content": "Sure! Where to?"},
        ]
        prompt = format_extraction_prompt("London please", history)
        assert "London please" in prompt
        assert "USER:" in prompt

    def test_format_no_history(self):
        prompt = format_extraction_prompt("Hello", [])
        assert "Hello" in prompt
        assert "No prior conversation" in prompt
