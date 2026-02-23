"""
brain/graph.py — LangGraph conversation state machine.

Graph Flow:
  START → extract_and_respond → (conditional) → clarify | respond → END

Nodes:
  extract_and_respond : Calls LLM with current transcript, returns LLMResponse
  clarify             : Formats clarification question as audio response
  respond             : Finalises a direct response (no clarification needed)
"""
import asyncio
from langgraph.graph import StateGraph, END

from app.brain.state import ConversationState
from app.brain.schemas import LLMResponse
from app.brain.prompts import SYSTEM_PROMPT, format_extraction_prompt
from app.brain.llm_factory import get_llm
from app.utils.logging import get_logger

logger = get_logger(__name__)

# Build LLM once at module load (reused across sessions)
_llm = None


def _get_llm():
    global _llm
    if _llm is None:
        _llm = get_llm()
    return _llm


# ── Nodes ──────────────────────────────────────────────────────────────────

def extract_and_respond(state: ConversationState) -> dict:
    """
    Core node: send transcript + history to LLM, get structured LLMResponse.
    Updates state with entities, sentiment, intent, missing_info, and response.
    """
    llm = _get_llm()
    transcript = state["current_transcript"]
    history = state.get("conversation_history", [])

    user_prompt = format_extraction_prompt(transcript, history)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]

    logger.info("brain_extract_start", transcript=transcript[:80])

    result: LLMResponse = llm.invoke(messages)

    logger.info(
        "brain_extract_done",
        entities=len(result.entities),
        sentiment=result.sentiment,
        trigger_clarification=result.trigger_clarification,
    )

    # Determine audio response text
    audio_text = (
        result.clarification_question
        if result.trigger_clarification and result.clarification_question
        else result.response_audio_text
    )

    return {
        "extracted_entities": [e.model_dump() for e in result.entities],
        "relationships": [r.model_dump() for r in result.relationships],
        "sentiment": result.sentiment,
        "intent": result.intent,
        "missing_info": result.missing_info,
        "response_audio_text": audio_text,
        "trigger_clarification": result.trigger_clarification,
        "pending_clarification": result.clarification_question,
        "conversation_history": [
            {"role": "user", "content": transcript},
            {"role": "assistant", "content": audio_text},
        ],
        "turn_count": state.get("turn_count", 0) + 1,
    }


def clarify(state: ConversationState) -> dict:
    """Clarification node — no extra processing needed, state already set."""
    logger.info("brain_clarify", question=state.get("pending_clarification", "")[:80])
    return {}


def respond(state: ConversationState) -> dict:
    """Direct response node — log and pass through."""
    logger.info("brain_respond", text=state.get("response_audio_text", "")[:80])
    return {}


# ── Conditional edge ───────────────────────────────────────────────────────

def route_after_extraction(state: ConversationState) -> str:
    """Route to 'clarify' if LLM triggered clarification, else 'respond'."""
    return "clarify" if state.get("trigger_clarification") else "respond"


# ── Graph Assembly ─────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    builder = StateGraph(ConversationState)

    builder.add_node("extract_and_respond", extract_and_respond)
    builder.add_node("clarify", clarify)
    builder.add_node("respond", respond)

    builder.set_entry_point("extract_and_respond")
    builder.add_conditional_edges(
        "extract_and_respond",
        route_after_extraction,
        {"clarify": "clarify", "respond": "respond"},
    )
    builder.add_edge("clarify", END)
    builder.add_edge("respond", END)

    return builder.compile()


# Singleton compiled graph
conversation_graph = build_graph()


async def run_brain(state: ConversationState) -> ConversationState:
    """
    Async wrapper: Run the LangGraph synchronously in a thread executor
    (LangGraph invoke is currently synchronous).
    """
    result = await asyncio.to_thread(conversation_graph.invoke, state)
    return result
