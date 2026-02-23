"""
brain/tools.py — LangChain tools available to the LLM.

Currently defines the ask_user_clarification tool, which the LLM can invoke
when it detects missing required entities. The graph handles the tool result
by setting trigger_clarification=True in state.
"""
from langchain_core.tools import tool


@tool
def ask_user_clarification(question: str, missing_fields: list[str]) -> dict:
    """
    Call this tool when the user's request is missing required information.

    Args:
        question: The natural-language question to ask the user.
        missing_fields: List of entity types that are still needed.
                        e.g. ["destination", "travel_date", "passenger_count"]

    Returns:
        A dict confirming the clarification was triggered.
    """
    return {
        "trigger_clarification": True,
        "clarification_question": question,
        "missing_fields": missing_fields,
    }


# Expose as a list for easy binding: llm.bind_tools(TOOLS)
TOOLS = [ask_user_clarification]
