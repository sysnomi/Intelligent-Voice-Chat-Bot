"""
brain/llm_factory.py — Return a LangChain ChatModel instance based on LLM_PROVIDER.

Supports: openai | gemini | groq | azure_openai
The model is pre-configured with .with_structured_output(LLMResponse)
so every call returns a validated Pydantic object.
"""
from app.config import settings
from app.brain.schemas import LLMResponse
from app.utils.logging import get_logger

logger = get_logger(__name__)


def get_llm():
    """
    Return a LangChain LLM instance bound to structured output (LLMResponse).
    The caller simply does: llm.invoke(messages) → LLMResponse
    """
    provider = settings.llm_provider.lower()
    model_name = settings.get_llm_model()

    logger.info("llm_factory_init", provider=provider, model=model_name)

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            model=model_name,
            api_key=settings.openai_api_key,
            temperature=0.3,
            streaming=True,
        )

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=settings.gemini_api_key,
            temperature=0.3,
        )

    elif provider == "groq":
        from langchain_groq import ChatGroq
        llm = ChatGroq(
            model=model_name,
            api_key=settings.groq_api_key,
            temperature=0.3,
        )

    elif provider == "azure_openai":
        from langchain_openai import AzureChatOpenAI
        llm = AzureChatOpenAI(
            azure_deployment=settings.azure_openai_deployment,
            azure_endpoint=settings.azure_openai_endpoint,
            api_key=settings.azure_openai_key,
            api_version="2024-02-01",
            temperature=0.3,
        )

    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER='{provider}'. "
            "Valid options: openai | gemini | groq | azure_openai"
        )

    # Bind structured output — every call returns a validated LLMResponse
    return llm.with_structured_output(LLMResponse)
