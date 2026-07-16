"""LLM helpers for rag-multimodal-2026.

The provider is chosen from the model name, so the same code serves OpenAI,
Anthropic, and local Ollama models. The Ollama context window is widened so
longer retrieved context, including image captions, fits comfortably.
"""

import logging
from typing import Any, List

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from src.core.config import settings

logger = logging.getLogger(__name__)


def _make_llm(model: str, temperature: float | None = None):
    """Return a chat model for the given model name, provider chosen by name."""
    name = model.lower()
    if any(
        tag in name for tag in ("llama", "qwen", "deepseek", "mistral", "gemma", "phi")
    ):
        from langchain_ollama import ChatOllama

        kwargs = {
            "model": model,
            "base_url": settings.ollama_base_url,
            "num_ctx": settings.ollama_num_ctx,
        }
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatOllama(**kwargs)
    if "claude" in name:
        from langchain_anthropic import ChatAnthropic

        kwargs = {"model": model}
        if temperature is not None:
            kwargs["temperature"] = temperature
        return ChatAnthropic(**kwargs)
    from langchain_openai import ChatOpenAI

    kwargs = {"model": model, "api_key": settings.openai_api_key}
    if temperature is not None:
        kwargs["temperature"] = temperature
    return ChatOpenAI(**kwargs)


def _to_lc_messages(chat_history) -> List[Any]:
    """Convert stored {role, content} dicts into langchain message objects."""
    messages: List[Any] = []
    for turn in chat_history or []:
        if turn.get("role") in ("ai", "assistant"):
            messages.append(AIMessage(content=turn["content"]))
        else:
            messages.append(HumanMessage(content=turn["content"]))
    return messages


_CONTEXTUALIZE_SYSTEM = (
    "Given a chat history and the latest user question which might reference "
    "context in the chat history, formulate a standalone question which can be "
    "understood without the chat history. Do NOT answer the question, just "
    "reformulate it if needed and otherwise return it as is."
)

_contextualize_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", _CONTEXTUALIZE_SYSTEM),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)


def _reformulate_query(llm, user_input: str, history: List[Any]) -> str:
    """Rewrite the question to be standalone, skipped when there is no history."""
    if not history:
        return user_input
    chain = _contextualize_prompt | llm | StrOutputParser()
    return chain.invoke({"input": user_input, "chat_history": history})
