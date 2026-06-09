from __future__ import annotations

import os

import dspy
from dotenv import load_dotenv


def _setup_lm() -> dspy.LM:
    """Configure the language model from available environment credentials.

    Priority: Anthropic → OpenAI → Mistral → Gemini → local Ollama fallback.
    """
    load_dotenv()
    if os.getenv("ANTHROPIC_API_KEY"):
        return dspy.LM("anthropic/claude-sonnet-4-6")
    if os.getenv("OPENAI_API_KEY"):
        #"gpt-4o"
        return dspy.LM("openai/gpt-4o", api_key=os.getenv("OPENAI_API_KEY"))
    if os.getenv("MISTRAL_API_KEY"):
        return dspy.LM("mistral/mistral-small-latest")
    if os.getenv("OPENROUTER_API_KEY"):
        return dspy.LM(model="openrouter/deepseek/deepseek-chat", api_base="https://operouter.ai/api/v1")
    if os.getenv("DEEPSEEK_API"):
        return dspy.LM(model="deepseek/deepseek-reasoner", api_key=os.getenv("DEEPSEEK_API"))
    #if os.getenv("GEMINI_API_KEY"):§
    #    return dspy.LM("gemini/gemini-2.0-flash-001", api_key=os.getenv("GEMINI_API_KEY"))
    # Local Ollama — no key required
    return dspy.LM("ollama_chat/LiquidAI/lfm2.5-1.2b-instruct:latest", api_base="http://localhost:11434")

