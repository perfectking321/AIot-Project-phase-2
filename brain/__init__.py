"""
VOXCODE Brain Module
LLM client (Ollama/Groq) and system prompts for Windows automation.
"""

from .llm import OllamaClient, GroqClient, get_llm_client, LLMResponse
from .prompts import SystemPrompts, PromptBuilder

__all__ = ["OllamaClient", "GroqClient", "get_llm_client", "LLMResponse", "SystemPrompts", "PromptBuilder"]
