"""
VOXCODE LLM Client
Supports both Ollama (local) and Groq (cloud) for LLM inference.
"""

import json
import requests
from typing import Optional, List, Dict, Generator, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

from config import config


@dataclass
class LLMResponse:
    """Response from LLM."""
    content: str
    model: str
    done: bool = True
    total_duration: Optional[int] = None
    load_duration: Optional[int] = None
    eval_count: Optional[int] = None
    eval_duration: Optional[int] = None

    @property
    def tokens_per_second(self) -> Optional[float]:
        if self.eval_count and self.eval_duration:
            return self.eval_count / (self.eval_duration / 1e9)
        return None


@dataclass
class Message:
    """Chat message."""
    role: str
    content: str


@dataclass
class Conversation:
    """Manages conversation history."""
    messages: List[Message] = field(default_factory=list)
    max_messages: int = 50

    def add(self, role: str, content: str) -> None:
        self.messages.append(Message(role=role, content=content))
        if len(self.messages) > self.max_messages:
            system_msgs = [m for m in self.messages if m.role == "system"]
            other_msgs = [m for m in self.messages if m.role != "system"]
            self.messages = system_msgs + other_msgs[-(self.max_messages - len(system_msgs)):]

    def to_list(self) -> List[Dict[str, str]]:
        return [{"role": m.role, "content": m.content} for m in self.messages]

    def clear(self, keep_system: bool = True) -> None:
        if keep_system:
            self.messages = [m for m in self.messages if m.role == "system"]
        else:
            self.messages = []


class BaseLLMClient(ABC):
    """Abstract base class for LLM clients."""

    @abstractmethod
    def generate(self, prompt: str, system: str = None, stream: bool = False) -> LLMResponse:
        pass

    @abstractmethod
    def chat(self, message: str, system: str = None, stream: bool = False, use_history: bool = True) -> LLMResponse:
        pass

    @abstractmethod
    def is_available(self) -> bool:
        pass

    @abstractmethod
    def clear_history(self) -> None:
        pass


class GroqClient(BaseLLMClient):
    """Client for Groq Cloud API - fast inference for Llama, Mixtral, etc."""

    def __init__(
        self,
        api_key: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None
    ):
        self.api_key = api_key or config.llm.groq_api_key
        self.model = model or config.llm.groq_model
        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.timeout = timeout or config.llm.timeout
        self.base_url = "https://api.groq.com/openai/v1"
        self._conversation = Conversation()

    def _make_request(self, messages: List[Dict], stream: bool = False) -> requests.Response:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "stream": stream
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
            stream=stream
        )
        response.raise_for_status()
        return response

    def generate(self, prompt: str, system: str = None, stream: bool = False) -> LLMResponse:
        """Generate a response (single-turn, no history)."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        if stream:
            return self._stream_response(messages)

        response = self._make_request(messages)
        data = response.json()

        content = data["choices"][0]["message"]["content"]
        return LLMResponse(
            content=content,
            model=data.get("model", self.model),
            done=True,
            eval_count=data.get("usage", {}).get("completion_tokens")
        )

    def chat(self, message: str, system: str = None, stream: bool = False, use_history: bool = True) -> LLMResponse:
        """Chat with conversation history."""
        if system and not any(m.role == "system" for m in self._conversation.messages):
            self._conversation.add("system", system)

        self._conversation.add("user", message)

        messages = self._conversation.to_list() if use_history else [{"role": "user", "content": message}]

        if stream:
            return self._stream_chat(messages)

        response = self._make_request(messages)
        data = response.json()

        assistant_message = data["choices"][0]["message"]["content"]
        self._conversation.add("assistant", assistant_message)

        return LLMResponse(
            content=assistant_message,
            model=data.get("model", self.model),
            done=True,
            eval_count=data.get("usage", {}).get("completion_tokens")
        )

    def _stream_response(self, messages: List[Dict]) -> Generator[str, None, None]:
        response = self._make_request(messages, stream=True)
        for line in response.iter_lines():
            if line:
                line = line.decode("utf-8")
                if line.startswith("data: "):
                    data_str = line[6:]
                    if data_str == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            yield delta["content"]
                    except json.JSONDecodeError:
                        continue

    def _stream_chat(self, messages: List[Dict]) -> Generator[str, None, None]:
        full_response = []
        for chunk in self._stream_response(messages):
            full_response.append(chunk)
            yield chunk
        self._conversation.add("assistant", "".join(full_response))

    def clear_history(self) -> None:
        self._conversation.clear()

    def is_available(self) -> bool:
        try:
            headers = {"Authorization": f"Bearer {self.api_key}"}
            response = requests.get(
                f"{self.base_url}/models",
                headers=headers,
                timeout=5
            )
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[Dict[str, Any]]:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        response = requests.get(
            f"{self.base_url}/models",
            headers=headers,
            timeout=self.timeout
        )
        response.raise_for_status()
        return response.json().get("data", [])


class OllamaClient(BaseLLMClient):
    """Client for Ollama local LLM API."""

    def __init__(
        self,
        host: str = None,
        model: str = None,
        temperature: float = None,
        max_tokens: int = None,
        timeout: int = None
    ):
        self.host = (host or config.llm.ollama_host).rstrip("/")
        self.model = model or config.llm.ollama_model
        self.temperature = temperature if temperature is not None else config.llm.temperature
        self.max_tokens = max_tokens or config.llm.max_tokens
        self.timeout = timeout or config.llm.timeout
        self._conversation = Conversation()

    def _make_request(self, endpoint: str, payload: Dict[str, Any], stream: bool = False) -> requests.Response:
        url = f"{self.host}/api/{endpoint}"
        response = requests.post(url, json=payload, stream=stream, timeout=self.timeout)
        response.raise_for_status()
        return response

    def generate(self, prompt: str, system: str = None, stream: bool = False):
        payload = {
            "model": self.model,
            "prompt": prompt,
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": stream
        }
        if system:
            payload["system"] = system

        if stream:
            return self._stream_response("generate", payload)

        response = self._make_request("generate", payload)
        data = response.json()
        return LLMResponse(
            content=data.get("response", ""),
            model=data.get("model", self.model),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration")
        )

    def chat(self, message: str, system: str = None, stream: bool = False, use_history: bool = True):
        if system and not any(m.role == "system" for m in self._conversation.messages):
            self._conversation.add("system", system)

        self._conversation.add("user", message)

        payload = {
            "model": self.model,
            "messages": self._conversation.to_list() if use_history else [{"role": "user", "content": message}],
            "options": {"temperature": self.temperature, "num_predict": self.max_tokens},
            "stream": stream
        }

        if stream:
            return self._stream_chat(payload)

        response = self._make_request("chat", payload)
        data = response.json()
        assistant_message = data.get("message", {}).get("content", "")
        self._conversation.add("assistant", assistant_message)

        return LLMResponse(
            content=assistant_message,
            model=data.get("model", self.model),
            done=data.get("done", True),
            total_duration=data.get("total_duration"),
            eval_count=data.get("eval_count"),
            eval_duration=data.get("eval_duration")
        )

    def _stream_response(self, endpoint: str, payload: Dict[str, Any]) -> Generator[str, None, None]:
        response = self._make_request(endpoint, payload, stream=True)
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                if "response" in data:
                    yield data["response"]
                if data.get("done"):
                    break

    def _stream_chat(self, payload: Dict[str, Any]) -> Generator[str, None, None]:
        response = self._make_request("chat", payload, stream=True)
        full_response = []
        for line in response.iter_lines():
            if line:
                data = json.loads(line)
                chunk = data.get("message", {}).get("content", "")
                if chunk:
                    full_response.append(chunk)
                    yield chunk
                if data.get("done"):
                    break
        self._conversation.add("assistant", "".join(full_response))

    def clear_history(self) -> None:
        self._conversation.clear()

    def list_models(self) -> List[Dict[str, Any]]:
        response = requests.get(f"{self.host}/api/tags", timeout=self.timeout)
        response.raise_for_status()
        return response.json().get("models", [])

    def is_available(self) -> bool:
        try:
            response = requests.get(f"{self.host}/api/tags", timeout=5)
            return response.status_code == 200
        except:
            return False

    def pull_model(self, model_name: str) -> Generator[Dict, None, None]:
        response = self._make_request("pull", {"name": model_name}, stream=True)
        for line in response.iter_lines():
            if line:
                yield json.loads(line)


def get_llm_client() -> BaseLLMClient:
    """Factory function to get the configured LLM client."""
    if config.llm.provider == "groq":
        return GroqClient()
    else:
        return OllamaClient()


# For backwards compatibility
def create_client() -> BaseLLMClient:
    """Create LLM client based on config."""
    return get_llm_client()
