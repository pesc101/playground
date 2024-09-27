"""Client for handling LLM API interactions."""

import os

from openai import OpenAI


class BaseClient:
    """Base client for handling OpenAI API interactions."""

    def __init__(self, api_key: str, base_url: str) -> None:
        if not api_key:
            raise ValueError("API key cannot be empty")
        self._client = OpenAI(api_key=api_key, base_url=base_url)

    @property
    def client(self) -> OpenAI:
        """Return the OpenAI client instance."""
        return self._client


class OpenAIClient(BaseClient):
    """Client for OpenAI's live API."""

    def __init__(self, api_key: str = None) -> None:
        # Allow for overriding API key and fallback to env variable
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        super().__init__(api_key=api_key, base_url="https://api.openai.com/v1")


class LocalClient(BaseClient):
    """Client for local OpenAI API instances, e.g., during development."""

    def __init__(self) -> None:
        super().__init__(api_key="EMPTY", base_url="http://localhost:8000/v1")


class ClientFactory:
    """Factory to create client instances based on model name.

    If the model is recognized as OpenAI's model, it will use OpenAIClient.
    If the model is a local model, it will use LocalClient.
    """

    openai_models = ["gpt-4o-mini", "gpt-4o", "gpt-4o-2024-08-06"]
    local_models = [
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        "VAGOsolutions/Llama-3.1-SauerkrautLM-8b-Instruct",
    ]

    @staticmethod
    def get_client(model_name: str, api_key: str = None) -> BaseClient:
        """Return a client instance based on the model name."""
        if model_name in ClientFactory.openai_models:
            return OpenAIClient(api_key)
        elif model_name in ClientFactory.local_models:
            return LocalClient()
        else:
            raise ValueError(f"Unknown model name: {model_name}")
