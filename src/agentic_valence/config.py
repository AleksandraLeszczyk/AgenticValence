"""Centralized, typed configuration for AgenticValence.

All environment variables and model choices live here. Import the singleton
``settings`` object anywhere in the codebase instead of reading ``os.environ``
or a loose ``CONFIG`` dict.

Values are resolved with the following precedence:
    1. Explicit environment variables (e.g. ``MODEL_CODE_WRITING=...``)
    2. Variables defined in a local ``.env`` file
    3. The defaults declared below

Example:
    from agentic_valence.config import settings
    ChatOpenAI(model=settings.model_code_writing, api_key=settings.openai_key)
"""

from functools import lru_cache

from dotenv import load_dotenv
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

# Load .env into os.environ once, centrally. ``Settings`` below reads typed
# values for our own code, while this keeps third-party libraries that read
# os.environ directly (e.g. the OpenAI client, HuggingFace's HF_TOKEN) working.
load_dotenv(override=True)


class Settings(BaseSettings):
    """Application settings sourced from the environment and ``.env``."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
        # ``model_*`` fields would otherwise collide with pydantic's protected
        # namespace and emit warnings.
        protected_namespaces=(),
    )

    # --- Secrets / API keys -------------------------------------------------
    openai_api_key: SecretStr | None = None
    gemini_api_key: SecretStr | None = None
    hf_token: SecretStr | None = None

    # --- Knowledge (literature) vector DB ----------------------------------
    knowledge_db_host: str = "localhost"
    knowledge_db_port: int = 80
    knowledge_db_collection: str = "ScientificArticles"

    # --- Code-snippet vector DB --------------------------------------------
    # Accepts either a remote Chroma host (``*.com``) or a local directory.
    code_db_host: str = "code_db"

    # --- Scientific computation backend ------------------------------------
    mcp_server_url: str | None = None

    # --- Models ------------------------------------------------------------
    model_embedding: str = "text-embedding-3-small"
    model_embedding_hf: str = "all-MiniLM-L6-v2"
    model_knowledge_summary: str = "gpt-5.4-mini"
    model_code_writing: str = "gpt-5.5"
    model_principal_investigator: str = "gpt-5.5"
    model_viz_creator: str = "gpt-5.4-mini"

    @property
    def openai_key(self) -> str | None:
        """Plain-text OpenAI key for clients that need a raw string."""
        return self.openai_api_key.get_secret_value() if self.openai_api_key else None

    @staticmethod
    def is_remote_host(host: str) -> bool:
        """Whether a DB host string points at a remote Chroma server.

        Remote hosts are addressed by domain (``*.com``); anything else is
        treated as a local on-disk ``persist_directory``.
        """
        return host.endswith(".com")


@lru_cache(maxsize=1)
def get_settings() -> "Settings":
    """Return the cached singleton ``Settings`` instance.

    Use this (rather than constructing ``Settings()`` directly) so tests can
    override the cache via ``get_settings.cache_clear()``.
    """
    return Settings()


# Module-level singleton for convenient imports.
settings = get_settings()
