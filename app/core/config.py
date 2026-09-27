from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    APP_NAME: str = "NextStep AI Interview Server"
    DEBUG: bool = True

    DATABASE_URL: str

    GROQ_API_KEY: str = ""
    CLOUDFLARE_ACCOUNT_ID: str = ""
    CLOUDFLARE_API_TOKEN: str = ""
    INCEPTION_API_KEY: str = ""
    OPENTYPHOON_API_KEY: str = ""
    OLLAMA_API_KEY: str = ""
    NVIDIA_API_KEY: str = ""
    GEMINI_API_KEY: str = ""

    GROQ_MODEL: str = "openai/gpt-oss-120b"
    CLOUDFLARE_MODEL: str = "@cf/qwen/qwen3-30b-a3b-fp8"
    INCEPTION_MODEL: str = "mercury-2.5"
    OPENTYPHOON_MODEL: str = "typhoon-v2.5-30b-a3b-instruct"
    OLLAMA_CLOUD_MODEL: str = "gemma4:31b"
    NVIDIA_NEMOTRON_MODEL: str = "nvidia/nemotron-3-ultra-550b-a55b"
    GEMINI_MODEL: str = "gemini-2.5-flash"
    DATASET_PROVIDER_TOKEN_BUDGET: int = 0
    CV_AI_ENRICHMENT_ENABLED: bool = True
    RESEARCH_PSEUDONYM_SALT: str = ""

    JWT_ACCESS_SECRET: str
    JWT_ACCESS_EXPIRES_IN: str

    model_config = SettingsConfigDict(
        env_file=(BASE_DIR / ".env", BASE_DIR / ".env.production"),
        extra="ignore",
    )

    @field_validator("DEBUG", mode="before")
    @classmethod
    def parse_debug_value(cls, value):
        if isinstance(value, bool):
            return value

        normalized = str(value or "").strip().lower()
        if normalized in {"1", "true", "yes", "on", "debug", "development", "dev"}:
            return True
        if normalized in {"0", "false", "no", "off", "release", "production", "prod"}:
            return False
        return value

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def sanitize_database_url(cls, value):
        raw = str(value or "").strip()
        if not raw:
            return raw

        # Some external tools append pgbouncer=true, which psycopg2 rejects.
        if "?" not in raw or "pgbouncer" not in raw.lower():
            return raw

        base, query = raw.split("?", 1)
        kept_parts: list[str] = []
        for part in query.split("&"):
            key = part.split("=", 1)[0].strip().lower()
            if key == "pgbouncer":
                continue
            if part.strip():
                kept_parts.append(part.strip())

        return f"{base}?{'&'.join(kept_parts)}" if kept_parts else base


settings = Settings()
