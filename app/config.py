"""
Centralised configuration via Pydantic Settings.
All constants live here — tuning the system means changing one file only.
"""

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Audio validation
    max_file_size_bytes: int = 10 * 1024 * 1024  # 10 MB
    min_file_size_bytes: int = 10 * 1024  # 10 KB
    max_audio_duration_seconds: int = 300  # 5 minutes
    min_audio_duration_seconds: int = 1
    max_transcript_length: int = 1000
    supported_extensions: list[str] = ["wav", "mp3"]
    supported_languages: list[str] = ["en", "it"]
    default_language: str = "en-US"

    # STT
    stt_confidence_threshold: float = 0.6
    stt_max_retries: int = 3
    stt_retry_base_delay_seconds: float = 1.0
    deepgram_model: str = "nova-3"
    sample_rate: int = 8000

    # LLM
    llm_confidence_threshold: float = 0.5
    llm_max_retries: int = 3
    llm_timeout_seconds: int = 10
    prompt_version: str = "v1.0"
    supported_intents: list[str] = [
        "account_support",
        "billing",
        "technical_support",
        "general_inquiry",
        "prompt_attack",
        "out_of_scope",
        "unclear",
    ]

    #LLM config
    temperature: float = 0.0


    # Orchestration
    job_ttl_seconds: int = 3600  # 1 hour
    max_manual_retries: int = 3
    checkpoint_dir: str = "./checkpoints"
    checkpoint_ttl_seconds: int = 3600
    output_dir: str = "./final_results"

    # API keys — loaded from environment, never hardcoded
    deepgram_api_key: str = ""
    azure_openai_key: str = ""
    azure_openai_endpoint: str = ""
    gemini_api_key: str = ""

    # Backend selection: 'azure_openai' or 'gemini'
    llm_backend: str = "gemini"

    # Azure OpenAI deployment names
    azure_openai_deployment_primary: str = "gpt-4.1"
    azure_openai_deployment_fallback: str = "gpt-4.1-mini"

    # Gemini model names
    gemini_model_primary: str = "gemini-2.5-flash"
    gemini_model_fallback: str = "gemini-2.5-flash"

    # Cost tracking (USD per token)
    gpt41_cost_per_input_token: float = 0.000003
    gpt41_cost_per_output_token: float = 0.000012
    gpt41_mini_cost_per_input_token: float = 0.0000008 
    gpt41_mini_cost_per_output_token: float = 0.0000032
    gemini_cost_per_input_token: float = 0.0000003
    gemini_cost_per_output_token: float = 0.0000025

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
