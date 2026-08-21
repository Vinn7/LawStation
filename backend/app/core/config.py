from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=Path(__file__).resolve().parents[3] / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    deepseek_api_key: str = ""
    deepseek_base_url: str = "https://api.deepseek.com"
    deepseek_model: str = "deepseek-chat"
    dashscope_api_key: str = ""
    dashscope_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "qwen3.7-text-embedding"
    embedding_dimension: int = 1024
    database_url: str = "sqlite:///./data/runtime/lawstation.db"
    law_data_path: str = "./data/knowledge/law/law_sample.json"
    index_dir: str = "./data/indexes"
    mcp_law_server_url: str = "http://127.0.0.1:8000/mcp/"
    mcp_debug_host: str = "127.0.0.1"
    mcp_debug_port: int = 8100
    mcp_tool_timeout_seconds: int = 30
    mcp_tool_discovery_retry_seconds: int = 30
    agent_max_tool_calls: int = 4
    agent_max_model_calls: int = 10
    agent_global_concurrency: int = 6
    agent_per_user_concurrency: int = 2
    agent_per_conversation_concurrency: int = 1
    agent_queue_timeout_seconds: int = 30
    llm_request_timeout_seconds: int = 60
    llm_max_retries: int = 2
    llm_temperature: float = 0.0
    memory_context_token_limit: int = 12000
    memory_compression_threshold: int = 9000
    memory_recent_message_count: int = 10
    memory_worker_poll_seconds: float = 1.0
    memory_job_max_attempts: int = 3
    memory_llm_model: str = ""
    memory_llm_thinking: bool = False
    memory_llm_temperature: float = 0.0
    memory_llm_max_tokens: int = 4096
    memory_llm_json_retry_count: int = 1
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    log_level: str = "INFO"
    log_dir: str = "./data/logs"
    log_max_bytes: int = 20971520
    log_backup_count: int = 10
    audit_summary_max_chars: int = 200
    index_auto_build: bool = True
    index_chunk_max_chars: int = 1000
    index_chunk_overlap_chars: int = 150
    index_build_batch_size: int = 20


@lru_cache
def get_settings() -> Settings:
    return Settings()
