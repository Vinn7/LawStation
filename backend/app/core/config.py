from functools import lru_cache
from pathlib import Path
from typing import Literal

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
    embedding_provider: str = "ollama"
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimension: int = 1024
    ollama_auto_start: bool = True
    ollama_command: str = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_startup_timeout_seconds: float = 60.0
    ollama_shutdown_timeout_seconds: float = 10.0
    ollama_request_timeout_seconds: float = 300.0
    ollama_keep_alive: str = "30m"
    ollama_embedding_batch_size: int = 8
    ollama_log_path: str = "./data/logs/ollama.log"
    ollama_query_instruction: str = (
        "Given a legal consultation query, retrieve relevant Chinese laws and regulations "
        "that answer the query"
    )
    database_url: str = "sqlite:///./data/runtime/lawstation.db"
    law_data_path: str = "./data/knowledge/law/law.json"
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
    index_build_batch_size: int = 8
    index_embedding_timeout_seconds: float = 120.0
    index_embedding_max_retries: int = 5
    index_embedding_retry_base_seconds: float = 2.0
    index_embedding_retry_max_seconds: float = 30.0
    rag_bm25_min_score: float = 0.01
    rag_dense_min_score: float = 0.20
    rag_rrf_min_score: float = 0.01
    rag_retrieval_mode: Literal["bm25", "hybrid"] = "hybrid"
    agent_review_mode: Literal["always-llm", "auto"] = "auto"
    sse_heartbeat_seconds: float = 15.0
    langsmith_enabled: bool = False
    langsmith_runtime_mode: Literal["config", "all", "off"] = "config"
    langsmith_session_trace_limit: int = 200
    langsmith_strict_startup: bool = False
    langsmith_api_key: str = ""
    langsmith_endpoint: str = "https://api.smith.langchain.com"
    langsmith_workspace_id: str = ""
    langsmith_project: str = "lawstation"
    langsmith_environment: str = "development"
    langsmith_capture_content: bool = True
    langsmith_id_hash_secret: str = ""
    langsmith_trace_sample_rate: float = 0.02
    langsmith_error_trace_enabled: bool = True
    langsmith_flush_timeout_seconds: float = 5.0
    langsmith_evaluator_model: str = ""
    langsmith_evaluator_base_url: str = ""
    langsmith_evaluator_api_key: str = ""
    langsmith_evaluator_temperature: float = 0.0
    langsmith_online_eval_sample_rate: float = 0.0
    langsmith_annotation_queue_id: str = ""
    langsmith_monthly_production_trace_budget: int = 20
    eval_monthly_trace_budget: int = 60
    eval_monthly_agent_model_call_budget: int = 80
    eval_monthly_judge_call_budget: int = 20
    eval_resource_budget_path: str = "./data/runtime/eval-resource-budget.json"
    langsmith_test_cache: str = "./data/runtime/langsmith-test-cache"
    eval_report_root: str = "./evals/reports/runs"
    eval_report_timezone: str = "Asia/Singapore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
