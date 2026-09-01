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
    ollama_max_loaded_models: int = 2
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
    agent_skills_enabled: bool = True
    agent_skill_root: str = "./skills/runtime"
    agent_max_active_skills: int = 2
    agent_skill_strict_validation: bool = True
    agent_global_concurrency: int = 6
    agent_per_user_concurrency: int = 2
    agent_per_conversation_concurrency: int = 1
    agent_queue_timeout_seconds: int = 30
    agent_run_worker_poll_seconds: float = 0.5
    agent_run_lease_seconds: int = 120
    agent_run_recovery_max_attempts: int = 2
    agent_run_event_retention_days: int = 7
    langgraph_checkpoint_enabled: bool = True
    langgraph_checkpoint_path: str = "./data/runtime/langgraph-checkpoints.db"
    langgraph_checkpoint_retention_days: int = 7
    langgraph_strict_msgpack: bool = True
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
    rag_match_gate_enabled: bool = True
    rag_match_gate_config_path: str = "./evals/config/retrieval-gate-v1.json"
    rag_match_gate_required: bool = False
    rag_retrieval_mode: Literal["bm25", "hybrid"] = "hybrid"
    rag_rerank_enabled: bool = True
    rag_rerank_provider: Literal["tei", "ollama"] = "tei"
    rag_rerank_model: str = "BAAI/bge-reranker-v2-m3"
    rag_rerank_model_revision: str = ""
    rag_rerank_base_url: str = "http://127.0.0.1:8081"
    rag_rerank_auto_start: bool = True
    rag_rerank_command: str = "text-embeddings-router"
    rag_rerank_model_cache: str = "./data/models/huggingface"
    rag_rerank_startup_timeout_seconds: float = 900.0
    rag_rerank_shutdown_timeout_seconds: float = 10.0
    rag_rerank_log_path: str = "./data/logs/reranker.log"
    rag_rerank_max_client_batch_size: int = 16
    rag_rerank_max_batch_requests: int = 1
    rag_rerank_candidate_count: int = 12
    rag_rerank_concurrency: int = 2
    rag_rerank_min_score: float = 0.0
    rag_rerank_stage_timeout_seconds: float = 20.0
    rag_rerank_request_timeout_seconds: float = 30.0
    rag_rerank_top_logprobs: int = 20
    rag_rerank_keep_alive: str = "30m"
    rag_rerank_retry_seconds: float = 60.0
    rag_rerank_required: bool = False
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
    eval_resource_budget_path: str = "./data/runtime/eval-resource-budget.json"
    langsmith_test_cache: str = "./data/runtime/langsmith-test-cache"
    eval_report_root: str = "./evals/reports/runs"
    eval_report_timezone: str = "Asia/Singapore"
    test_scenarios_enabled: bool = False
    test_scenario_data_path: str = "./evals/conversations/lawstation-dialogue-scenarios-v1.jsonl"
    test_scenario_data_paths: list[str] = []
    test_scenario_step_timeout_seconds: float = 60.0
    test_scenario_generator_model: str = ""
    test_scenario_generator_temperature: float = 0.7
    test_scenario_generator_max_calls: int = 18
    test_scenario_variants_per_blueprint: int = 2
    test_scenario_generator_seed: int = 42


@lru_cache
def get_settings() -> Settings:
    return Settings()
