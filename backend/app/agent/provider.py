from langchain_openai import ChatOpenAI

from backend.app.core.config import Settings, get_settings


class AgentConfigurationError(RuntimeError):
    pass


class LLMProvider:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: ChatOpenAI | None = None

    def get_chat_model(self) -> ChatOpenAI:
        if not self.settings.deepseek_api_key:
            raise AgentConfigurationError("尚未配置 DEEPSEEK_API_KEY，暂时无法生成回答。")
        if self._model is None:
            self._model = ChatOpenAI(
                model=self.settings.deepseek_model,
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                streaming=True,
                temperature=self.settings.llm_temperature,
                timeout=self.settings.llm_request_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
            )
        return self._model
