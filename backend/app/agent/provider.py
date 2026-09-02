"""集中构造可复用的 LangChain ChatOpenAI 客户端。"""

from langchain_openai import ChatOpenAI

from backend.app.core.config import Settings, get_settings


class AgentConfigurationError(RuntimeError):
    pass


class LLMProvider:
    """缓存模型客户端配置，但不保存消息、用户或 Agent State。"""
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._model: ChatOpenAI | None = None
        self._memory_model: ChatOpenAI | None = None

    def get_chat_model(self) -> ChatOpenAI:
        """返回连接 DeepSeek OpenAI-compatible API 的主 Agent 模型。"""
        if not self.settings.deepseek_api_key:
            raise AgentConfigurationError("尚未配置 DEEPSEEK_API_KEY，暂时无法生成回答。")
        if self._model is None:
            # ChatOpenAI 是 LangChain Provider 封装；之后可被直接 ainvoke，也可由
            # create_agent 绑定 MCP BaseTool。streaming 不代表所有节点都输出原始 token。
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

    def get_memory_model(self) -> ChatOpenAI:
        """返回记忆提取/摘要专用的无工具、非 Thinking 模型。"""
        if not self.settings.deepseek_api_key:
            raise AgentConfigurationError("尚未配置 DEEPSEEK_API_KEY，暂时无法整理记忆。")
        if self.settings.memory_llm_thinking:
            raise AgentConfigurationError("MEMORY_LLM_THINKING 必须为 false。")
        if self._memory_model is None:
            # 记忆模型绝不 bind_tools，显式关闭 Thinking，避免 JSON Output 与
            # tool_choice 冲突；它与法律咨询 Graph 是独立调用链。
            #
            # langchain-openai 1.6 会把构造器参数 max_tokens 重命名为
            # max_completion_tokens，但 DeepSeek Chat Completions 接口接受的是
            # max_tokens。通过 extra_body 传递可保留 DeepSeek 原生字段名，同时
            # 避免影响主 Agent 的 Thinking 和工具调用配置。
            self._memory_model = ChatOpenAI(
                model=self.settings.memory_llm_model or self.settings.deepseek_model,
                api_key=self.settings.deepseek_api_key,
                base_url=self.settings.deepseek_base_url,
                streaming=False,
                temperature=self.settings.memory_llm_temperature,
                timeout=self.settings.llm_request_timeout_seconds,
                max_retries=self.settings.llm_max_retries,
                extra_body={
                    "thinking": {"type": "disabled"},
                    "max_tokens": self.settings.memory_llm_max_tokens,
                },
            )
        return self._memory_model
