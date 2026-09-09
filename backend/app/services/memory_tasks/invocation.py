"""Memory Tasks 的无工具结构化模型调用。"""

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import MemoryProcessingError
from .helpers import _response_text

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class InvocationMixin:
    @property
    def _memory_model_name(self) -> str:
        """返回配置中的模型标识；审计只记录名称，不记录地址或凭证。"""
        return self.settings.memory_llm_model or self.settings.deepseek_model

    async def _invoke_structured_json(
        self,
        model: Any,
        schema: type[StructuredResult],
        system_prompt: str,
        payload: dict[str, Any],
        example: dict[str, Any],
        trace_config: dict[str, Any] | None = None,
    ) -> StructuredResult:
        schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
        example_json = json.dumps(example, ensure_ascii=False)
        prompt = (
            f"{system_prompt}\n\n"
            "必须只返回一个符合 JSON Schema 的 JSON 对象，不得输出 Markdown、代码围栏或解释。\n"
            f"JSON Schema：{schema_json}\n"
            f"合法 JSON 示例：{example_json}"
        )
        runner = model.bind(response_format={"type": "json_object"})
        last_category = "empty_response"
        attempts = max(0, self.settings.memory_llm_json_retry_count) + 1
        for attempt in range(attempts):
            messages = [
                ("system", prompt),
                ("human", json.dumps(payload, ensure_ascii=False)),
            ]
            response = (
                await runner.ainvoke(messages, config=trace_config)
                if trace_config else await runner.ainvoke(messages)
            )
            raw = _response_text(response)
            if not raw:
                last_category = "empty_response"
            else:
                try:
                    decoded = json.loads(raw)
                except json.JSONDecodeError:
                    last_category = "invalid_json"
                else:
                    try:
                        return schema.model_validate(decoded)
                    except ValidationError as exc:
                        raise MemoryProcessingError(
                            "schema_validation_error",
                            "记忆模型返回的数据不符合结构要求",
                        ) from exc
            if attempt + 1 < attempts:
                continue
        message = "记忆模型返回空响应" if last_category == "empty_response" else "记忆模型返回无效 JSON"
        raise MemoryProcessingError(last_category, message)
