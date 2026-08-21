import json
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from backend.app.core.config import Settings, get_settings


class JudgeScores(BaseModel):
    legal_issue_coverage: float = Field(ge=1, le=5)
    evidence_consistency: float = Field(ge=1, le=5)
    factual_fidelity: float = Field(ge=1, le=5)
    risk_calibration: float = Field(ge=1, le=5)
    completeness: float = Field(ge=1, le=5)
    actionability: float = Field(ge=1, le=5)
    clarity: float = Field(ge=1, le=5)
    helpfulness: float = Field(ge=1, le=5)
    comment: str = Field(default="", max_length=500)


class LegalQualityJudge:
    def __init__(self, settings: Settings | None = None) -> None:
        settings = settings or get_settings()
        key = settings.langsmith_evaluator_api_key or settings.deepseek_api_key
        if not key:
            raise RuntimeError("未配置 LANGSMITH_EVALUATOR_API_KEY 或 DEEPSEEK_API_KEY")
        self.model = ChatOpenAI(
            model=settings.langsmith_evaluator_model or settings.deepseek_model,
            api_key=key,
            base_url=settings.langsmith_evaluator_base_url or settings.deepseek_base_url,
            temperature=settings.langsmith_evaluator_temperature,
            streaming=False,
            extra_body={"thinking": {"type": "disabled"}},
        ).bind(response_format={"type": "json_object"})

    async def __call__(self, run: Any, example: Any) -> list[dict[str, Any]]:
        outputs = getattr(run, "outputs", {}) or {}
        inputs = getattr(example, "inputs", {}) or {}
        reference = getattr(example, "outputs", {}) or {}
        schema = JudgeScores.model_json_schema()
        prompt = (
            "你是法律咨询质量评审员。只能根据用户输入、EvidencePacket、最终回答和参考要求评分，"
            "不得用自身法律常识替代证据包判断具体法条。每项1到5分，只返回JSON，不输出推理过程。\n"
            f"JSON Schema: {json.dumps(schema, ensure_ascii=False)}"
        )
        response = await self.model.ainvoke([
            ("system", prompt),
            ("human", json.dumps({"inputs": inputs, "outputs": outputs, "reference": reference}, ensure_ascii=False, default=str)),
        ])
        raw = response.content if isinstance(response.content, str) else ""
        scores = JudgeScores.model_validate(json.loads(raw))
        values = scores.model_dump(exclude={"comment"})
        return [
            {"key": f"judge_{key}", "score": value / 5, "comment": scores.comment}
            for key, value in values.items()
        ]
