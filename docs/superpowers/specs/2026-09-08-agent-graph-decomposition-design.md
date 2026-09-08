# LawStation Agent Graph 模块拆分设计

> 状态：已通过用户确认的设计（brainstorming 阶段产出），尚未进入实施计划。
> 范围：仅 `backend/app/agent/graph.py` 一个文件的内部结构重组。

## 1. 背景与动机

LawStation 已经在用 Spec-Driven Development（`ai-context/SPEC.md` 是 Living Spec，`.agents/skills/lawstation-spec-change` 固化了"改动前读 Spec、改动后同步回写"的闭环）。用户希望在继续按 SDD 节奏迭代新功能之前，先解决现有代码里"单文件职责过重"的问题，让高频改动的核心模块更符合高内聚、低耦合的原则。

排查确认了四个候选文件：`backend/app/agent/graph.py`（1170 行）、`backend/app/api/routes.py`（约 756 行）、`backend/app/services/memory_tasks.py`（约 900 行）、`frontend/src/App.tsx`（1315 行）。这四者职责相对独立，一次性通盘重构会让单份 spec 过于庞大、也违背"最小变更"原则，因此决定拆成多个独立子项目，每个单独走一轮设计。**本文档只覆盖第一个子项目：`backend/app/agent/graph.py`**，其余三个留作后续独立的 brainstorming 迭代（见第 7 节）。

## 2. 目标与非目标

**目标：**
- 把 `graph.py` 按职责拆分为多个高内聚的小文件，其中优先解决占比最大的 `legal_researcher` 节点（243 行，占全文件 21%）。
- 拆分后不改变任何运行时行为——这是一次纯结构重组，不是功能修改。
- 外部对这个模块的 3 处依赖（`backend/app/agent/runtime.py`、`backend/app/evaluation/targets.py`、`tests/test_agent_runtime.py`）保持现有 import 路径不变。

**非目标（本次不做）：**
- 不涉及 `routes.py`/`memory_tasks.py`/`App.tsx` 的拆分（留作后续子项目）。
- 不改变 `LegalConsultationGraph` 对外的公共接口、构造参数或方法签名。
- 不引入新的类型检查工具（mypy/pyright）作为本次交付的一部分，即便下文提到的 Mixin 属性可见性问题在严格类型检查下会报警。
- 不新增业务逻辑、不修改 Prompt 文案、不调整证据校验规则。

## 3. 现状测量

| 部分 | 行数 | 占比 |
|---|---|---|
| 6 个节点方法 + 3 个路由函数 | ~678 行 | 58% |
| 其中单独 `legal_researcher` | 243 行 | 21% |
| 5 个模块级纯函数（JSON 解析/证据映射/边界校验） | ~236 行 | 20% |
| 5 个 Prompt 常量 | ~52 行 | 4% |
| 类脚手架（`__init__`/`_compile`/`_invoke_json`） | ~124 行 | 11% |

外部依赖排查结果：

- `backend/app/agent/runtime.py` — 仅导入 `LegalConsultationGraph`
- `backend/app/evaluation/targets.py` — 仅导入 `LegalConsultationGraph`
- `tests/test_agent_runtime.py` — 导入 `LegalConsultationGraph`、`ANALYST_PROMPT`、`COUNSEL_PROMPT`、`REVIEW_PROMPT`，以及三个下划线开头的私有辅助函数：`_authoritative_evidence`、`_citation_errors`、`_no_match_violations`（作为独立单元测试目标）

## 4. 方案比较

评估了三个方案，选定方案 B：

- **方案 A（最小拆分）**：只把纯函数（`evidence.py`）和 Prompt 常量（`prompts.py`）挪出去，`LegalConsultationGraph` 类本体留在原文件。改动最小，但拆完仍剩 ~880 行，`legal_researcher` 的 243 行没有解决，未命中核心痛点。
- **方案 B（采用）：按 Agent 阶段拆分节点，Mixin 组合类**。见第 5 节。
- **方案 C（完全函数式）**：节点从绑定方法改成显式传参的普通函数，解耦最彻底但改动面最大，与项目"最小必要改动"的既有原则冲突较多，作为后续可选的更激进重构保留，本次不采用。

方案 B 被选中的原因：真正解决 `legal_researcher` 这个最大的单一职责块，同时外部三处依赖零改动，`_compile()` 的节点注册代码不需要任何调整，符合"最小必要改动"原则。已知取舍：Mixin 里引用只在最终类 `__init__` 定义的 `self.settings`/`self.model` 等属性，在未来若接入 mypy/pyright 严格模式会报"属性未定义"，需要额外补一个 Protocol 基类声明共享属性；本项目目前没有类型检查步骤，此问题暂不影响交付。

## 5. 目标架构

```text
backend/app/agent/graph/
    __init__.py        # 门面：只做 re-export，不放逻辑
    prompts.py          # 5 个 Prompt 常量
    evidence.py           # 12 个模块级纯函数
    orchestrator.py        # LegalConsultationGraph 主体：__init__ / _compile / _invoke_json + 3 个路由函数
    nodes/
        __init__.py         # 空文件，仅用于把 nodes 标记为包
        case_analyst.py    # CaseAnalystNode（91 行）
        research.py          # ResearchNode（243 行）
        counsel.py             # CounselNode（63 行）
        review.py                # ReviewNode：review_gate + reviewer（147 行）
        finalize.py                # FinalizeNode（92 行）
```

原 `backend/app/agent/graph.py` 单文件删除，被上述目录取代。

### 5.1 各文件职责

- **`prompts.py`**：`ANALYST_PROMPT`、`RESEARCH_PROMPT`、`COUNSEL_PROMPT`、`REVIEW_PROMPT`、`EVIDENCE_SELECTOR_PROMPT` 五个常量，零依赖，原样从 `graph.py` 迁移，不改文案。
- **`evidence.py`**：`_message_text`、`_extract_json`、`_json_values`、`_tool_documents`、`_authoritative_evidence`、`_no_match_disclosure_present`、`_no_match_violations`、`_no_match_safe_answer`、`_validate_fact_overrides`、`_fact_boundary_errors`、`_citation_errors`、`_payload`——12 个纯函数，不依赖 `LegalConsultationGraph` 实例，原样迁移，函数体不改。
- **`nodes/case_analyst.py`**：`class CaseAnalystNode:` 内含 `async def case_analyst(self, state, runtime)`，从 `..evidence`/`..prompts` 按需导入。
- **`nodes/research.py`**：`class ResearchNode:` 内含 `async def legal_researcher(self, state, runtime)`，是本次拆分收益最大的一块。
- **`nodes/counsel.py`**：`class CounselNode:` 内含 `async def legal_counsel(self, state, runtime)`。
- **`nodes/review.py`**：`class ReviewNode:` 内含 `async def review_gate(self, state, runtime)` 和 `async def reviewer(self, state, runtime)`——两者放同一文件是因为它们同属"复核阶段"这一个业务概念，只是一个是确定性快速路径、一个是真正的 LLM 复核，机制不同但阶段相同。
- **`nodes/finalize.py`**：`class FinalizeNode:` 内含 `async def finalize(self, state, runtime)`。
- **`orchestrator.py`**：
  ```python
  from .nodes.case_analyst import CaseAnalystNode
  from .nodes.research import ResearchNode
  from .nodes.counsel import CounselNode
  from .nodes.review import ReviewNode
  from .nodes.finalize import FinalizeNode

  class LegalConsultationGraph(
      CaseAnalystNode, ResearchNode, CounselNode, ReviewNode, FinalizeNode,
  ):
      def __init__(self, model, tools, registry, settings, checkpointer=None) -> None: ...
      def _compile(self): ...
      async def _invoke_json(self, runtime, agent_name, prompt, payload, schema): ...
      def after_analysis(self, state) -> str: ...
      @staticmethod
      def after_review_gate(state) -> str: ...
      def after_review(self, state) -> str: ...
  ```
  路由函数紧挨着 `_compile()`，保持"图拓扑 + 路由决策"在同一文件里可以一眼读完的特性。
- **`__init__.py`**：
  ```python
  from .orchestrator import LegalConsultationGraph
  from .prompts import (
      ANALYST_PROMPT, RESEARCH_PROMPT, COUNSEL_PROMPT, REVIEW_PROMPT, EVIDENCE_SELECTOR_PROMPT,
  )
  from .evidence import _authoritative_evidence, _citation_errors, _no_match_violations
  ```
  只做 re-export，不放任何逻辑；导出的私有函数集合与第 3 节测出的"测试实际使用的符号"完全一致，不多不少。

### 5.2 运行时组合机制

各 Mixin 方法签名与现状完全一致（`self, state, runtime`）。运行时 `self` 是组合出的完整 `LegalConsultationGraph` 实例，`self.model`/`self.settings`/`self.registry`/`self._invoke_json` 等属性/方法由 `orchestrator.py::__init__` 统一赋值，各 Mixin 文件不重复定义 `__init__`，也不定义任何状态。Mixin 之间没有同名方法冲突，MRO（方法解析顺序）不影响实际行为。`_compile()` 中的节点注册代码（如 `graph.add_node("case_analyst", self.case_analyst)`）不需要任何修改。

## 6. 错误处理与测试

**错误处理**：本次是纯结构重组，不引入任何新行为。每个节点方法内部现有的 `try/except` 逻辑原样跟随该节点迁移到对应 Mixin 文件，不做任何调整、不改变异常类型或降级路径。

**测试**：
- `tests/test_agent_runtime.py` 中 `from backend.app.agent.graph import (...)` 一行不需要修改，依赖门面导出的符号与现状完全一致。
- 迁移完成后必须执行的验证：
  1. `pytest -v`（真实环境：`/opt/anaconda3/envs/LawStation/bin/python -m pytest -q`）全部通过，数量与迁移前一致。
  2. `ruff check backend/app/agent/` 干净。
  3. 手工确认 `backend/app/agent/graph.py`（旧文件）已被删除，不再存在新旧两份并存的情况。
  4. `grep -rn "from backend.app.agent.graph import\|from backend\.app\.agent import graph"` 全仓扫描，确认所有引用路径依旧解析成功（Python import 是否报错以 pytest 收集阶段是否失败为准）。
- 可选、非本次强制项：把 `test_agent_runtime.py` 里针对 `_authoritative_evidence`/`_citation_errors`/`_no_match_violations` 的用例迁到独立的 `tests/test_graph_evidence.py`，让测试文件边界跟随源码边界对齐。这不阻塞本次交付，留给用户后续按需决定。

## 7. 后续子项目（不在本次范围内）

以下三个文件的拆分作为独立子项目，各自需要单独走一轮 brainstorming（问题澄清 → 方案比较 → 设计 → spec），不在本设计文档覆盖范围内：

1. `backend/app/api/routes.py`（约 756 行）——CRUD、SSE 长连接主链路、审计埋点、LangSmith trace 混在一起；`stream_message` 内部还有五处重复的错误收尾逻辑。
2. `backend/app/services/memory_tasks.py`（约 900 行）——记忆抽取状态机、乐观锁原位替换、增量摘要全在一处。
3. `frontend/src/App.tsx`（1315 行）——用户/会话路由、多会话后台 SSE 流状态管理、场景观察执行器三种职责糅合在一个组件里。

## 8. 交付后需要同步的文档

按 `.agents/skills/lawstation-spec-change` 和 `ai-context/SPEC.md` 第 18 节的规则，实施完成后需要：

- 更新 `ai-context/SPEC.md` 第 4 节（目录和模块职责表）中 `backend/app/agent/` 一行的关键 symbol 描述，反映新的文件结构。
- 核验并按需更新 `resume/agent.md`、`resume/three-agent-behavior.md`、`resume/architecture.md`（这几篇提到了 `graph.py` 的具体结构）。
- 若最终确认无需修改 `resume/architecture.md` 的总体架构描述，也需要在交付说明里明确标注"架构文档已核验，无需修改"，而不是默认跳过。

这些文档同步动作属于实施计划（下一步 `writing-plans`）的验收项之一，本设计文档只做提醒，不在此处展开具体文案。
