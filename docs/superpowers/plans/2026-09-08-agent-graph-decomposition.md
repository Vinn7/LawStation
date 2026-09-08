# Agent Graph 模块拆分 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 把 `backend/app/agent/graph.py`（1170 行）拆分成 `backend/app/agent/graph/` 包（9 个文件），不改变任何运行时行为，外部 3 处 import（`runtime.py`、`evaluation/targets.py`、`tests/test_agent_runtime.py`）保持不变。

**Architecture:** 5 个 Prompt 常量 → `prompts.py`；12 个模块级纯函数 → `evidence.py`；6 个节点方法按业务阶段拆成 5 个 Mixin 类（`nodes/*.py`）；`__init__`/`_compile`/`_invoke_json`/3 个路由函数留在 `orchestrator.py`，通过多继承组合出最终的 `LegalConsultationGraph(CaseAnalystNode, ResearchNode, CounselNode, ReviewNode, FinalizeNode)`；`__init__.py` 只做 re-export 门面。

**Tech Stack:** Python 3.12、LangGraph `StateGraph`、LangChain `create_agent`、Pydantic、pytest。

**Spec:** [docs/superpowers/specs/2026-09-08-agent-graph-decomposition-design.md](../specs/2026-09-08-agent-graph-decomposition-design.md)

## Global Constraints

- 这是纯结构重组，**不允许修改任何被搬移代码的行为**——函数体、Prompt 文案、注释、docstring 一律逐字保留，只允许改动 import 语句和外层 `class` 包装。
- 外部 import 路径不变：`from backend.app.agent.graph import LegalConsultationGraph` 和 `tests/test_agent_runtime.py` 里现有的 `from backend.app.agent.graph import (ANALYST_PROMPT, COUNSEL_PROMPT, REVIEW_PROMPT, _authoritative_evidence, _citation_errors, _no_match_violations)` 必须继续解析成功，一个字符都不改。
- 真实验证环境：`/opt/anaconda3/envs/LawStation/bin/python`（本机唯一装了项目全部依赖的解释器），本任务所有 `pytest`/`ruff` 命令都必须用这个解释器执行，不要用系统默认 `python`/`python3`。
- **关键的 Python 导入约束（已实测验证）**：`backend/app/agent/graph.py`（文件）和 `backend/app/agent/graph/`（目录）不能同时被当作同一个可导入模块解析——Python 会让**扁平 `.py` 文件优先于同名包目录**（已用一个隔离的临时脚本验证过这个行为）。这意味着：
  - Task 1-7 期间，`backend/app/agent/graph.py` 保持原样不动、继续是唯一生效的导入目标；新建的 `backend/app/agent/graph/` 目录及其文件在此期间是"死代码"，不会被任何真实代码路径导入，因此这几个任务**无法用 pytest 做行为验证**，只能做语法/静态检查（`ruff check` + `python3 -m py_compile`）。这不是偷懒，是这个迁移方式本身的限制，第 8 个任务（删除旧文件、激活新包）才是第一次能跑通全量 pytest 的时间点。
  - 删除 `backend/app/agent/graph.py` 的那一刻，`backend.app.agent.graph` 才会开始解析为新包——这个删除动作必须和"新包已经完全就绪"发生在同一个任务里，不能提前删、也不能拖后。

## 摘录约定

以下任务里"从 `graph.py` 第 X-Y 行摘录"的意思是：用 Read 工具读取当前 `backend/app/agent/graph.py` 的第 X 到 Y 行（`offset=X limit=(Y-X+1)`），**逐字**拷贝到新文件里对应位置，不得从记忆重新打字、不得意译、不得"顺手"修正措辞或格式。每个任务里给出的 import 头和 `class` 包装是本计划新决定的内容，必须按给定内容原样写入。

---

### Task 1: 创建 `prompts.py`

**Files:**
- Create: `backend/app/agent/graph/prompts.py`
- Modify: 无（本任务不动 `graph.py`）

**Interfaces:**
- Consumes: 无
- Produces: `ANALYST_PROMPT`、`RESEARCH_PROMPT`、`COUNSEL_PROMPT`、`REVIEW_PROMPT`、`EVIDENCE_SELECTOR_PROMPT`（5 个字符串常量，供后续所有任务导入）

- [ ] **Step 1: 建目录**

```bash
mkdir -p backend/app/agent/graph
```

- [ ] **Step 2: 摘录并写入 `prompts.py`**

从当前 `backend/app/agent/graph.py` 第 74-126 行（5 个 Prompt 常量及其上方注释，从 `# case_analyst 使用；不绑定工具` 开始到 `EVIDENCE_SELECTOR_PROMPT` 三重引号结束）逐字摘录，写入 `backend/app/agent/graph/prompts.py`，文件顶部加一行模块 docstring：

```python
"""三 Agent 与证据选择器使用的 Prompt 常量。"""

# ...此处粘贴第 74-126 行摘录内容，逐字不改...
```

- [ ] **Step 3: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/prompts.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/prompts.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 4: Commit**

```bash
git add backend/app/agent/graph/prompts.py
git commit -m "refactor: extract prompt constants from graph.py into graph/prompts.py"
```

---

### Task 2: 创建 `evidence.py`

**Files:**
- Create: `backend/app/agent/graph/evidence.py`

**Interfaces:**
- Consumes: 无（不依赖 Task 1 产物，`evidence.py` 与 `prompts.py` 互相独立）
- Produces: `_message_text`、`_extract_json`、`_json_values`、`_tool_documents`、`_authoritative_evidence`、`_no_match_disclosure_present`、`_no_match_violations`、`_no_match_safe_answer`、`_validate_fact_overrides`、`_fact_boundary_errors`、`_citation_errors`、`_payload`（12 个函数）

- [ ] **Step 1: 摘录并写入 `evidence.py`**

从当前 `backend/app/agent/graph.py` 第 128-365 行（`_message_text` 到 `_payload` 共 12 个函数）逐字摘录函数体，文件结构为：

```python
"""Graph 节点共用的纯函数：JSON 解析、证据权威映射、边界校验。"""

import json
import logging
import re
from typing import Any

from langchain_core.messages import BaseMessage, ToolMessage
from sqlalchemy import or_, select

from backend.app.agent.schemas import CounselDraft, EvidenceItem, EvidencePacket
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit
from backend.app.db.models import UserMemory
from backend.app.db.session import SessionLocal

# ...此处粘贴第 128-365 行摘录内容，逐字不改（12 个函数原样保留，包括
# _payload 内部对同文件 _message_text 的调用，不需要额外处理）...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/evidence.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/evidence.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/graph/evidence.py
git commit -m "refactor: extract evidence/JSON helper functions from graph.py into graph/evidence.py"
```

---

### Task 3: 创建 `nodes/case_analyst.py`

**Files:**
- Create: `backend/app/agent/graph/nodes/__init__.py`（空文件，仅用于把 `nodes` 标记为包）
- Create: `backend/app/agent/graph/nodes/case_analyst.py`

**Interfaces:**
- Consumes: `evidence.py` 的 `_message_text`、`_payload`、`_validate_fact_overrides`；`prompts.py` 的 `ANALYST_PROMPT`
- Produces: `CaseAnalystNode` Mixin 类，带一个方法 `async def case_analyst(self, state, runtime) -> dict[str, Any]`。运行时依赖 `self._invoke_json`（由 Task 8 的 `orchestrator.py` 提供，本任务不需要、也不应该定义它）。

- [ ] **Step 1: 建空的包标记文件**

```bash
touch backend/app/agent/graph/nodes/__init__.py
```

- [ ] **Step 2: 摘录并写入 `case_analyst.py`**

从当前 `backend/app/agent/graph.py` 第 492-581 行（`case_analyst` 方法完整内容，含 docstring）逐字摘录，写入：

```python
"""Case Analyst 节点：案情分类、事实整理与研究规划。"""

import asyncio
from typing import Any

from langgraph.runtime import Runtime
from pydantic import ValidationError

from backend.app.agent.schemas import AgentError, CaseAnalysis, ResearchTask
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _message_text, _payload, _validate_fact_overrides
from ..prompts import ANALYST_PROMPT


class CaseAnalystNode:
    # ...此处粘贴第 492-581 行摘录内容，缩进不变（仍是 4 空格 async def、
    # 8 空格方法体），逐字不改...
```

- [ ] **Step 3: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/nodes/case_analyst.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/nodes/case_analyst.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 4: Commit**

```bash
git add backend/app/agent/graph/nodes/__init__.py backend/app/agent/graph/nodes/case_analyst.py
git commit -m "refactor: extract case_analyst node into graph/nodes/case_analyst.py"
```

---

### Task 4: 创建 `nodes/research.py`

**Files:**
- Create: `backend/app/agent/graph/nodes/research.py`

**Interfaces:**
- Consumes: `evidence.py` 的 `_authoritative_evidence`、`_payload`、`_tool_documents`；`prompts.py` 的 `RESEARCH_PROMPT`、`EVIDENCE_SELECTOR_PROMPT`
- Produces: `ResearchNode` Mixin 类，带一个方法 `async def legal_researcher(self, state, runtime) -> dict[str, Any]`。运行时依赖 `self.research_agent`、`self.tool_names`、`self.settings`、`self._invoke_json`（均由 `orchestrator.py` 提供）。

这是本次拆分收益最大的一块（243 行，占原文件 21%），必须逐字摘录，不做任何"顺手优化"。

- [ ] **Step 1: 摘录并写入 `research.py`**

从当前 `backend/app/agent/graph.py` 第 590-831 行（`legal_researcher` 方法完整内容，含 docstring 和 try/except）逐字摘录，写入：

```python
"""Legal Research 节点：唯一绑定 MCP 工具的 Agent，生成可验证 EvidencePacket。"""

import json
import logging
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langgraph.runtime import Runtime

from backend.app.agent.schemas import (
    AgentError,
    EvidenceItem,
    EvidencePacket,
    EvidenceSelectionResult,
    UnresolvedIssue,
)
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit, summary

from ..evidence import _authoritative_evidence, _payload, _tool_documents
from ..prompts import EVIDENCE_SELECTOR_PROMPT, RESEARCH_PROMPT


class ResearchNode:
    # ...此处粘贴第 590-831 行摘录内容，缩进不变，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/nodes/research.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/nodes/research.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/graph/nodes/research.py
git commit -m "refactor: extract legal_researcher node into graph/nodes/research.py"
```

---

### Task 5: 创建 `nodes/counsel.py`

**Files:**
- Create: `backend/app/agent/graph/nodes/counsel.py`

**Interfaces:**
- Consumes: `evidence.py` 的 `_no_match_disclosure_present`、`_no_match_safe_answer`、`_payload`；`prompts.py` 的 `COUNSEL_PROMPT`
- Produces: `CounselNode` Mixin 类，带一个方法 `async def legal_counsel(self, state, runtime) -> dict[str, Any]`。运行时依赖 `self._invoke_json`。

- [ ] **Step 1: 摘录并写入 `counsel.py`**

从当前 `backend/app/agent/graph.py` 第 833-894 行（`legal_counsel` 方法完整内容）逐字摘录，写入：

```python
"""Legal Counsel 节点：基于案情与 EvidencePacket 生成待复核法律意见草稿。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import AgentError, CounselDraft
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _no_match_disclosure_present, _no_match_safe_answer, _payload
from ..prompts import COUNSEL_PROMPT


class CounselNode:
    # ...此处粘贴第 833-894 行摘录内容，缩进不变，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/nodes/counsel.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/nodes/counsel.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/graph/nodes/counsel.py
git commit -m "refactor: extract legal_counsel node into graph/nodes/counsel.py"
```

---

### Task 6: 创建 `nodes/review.py`

**Files:**
- Create: `backend/app/agent/graph/nodes/review.py`

**Interfaces:**
- Consumes: `evidence.py` 的 `_citation_errors`、`_fact_boundary_errors`、`_no_match_violations`、`_payload`；`prompts.py` 的 `REVIEW_PROMPT`
- Produces: `ReviewNode` Mixin 类，带两个方法 `async def reviewer(self, state, runtime)` 和 `async def review_gate(self, state, runtime)`。运行时依赖 `self._invoke_json`、`self.settings`。

`reviewer`（第 896-966 行）和 `review_gate`（第 968-1040 行）在原文件里是紧挨着的两个方法（中间只隔一个空行），一次性摘录。

- [ ] **Step 1: 摘录并写入 `review.py`**

从当前 `backend/app/agent/graph.py` 第 896-1040 行（`reviewer` 和 `review_gate` 两个方法完整内容）逐字摘录，写入：

```python
"""Review 阶段：确定性快速路径（review_gate）与 LLM 复核（reviewer）。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import AgentError, ReviewResult
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.logging import audit

from ..evidence import _citation_errors, _fact_boundary_errors, _no_match_violations, _payload
from ..prompts import REVIEW_PROMPT


class ReviewNode:
    # ...此处粘贴第 896-1040 行摘录内容（reviewer 方法 + review_gate 方法，
    # 顺序与原文件一致），缩进不变，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/nodes/review.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/nodes/review.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/graph/nodes/review.py
git commit -m "refactor: extract reviewer and review_gate nodes into graph/nodes/review.py"
```

---

### Task 7: 创建 `nodes/finalize.py`

**Files:**
- Create: `backend/app/agent/graph/nodes/finalize.py`

**Interfaces:**
- Consumes: `evidence.py` 的 `_no_match_safe_answer`、`_no_match_violations`
- Produces: `FinalizeNode` Mixin 类，带一个方法 `async def finalize(self, state, runtime) -> dict[str, Any]`。不依赖 `self.` 上任何其它属性/方法。

- [ ] **Step 1: 摘录并写入 `finalize.py`**

从当前 `backend/app/agent/graph.py` 第 1079-1170 行（`finalize` 方法完整内容，到文件末尾）逐字摘录，写入：

```python
"""Finalize 节点：纯代码安全出口，收敛回答并生成可追踪 Citation。"""

from typing import Any

from langgraph.runtime import Runtime

from backend.app.agent.schemas import Citation
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState

from ..evidence import _no_match_safe_answer, _no_match_violations


class FinalizeNode:
    # ...此处粘贴第 1079-1170 行摘录内容，缩进不变，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/nodes/finalize.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/nodes/finalize.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/agent/graph/nodes/finalize.py
git commit -m "refactor: extract finalize node into graph/nodes/finalize.py"
```

---

### Task 8: 创建 `orchestrator.py` + `__init__.py`，删除旧 `graph.py`，全量验证（激活任务）

这是唯一一个会真正改变 `backend.app.agent.graph` 导入解析目标的任务，前 7 个任务产出的文件在此之前都是"未激活"的。本任务必须一次性完成"新包补全 + 旧文件删除"，中间不能拆分成多次独立 commit（拆分会造成一个两者都不完整解析的中间状态）。

**Files:**
- Create: `backend/app/agent/graph/orchestrator.py`
- Create: `backend/app/agent/graph/__init__.py`
- Delete: `backend/app/agent/graph.py`

**Interfaces:**
- Consumes: Task 1-7 产出的全部符号（`prompts.py` 的 5 个常量、`evidence.py` 的 `_extract_json`/`_message_text`、5 个 `nodes/*.py` 的 5 个 Mixin 类）
- Produces: `LegalConsultationGraph` 类（供 `runtime.py`、`evaluation/targets.py`、测试导入）；`backend/app/agent/graph` 包对外的完整门面

- [ ] **Step 1: 摘录并写入 `orchestrator.py`**

从当前 `backend/app/agent/graph.py` 摘录以下四段内容（顺序为：`__init__` → `_compile` → `_invoke_json` → 三个路由函数），逐字不改，组装进一个文件：

1. 第 376-423 行（`__init__` 方法）
2. 第 425-463 行（`_compile` 方法）
3. 第 465-490 行（`_invoke_json` 方法）
4. 第 583-588 行（`after_analysis` 方法）
5. 第 1042-1049 行（`after_review_gate` 静态方法，含 `@staticmethod` 装饰器）
6. 第 1051-1077 行（`after_review` 方法）

写入 `backend/app/agent/graph/orchestrator.py`：

```python
"""LawStation 法律咨询的 LangGraph 编排入口：图拓扑组装与跨节点共享调用。

本模块只负责“图怎么连”和节点间共享的基础设施（无工具结构化调用），六个节点
的业务逻辑分别在 nodes/ 目录下按阶段拆分，通过多继承组合进本文件的
LegalConsultationGraph。

``legal_researcher``（nodes/research.py）节点内部使用 LangChain
``create_agent``。该 Agent 才会让 DeepSeek 自主产生 tool_calls、经 MCP 执行
工具、接收 ToolMessage，并继续调用模型形成研究结论；模型自主决定检索次数和
参数，不由代码替它决定。``response_format=ToolStrategy(EvidencePacket)`` 强制
它在结束检索后必须通过结构化工具（而不是自由文本）汇报结果：LangChain 在这种
模式下对每一轮模型调用都设置 ``tool_choice="required"``，模型在这个子 Agent
里物理上不能返回纯文本。

主路径为 ``START -> case_analyst -> [finalize | legal_researcher] ->
legal_counsel -> review_gate -> [finalize | reviewer]``。Reviewer 最多把状态送回
Research 或 Counsel 各一次，最后由 ``finalize`` 执行确定性证据边界校验。

``LegalConsultationState`` 是节点间传递且可被 LangGraph Checkpoint 持久化的数据；
``AgentInvocationContext`` 则保存当前请求的身份、调用计数和审计对象，不属于跨节点
业务状态，也不得成为跨轮会话记忆。
"""

import json
import logging
import time
from typing import Any, TypeVar

from langchain.agents import create_agent
from langchain.agents.middleware import ModelCallLimitMiddleware
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from pydantic import BaseModel

from backend.app.agent.middleware import InvocationModelLimitMiddleware, ToolAuditMiddleware
from backend.app.agent.registry import MCPToolRegistry
from backend.app.agent.schemas import EvidencePacket
from backend.app.agent.state import AgentInvocationContext, LegalConsultationState
from backend.app.core.config import Settings
from backend.app.core.logging import audit, summary

from .evidence import _extract_json, _message_text
from .nodes.case_analyst import CaseAnalystNode
from .nodes.counsel import CounselNode
from .nodes.finalize import FinalizeNode
from .nodes.research import ResearchNode
from .nodes.review import ReviewNode

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LegalConsultationGraph(
    CaseAnalystNode, ResearchNode, CounselNode, ReviewNode, FinalizeNode,
):
    """编译并持有可复用的三 Agent LangGraph。

    类实例可以由多个请求共享；每次 ``compiled.astream`` 接收的 State 与
    ``AgentInvocationContext`` 仍相互隔离。只有 Research 节点需要 LangChain
    create_agent，因为只有它允许模型自主选择 MCP 工具。
    """

    # ...此处依次粘贴第 376-423 行（__init__）、第 425-463 行（_compile）、
    # 第 465-490 行（_invoke_json）、第 583-588 行（after_analysis）、
    # 第 1042-1049 行（after_review_gate，含 @staticmethod）、
    # 第 1051-1077 行（after_review），六段内容逐字不改，方法之间保留一个空行...
```

- [ ] **Step 2: 摘录并写入 `__init__.py`**

```python
"""Agent Graph 包的对外门面：只做 re-export，不放任何逻辑。"""

from .evidence import _authoritative_evidence, _citation_errors, _no_match_violations
from .orchestrator import LegalConsultationGraph
from .prompts import (
    ANALYST_PROMPT,
    COUNSEL_PROMPT,
    EVIDENCE_SELECTOR_PROMPT,
    RESEARCH_PROMPT,
    REVIEW_PROMPT,
)

__all__ = [
    "LegalConsultationGraph",
    "ANALYST_PROMPT",
    "COUNSEL_PROMPT",
    "EVIDENCE_SELECTOR_PROMPT",
    "RESEARCH_PROMPT",
    "REVIEW_PROMPT",
    "_authoritative_evidence",
    "_citation_errors",
    "_no_match_violations",
]
```

- [ ] **Step 3: 语法与 lint 检查（先于删除旧文件）**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/agent/graph/orchestrator.py backend/app/agent/graph/__init__.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/orchestrator.py backend/app/agent/graph/__init__.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 4: 删除旧文件**

```bash
git rm backend/app/agent/graph.py
```

- [ ] **Step 5: 全量测试验证（这是第一次能真正验证行为的时刻）**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m pytest -q`
Expected: 全部通过，输出末尾为 `176 passed`（这是本计划编写时用同一命令测得的改动前基线，2026-09-08）——测试数量不多不少才算通过，不能是"大部分通过"。

若失败，最常见原因排查顺序：
1. `ImportError`/`ModuleNotFoundError` → 检查 `orchestrator.py` 或某个 `nodes/*.py` 的 import 路径是否和本任务给定内容完全一致（相对导入层级 `.`/`..` 是否写对）。
2. `AttributeError: 'LegalConsultationGraph' object has no attribute 'xxx'` → 说明某个 Mixin 类没有被正确组合进 `LegalConsultationGraph` 的继承列表，检查 `orchestrator.py` 的 class 声明。
3. 测试断言内容不一致（不是 Error 而是 Failure）→ 说明摘录时改动了原始逻辑，回去逐行比对 `git show HEAD~8:backend/app/agent/graph.py`（改动前的旧文件内容）与新文件是否真的逐字一致。

- [ ] **Step 6: Lint 检查整个包**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/agent/graph/`
Expected: `All checks passed!`

- [ ] **Step 7: 全仓扫描确认无孤立引用**

Run: `grep -rn "from backend.app.agent.graph import\|from backend\.app\.agent import graph\b" backend/ tests/ scripts/ mcp_servers/`
Expected: 只剩 3 处（`runtime.py`、`evaluation/targets.py`、`tests/test_agent_runtime.py`），且都能正常解析（Step 5 的 pytest 已经间接验证了这点）。

- [ ] **Step 8: 确认旧文件确实已经不存在**

Run: `ls backend/app/agent/graph.py 2>&1; ls backend/app/agent/graph/`
Expected: 第一条命令报 `No such file or directory`；第二条命令列出 9 个新文件（`__init__.py`、`prompts.py`、`evidence.py`、`orchestrator.py`、`nodes/__init__.py`、`nodes/case_analyst.py`、`nodes/research.py`、`nodes/counsel.py`、`nodes/review.py`、`nodes/finalize.py`——共 10 项，含 `nodes` 子目录本身显示为一行）。

- [ ] **Step 9: Commit**

```bash
git add backend/app/agent/graph/orchestrator.py backend/app/agent/graph/__init__.py
git commit -m "refactor: assemble LegalConsultationGraph via mixins, remove old graph.py"
```

---

### Task 9: 同步文档

按 `ai-context/SPEC.md` 第 18 节的 Spec 维护规则和 `.agents/skills/lawstation-spec-change` 的交付闭环要求，模块结构变化必须同步回写文档。

**Files:**
- Modify: `ai-context/SPEC.md`（第 4 节目录和模块职责表）
- Modify（按需核验）: `resume/agent.md`、`resume/three-agent-behavior.md`、`resume/architecture.md`

**Interfaces:**
- Consumes: 无（纯文档任务）
- Produces: 无新代码符号

- [ ] **Step 1: 更新 SPEC.md 模块职责表**

在 `ai-context/SPEC.md` 第 4 节的表格里，找到 `backend/app/agent/` 这一行，把"关键 symbol / 文件"列里对 `graph.py` 单文件的描述，改为反映新的包结构（`graph/orchestrator.py::LegalConsultationGraph`、`graph/nodes/`、`graph/prompts.py`、`graph/evidence.py`）。

- [ ] **Step 2: 核验 resume 文档**

打开 `resume/agent.md`、`resume/three-agent-behavior.md`、`resume/architecture.md`，检查是否有直接提到 `backend/app/agent/graph.py` 具体行号或"单文件"描述的地方。如果有，同步改成新路径；如果这几篇本来就只在概念层面描述三 Agent 行为、没有提具体文件结构，则在交付说明里明确写"架构文档已核验，无需修改"，不能默认跳过不提。

- [ ] **Step 3: SPEC.md 变更记录追加一条**

在 `ai-context/SPEC.md` 第 18 节"变更记录"末尾追加一条新版本号 + 日期 + 摘要，说明"`backend/app/agent/graph.py` 拆分为 `graph/` 包，外部接口不变"。

- [ ] **Step 4: Commit**

```bash
git add ai-context/SPEC.md resume/
git commit -m "docs: sync SPEC.md and resume/ with graph/ package split"
```

---

## Self-Review 记录

- **Spec coverage**：spec 第 5 节的 9 个文件（`__init__.py`/`prompts.py`/`evidence.py`/`orchestrator.py`/`nodes/` 下 6 个）分别对应 Task 1-2-3-4-5-6-7-8，spec 第 8 节的文档同步对应 Task 9，无遗漏。
- **Placeholder scan**：每个 Task 的新增内容（import 头、class 声明、verification 命令）都是完整可执行的具体内容；被搬移的既有代码用"从第 X-Y 行摘录"这种可机械执行、无歧义的方式指定，不是含糊描述。
- **Type consistency**：所有节点方法签名统一为 `(self, state, runtime)`，`self._invoke_json` 的签名 `(self, runtime, agent_name, prompt, payload, schema)` 在 Task 8 定义、在 Task 3-7 的各节点里原样调用，未出现改名不一致。
- **已知风险**：Task 8 是唯一一个"改动量大、且必须原子完成"的任务，如果中途失败，`git status` 检查未提交的改动、必要时 `git checkout -- backend/app/agent/graph.py` 恢复旧文件（因为 Task 8 之前它还在 git 历史里），不要在半完成状态下继续。
