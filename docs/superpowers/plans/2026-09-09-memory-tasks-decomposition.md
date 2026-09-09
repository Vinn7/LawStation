# memory_tasks.py Decomposition Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Split `backend/app/services/memory_tasks.py` (902 lines, single file) into a package `backend/app/services/memory_tasks/` using the same Mixin composition pattern already validated in the `backend/app/agent/graph.py` decomposition (sub-project 1), with zero runtime behavior change.

**Architecture:** Extract module-level constants/errors/helpers into standalone files, extract three logically distinct groups of `MemoryTaskManager` methods into three Mixin classes (`PersistenceMixin`, `SummaryMixin`, `InvocationMixin`), then assemble `MemoryTaskManager(PersistenceMixin, SummaryMixin, InvocationMixin)` in `manager.py` with a thin `__init__.py` facade re-exporting only `MemoryTaskManager`. The three real external call sites (`backend/app/main.py`, `backend/app/api/routes.py`, `backend/app/services/agent_runs.py`) only ever import `MemoryTaskManager` and never need to change.

**Tech Stack:** Python 3.12, SQLAlchemy (optimistic locking via `version` column), Pydantic, pytest + `monkeypatch`.

**Spec:** `docs/superpowers/specs/2026-09-09-memory-tasks-decomposition-design.md`

## Global Constraints

- **Byte-for-byte extraction.** Every method/function body moved by this plan must be copied verbatim from the current `backend/app/services/memory_tasks.py` — no reformatting, no "cleanup," no behavior change. (Verified via `grep`: this file, unlike `graph.py`, contains **zero** Chinese curly quotation marks (U+201C/U+201D) in any of the ranges this plan touches — there is nothing to accidentally normalize on that front, but byte-for-byte discipline still applies to every other character.)
- **Flat-module-shadows-package precedence (reused finding, not re-verified here).** CPython resolves a flat `x.py` module ahead of a same-named `x/` package directory. This was empirically verified during the `graph.py` decomposition (`docs/superpowers/plans/2026-09-08-agent-graph-decomposition.md`, Global Constraints section) and the same conclusion applies here unchanged: the task that creates `memory_tasks/__init__.py` alongside the still-existing `memory_tasks.py` must, in the SAME commit, also delete the old file — otherwise imports keep silently resolving to the stale flat file.
- **Baseline: 176 passed** (confirmed via `/opt/anaconda3/envs/LawStation/bin/python -m pytest -q` on `main` immediately before this plan was written, 2026-09-09). The full suite must show exactly `176 passed` after the activation task (Task 7) — this refactor changes zero test *count*, only *how* 10 of `tests/test_memory.py`'s `monkeypatch.setattr` calls target module paths.
- **Python executable:** always use `/opt/anaconda3/envs/LawStation/bin/python` — the sandbox default lacks this project's dependencies.
- **External interface unchanged.** `backend.app.services.memory_tasks.MemoryTaskManager` (import path + class name) must resolve exactly as before. `start`/`close`/`enqueue` signatures do not change.

---

### Task 1: 创建 `prompts.py`

**Files:**
- Create: `backend/app/services/memory_tasks/prompts.py`

**Interfaces:**
- Consumes: 无
- Produces: `SUMMARY_SYSTEM`、`EXTRACTION_SYSTEM` 两个字符串常量

- [ ] **Step 1: 建包目录和摘录写入 `prompts.py`**

```bash
mkdir -p backend/app/services/memory_tasks
```

从当前 `backend/app/services/memory_tasks.py` 第 34-45 行（`SUMMARY_SYSTEM` 34-35、`EXTRACTION_SYSTEM` 37-45，含两常量间的空行）逐字摘录，写入 `backend/app/services/memory_tasks/prompts.py`：

```python
"""Memory Tasks 使用的 Prompt 常量。"""

# ...此处粘贴第 34-45 行摘录内容，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/prompts.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/prompts.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/prompts.py
git commit -m "refactor: extract prompt constants from memory_tasks.py into memory_tasks/prompts.py"
```

---

### Task 2: 创建 `errors.py`

**Files:**
- Create: `backend/app/services/memory_tasks/errors.py`

**Interfaces:**
- Consumes: 无
- Produces: `MemoryProcessingError`（异常类）、`MemoryPersistStats`（dataclass）、`MemoryFailureDetails`（frozen dataclass）

- [ ] **Step 1: 摘录并写入 `errors.py`**

从当前 `backend/app/services/memory_tasks.py` 第 50-85 行（`MemoryProcessingError` 50-54、`MemoryPersistStats` 57-65、`MemoryFailureDetails` 68-85，含类间空行）逐字摘录，写入：

```python
"""Memory Tasks 的异常与结果数据类型。"""

from dataclasses import dataclass
from typing import Any

# ...此处粘贴第 50-85 行摘录内容，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/errors.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/errors.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/errors.py
git commit -m "refactor: extract error/result types from memory_tasks.py into memory_tasks/errors.py"
```

---

### Task 3: 创建 `helpers.py`

**Files:**
- Create: `backend/app/services/memory_tasks/helpers.py`

**Interfaces:**
- Consumes: `errors.py` 的 `MemoryProcessingError`、`MemoryFailureDetails`
- Produces: `_response_text`、`_safe_upstream_value`、`_failure_details`、`_utcnow`、`_canonical_key`（5 个纯函数）

- [ ] **Step 1: 摘录并写入 `helpers.py`**

从当前 `backend/app/services/memory_tasks.py` 第 88-198 行（`_response_text` 88-100、`_safe_upstream_value` 103-108、`_failure_details` 111-186、`_utcnow` 189-190、`_canonical_key` 193-198，含函数间空行）逐字摘录，写入：

```python
"""Memory Tasks 共用的纯函数：响应解析、失败分类、规范化 key。"""

import hashlib
import re
from datetime import UTC, datetime
from typing import Any

from sqlalchemy.exc import SQLAlchemyError

from backend.app.agent.provider import AgentConfigurationError

from .errors import MemoryFailureDetails, MemoryProcessingError

# ...此处粘贴第 88-198 行摘录内容，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/helpers.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/helpers.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/helpers.py
git commit -m "refactor: extract pure helper functions from memory_tasks.py into memory_tasks/helpers.py"
```

---

### Task 4: 创建 `invocation.py`

**Files:**
- Create: `backend/app/services/memory_tasks/invocation.py`

**Interfaces:**
- Consumes: `errors.py` 的 `MemoryProcessingError`；`helpers.py` 的 `_response_text`
- Produces: `InvocationMixin` Mixin 类，带 `_memory_model_name`（`@property`）和 `async def _invoke_structured_json(self, model, schema, system_prompt, payload, example, trace_config=None)` 两个方法；模块级 `StructuredResult = TypeVar("StructuredResult", bound=BaseModel)`。运行时依赖 `self.settings`（由 Task 7 的 `manager.py` 提供，本任务不需要、也不应该定义它）。

- [ ] **Step 1: 摘录并写入 `invocation.py`**

从当前 `backend/app/services/memory_tasks.py` 摘录以下两段内容：
1. 第 47 行（`StructuredResult = TypeVar(...)`）
2. 第 448-501 行（`_memory_model_name` 448-451，`_invoke_structured_json` 453-501，含中间空行）

逐字不改，组装进：

```python
"""Memory Tasks 的无工具结构化模型调用。"""

import json
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from .errors import MemoryProcessingError
from .helpers import _response_text

StructuredResult = TypeVar("StructuredResult", bound=BaseModel)


class InvocationMixin:
    # ...此处依次粘贴第 448-451 行（_memory_model_name）、第 453-501 行
    # （_invoke_structured_json），缩进不变（4 空格 def/async def、
    # 8 空格方法体），逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/invocation.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/invocation.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/invocation.py
git commit -m "refactor: extract structured invocation into memory_tasks/invocation.py"
```

---

### Task 5: 创建 `persistence.py`

**Files:**
- Create: `backend/app/services/memory_tasks/persistence.py`

**Interfaces:**
- Consumes: `errors.py` 的 `MemoryPersistStats`；`helpers.py` 的 `_canonical_key`、`_utcnow`
- Produces: `PersistenceMixin` Mixin 类，带 5 个方法：`_load_job`、`_complete_job`（`@staticmethod`）、`_persist_candidates`、`_replace_memory`、`_fail`。运行时依赖 `self.settings`（由 Task 7 的 `manager.py` 提供，`_fail` 用到 `self.settings.memory_job_max_attempts`）和 `self._wake`（`_fail` 用到 `self._wake.set()`），本任务不需要、也不应该定义它们。

- [ ] **Step 1: 摘录并写入 `persistence.py`**

从当前 `backend/app/services/memory_tasks.py` 摘录以下两段内容（顺序：`_load_job` → `_complete_job` → `_persist_candidates` → `_replace_memory` → `_fail`）：
1. 第 503-762 行（`_load_job` 503-556、`_complete_job` 558-571、`_persist_candidates` 573-697、`_replace_memory` 699-762，含方法间空行）
2. 第 886-902 行（`_fail`，文件末尾最后一个方法）

逐字不改，组装进：

```python
"""Memory Tasks 的任务状态 I/O 与乐观锁记忆持久化。"""

import logging
from typing import Any

from sqlalchemy import and_, or_, select, update
from sqlalchemy.exc import IntegrityError

from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob, MemoryRevision, Message, UserMemory
from backend.app.db.session import SessionLocal

from .errors import MemoryPersistStats
from .helpers import _canonical_key, _utcnow


class PersistenceMixin:
    # ...此处依次粘贴第 503-556 行（_load_job）、第 558-571 行（_complete_job，
    # 含 @staticmethod）、第 573-697 行（_persist_candidates）、第 699-762 行
    # （_replace_memory）、第 886-902 行（_fail），缩进不变（4 空格 def/async def、
    # 8 空格方法体），方法之间保留一个空行，逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/persistence.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/persistence.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/persistence.py
git commit -m "refactor: extract job persistence + optimistic-lock replacement into memory_tasks/persistence.py"
```

---

### Task 6: 创建 `summary.py`

**Files:**
- Create: `backend/app/services/memory_tasks/summary.py`

**Interfaces:**
- Consumes: `helpers.py` 的 `_utcnow`
- Produces: `SummaryMixin` Mixin 类，带一个方法 `async def _update_summary(self, model, job_data, *, trace_config=None) -> bool`。运行时依赖 `self.settings`（`memory_recent_message_count`/`memory_compression_threshold`）和 `self._invoke_structured_json`（由 `InvocationMixin` 提供，通过 Task 7 的多继承组合获得），本任务不需要、也不应该定义它们。

- [ ] **Step 1: 摘录并写入 `summary.py`**

从当前 `backend/app/services/memory_tasks.py` 第 764-884 行（`_update_summary` 方法完整内容，含 docstring 位置——该方法无独立 docstring，直接是代码）逐字摘录，写入：

```python
"""Memory Tasks 的增量结构化摘要生成。"""

import json

from sqlalchemy import and_, or_, select

from backend.app.core.logging import audit
from backend.app.db.models import ConversationSummary, Message
from backend.app.db.session import SessionLocal
from backend.app.services.memory import estimate_tokens
from backend.app.services.memory_schemas import StructuredConversationSummary

from .helpers import _utcnow


class SummaryMixin:
    # ...此处粘贴第 764-884 行摘录内容，缩进不变（4 空格 async def、
    # 8 空格方法体），逐字不改...
```

- [ ] **Step 2: 语法与 lint 检查**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/summary.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/summary.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 3: Commit**

```bash
git add backend/app/services/memory_tasks/summary.py
git commit -m "refactor: extract incremental summary generation into memory_tasks/summary.py"
```

---

### Task 7: 创建 `manager.py` + `__init__.py`，删除旧 `memory_tasks.py`，修复测试，全量验证（激活任务）

这是唯一一个会真正改变 `backend.app.services.memory_tasks` 导入解析目标的任务，前 6 个任务产出的文件在此之前都是"未激活"的。本任务必须一次性完成"新包补全 + 旧文件删除 + 测试文件修复"，中间不能拆分成多次独立 commit（拆分会造成一个两者都不完整解析、或测试全部报 `AttributeError` 的中间状态）。

**Files:**
- Create: `backend/app/services/memory_tasks/manager.py`
- Create: `backend/app/services/memory_tasks/__init__.py`
- Delete: `backend/app/services/memory_tasks.py`
- Modify: `tests/test_memory.py`

**Interfaces:**
- Consumes: Task 1-6 产出的全部符号（`prompts.py` 的 `EXTRACTION_SYSTEM`；`helpers.py` 的 `_failure_details`；`persistence.py` 的 `PersistenceMixin`；`summary.py` 的 `SummaryMixin`；`invocation.py` 的 `InvocationMixin`）
- Produces: `MemoryTaskManager` 类（供 `main.py`、`routes.py`、`agent_runs.py`、测试导入）；`backend/app/services/memory_tasks` 包对外的完整门面

- [ ] **Step 1: 摘录并写入 `manager.py`**

从当前 `backend/app/services/memory_tasks.py` 第 202-447 行（`__init__` 202-214、`start` 215-225、`close` 226-231、`enqueue` 232-263、`_run` 264-277、`_claim_next` 278-292、`_process` 293-447，含方法间空行）逐字摘录，写入 `backend/app/services/memory_tasks/manager.py`：

```python
"""Memory Tasks 包的编排入口：后台队列生命周期与单任务处理流程。

本模块只负责队列生命周期（start/close/enqueue/_run/_claim_next）和单任务
编排（_process：加载 -> 抽取 -> 持久化 -> 摘要 -> 完成/失败），具体的模型调用、
记忆持久化和摘要生成分别在 invocation.py/persistence.py/summary.py 里按职责
拆分，通过多继承组合进本文件的 MemoryTaskManager。
"""

import asyncio
import logging
import time
from typing import Any

from sqlalchemy import select, update
from sqlalchemy.exc import IntegrityError

from backend.app.agent.provider import LLMProvider
from backend.app.core.config import Settings, get_settings
from backend.app.core.context import RequestUserContext
from backend.app.core.logging import audit
from backend.app.db.models import MemoryJob
from backend.app.db.session import SessionLocal
from backend.app.observability import LangSmithObservability
from backend.app.services.memory_schemas import MemoryExtractionResult

from .helpers import _failure_details
from .invocation import InvocationMixin
from .persistence import PersistenceMixin
from .prompts import EXTRACTION_SYSTEM
from .summary import SummaryMixin


class MemoryTaskManager(PersistenceMixin, SummaryMixin, InvocationMixin):
    """管理记忆抽取后台任务队列，持有可复用的 Provider/Settings/Observability。

    类实例可以由多个请求共享；`enqueue` 只做去重和入队，实际处理由内部
    worker 任务串行执行。
    """

    # ...此处依次粘贴第 202-214 行（__init__）、第 215-225 行（start）、
    # 第 226-231 行（close）、第 232-263 行（enqueue）、第 264-277 行（_run）、
    # 第 278-292 行（_claim_next）、第 293-447 行（_process），六段内容逐字
    # 不改，方法之间保留一个空行...
```

- [ ] **Step 2: 摘录并写入 `__init__.py`**

```python
"""Memory Tasks 包的对外门面：只做 re-export，不放任何逻辑。"""

from .manager import MemoryTaskManager

__all__ = ["MemoryTaskManager"]
```

- [ ] **Step 3: 语法与 lint 检查（先于删除旧文件）**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m py_compile backend/app/services/memory_tasks/manager.py backend/app/services/memory_tasks/__init__.py && /opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/manager.py backend/app/services/memory_tasks/__init__.py`
Expected: 两条命令都无输出、退出码 0。

- [ ] **Step 4: 删除旧文件**

```bash
git rm backend/app/services/memory_tasks.py
```

- [ ] **Step 5: 修复 `tests/test_memory.py` 的 import 与 monkeypatch 路径**

这是本任务最容易出错的一步——**下表是逐个测试函数核对真实调用链后得出的精确映射，直接照做，不要重新猜测**：

**Step 5A: 顶部 import（第 28 行）**

```python
from backend.app.services.memory_tasks import MemoryTaskManager, _failure_details
```

改为：

```python
from backend.app.services.memory_tasks import MemoryTaskManager
from backend.app.services.memory_tasks.helpers import _failure_details
```

**Step 5B: 逐个测试函数的 monkeypatch 路径**（按当前文件行号定位，都是把字符串前缀 `"backend.app.services.memory_tasks."` 改成 `"backend.app.services.memory_tasks.<子模块>."`，属性名不变）：

| 当前行号 | 测试函数 | 调用链 | 原字符串 | 改为 |
|---|---|---|---|---|
| 200 | `test_background_extraction_auto_activates_all_valid_memories` | 调用 `manager._process(job.id)`；`_process` 本身不直接用 `SessionLocal`，但依次经过 `_load_job`（persistence）→`_persist_candidates`/`_replace_memory`（persistence）→`_update_summary`（summary，即使提前 return 也会先 `with SessionLocal()` 读消息数） | `"backend.app.services.memory_tasks.SessionLocal"` | 改成两行：`monkeypatch.setattr("backend.app.services.memory_tasks.persistence.SessionLocal", local_session)` 和 `monkeypatch.setattr("backend.app.services.memory_tasks.summary.SessionLocal", local_session)` |
| 268-271 | `test_model_target_replaces_different_canonical_key_in_place` | 直接调用 `manager._persist_candidates(...)`（persistence） | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 273-276 | 同上 | `_persist_candidates` 内部调用 `audit(...)`（persistence） | `"backend.app.services.memory_tasks.audit"` | `"backend.app.services.memory_tasks.persistence.audit"` |
| 321-324 | `test_invalid_cross_conversation_replacement_is_rejected` | 直接调用 `manager._persist_candidates(...)`（persistence） | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 359-362 | `test_user_memory_can_be_replaced_from_another_owned_conversation` | 同上 | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 401-404 | `test_model_cannot_replace_another_users_memory` | 同上 | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 437-440 | `test_timeline_events_with_different_keys_coexist` | 同上 | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 472-475 | `test_identical_active_memory_is_a_noop` | 同上 | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |
| 512-513 | `test_summary_uses_only_messages_after_previous_coverage` | 直接调用 `manager._update_summary(...)`（summary，不经过 `_process`） | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.summary.SessionLocal"` |
| 578-579 | `test_empty_memory_extraction_is_success` | 调用 `manager._process(job.id)`；经过 `_load_job`+`_persist_candidates`（persistence，即使 `memories=[]` 也会 `with SessionLocal()`）和 `_update_summary`（summary，提前 return 前先读消息数） | `"backend.app.services.memory_tasks.SessionLocal"` | 改成两行：`monkeypatch.setattr("backend.app.services.memory_tasks.persistence.SessionLocal", local_session)` 和 `monkeypatch.setattr("backend.app.services.memory_tasks.summary.SessionLocal", local_session)` |
| 678-679 | `test_tool_choice_compatibility_error_is_not_retried` | 调用 `manager._process(job.id)`；`provider.get_memory_model()` 在抽取阶段之前就抛异常，只经过 `_load_job`（persistence）和随后的 `_fail`（persistence），从未到达 `_persist_candidates`/`_update_summary` | `"backend.app.services.memory_tasks.SessionLocal"` | `"backend.app.services.memory_tasks.persistence.SessionLocal"` |

`test_invalid_json_is_retried_once`（第 634 行）和 `test_memory_failure_details_classifies_known_parameter_and_transport_errors`（第 696 行）本来就没有 `monkeypatch` 参数，不需要改动。

- [ ] **Step 6: 全量测试验证（这是第一次能真正验证行为的时刻）**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m pytest -q`
Expected: 全部通过，输出末尾为 `176 passed`（本计划编写时用同一命令测得的改动前基线，2026-09-09）。

若失败，最常见原因排查顺序：
1. `ImportError`/`ModuleNotFoundError` → 检查 `manager.py` 或某个子模块的 import 路径是否和本任务给定内容完全一致（相对导入层级 `.` 是否写对）。
2. `AttributeError: 'MemoryTaskManager' object has no attribute 'xxx'` → 说明某个 Mixin 类没有被正确组合进 `MemoryTaskManager` 的继承列表，检查 `manager.py` 的 class 声明。
3. `tests/test_memory.py` 里的测试报 `SessionLocal`/`audit` 相关的 `AttributeError` 或 patch 后行为看起来没生效（例如实际操作了真实数据库而不是测试用的内存库）→ 回去核对 Step 5B 的映射表，确认 patch 路径真的对应该测试实际会执行到的子模块。
4. 测试断言内容不一致（不是 Error 而是 Failure）→ 说明摘录时改动了原始逻辑，回去用 `git show <删除 memory_tasks.py 之前的 commit>:backend/app/services/memory_tasks.py`（可用 `git log --oneline -- backend/app/services/memory_tasks.py` 找到删除前最后一个 commit）与新文件逐行比对是否真的逐字一致。

- [ ] **Step 7: Lint 检查整个包**

Run: `/opt/anaconda3/envs/LawStation/bin/python -m ruff check backend/app/services/memory_tasks/`
Expected: `All checks passed!`

- [ ] **Step 8: 全仓扫描确认无孤立引用**

Run: `grep -rn "from backend.app.services.memory_tasks import\|from backend\.app\.services import memory_tasks\b" backend/ tests/ scripts/`
Expected: 只剩 4 处（`backend/app/main.py`、`backend/app/api/routes.py`（间接访问 `.enqueue`，不是 import）、`backend/app/services/agent_runs.py`（同样是间接访问）、`tests/test_memory.py`），且都能正常解析（Step 6 的 pytest 已经间接验证了这点）。实际直接写 `import` 语句的只有 `main.py` 和 `test_memory.py` 两处，`routes.py`/`agent_runs.py` 通过 `app.state.memory_tasks`/`self.memory_tasks` 属性访问，不在这个 grep 命中范围内，属于正常情况。

- [ ] **Step 9: 确认旧文件确实已经不存在**

Run: `ls backend/app/services/memory_tasks.py 2>&1; ls backend/app/services/memory_tasks/`
Expected: 第一条命令报 `No such file or directory`；第二条命令列出 8 个新文件（`__init__.py`、`prompts.py`、`errors.py`、`helpers.py`、`invocation.py`、`persistence.py`、`summary.py`、`manager.py`）。

- [ ] **Step 10: Commit**

```bash
git add backend/app/services/memory_tasks/manager.py backend/app/services/memory_tasks/__init__.py tests/test_memory.py
git commit -m "refactor: assemble MemoryTaskManager via mixins, remove old memory_tasks.py, fix test monkeypatch paths"
```

**如果本任务中途失败（例如 pytest 未达到 176 passed 且短时间内无法定位原因）：** 先运行 `git status` 确认工作区状态，不要在半完成状态下继续摸索。如果新文件已创建但 `memory_tasks.py` 尚未 `git rm`，可以先把发现记录进报告，交回给 dispatcher 决定下一步。

---

### Task 8: 同步文档

按 `ai-context/SPEC.md` 第 18 节的 Spec 维护规则，模块结构变化必须同步回写文档。

**Files:**
- Modify: `ai-context/SPEC.md`（第 18 节"关键模块阅读列表"第 7 项；第 18 节变更记录追加一条）
- Modify: `resume/review-findings.md`（第 154 行表格行）
- Modify: `resume/langsmith.md`（第 130 行）
- Modify: `resume/memory.md`（第 510-514 行，5 行表格）

**Interfaces:**
- Consumes: 无（纯文档任务）
- Produces: 无新代码符号

**特别注意（吸取子项目 1 的教训）：** 子项目 1（graph.py 拆分）的文档同步任务第一次只改了 3 篇明确列出的文件，遗漏了另外 3 篇引用旧路径的文件，被最终全分支评审发现后才补一轮修复。本任务已经用 `grep -rln "memory_tasks\.py" --include="*.md"` 提前扫描过全仓，下面列出的文件就是全部需要改的地方——不需要再自己重新扫描，但改完之后仍需按 Step 5 的 grep 做一次收尾确认。

- [ ] **Step 1: 更新 SPEC.md 关键模块阅读列表**

打开 `ai-context/SPEC.md`，找到第 611 行：

```
7. `backend/app/services/memory_tasks.py::MemoryTaskManager`：持久后台抽取与增量摘要。
```

改为：

```
7. `backend/app/services/memory_tasks/manager.py::MemoryTaskManager`：持久后台抽取与增量摘要。
```

（第 4 节第 80 行的模块职责表这一行本来就只列 `OwnedRepository`、`MemoryService` 两个符号，不提 `MemoryTaskManager`，不需要改。）

- [ ] **Step 2: 更新 `resume/review-findings.md`**

第 154 行：

```
| 4 | `backend/app/services/memory_tasks.py` | 异步记忆、最新事实覆盖和增量摘要 |
```

改为（整包路径，因为这行描述的是模块整体阅读顺序，不是单个 symbol）：

```
| 4 | `backend/app/services/memory_tasks/` | 异步记忆、最新事实覆盖和增量摘要 |
```

- [ ] **Step 3: 更新 `resume/langsmith.md`**

第 130 行：

```
- `backend/app/services/memory_tasks.py::MemoryTaskManager._process`
```

改为：

```
- `backend/app/services/memory_tasks/manager.py::MemoryTaskManager._process`
```

- [ ] **Step 4: 更新 `resume/memory.md`**

第 510-514 行，5 行表格，按 symbol 实际所在新文件分别修正路径（内容/说明列不变）：

```
| `backend/app/services/memory_tasks.py` | `MemoryTaskManager` | 持久化后台任务 Worker |
| `backend/app/services/memory_tasks.py` | `_invoke_structured_json` | JSON Output 与 Pydantic 校验 |
| `backend/app/services/memory_tasks.py` | `_persist_candidates` | 新增、去重和冲突判定 |
| `backend/app/services/memory_tasks.py` | `_replace_memory` | 原位替换和修订审计 |
| `backend/app/services/memory_tasks.py` | `_update_summary` | 增量结构化摘要 |
```

改为：

```
| `backend/app/services/memory_tasks/manager.py` | `MemoryTaskManager` | 持久化后台任务 Worker |
| `backend/app/services/memory_tasks/invocation.py` | `_invoke_structured_json` | JSON Output 与 Pydantic 校验 |
| `backend/app/services/memory_tasks/persistence.py` | `_persist_candidates` | 新增、去重和冲突判定 |
| `backend/app/services/memory_tasks/persistence.py` | `_replace_memory` | 原位替换和修订审计 |
| `backend/app/services/memory_tasks/summary.py` | `_update_summary` | 增量结构化摘要 |
```

- [ ] **Step 5: 收尾扫描确认无遗漏**

Run: `grep -rn "backend/app/services/memory_tasks\.py" --include="*.md" .`
Expected: 只剩 `docs/agent-learning/03-memory-context-engineering.md` 和两份 `docs/superpowers/specs/*.md`（`2026-09-08-agent-graph-decomposition-design.md`、`2026-09-09-memory-tasks-decomposition-design.md`）——`docs/agent-learning/` 下的文档是带时间戳的学习笔记快照，按子项目 1 全分支评审时的既定原则不做更新（更新行号锚点会让它们看起来像持续维护的文档，而不是特定时间点的快照）；两份 spec 设计文档描述的是拆分之前的状态，属于历史记录，同样不改。如果 grep 结果出现这 3 个文件之外的任何其它文件，说明本任务遗漏了引用，需要回去补上。

- [ ] **Step 6: SPEC.md 变更记录追加一条**

在 `ai-context/SPEC.md` 第 18 节"变更记录"末尾追加一条新记录，格式与该节现有条目保持一致（版本号在 4.5 基础上递增为 4.6、日期用今天，摘要说明）：

```
- **4.6 / 2026-09-09**：`backend/app/services/memory_tasks.py` 拆分为 `memory_tasks/` 包（`manager.py` + `persistence.py` + `summary.py` + `invocation.py` + `prompts.py` + `errors.py` + `helpers.py`），外部接口 `MemoryTaskManager` 和导入路径 `backend.app.services.memory_tasks` 不变，`tests/test_memory.py` 的 10 处 `SessionLocal`/`audit` monkeypatch 已同步改为具体子模块路径，全量测试 176 passed 验证通过。
```

- [ ] **Step 7: Commit**

```bash
git add ai-context/SPEC.md resume/review-findings.md resume/langsmith.md resume/memory.md
git commit -m "docs: sync SPEC.md and resume/ with memory_tasks/ package split"
```

---

## Self-Review 记录

- **Spec coverage**：spec 第 5 节（目标目录结构）的 8 个文件（`__init__.py`/`prompts.py`/`errors.py`/`helpers.py`/`persistence.py`/`summary.py`/`invocation.py`/`manager.py`）分别对应 Task 1-7，spec 第"关键设计决策"第 2 点（测试 monkeypatch 路径修复）对应 Task 7 Step 5，文档同步对应 Task 8，无遗漏。
- **Placeholder scan**：每个 Task 的新增内容（import 头、class 声明、verification 命令）都是完整可执行的具体内容；被搬移的既有代码用"从第 X-Y 行摘录"这种可机械执行、无歧义的方式指定；Task 7 Step 5B 的 monkeypatch 映射表是逐个测试函数读过真实调用链后得出的具体值，不是猜测。
- **Type consistency**：所有 Mixin 方法签名与当前源码逐字一致（未改名、未改参数顺序）；`self._invoke_structured_json` 在 Task 4 定义、在 Task 6 的 `_update_summary` 里原样调用（通过多继承在 Task 7 组合到位）；`self._replace_memory`/`self._load_job`/`self._complete_job`/`self._fail` 均在 Task 5 同一个 `PersistenceMixin` 内定义并互相调用，无跨文件方法名不一致问题。
- **已知风险**：Task 7 是唯一一个"改动量大、且必须原子完成"的任务（新增 2 文件 + 删除 1 文件 + 修改 1 个测试文件，必须同一个 commit），如果中途失败，`git status` 检查未提交的改动，必要时 `git checkout -- backend/app/services/memory_tasks.py` 恢复旧文件（因为 Task 7 之前它还在 git 历史里），不要在半完成状态下继续。Task 7 Step 5 的 monkeypatch 路径映射是本计划里认知负荷最高的一步，如果实施后 `test_background_extraction_auto_activates_all_valid_memories` 或 `test_empty_memory_extraction_is_success`（两个需要同时 patch 两个子模块的测试）失败，优先怀疑是否漏 patch 了 `summary.SessionLocal` 或 `persistence.SessionLocal` 其中一个。
