# memory_tasks.py 模块拆分设计

## Context

这是 2026-09-08 `backend/app/agent/graph.py` 拆分（子项目 1，已完成并合并）确认的"排查确认了四个候选文件"计划中的子项目 2。原设计文档（`docs/superpowers/specs/2026-09-08-agent-graph-decomposition-design.md`）已记录：`backend/app/services/memory_tasks.py`（约 900 行）——记忆抽取状态机、乐观锁原位替换、增量摘要全在一处，留作后续独立的 brainstorming 迭代。

`memory_tasks.py` 现状（902 行，单文件，无子目录）：
- 2 个 Prompt 常量（`SUMMARY_SYSTEM`、`EXTRACTION_SYSTEM`）
- 3 个模块级错误/数据类（`MemoryProcessingError`、`MemoryPersistStats`、`MemoryFailureDetails`）
- 5 个模块级纯函数（`_response_text`、`_safe_upstream_value`、`_failure_details`、`_utcnow`、`_canonical_key`）
- 1 个大类 `MemoryTaskManager`，包含队列生命周期（`__init__`/`start`/`close`/`enqueue`/`_run`/`_claim_next`）、单任务编排（`_process`，156 行）、无工具结构化调用（`_memory_model_name`/`_invoke_structured_json`）、任务状态 I/O（`_load_job`/`_complete_job`）、乐观锁记忆持久化（`_persist_candidates`/`_replace_memory`，约 190 行）、增量摘要（`_update_summary`，122 行）、失败处理（`_fail`）

外部消费者（唯一真实调用方）：
- `backend/app/main.py`：`from backend.app.services.memory_tasks import MemoryTaskManager`，调用 `.start()`/`.close()`
- `backend/app/api/routes.py`：`request.app.state.memory_tasks.enqueue(...)`
- `backend/app/services/agent_runs.py`：`self.memory_tasks.enqueue(...)`
- `tests/test_memory.py`：`from backend.app.services.memory_tasks import MemoryTaskManager, _failure_details`，另有 10 处 `monkeypatch.setattr("backend.app.services.memory_tasks.SessionLocal"/".audit", ...)` 直接 patch 模块级名字

## 目标

- 把 `memory_tasks.py` 按职责拆分为多个高内聚的小文件，优先解决占比最大的 `_process`（156 行，单任务编排）与 `_persist_candidates`/`_replace_memory`（约 190 行，乐观锁记忆持久化）两块。
- 与子项目 1（graph.py）保持同样的 Mixin 组合模式：`MemoryTaskManager` 从多个 Mixin 类继承，行为零变化，纯结构重组。
- 外部接口不变：`backend.app.services.memory_tasks.MemoryTaskManager` 这个导入路径和类名必须继续可用（`memory_tasks.py` 变成 `memory_tasks/` 包，`__init__.py` 门面 re-export）。
- 全量测试必须在拆分前后行为完全一致（改动前基线由 Task 0 实测记录，见实施计划）。

## 不做的事

- 不涉及 `routes.py`/`App.tsx` 的拆分（留作后续独立子项目）。
- 不改变记忆抽取、乐观锁替换、增量摘要的任何业务逻辑或 SQL 语句。
- 不改变 `MemoryJob`/`UserMemory`/`ConversationSummary`/`MemoryRevision` 等 ORM 模型。
- 不改变对外可见的方法签名（`start`/`close`/`enqueue`）。

## 目标目录结构

```
backend/app/services/memory_tasks/
├── __init__.py       # 门面：re-export MemoryTaskManager
├── prompts.py         # SUMMARY_SYSTEM, EXTRACTION_SYSTEM
├── errors.py           # MemoryProcessingError, MemoryPersistStats, MemoryFailureDetails
├── helpers.py           # _response_text, _safe_upstream_value, _failure_details, _utcnow, _canonical_key
├── persistence.py        # PersistenceMixin: _load_job, _complete_job, _persist_candidates, _replace_memory, _fail
├── summary.py              # SummaryMixin: _update_summary
├── invocation.py            # InvocationMixin: _memory_model_name, _invoke_structured_json
└── manager.py                # MemoryTaskManager(PersistenceMixin, SummaryMixin, InvocationMixin): __init__/start/close/enqueue/_run/_claim_next/_process
```

每个文件的精确源码行范围（对照当前 `backend/app/services/memory_tasks.py`，逐字摘录搬移，不改动任何代码逻辑）：

| 目标文件 | 源文件行范围 | 内容 |
|---|---|---|
| `prompts.py` | 34-45 | `SUMMARY_SYSTEM`（34-35）、`EXTRACTION_SYSTEM`（37-45） |
| `errors.py` | 50-85 | `MemoryProcessingError`（50-54）、`MemoryPersistStats`（57-65）、`MemoryFailureDetails`（68-85） |
| `helpers.py` | 88-198 | `_response_text`（88-100）、`_safe_upstream_value`（103-108）、`_failure_details`（111-186）、`_utcnow`（189-190）、`_canonical_key`（193-198） |
| `persistence.py`（`PersistenceMixin`） | 503-762, 886-902 | `_load_job`（503-556）、`_complete_job`（558-571，`@staticmethod`）、`_persist_candidates`（573-697）、`_replace_memory`（699-762）、`_fail`（886-902） |
| `summary.py`（`SummaryMixin`） | 764-884 | `_update_summary` |
| `invocation.py`（`InvocationMixin`） | 448-501 | `_memory_model_name`（448-451，`@property`）、`_invoke_structured_json`（453-501） |
| `manager.py`（`MemoryTaskManager`） | 202-447 | `__init__`（202-214）、`start`（215-225）、`close`（226-231）、`enqueue`（232-263）、`_run`（264-277）、`_claim_next`（278-292）、`_process`（293-447） |

`StructuredResult = TypeVar("StructuredResult", bound=BaseModel)`（第 47 行）随 `_invoke_structured_json` 一起搬到 `invocation.py`。

## 关键设计决策

### 1. Mixin 组合，与子项目 1 一致

`manager.py` 里的 `MemoryTaskManager(PersistenceMixin, SummaryMixin, InvocationMixin)` 持有 `self.provider`/`self.settings`/`self.observability`/`self._wake`/`self._worker`/`self._stopping` 等实例状态；三个 Mixin 通过 `self.xxx` 访问这些属性和彼此的方法（例如 `_persist_candidates` 调用 `self._replace_memory`，`_process` 调用 `self._load_job`/`self._invoke_structured_json`/`self._persist_candidates`/`self._update_summary`/`self._complete_job`/`self._fail`）。这是子项目 1 已经验证过、评审通过的模式，不再重新论证。

### 2. `tests/test_memory.py` 的 monkeypatch 路径必须同步更新

这是与子项目 1 不同的、真实存在的复杂点。当前测试文件里有 10 处直接 patch 模块级名字：

```python
monkeypatch.setattr("backend.app.services.memory_tasks.SessionLocal", local_session)  # ×9
monkeypatch.setattr("backend.app.services.memory_tasks.audit", ...)                    # ×1
```

拆分后 `SessionLocal` 会被 `manager.py`（`start`/`enqueue`/`_claim_next`）、`persistence.py`（`_load_job`/`_complete_job`/`_persist_candidates`/`_fail`）、`summary.py`（`_update_summary`）三个文件各自独立 `from backend.app.db.session import SessionLocal`。Python 的 `from X import Y` 在导入时把值复制进当前模块的命名空间，之后 `monkeypatch.setattr("...memory_tasks.SessionLocal", ...)`（只 patch 包门面 `__init__.py` 里的名字）**不会**影响三个子文件里已经各自绑定的 `SessionLocal`。

**决策（已与用户确认）：不引入间接层，保持生产代码的常规写法**（每个文件按需直接 import，不做延迟属性访问之类的技巧）。测试文件按实际命中的代码路径改成 patch 具体子模块：

- 只调用 `enqueue`/`start`/`_claim_next` 的测试 → patch `backend.app.services.memory_tasks.manager.SessionLocal`
- 只测 `_persist_candidates`/`_replace_memory`/`_load_job`/`_complete_job`/`_fail` 的测试 → patch `backend.app.services.memory_tasks.persistence.SessionLocal`
- 只测 `_update_summary` 的测试 → patch `backend.app.services.memory_tasks.summary.SessionLocal`
- 端到端跑 `_process`（例如 `test_background_extraction_auto_activates_all_valid_memories`）的测试 → 按该测试实际会经过的模块组合，同时 patch 多个路径（`manager` 一定要 patch，因为 `_process`/`_claim_next` 在其中；`persistence`、`summary` 视测试是否会真正走到 `_persist_candidates`/`_update_summary` 而定）
- `audit` 的 1 处 patch 同理，改成 patch 该测试实际命中方法所在的子模块路径

具体每条 `monkeypatch.setattr` 改成哪个路径，需要在实施阶段逐个测试用例核对其调用链后确定（不能靠猜测），实施计划里会把这个核对工作列成显式步骤。

`from backend.app.services.memory_tasks import MemoryTaskManager, _failure_details` 这行也要改：`_failure_details` 挪到 `helpers.py` 后，门面 `__init__.py` 不 re-export 它（只有测试直接用，遵循子项目 1"门面只导出真正外部消费者需要的符号"的先例），测试改成 `from backend.app.services.memory_tasks.helpers import _failure_details`。

### 3. `__init__.py` 门面

```python
"""Memory Tasks 包的对外门面：只做 re-export，不放任何逻辑。"""

from .manager import MemoryTaskManager

__all__ = ["MemoryTaskManager"]
```

`main.py`/`routes.py`/`agent_runs.py` 三处真实消费者只用到 `MemoryTaskManager` 这一个符号，门面只需要 re-export 它。

### 4. Python "同名文件优先于同名包" 约束（沿用子项目 1 已验证的全局约束）

激活任务（新建 `memory_tasks/` 包 + 删除旧 `memory_tasks.py`）必须是单次原子 commit，理由与子项目 1 完全相同：CPython 里同名的 `.py` 模块文件优先于同名包目录被导入，中途状态下二者同时存在会导致导入解析到旧文件而不是新包，讨论过程已经在子项目 1 的实施计划里用真实实验验证过，此处直接复用结论。

## 验证方式

1. 全量 `pytest` 通过，测试数量与拆分前基线完全一致（不多不少）。
2. `ruff check backend/app/services/memory_tasks/` 全部通过。
3. `ls backend/app/services/memory_tasks.py` 报错不存在；`ls backend/app/services/memory_tasks/` 显示 8 个新文件。
4. 全仓 `grep` 确认无 `from backend.app.services.memory_tasks import` 之外的孤立引用残留（例如某处直接 `import backend.app.services.memory_tasks` 后又访问已经不存在于门面里的符号）。
5. 中文注释里的弯引号（U+201C/U+201D）逐字保留——子项目 1 的实施过程反复踩过这个坑，实施计划会像子项目 1 一样在开工前先跑一次全文件弯引号位置核对表。

## Self-Review

- **占位符扫描**：本文档每个文件的行范围、导入列表、Mixin 名称都是从当前真实文件内容核实得出的具体值，没有"待补充"类描述。
- **内部一致性**：`persistence.py`/`summary.py`/`manager.py` 之间的方法调用关系（`self.xxx`）与当前源码逐一核对一致；`invocation.py` 的 `_invoke_structured_json` 被 `manager.py`（`_process` 内的抽取阶段）和 `summary.py`（`_update_summary`）两处调用，两者都在 Mixin 组合后的同一个类上，行为不变。
- **范围检查**：聚焦单个子项目（`memory_tasks.py`），不涉及 `routes.py`/`App.tsx`，符合"每个子项目独立走完整设计+实施周期"的既定安排。
