# 分层记忆系统与 Context 工程（Memory / Context Engineering）

> 面向：熟悉"缓存"、"过期策略"这类概念，但没接触过 LLM 应用里"记忆"设计的后端工程师。
> 目标：理解为什么记忆不能简单等于"把历史消息存进数据库再原样塞回 Prompt"，以及本项目怎么做隔离和冲突消解。

## 1. 要解决的问题

LLM 每次调用的上下文窗口（能塞进 Prompt 的 token 数）是有限的，但用户体验上又希望它"记得"：
- 这个会话之前聊过的内容（避免重复问同样的问题）
- 跨会话的偏好/背景（比如用户是租客还是房东，之前说过自己在哪个城市）

同时又要防止：
- 一个用户的会话内容"串"进另一个用户的上下文（隐私/数据隔离）
- 案件 A 的具体事实被错误地带进案件 B 的分析里（不同案件之间的事实污染）
- 历史记忆和用户本轮刚说的话冲突时，模型该信谁

## 2. 行业内一般怎么做

- **最简单粗暴**：把完整历史对话原样拼进每次 Prompt。短会话可行，长会话很快就超预算，而且历史越长、无关信息越多，反而稀释模型对当前问题的注意力（业内常说的"lost in the middle"问题）。
- **滑动窗口**：只保留最近 N 条消息，早期的直接丢弃。简单，但会丢掉早期提到的、之后仍然重要的信息（比如第一条消息里说的关键背景）。
- **摘要压缩（Summarization）**：定期把较早的对话压缩成一段摘要，滚动更新，替代原始消息塞进 Prompt。信息密度更高，但压缩是有损的，摘要生成本身还要额外调用一次模型。
- **向量检索式记忆（RAG-as-Memory）**：把历史对话/事实向量化存进向量库，每次按当前问题做语义检索，只取相关的片段。灵活，但"相关性"判断依赖检索质量，可能漏掉语义上不直接相似但业务上重要的信息。
- **结构化长期记忆**：把"记忆"当成结构化数据显式抽取和管理（用户偏好、关键事实各自建模，带状态/版本），而不是简单存一堆文本片段。管理成本更高，但可控性、可解释性最好。

本项目组合使用了**滑动窗口 + 摘要压缩 + 结构化长期记忆**，并且额外加了严格的**作用域隔离**——这是本项目区别于很多简化版 Demo 的地方。

## 3. 核心机制原理

四层记忆，从"临时"到"持久"：

1. **原始消息**：完整存库，作为事实源头
2. **近期对话**：按 token 预算选取当前会话最近若干条
3. **工作记忆（会话摘要）**：当前会话的结构化滚动摘要，超过压缩阈值时更新
4. **长期记忆**：区分"用户级偏好"（跨会话复用）和"案件级事实"（只在所属会话使用）

四层各自分配一部分 token 预算，拼进最终 Prompt；后台异步任务负责把每轮对话里值得沉淀的内容抽取进第 3、4 层，不阻塞主回答。

## 4. 本项目具体实现（函数级）

### 4.1 预算分配：`MemoryService.snapshot()`

[memory.py:85-185](../../backend/app/services/memory.py#L85)：

```python
total_budget = self.settings.memory_context_token_limit
available = max(0, total_budget - estimate_tokens(question))
content_budget = max(0, available - min(200, available))
recent_budget = int(content_budget * 0.50)
case_budget = int(content_budget * 0.25)
profile_budget = int(content_budget * 0.10)
summary_budget = content_budget - recent_budget - case_budget - profile_budget
```

- **`estimate_tokens(text)`**（[memory.py:17-22](../../backend/app/services/memory.py#L17)）：本项目自己写的一个粗略 token 估算函数，对中文字符（CJK 范围）按 1 字符≈1 token 算，其余按约 4 字符 1 token 估算——**不是精确分词**，只是一个偏保守的近似值，够用来做预算切分，不追求和真实 tokenizer 完全一致。
- 拿到总预算后先减去"当前问题"本身占用的 token，剩下的按固定比例（近期对话 50%、案件记忆 25%、用户偏好 10%、摘要占剩下部分）切给四层——这是一个**硬编码的静态分配策略**，不是动态根据"这一层信息量大小"自适应调整。
- **`_pack_memories(memories, token_budget)`**（[memory.py:52-66](../../backend/app/services/memory.py#L52)）：贪心地往预算里塞记忆条目，一旦某条塞进去会超预算就跳过（`truncated = True`）、继续尝试后面的——**不是"超了就整体截断"，而是"跳过超预算的单条，尽量多塞进去几条"**。
- **`fit_text(text, token_budget)`**（[memory.py:25-37](../../backend/app/services/memory.py#L25)）：对摘要文本做**二分查找**，找到"在预算内能保留的最长前缀"——这是一个通用的"按估算 token 数截断文本"小技巧，值得记住这个写法。

### 4.2 作用域隔离：`case_memories` vs `profile_memories`

[memory.py:103](../../backend/app/services/memory.py#L103) 调用 `self.repo.context_memories(conversation_id)`，返回两个分开的列表：**案件级记忆只属于发起它的那个会话，用户级偏好才允许跨会话加载**——这个区分是在 SQL 查询层面就做好的（[repositories.py::OwnedRepository](../../backend/app/services/repositories.py)），不是在应用层"过滤一下"，避免任何遗漏路径导致案件 A 的具体事实泄露进案件 B 的上下文。

### 4.3 后台异步抽取：不阻塞主回答

主回答保存完之后才 `enqueue` 一个记忆整理任务（[routes.py:678-684](../../backend/app/api/routes.py#L678)），真正的抽取逻辑在独立的后台 Worker 里跑（[memory_tasks.py::MemoryTaskManager](../../backend/app/services/memory_tasks.py)）。这里最值得看的是**结构化输出的实现方式**：

[memory_tasks.py:453-501](../../backend/app/services/memory_tasks.py#L453) `_invoke_structured_json`：

```python
schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
...
runner = model.bind(response_format={"type": "json_object"})
...
response = await runner.ainvoke(messages, config=trace_config) if trace_config else await runner.ainvoke(messages)
raw = _response_text(response)
decoded = json.loads(raw)
return schema.model_validate(decoded)
```

- **`schema.model_json_schema()`**：Pydantic 模型自带的方法，把一个 Pydantic 类转成标准 JSON Schema 字典——用来直接拼进 Prompt 告诉模型"你必须返回符合这个结构的 JSON"，比用自然语言描述字段更精确。
- **`model.bind(response_format={"type": "json_object"})`**：`.bind()` 是 LangChain Runnable 的方法，返回一个"预先绑定了额外参数"的新 Runnable，不修改原对象。`response_format={"type": "json_object"}` 是 DeepSeek/OpenAI 兼容 API 的**原生 JSON 输出模式**——从接口层面强制模型只能返回一个合法 JSON 对象，不能夹带解释性文字。这比"在 Prompt 里拜托模型只返回 JSON"这种纯文字约束可靠得多（相关的真实案例和踩坑经历见 [05-tool-calling-mcp.md](05-tool-calling-mcp.md)）。
- 拿到文本后先 `json.loads` 解析语法，再用 `schema.model_validate(decoded)` 做 Pydantic 校验——**两层校验**：JSON 语法是否合法、字段是否符合 Schema，任何一层失败都会分类记录失败原因（`empty_response`/`invalid_json`/`schema_validation_error`），供上层决定要不要重试。

### 4.4 记忆冲突的"原位替换"：不信任模型直接改数据

这是本项目记忆系统里工程质量最高的一处。当模型判断某条新提取的事实和已有记忆冲突、建议替换时，服务端**不会直接相信模型给的目标 ID**，而是重新做一次原子校验：

[memory_tasks.py:699-762](../../backend/app/services/memory_tasks.py#L699) `_replace_memory`：

```python
result = db.execute(
    update(UserMemory)
    .where(
        UserMemory.id == target.id,
        UserMemory.tenant_id == job_data["tenant_id"],
        UserMemory.user_id == job_data["user_id"],
        UserMemory.status == "active",
        UserMemory.version == expected_version,
    )
    .values(..., version=expected_version + 1)
)
if not result.rowcount:
    continue   # 版本冲突，重试一次；重试仍失败则返回 version_conflict
```

- 这是一次**原子 `UPDATE ... WHERE`**：所有权（`tenant_id`/`user_id`）、状态（必须是 `active`）、乐观锁版本号（`version == expected_version`）全部写进同一条 SQL 的 `WHERE` 子句，而不是"先 `SELECT` 出来判断一遍，再单独 `UPDATE`"——避免两次数据库往返之间被并发请求抢先修改（TOCTOU：Time-Of-Check to Time-Of-Use 竞态）。
- `result.rowcount` 为 0 说明这次 `UPDATE` 没有真正命中任何行（版本号已经被别的并发请求改过），会重试一次；仍然失败就放弃这次替换，返回 `version_conflict`，不会用一个过期的判断强行覆盖。
- 旧内容会写进 `MemoryRevision` 表（`action="auto_replace"`）而不是直接消失——保留可追溯的修订历史。

## 5. 对比：本项目 vs 行业常规方案

- 相比"直接把全部历史扔进 Prompt"：本项目用固定比例预算 + 摘要压缩，控制了单次请求的 Prompt 大小，且不会随对话变长而线性膨胀。
- 相比很多 Demo 项目里"模型说改就改"的简化实现：本项目在**持久化层面**用乐观锁 + 所有权重新校验，杜绝了"模型产生的建议被直接当成事实写库"这种信任链风险——这是一个通用的工程原则："模型的输出永远只是建议，最终决定权和校验必须在确定性代码里"，本项目的 Tool Calling 部分（见 [05-tool-calling-mcp.md](05-tool-calling-mcp.md)）也贯彻了同样的原则。
- 相比向量检索式记忆：本项目用结构化字段（`memory_type`/`scope`/`canonical_key`）显式建模记忆，而不是"存一堆文本片段靠语义检索"——可解释性更强（能明确说出"这条记忆是什么类型、属于哪个作用域"），但需要模型在抽取阶段就把内容结构化，抽取质量直接决定记忆质量。

## 6. 本项目内部的关键设计取舍与易错点

- 记忆抽取用**独立的、非流式、非 Thinking** 的模型配置（`LLMProvider.get_memory_model()`），和主 Agent 的模型调用完全分开——避免记忆整理的模型行为（比如输出格式要求）影响到主对话链路。
- `MemoryService.consolidate()` 已经被显式废弃（[memory.py:187-191](../../backend/app/services/memory.py#L187)，调用直接 `raise RuntimeError`）——这是一种"用代码强制淘汰旧接口"的做法：不只是文档说"别用这个了"，而是让旧入口物理上无法被误用。
- token 估算是近似值，预算分配比例是硬编码常量——如果未来发现某一层经常被过度截断，这是第一个该去调的地方。

## 7. 动手验证方式

1. 在对话里明确提一个事实（比如"我是房东"），再在后续消息里修正它（"抱歉说错了，我是租客"），通过前端"管理我的记忆"面板观察这条记忆是否被原位替换而不是新增一条。
2. 查一下 `memory_revisions` 表，看修订历史是否被完整保留：
   ```bash
   sqlite3 data/runtime/lawstation.db "select action, previous_content, new_content from memory_revisions order by created_at desc limit 5;"
   ```
3. 读一遍 [memory_schemas.py](../../backend/app/services/memory_schemas.py) 里 `MemorySnapshot` 的字段定义，对照 `MemoryService.snapshot()` 的返回值，确认自己能说清楚每个字段是从哪一步计算出来的。
