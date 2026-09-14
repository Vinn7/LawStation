# 分层记忆系统与 Context 工程（Memory / Context Engineering）

> 面向：熟悉"缓存"、"过期策略"这类概念，但没接触过 LLM 应用里"记忆"设计的后端工程师。
> 目标：理解为什么记忆不能简单等于"把历史消息存进数据库再原样塞回 Prompt"，本项目怎么做隔离和冲突消解，以及这套记忆系统离生产级 Memory 服务还差什么。

## 0. 前置知识

不需要提前读其他篇。如果想理解记忆抽取时用到的"强制模型返回结构化 JSON"这个技巧背后的原理，可以对照 [05-tool-calling-mcp.md](05-tool-calling-mcp.md) §3.5 一起看——两处用的是同一类思路的不同强度版本。

## 1. 要解决的问题

LLM 每次调用的上下文窗口（能塞进 Prompt 的 token 数）是有限的，但用户体验上又希望它"记得"：
- 这个会话之前聊过的内容（避免重复问同样的问题）
- 跨会话的偏好/背景（比如用户是租客还是房东，之前说过自己在哪个城市）

同时又要防止：
- 一个用户的会话内容"串"进另一个用户的上下文（隐私/数据隔离）
- 案件 A 的具体事实被错误地带进案件 B 的分析里（不同案件之间的事实污染）
- 历史记忆和用户本轮刚说的话冲突时，模型该信谁

## 2. 核心机制原理

### 2.1 五条常见路线，以及本项目组合了哪些

- **最简单粗暴**：把完整历史对话原样拼进每次 Prompt。短会话可行，长会话很快就超预算，而且历史越长、无关信息越多，反而稀释模型对当前问题的注意力（业内常说的"lost in the middle"问题）。
- **滑动窗口**：只保留最近 N 条消息，早期的直接丢弃。简单，但会丢掉早期提到的、之后仍然重要的信息（比如第一条消息里说的关键背景）。
- **摘要压缩（Summarization）**：定期把较早的对话压缩成一段摘要，滚动更新，替代原始消息塞进 Prompt。信息密度更高，但压缩是有损的，摘要生成本身还要额外调用一次模型。
- **向量检索式记忆（RAG-as-Memory）**：把历史对话/事实向量化存进向量库，每次按当前问题做语义检索，只取相关的片段。灵活，但"相关性"判断依赖检索质量，可能漏掉语义上不直接相似但业务上重要的信息——这正是 [06-rag-hybrid-retrieval.md](06-rag-hybrid-retrieval.md) 讲的那套检索机制如果被用来做记忆会遇到的局限。
- **结构化长期记忆**：把"记忆"当成结构化数据显式抽取和管理（用户偏好、关键事实各自建模，带状态/版本），而不是简单存一堆文本片段。管理成本更高，但可控性、可解释性最好。

本项目组合使用了**滑动窗口 + 摘要压缩 + 结构化长期记忆**，并且额外加了严格的**作用域隔离**——这是本项目区别于很多简化版 Demo 的地方。放弃"向量检索式记忆"的原因很直接：结构化字段（下面会讲的 `memory_type`/`scope`/`canonical_key`）能明确说出"这条记忆是什么类型、属于哪个作用域"，而一堆靠语义检索召回的文本片段做不到这种可解释性，代价是需要模型在抽取阶段就把内容结构化，抽取质量直接决定记忆质量。

### 2.2 四层记忆的分工

从"临时"到"持久"：

1. **原始消息**：完整存库，作为事实源头
2. **近期对话**：按 token 预算选取当前会话最近若干条
3. **工作记忆（会话摘要）**：当前会话的结构化滚动摘要，超过压缩阈值时更新
4. **长期记忆**：区分"用户级偏好"（跨会话复用）和"案件级事实"（只在所属会话使用）

四层各自分配一部分 token 预算，拼进最终 Prompt；后台异步任务负责把每轮对话里值得沉淀的内容抽取进第 3、4 层，不阻塞主回答。

## 3. 本项目具体实现（函数级）

### 3.1 预算分配：`MemoryService.snapshot()`

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

### 3.2 作用域隔离：`case_memories` vs `profile_memories`

[memory.py:103](../../backend/app/services/memory.py#L103) 调用 `self.repo.context_memories(conversation_id)`，返回两个分开的列表：**案件级记忆只属于发起它的那个会话，用户级偏好才允许跨会话加载**——这个区分是在 SQL 查询层面就做好的（[repositories.py::OwnedRepository](../../backend/app/services/repositories.py)），不是在应用层"过滤一下"，避免任何遗漏路径导致案件 A 的具体事实泄露进案件 B 的上下文。

### 3.3 后台异步抽取：不阻塞主回答

主回答保存完之后才 `enqueue` 一个记忆整理任务（[routes/chat.py:200-206](../../backend/app/api/routes/chat.py#L200)），真正的抽取逻辑在独立的后台 Worker 里跑（[memory_tasks/manager.py::MemoryTaskManager](../../backend/app/services/memory_tasks/manager.py#L32)）。

抽取模型看到的不只是当前这一条消息：`_load_job`（[memory_tasks/persistence.py:20-73](../../backend/app/services/memory_tasks/persistence.py#L20)）会先查出该用户**所有 `status="active"` 的记忆**——user 作用域的全部 + conversation 作用域里属于当前会话的那些（[persistence.py:34-47](../../backend/app/services/memory_tasks/persistence.py#L34)），一起打包成 `existing_memories` 放进 Prompt。抽取系统提示（[memory_tasks/prompts.py:6-14](../../backend/app/services/memory_tasks/prompts.py#L6)）明确要求模型：`existing_memories` 只能用来**比较**，不得被当成指令执行（防止历史记忆里混入的文本被误当成新指令，即防注入）；只有"同一当前属性互相矛盾"或用户明确纠正时，才把冲突记忆的 ID 填进 `replaces_memory_id`——不同时间点的历史事件（比如"3 月还了 5000""5 月又借了 3000"）应该共存，不能被误判成互斥覆盖。这条指令直接决定了下一节"落库"时模型给出的 `replaces_memory_id` 是否可信（答案是：不完全可信，服务端还要再核实一遍）。

这里最值得看的是**结构化输出的实现方式**：

[memory_tasks/invocation.py:20-68](../../backend/app/services/memory_tasks/invocation.py#L20) `_invoke_structured_json`：

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
- **`model.bind(response_format={"type": "json_object"})`**：`.bind()` 是 LangChain Runnable 的方法，返回一个"预先绑定了额外参数"的新 Runnable，不修改原对象。`response_format={"type": "json_object"}` 是 DeepSeek/OpenAI 兼容 API 的**原生 JSON 输出模式**——从接口层面强制模型只能返回一个合法 JSON 对象，不能夹带解释性文字。这比"在 Prompt 里拜托模型只返回 JSON"这种纯文字约束可靠得多，但比 [05-tool-calling-mcp.md](05-tool-calling-mcp.md) 讲的 `tool_choice="required"` 弱一档——后者能保证返回的 JSON 一定符合特定 Schema，这里只保证"是一个合法 JSON 对象"，字段是否符合业务 Schema 还要靠下一步的 Pydantic 校验。
- 拿到文本后先 `json.loads` 解析语法，再用 `schema.model_validate(decoded)` 做 Pydantic 校验——**两层校验**：JSON 语法是否合法、字段是否符合 Schema，任何一层失败都会分类记录失败原因（`empty_response`/`invalid_json`/`schema_validation_error`），供上层决定要不要重试。

### 3.4 候选落库：写库前怎么和陈旧记忆比对

模型给出的候选只是建议，真正决定"是新增、替换、跳过还是拒绝"的逻辑全部在 `_persist_candidates`（[memory_tasks/persistence.py:90-214](../../backend/app/services/memory_tasks/persistence.py#L90)）里，对每条候选按顺序做以下判断：

**① 幂等去重**：先用 `_canonical_key(memory_type, key, content)`（[helpers.py:120-125](../../backend/app/services/memory_tasks/helpers.py#L120)）算出规范化 key——把模型给的 `canonical_key` 转小写、合并空白、截断到 160 字符；如果模型给的 key 是空字符串，就退化成对 `content` 做 SHA256 取前 24 位当 key。查"同一条源消息 + 同一 canonical_key"是否已经存在，存在就跳过（`continue`）——这防的是同一个 job 被 worker 重启后重放、或者失败重试时重复插入同一条记忆；`UserMemory` 表本身也有 `(tenant_id, user_id, source_message_id, canonical_key)` 唯一约束兜底（[models.py:135-138](../../backend/app/db/models.py#L135)）。

**② 定位"目标"（这条候选可能要替换的陈旧记忆）**：
- 模型给了 `replaces_memory_id` → 查询条件锁定这个具体 ID，同时仍然要求 `tenant_id`/`user_id`/`scope`/`status="active"` 全部匹配（conversation 作用域还要求 `conversation_id` 一致）——记作 `replacement_source="model"`
- 模型没给（`null`）→ 退化成按 `canonical_key` 匹配同类型、同语义键的现有记忆——记作 `replacement_source="canonical_key"`

两种情况都按 `updated_at desc, id desc` 取最新一条（[persistence.py:109-129](../../backend/app/services/memory_tasks/persistence.py#L109)）。

**③ 校验目标是否真实存在**：如果模型给了 `replaces_memory_id` 但查不到匹配行（ID 是模型编造的、状态不是 `active`、作用域或所有权对不上）→ 直接拒绝这次替换（`rejected_count++`，审计事件 `memory.replacement.rejected`，`reason="invalid_or_unowned_target"`，[persistence.py:130-142](../../backend/app/services/memory_tasks/persistence.py#L130)）。**这是不信任模型 ID 的第一道关**——下一节讲的"原子 UPDATE"是第二道关，两道关分别防的是"模型编造了一个不存在/不属于自己的 ID"和"目标存在，但在核验之后、真正写入之前被并发请求抢先改过"这两类不同的问题。

**④ 内容完全相同就跳过**：`target.content == candidate.content` 时直接 `continue`（[persistence.py:144-145](../../backend/app/services/memory_tasks/persistence.py#L144)），不产生一次空替换——避免同样的事实被反复抽取时，每次都无意义地刷新 `version` 并多写一条 `MemoryRevision`。

**⑤ 三种结局**：
- 找到目标且内容不同 → 调用 `_replace_memory` 做原子替换（见下一节）
- 完全没找到目标（模型判断"这是全新事实"，或者按 `canonical_key` 也没匹配上）→ 当作全新记忆 `INSERT`，用 `db.begin_nested()` 包一层 savepoint，吞掉 `IntegrityError` 当作"并发下已经被别的请求插入过，跳过"处理（[persistence.py:182-204](../../backend/app/services/memory_tasks/persistence.py#L182)）
- 替换过程本身失败（乐观锁版本冲突、完整性冲突）→ 计入 `rejected_count`，不强行覆盖

### 3.5 原子替换的具体实现：并发下的第二道防线

上一节的③已经挡掉了"目标不存在"的情况；这里处理的是"目标存在，但可能在核验之后、真正写入之前，被另一个并发的记忆整理任务抢先修改"——服务端同样不直接执行 `UPDATE`，而是把所有权、状态、乐观锁版本号一起写进 `WHERE` 子句：

[memory_tasks/persistence.py:216-279](../../backend/app/services/memory_tasks/persistence.py#L216) `_replace_memory`：

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

### 3.6 状态字段的实现现状：schema 预留的 superseded 链没有被启用

`UserMemory.status` 的类型定义了五种值：`pending/active/superseded/rejected/expired`（[memory_schemas.py:10](../../backend/app/services/memory_schemas.py#L10)），`superseded_by_id` 这一列（[models.py:153-155](../../backend/app/db/models.py#L153)）从命名上看，是为"旧记忆保留一行、标记 `superseded`、`superseded_by_id` 指向替换它的新记忆"这种典型的多版本链设计准备的。

但实际写路径不是这样跑的：新建记忆时永远直接写 `status="active"`（[persistence.py:188](../../backend/app/services/memory_tasks/persistence.py#L188)），从不经过 `pending`；`_replace_memory` 的 `UPDATE` 语句**原地覆写同一行的 `content`**，`status` 根本不出现在 `.values(...)` 里（保持原来的 `active`），`superseded_by_id` 还被显式设成 `None`（[persistence.py:255](../../backend/app/services/memory_tasks/persistence.py#L255)）。也就是说：

- `superseded`/`rejected`/`expired` 这三个状态值在当前代码里**从未被真正写入过**——它们是 `MemoryStatus` 类型定义里"预留但未使用"的字面量，全仓搜索也找不到任何一处对 `UserMemory.status` 赋这三个值的代码。
- 版本历史不是靠"旧行标 `superseded`、留在表里"实现的，而是靠**同一行原地更新 + 另开一张 `MemoryRevision` 表记流水**（`action="auto_replace"`，记 `previous_content`/`new_content`/`previous_status`/`new_status`）。

这两种模式能达到的效果不一样：当前的"原地覆写 + 流水表"能回答"这条记忆的内容变更历史是什么"，但做不到"让某条历史版本重新变回 active"——因为旧内容已经不在 `UserMemory` 表的任何一行里，只作为一段文本存在于 `MemoryRevision.previous_content` 字段里，不是一条可查询、可重新激活的记忆记录。真正启用 `superseded` 状态链的做法是**插入新行而不是覆写旧行**：旧行 `status` 改成 `superseded`、`superseded_by_id` 指向新行——这样旧版本依然是一条完整的、可以重新激活的 `UserMemory` 记录，代价是每次按 `canonical_key`/`scope` 查"当前有效版本"都要多加一个 `status != 'superseded'` 的过滤条件。

## 4. 设计取舍

**为什么不直接把全部历史扔进 Prompt？** 会随对话变长而线性膨胀，超预算是必然的事，而且历史越长模型对当前问题的注意力越容易被稀释。固定比例预算 + 摘要压缩把这件事变成了可控的、不随对话长度线性增长的开销。

**为什么记忆冲突要在持久化层面用乐观锁 + 所有权重新校验，而不是直接相信模型给的判断？** 这是一个通用的工程原则："模型的输出永远只是建议，最终决定权和校验必须在确定性代码里"——本项目的 Tool Calling 部分（见 [05-tool-calling-mcp.md](05-tool-calling-mcp.md)）也贯彻了同样的原则。很多简化版 Demo 会"模型说改就改"，直接拿模型给的目标 ID 去更新，这里选择多一层原子校验，换来的是不会被一个过期或错误的模型判断污染数据。

**为什么记忆抽取用独立的模型配置，而不是复用主 Agent 的模型调用？** `LLMProvider.get_memory_model()` 是一个独立的、非流式、非 Thinking 的模型配置，和主对话链路的模型调用完全分开——这样记忆整理这个后台任务的输出格式要求、超时策略变化，不会影响主对话链路的行为，两者可以独立演进。

**为什么记忆替换选择"原地覆写 + 独立流水表"，而不是"新增行 + `superseded` 状态链"（见 §3.6）？** 前者实现更简单：一次 `UPDATE` 就能同时完成内容替换和乐观锁校验，读取时也不用关心"这个 `canonical_key` 下可能有多行，只有一行是当前有效版本"；后者需要在所有按 `scope`/`canonical_key` 查询"当前有效记忆"的地方都加上过滤条件，查询逻辑更复杂，但换来的是旧版本本身仍是一条可独立查询、可重新激活的记忆记录。这是一个**用能力换简单性**的取舍——本项目目前的场景只需要"看到变更历史用于审计/调试"，不需要"回滚到某个历史版本"，前者够用；如果后续要支持回滚，这里需要重新设计。

## 5. 易错点

- **以为 `MemoryService.consolidate()` 只是被废弃的旧文档说法，实际还能调用**：这个方法已经被显式改成调用直接 `raise RuntimeError`（[memory.py:187-191](../../backend/app/services/memory.py#L187)）——这是一种"用代码强制淘汰旧接口"的做法，不只是文档说"别用这个了"，而是让旧入口物理上无法被误用。如果看到还有代码路径调用它，那是一个需要立刻修的 Bug，不是"遗留但还能跑"。
- **改了预算分配比例常量却没有验证效果**：token 估算本身是近似值，预算分配比例是硬编码常量，改动前后需要实际观察某一层是否经常被过度截断（`truncated=True`），而不是凭直觉调整数字。
- **混淆"案件级记忆"和"用户级偏好"的隔离边界**：案件级记忆只属于发起它的那个会话，如果在查询时误用了跨会话的查询条件，会导致案件 A 的具体事实泄露进案件 B——这个隔离是在 SQL 层面做的，业务代码不应该在应用层"再过滤一次"来兜底，那样反而掩盖了本该在数据层拦住的错误。
- **误以为 `MemoryStatus` 定义的 `superseded`/`rejected`/`expired` 在数据库里真的会出现**：这几个状态值目前从未被任何写路径赋值过（见 §3.6）——如果按这些状态写监控指标或查询过滤条件，永远查不到结果；这不是查询写错了，是这套状态机在当前实现里根本没有被启用到那个程度，看类型定义容易误判实现完整度。

## 6. 生产化差距与面试应对

这套分层记忆在隔离和冲突消解上已经做得比很多简化 Demo 严谨，但离生产级 Memory 服务还有几处差距：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| Token 估算 | 自实现的近似估算函数（CJK 按字符数、其余按 4 字符 1 token） | 使用真实 tokenizer（如 `tiktoken` 对应模型的编码器）精确计算，避免估算误差导致预算超支或浪费 | "当前用近似估算是为了避免引入额外依赖和计算开销，如果发现预算切分经常不准，第一步是换成真实 tokenizer" |
| 预算分配策略 | 四层固定比例（50/25/10/摘要），硬编码常量 | 生产系统常见做法是根据历史数据动态调整分配比例，或允许按会话/用户类型配置不同策略 | "当前是静态分配，没有做到'哪层信息量大就多分配'的自适应；这是从'能用的默认值'到'数据驱动调优'之间的差距" |
| 记忆抽取质量监控 | 抽取失败分类记录（`empty_response`/`invalid_json`/`schema_validation_error`），但没有持续的抽取质量评测 | 生产级记忆系统通常会持续采样评测"抽取的记忆是否准确、是否遗漏关键信息"，形成质量回归基线 | "当前只监控抽取失败率，还没有对'抽取内容准确性'做持续评测；这块可以接入 [08-agent-rag-eval-methodology.md](08-agent-rag-eval-methodology.md) 讲的评测方法论" |
| 存储规模 | 单机 SQLite，记忆条目和摘要都在业务库里 | 记忆条目量级大之后，通常会拆分成独立的存储服务，甚至引入向量索引做混合检索（结构化字段 + 语义相似度） | "当前规模下结构化字段查询足够快；如果单用户记忆条目数量级上升，可能需要引入检索索引辅助召回相关记忆，而不是每次全量拉取" |
| 记忆治理 UI | 有前端"管理我的记忆"面板支持确认/拒绝，但主要服务历史/兼容场景 | 生产系统通常会有更完整的用户数据控制能力（导出、批量删除、按类型/时间范围管理），配合数据合规要求 | "当前的治理能力覆盖了基本场景，如果要满足更严格的数据合规要求（比如用户要求彻底删除某类记忆），还需要补充批量管理能力" |
| 记忆版本链 | `status` 字段预留了 `superseded`/`rejected`/`expired`，但替换只做原地覆写 + `MemoryRevision` 流水表，这三个状态从未被写入（见 §3.6） | 生产级记忆系统通常需要支持"查看/回滚到历史版本"，这要求旧版本本身是一条可查询、可重新激活的记录，而不只是流水表里的一段文本 | "当前的原地覆写模式能满足审计诉求，但不支持回滚；如果要支持回滚，需要改成'新增行 + 状态链'模式，旧行标记 `superseded` 而不是被覆写，查询逻辑也要相应加上状态过滤" |

## 7. 动手验证方式

1. 在对话里明确提一个事实（比如"我是房东"），再在后续消息里修正它（"抱歉说错了，我是租客"），通过前端"管理我的记忆"面板观察这条记忆是否被原位替换而不是新增一条。
2. 查一下 `memory_revisions` 表，看修订历史是否被完整保留：
   ```bash
   sqlite3 data/runtime/lawstation.db "select action, previous_content, new_content from memory_revisions order by created_at desc limit 5;"
   ```
3. 读一遍 [memory_schemas.py](../../backend/app/services/memory_schemas.py) 里 `MemorySnapshot` 的字段定义，对照 `MemoryService.snapshot()` 的返回值，确认自己能说清楚每个字段是从哪一步计算出来的。

**自测题：**

- 如果两个并发的记忆整理任务同时试图替换同一条记忆，`_replace_memory` 是怎么保证不会有一个任务的更新被另一个悄悄覆盖的？（提示：想想乐观锁版本号在 `WHERE` 子句里的作用）
- 为什么案件级记忆的隔离要做在 SQL 查询条件里，而不是查出来之后在 Python 代码里过滤掉不属于当前会话的记忆？两种做法在"漏改一处代码"时的后果有什么区别？
- 如果要让 `UserMemory.status` 里的 `superseded` 状态真正生效（支持"回滚到某个历史版本"），`_replace_memory` 的实现需要怎么改？现在的"原地覆写"为什么做不到这一点？
