# RAG 混合检索与 Reranker（BM25 + Dense + RRF + BGE）

> 面向：知道"RAG 就是检索加生成"，但没有亲手调过混合检索权重、没有踩过 Reranker 延迟坑的后端工程师。
> 目标：搞清楚为什么两路召回要用排名融合而不是分数加权、精排到底解决什么问题、以及"向量数据库"这个词在本项目里该怎么讲才不算夸大。

## 0. 前置知识

读这篇前，应该已经知道：Embedding 是把文本映射成向量、向量距离衡量语义相似度这个基本直觉。不需要提前读其他篇——本篇是检索链路的起点，[04-langgraph-stategraph.md](04-langgraph-stategraph.md) 的 Research 节点会调用这里讲的工具，[05-tool-calling-mcp.md](05-tool-calling-mcp.md) 讲的是"模型怎么决定调用"，本篇讲的是"调用之后到底怎么找到正确法条"。

## 1. 要解决的问题

用户的咨询用口语表达（"公司一直没和我签合同"），法条用正式表述（"未订立书面劳动合同"）；同时用户也会直接提到法律名称和条号（"劳动合同法第八十二条怎么算"），这种精确匹配又是自然语言语义检索的弱项。只选一种检索方式，必然在另一种查询上失分：

- 只用关键词/词法检索（如 BM25）：精确匹配法律名称、条号、专业术语很强，但"公司一直不给我签合同"这种口语化表达很难命中"未订立书面劳动合同"。
- 只用向量语义检索：能理解口语和正式表述的语义接近，但对法律名称、条号这类需要精确复现的字符串反而不敏感，容易被"意思相近但法律实体不对"的候选干扰。

法律场景还有一条更硬的约束：**回答里的每一条法条引用都必须能追溯到真实检索到的证据片段**，检索环节的候选边界直接决定了后面能不能防止模型编造法条。

## 2. 核心机制原理

### 2.1 词法检索 vs 语义检索，为什么要两路并存

行业里成熟的做法是"混合检索"（Hybrid Search）：词法路（BM25/TF-IDF 类算法）和向量路（Dense Embedding + 近似最近邻）各自独立召回一批候选，再融合成一个排序。这不是新概念——Elasticsearch/OpenSearch 的 `rank_features` 混合查询、Vespa 的多阶段排序、Weaviate 和 Qdrant 的原生 Hybrid API，本质都是同一个思路的不同实现。

### 2.2 两路结果怎么合并：加权求和 vs 排名融合

合并两路分数最直觉的做法是加权求和：`score = α · bm25_score + β · cosine_similarity`。问题是 BM25 分数没有固定上界（和词频、文档长度相关），余弦相似度天然落在 `[-1, 1]`，两者不在同一量纲，需要额外做归一化，而归一化方式本身又要针对数据集调参，容易在换一批数据后重新失效。

行业里更常用、更稳的替代方案是 **RRF（Reciprocal Rank Fusion）**：不看分数本身，只看每路里的排名，按排名的倒数打分再相加。因为只依赖排名（一个自然的相对顺序），不需要关心两路分数的量纲差异，这是 RRF 相比线性加权最大的优势，也是 Elasticsearch 8.8+、OpenSearch 2.11+ 把 RRF 作为内置混合检索融合方式的原因。

### 2.3 精排（Reranking）解决什么问题

粗排（BM25/Dense/RRF）为了速度，通常用"查询和文档各自独立编码、算一个相似度分数"的方式（表示学习，representation-based），计算便宜但精度有限。精排环节换用 **Cross-Encoder**：把查询和候选文档拼在一起，一起输入模型做联合编码，能捕捉更细粒度的语义交互，代价是必须对每个候选单独跑一次模型推理，不能像向量检索一样提前离线计算好、只留查询时做最近邻搜索。所以精排几乎总是放在"粗排先筛出一小批候选，精排只处理这一小批"的两阶段架构里，行业里 Cohere Rerank、Voyage Rerank、开源的 BGE-Reranker/Jina Reranker 都是这个定位。

## 3. 本项目具体实现（函数级）

> 本节所有代码引用都对照当前工作区源码逐行核实过；行号会随代码演进漂移，读到行号对不上时以源文件为准。

### 3.1 法条切分与稳定 ID：`load_chunks`

[load_chunks（engine.py:65-85）](../../mcp_servers/law_rag/engine.py#L65) 把法规 JSON 的每个键解析为法律名称和条号：用正则 `(第[...]条)$` 从键尾部截出条号，剩下的部分是法律名称（[engine.py:69-71](../../mcp_servers/law_rag/engine.py#L69)）。真正的切分逻辑在 [split_text（engine.py:47-62）](../../mcp_servers/law_rag/engine.py#L47)：只有超过 `maximum` 字才切分；边界优先在 `[start + maximum // 2, end)` 这段区间里找最靠右的换行 / 句号 / 分号（[engine.py:54-55](../../mcp_servers/law_rag/engine.py#L54)），找不到就直接在 `maximum` 处硬切——也就是说如果一段条文连续几百字没有这三种标点，切片会从字中间断开，这是当前实现里完全没有兜底的边界情况，值得在设计评审时点出来。`maximum`/`overlap` 不是 `load_chunks` 里的字面量，而是从 `Settings.index_chunk_max_chars`/`index_chunk_overlap_chars` 传入（[config.py:80-81](../../backend/app/core/config.py#L80)），只是默认值恰好是 1000 字和 150 字重叠。

`document_id` 是对 JSON 键原文（法律名称 + 条号拼接后的完整字符串，例如"劳动合同法第八十二条"）做 SHA-1（[engine.py:72](../../mcp_servers/law_rag/engine.py#L72)）——它标识的是"一条具体条文"这个原始条目本身，不是整部法律；`chunk_id` 由 `document_id:chunk_index:content_hash` 拼接后再做一次 SHA-1（[engine.py:74-75](../../mcp_servers/law_rag/engine.py#L74)），其中 `content_hash` 是切片正文的 SHA-256。这两层 ID 拆开，是为了让"长法条命中了后半部分"这件事精确记录下来：同一条文的多个切片共享 `document_id`，但各自有独立 `chunk_id`，不会在引用时错误关联到该法条的第一个片段。

### 3.2 索引指纹与幂等建库

[LawSearchEngine._fingerprint（engine.py:150-164）](../../mcp_servers/law_rag/engine.py#L150) 实际参与哈希的是 9 个字段：`source_sha256`（数据 SHA-256）、`chunker_version`（固定字符串 `"law-article-v1"`）、`max_chars`、`overlap_chars`、`embedding_provider`、`embedding_model`、`embedding_model_digest`、`embedding_dimension`、`query_instruction_version`（固定字符串 `"legal-query-v1"`）。"切分参数"其实是版本号 + 两个数字三个独立字段，"模型标签"和"模型 digest"也是分开的两个字段——任意一个变化都会让指纹变化并触发重建。模型 digest 进指纹很关键：即使模型名字没变，本地模型被重新下载或悄悄升级了版本，指纹也会变化，避免"索引里的旧版本向量"和"查询时的新版本向量"来自不同模型却被当成同一空间比较。

建库 [_build（engine.py:375-521）](../../mcp_servers/law_rag/engine.py#L375) 是**原子化、可恢复的**：新索引先写到 `{index_dir}/.staging-<fingerprint>/`（`index_dir` 默认 `./data/indexes`，[config.py:39](../../backend/app/core/config.py#L39)），每批调用一次 Embedding。批大小不是硬编码，是三方取最小值 `max(1, min(index_build_batch_size, provider_limit, 20))`（[engine.py:407-412](../../mcp_servers/law_rag/engine.py#L407)），Ollama Provider 下 `provider_limit` 取 `ollama_embedding_batch_size`；当前两个配置默认值都是 8（[config.py:31](../../backend/app/core/config.py#L31)、[config.py:82](../../backend/app/core/config.py#L82)），所以实际生效批大小恰好是 8——这是两个默认值巧合相等的结果，不是代码里写死的"8"。每批先写临时 `.npy` 文件再 `os.replace` 原子替换（[engine.py:428-430](../../mcp_servers/law_rag/engine.py#L428)），重启时会校验已有批次文件的 shape/dtype/有限性，只重新生成缺失或损坏的批次（[_load_batch，engine.py:352-363](../../mcp_servers/law_rag/engine.py#L352)）。

全部批次生成完毕、写完 FAISS 索引和 `manifest.json`、并通过一次独立的完整性重读校验（[engine.py:490-492](../../mcp_servers/law_rag/engine.py#L490)）之后，才会把 staging 目录切到正式路径。这一步实际是**两跳**，文档旧版本没有提到：先把已存在的旧 `final_dir` 用 `os.replace` 挪到 `.law-backup`（[engine.py:496-500](../../mcp_servers/law_rag/engine.py#L496)），再把 staging 换到 `final_dir`；如果第二次 `os.replace` 本身抛异常（比如目标路径跨设备、权限问题），会把 backup 换回来（[engine.py:501-506](../../mcp_servers/law_rag/engine.py#L501)），只有整个流程成功才删除 backup。也就是说不仅"新建库失败不会覆盖旧索引"，"新索引已经生成、但最后切换这一步本身失败"这种更极端的情况也有恢复路径。内存中的 FAISS 实例最后在 `_dense_lock` 锁内热切换（[engine.py:509-510](../../mcp_servers/law_rag/engine.py#L509)）。

### 3.3 双路召回：BM25 + Dense

- BM25 路：[_lexical（engine.py:572-601）](../../mcp_servers/law_rag/engine.py#L572) 用 Jieba 分词后交给 `rank_bm25.BM25Okapi` 打分，按分数降序取前 `pool` 个，再过滤掉低于 `RAG_BM25_MIN_SCORE`（默认 `0.01`，[config.py:87](../../backend/app/core/config.py#L87)）的候选。
- Dense 路：查询文本套一层检索指令模板 `Instruct: {ollama_query_instruction}\nQuery: {query}`（[embeddings.py:107-109](../../mcp_servers/law_rag/embeddings.py#L107)）后送入 Ollama `qwen3-embedding:0.6b` 编码成 1024 维向量，L2 归一化后用 `faiss.IndexFlatIP`（[_faiss_search，engine.py:603-627](../../mcp_servers/law_rag/engine.py#L603)）做内积搜索——文档和查询都归一化后，内积在数学上等价于余弦相似度。低于 `RAG_DENSE_MIN_SCORE`（默认 `0.20`，[config.py:88](../../backend/app/core/config.py#L88)）的候选不进入融合（[engine.py:697-707](../../mcp_servers/law_rag/engine.py#L697)）。

两路候选池宽度由 `pool = min(candidate_count, max(30, top_k * 4))`（[engine.py:681](../../mcp_servers/law_rag/engine.py#L681)）决定，也就是说 `top_k` 不只影响最终返回数量，还同时影响两路各自召回多宽——这是做对比实验时必须固定 `top_k` 的原因，否则连候选池大小都变了，就不是单变量对比。

**`filters` 会改变 Dense 路实际扫描的范围**（见 §3.7），但这里先说一个容易被忽略的联动：`filters` 生效时，传给 FAISS 的 `pool` 参数不是上面算出来的 `pool`，而是 `len(self.docs)`——即请求 FAISS 返回**全部**文档的相似度排序（[engine.py:695](../../mcp_servers/law_rag/engine.py#L695)），再在应用层逐个过滤是否属于 `eligible` 集合、是否达到最低分，凑够 `pool` 个接受的候选才停止（[engine.py:696-713](../../mcp_servers/law_rag/engine.py#L696)）。原因是 `IndexFlatIP` 本身不支持按元数据过滤检索：要保证"只在某部法律范围内检索"仍能拿到该法律下语义最相关的候选，只能先做一次全量排序再筛选。因为 `IndexFlatIP` 精确搜索本来就要和全部向量算一遍距离，这里没有引入额外的近似损失，但确实意味着"加了 `law_name` 过滤之后，Dense 检索这一路的计算量固定是全量扫描，不会因为过滤范围变小而变小"。

### 3.4 RRF 融合

```text
rrf(chunk) = Σ 1 / (61 + zero_based_rank_in_source)
```

常数 `61` 在两路累加里都是同一个值（[engine.py:687](../../mcp_servers/law_rag/engine.py#L687)、[engine.py:708](../../mcp_servers/law_rag/engine.py#L708)）。同一个 chunk 如果同时被两路召回，两边的贡献会累加（约 `0.0328`），只被一路召回的贡献单一路（约 `0.0164`）。这一步不看 BM25 原始分数和 Dense 余弦分数的具体数值，只看各自的名次（这里的"名次"是过阈值筛选之后、在各自候选列表里的顺序，不是全库排名）——这正是 §2.2 里说的"避免量纲不一致"在代码里的落地。融合结果低于 `RAG_RRF_MIN_SCORE`（默认 `0.01`，[config.py:89](../../backend/app/core/config.py#L89)）会被过滤（[_rrf_order，engine.py:629-652](../../mcp_servers/law_rag/engine.py#L629)）。

**精排候选保留数不是写死的 12**：实际值是 `rerank_limit = max(top_k, RAG_RERANK_CANDIDATE_COUNT)`（[engine.py:714-718](../../mcp_servers/law_rag/engine.py#L714)）。`RAG_RERANK_CANDIDATE_COUNT` 默认确实是 `12`（[config.py:107](../../backend/app/core/config.py#L107)），多数调用（`top_k` 默认 `8`）确实会保留 12 个候选给精排；但 `search()` 允许 `top_k` 传到 `20`（[engine.py:672](../../mcp_servers/law_rag/engine.py#L672)），一旦调用方把 `top_k` 设到大于 12，精排候选池会跟着涨到 `top_k`，不会停在 12。§3.5 末尾会说明这个"跟涨"和 TEI 自身批量上限之间没有被显式关联校验的耦合点。精排完成后才把结果截断成最终的 `top_k`。

### 3.5 TEI + BGE Cross-Encoder 精排

[TEIReranker.rerank（reranker.py:523-613）](../../mcp_servers/law_rag/reranker.py#L523) 调用独立部署的 Hugging Face TEI 服务，一次 `POST /rerank` 请求把查询和全部候选（数量就是 §3.4 算出来的 `rerank_limit`）一起提交（[reranker.py:553-562](../../mcp_servers/law_rag/reranker.py#L553)），拿到 Cross-Encoder 相关性分数重新排序。

响应校验由 [parse_tei_ranks（reranker.py:134-160）](../../mcp_servers/law_rag/reranker.py#L134) 完成，不接受任何部分成功：`ranks` 数组长度必须严格等于候选数（[reranker.py:138](../../mcp_servers/law_rag/reranker.py#L138)）；每个 `index` 必须是非布尔整数、在合法范围内且不重复（[reranker.py:144-149](../../mcp_servers/law_rag/reranker.py#L144)）；`score` 必须是非布尔有限数值且落在 `[0, 1]`（[reranker.py:150-156](../../mcp_servers/law_rag/reranker.py#L150)）；最后还要求解析出来的索引集合恰好等于 `range(expected_count)`（[reranker.py:158](../../mcp_servers/law_rag/reranker.py#L158)）——任何一处不满足都抛异常。

**精排有两条完全不同性质的"清空"路径，容易被混为一谈：**

1. **服务故障**：任一候选缺失、响应格式异常、HTTP 失败或整体超时，都会在 `except` 分支里调用 `_degrade()` 并抛出 `RerankerUnavailable`（[reranker.py:566-575](../../mcp_servers/law_rag/reranker.py#L566)）。`search()` 捕获这个异常后（[engine.py:738-739](../../mcp_servers/law_rag/engine.py#L738)），直接保留精排前基于 RRF 截断的 `top_k` 结果，`rerank_applied=False`——这是**降级**，用户仍能拿到结果，只是没有精排。
2. **精排判定确实不相关**：TEI 正常返回了合法分数，但普遍低于 `RAG_RERANK_MIN_SCORE`（默认 `0.0`，[config.py:109](../../backend/app/core/config.py#L109)，即默认不生效，调高后才会触发），`ordered` 会被过滤成空列表（[engine.py:741-753](../../mcp_servers/law_rag/engine.py#L741)），此时 `rerank_applied=True`——这是一次**真实完成的精排判定**，结论是"确实没有相关的"，不是故障。

测试 [`test_reranker_failure_falls_back_to_rrf_not_no_match`（tests/test_index_manager.py:438-466）](../../tests/test_index_manager.py#L438) 覆盖路径 1（最终有结果、`rerank_applied=False`），[`test_reranker_threshold_can_return_normal_no_match`（tests/test_index_manager.py:470-499）](../../tests/test_index_manager.py#L470) 覆盖路径 2（最终 `search()` 返回 `[]`）。把这两条路径混为一谈，会把"精排服务挂了但检索还能用"和"精排正常工作、只是判定没有相关证据"这两种性质完全不同的问题误诊成同一件事。

**冷却机制**：`_cooldown_until = time.monotonic() + RAG_RERANK_RETRY_SECONDS`（默认 `60.0` 秒，[config.py:114](../../backend/app/core/config.py#L114)）只在 `_degrade()`（TEI：[reranker.py:487-509](../../mcp_servers/law_rag/reranker.py#L487)；Ollama：[reranker.py:296-315](../../mcp_servers/law_rag/reranker.py#L296)）里因为一次真实失败被设置。冷却期内如果又来了一次精排请求，`rerank()` 一进来就检查 `self._cooldown_until > time.monotonic()`（[reranker.py:528-529](../../mcp_servers/law_rag/reranker.py#L528)），直接抛出带"冷却期"字样的 `RerankerUnavailable`，**不会发起任何网络请求，也不会重置或延长冷却窗口**——冷却的起点和终点只由上一次真实失败决定，这一点被测试 [`test_invalid_candidate_response_degrades_entire_rerank_batch`（tests/test_reranker.py:108-126）](../../tests/test_reranker.py#L108) 显式验证：冷却期内的第二次调用立刻拿到"冷却期"异常。这个冷却状态是 reranker 实例的普通进程内属性，不是跨进程共享的分布式状态——多 worker 进程部署下，每个进程会独立进入和退出冷却，这也是 §6 表格"精排部署"一行需要展开的一个具体原因。

**一个尚未被显式校验、值得在设计评审里主动点出的耦合**：精排请求携带的候选数是 §3.4 算出来的 `max(top_k, RAG_RERANK_CANDIDATE_COUNT)`，`top_k` 允许调用方传到 `20`；但如果本项目自己拉起 TEI（`rag_rerank_auto_start=True`），启动参数会把 TEI 服务自身的 `--max-client-batch-size` 设成 `RAG_RERANK_MAX_CLIENT_BATCH_SIZE`，默认也是 `16`（[config.py:105](../../backend/app/core/config.py#L105)，实际传参见 [tei.py:167-170](../../backend/app/core/tei.py#L167)）。也就是说 `top_k` 落在 `17~20` 区间时，精排请求的候选数会超过 TEI 自己配置的批量上限，这次请求会在 TEI 侧被拒绝，再经过上面路径 1 的降级逻辑退化为未精排的 RRF 结果——不会报错或返回空结果，但精排的效果会静默丢失，且代码里目前没有任何断言或告警把这两个默认值关联起来。

### 3.6 证据链闭环：检索结果不能直接变成引用

检索只是候选来源，真正防止模型编造法条的收敛逻辑分布在四个位置，不止旧版本文档描述的"两层"：

1. **Research 节点第一次权威映射**：[nodes/research.py:141](../../backend/app/agent/graph/nodes/research.py#L141) 在 Research Agent 产出 `structured_response` 之后，立即调用 [`_authoritative_evidence`（graph/evidence.py:83-125）](../../backend/app/agent/graph/evidence.py#L83)。它按 `chunk_id`（兼容旧格式时按"某个 `document_id` 在本轮恰好只对应一个 chunk"这一条件退化用 `document_id`）把模型选择映射回本轮 MCP 工具返回的真实候选，`law_name`/`article_number`/`content` 全部从候选对象回填，模型自己写的正文不会进入 `EvidenceItem`；伪造 ID、歧义 `document_id`、本轮不存在的 ID 都被静默丢弃。
2. **候选存在但模型既未接受也未拒绝时的兜底 Selector**：[nodes/research.py:142-187](../../backend/app/agent/graph/nodes/research.py#L142) 会额外发起一次不带工具的"Evidence Selector"子调用，它只能从已有候选的 `chunk_id` 里选择，不能重新检索或产生新证据（[research.py:163-171](../../backend/app/agent/graph/nodes/research.py#L163) 还会用候选 ID 集合再裁剪一次防止越权）；选完之后仍会再跑一遍 `_authoritative_evidence`（[research.py:186](../../backend/app/agent/graph/nodes/research.py#L186)）做同样的权威回填。
3. **Review 节点的确定性引用校验**：[nodes/review.py:47](../../backend/app/agent/graph/nodes/review.py#L47) 调用 [`_citation_errors`（graph/evidence.py:211-239）](../../backend/app/agent/graph/evidence.py#L211)，用代码检查草稿 `claims` 里引用的 `chunk_id`/`document_id` 是否都能在 `EvidencePacket` 里找到，以及正文里出现的《法律名称》第 N 条组合是否有证据支持。这一步只产出错误列表、影响 `review_result.approved`，**不生成最终引用**，和第 4 步是两个独立机制。
4. **Finalize 节点的真正交集**：这一步的实现不在 `evidence.py`，而在 [`FinalizeNode.finalize`（graph/nodes/finalize.py:14-105）](../../backend/app/agent/graph/nodes/finalize.py#L14)。具体顺序是：先收集 `counsel_draft.claims` 里实际引用到的全部 `chunk_id`/`document_id`（[finalize.py:64-68](../../backend/app/agent/graph/nodes/finalize.py#L64)）；再统计 `EvidencePacket` 里每个 `document_id` 对应几个 chunk（[finalize.py:70-73](../../backend/app/agent/graph/nodes/finalize.py#L70)）；只遍历 `retrieval_status == "matched"` 的 `EvidencePacket`（[finalize.py:77](../../backend/app/agent/graph/nodes/finalize.py#L77)，no_match 恒产出零条引用，和安全模板替换逻辑保持一致）；对其中被草稿引用、或满足"该 `document_id` 本轮只对应一个 chunk"这一兼容旧格式条件的条目去重后（[finalize.py:80-88](../../backend/app/agent/graph/nodes/finalize.py#L80)）才构造最终 `Citation`，摘录截断到 240 字（[finalize.py:91](../../backend/app/agent/graph/nodes/finalize.py#L91)）。检索找到但回答没用到的候选，不会出现在引用列表里。

### 3.7 `filters` 参数与 `search()` 的整体边界

MCP 工具 [`search_laws`（server.py:66）](../../mcp_servers/law_rag/server.py#L66) 暴露了 `filters: dict | None` 参数，但旧版本文档完全没解释它的语义——实际支持面很窄，也没有做成通用过滤器：

- **只支持一个过滤维度**：[_filtered_indices（engine.py:542-570）](../../mcp_servers/law_rag/engine.py#L542) 里 `unknown = set(filters) - {"law_name"}`（[engine.py:549](../../mcp_servers/law_rag/engine.py#L549)），传入 `law_name` 以外的任何字段都会直接抛 `ValueError`（测试 [`test_unknown_filter_is_rejected`（tests/test_index_manager.py:349-356）](../../tests/test_index_manager.py#L349) 覆盖了这一点）。
- **匹配方式是归一化后的双向子串**，不是精确匹配：先用 `normalize_law_name`（[engine.py:38-40](../../mcp_servers/law_rag/engine.py#L38)）去空格、去书名号、转小写、去掉"中华人民共和国"前缀，再判断 `normalized in stored_name or stored_name in normalized`（[engine.py:561-563](../../mcp_servers/law_rag/engine.py#L561)）——传入"劳动"这样的短词也能匹配到"中华人民共和国劳动合同法"，模糊度相当高，没有精确匹配的选项。
- **过滤在检索之前生效，不是检索之后再筛**：`eligible_indices` 先算出来，BM25 只在这个子集里打分排序（测试 [`test_law_name_filter_is_applied_before_ranking`（tests/test_index_manager.py:333-345）](../../tests/test_index_manager.py#L333)），Dense 则如 §3.3 所述改为全量扫描后再筛选。过滤后如果候选为空列表，`search()` 会提前短路，返回专门的 `no_match_reason="law_name_filter_has_no_candidates"`（[engine.py:674-678](../../mcp_servers/law_rag/engine.py#L674)），BM25/Dense 都不会再跑。

`search()` 的其他两个边界值得一起记下：`retrieval_mode` 只有两个合法取值 `bm25` 和 `hybrid`（[engine.py:670-671](../../mcp_servers/law_rag/engine.py#L670)，`config.py:93` 用 `Literal["bm25", "hybrid"]` 做了静态约束）——**没有"dense-only"模式**，即使只想单独验证 Dense 效果，BM25 这一路的代码也一定会跑；`top_k` 会被无提示地夹到 `[1, 20]`（[engine.py:672](../../mcp_servers/law_rag/engine.py#L672)），传 0、负数或 100 都不会报错，只会被静默 clamp。

### 3.8 阈值分散在四处，以及只看 Top-1 的置信度门

本项目的检索链路里实际上有 **四层独立的分数阈值**，而不是"粗排 + 精排"两层：BM25 单候选阈值（`RAG_BM25_MIN_SCORE`，§3.3）、Dense 单候选阈值（`RAG_DENSE_MIN_SCORE`，§3.3）、RRF 融合后阈值（`RAG_RRF_MIN_SCORE`，§3.4）、以及精排阈值（`RAG_RERANK_MIN_SCORE`，§3.5）。每一层都独立生效、互不知晓——这意味着一个候选可能在"综合来看还不错"，却在某一路单独判定里被拒收：比如它的 Dense 相似度略低于 `0.20` 被 Dense 路直接丢弃，即使它的 BM25 分数很高，也只能靠 BM25 单路的 RRF 贡献值挤进最终排序，损失了"两路都命中"应有的分数叠加。四处阈值各自都是经验值，互相之间没有做过联合校准。

在这四层之上，`search()` 还会跑一次 [`RetrievalConfidenceGate.evaluate`（mcp_servers/law_rag/confidence.py:59-86）](../../mcp_servers/law_rag/confidence.py#L59)（默认开启，`RAG_MATCH_GATE_ENABLED` 默认 `True`，[config.py:90](../../backend/app/core/config.py#L90)），如果判定不通过，会把**整个结果列表清空**（[engine.py:788-792](../../mcp_servers/law_rag/engine.py#L788)）。它的机制是：只取 Top-1 候选（[confidence.py:63](../../mcp_servers/law_rag/confidence.py#L63)），把它的 `bm25`/`dense`/`rrf`/`rerank` 分数各自用不同公式归一化到 `[0,1]`（[confidence.py:50-57](../../mcp_servers/law_rag/confidence.py#L50)，比如 RRF 是按"两路都命中且都排第一"时的理论最大值 `2/61` 做归一化），再加上"两路是否同时命中 Top-1"（`source_agreement`）、"Top-1 正文覆盖了多少查询关键词"（`query_coverage`）、"Top-1 和 Top-2 的分数差距"（`top_margin`）三个特征，按 [`DEFAULTS.weights`（confidence.py:26-38）](../../mcp_servers/law_rag/confidence.py#L26)（`rerank 0.30 / dense 0.15 / bm25 0.15 / rrf 0.10 / source_agreement 0.15 / query_coverage 0.10 / top_margin 0.05`）加权求和，和阈值 `0.25` 比较。实际生效的权重和阈值来自 [`evals/config/retrieval-gate-v1.json`](../../evals/config/retrieval-gate-v1.json)，文件里当前的数值和代码里的 `DEFAULTS` 完全一致，并显式标了 `"calibration_status": "provisional"`。

这带来一个和"阈值分散在四处"同源、但更容易被忽略的边界情况：**置信度门只评估 Top-1，但清空的是整个列表**。如果 Top-1 恰好是一个"综合分不上不下"的候选（比如只被一路命中、和 Top-2 分差很小），即使 Top-2、Top-3 单独看质量都不错、且已经分别通过了各自那一路的阈值，也会被这次基于 Top-1 的单点判定一起清空。这是"阈值分散在多处独立生效"这个设计选择在系统边界上的一个具体后果，面试被问"你们的检索有没有做置信度校准"时，比笼统回答"有"更有说服力的答法是讲清楚这个机制和它的已知局限。

> 索引/语料库规模：工作区里最近一次建库的 [manifest.json](../../data/indexes/law/manifest.json) 记录 `source_document_count: 55348`、`chunk_count: 55374`、`built_at: 2026-08-21`。这是当前工作区实际落盘的数字，不是理论估算，但也不是一个恒定的产品规模承诺——法规源数据变化后这两个数字会随之重新计算，讲述时应带上"截至上次建库"这个限定，而不是当成固定不变的项目规模。

## 4. 设计取舍

**为什么用 RRF 而不是加权求和？** BM25 分数和余弦相似度不在同一量纲，直接线性加权需要额外归一化和参数标定；RRF 只依赖排名，首版实现更稳定、更容易向别人解释清楚"为什么这条排在前面"。代价是丢失了"这条候选到底有多好"的绝对强度信息，只保留了相对顺序——这也是 §3.8 里"阈值分散在多处"这个问题的根源之一：既然融合阶段本身已经放弃了绝对分数，各路只能各自在自己的量纲里独立设阈值，没有办法在融合后统一做一次"综合分是否够好"的判断（置信度门算是对这个缺口的一次补救，但它自己又引入了"只看 Top-1"的新局限）。

**为什么用 `IndexFlatIP` 而不是近似索引？** 按当前工作区最近一次建库的规模（约 5.5 万个 chunk，见 §3.8 末尾的限定说明），精确内积搜索的延迟和内存都还能接受，换来的是**不会有 ANN 近似索引本身带来的召回损失**——`IndexFlatIP` 里的 `Flat` 就是"不训练、不压缩、不过滤，每次查询都和全部向量算一遍距离"。这是用查询延迟换检索质量的确定性，规模继续增长后才需要考虑近似索引。

**为什么精排候选数默认是 12，又不是真正"固定"的？** 精排是一次 HTTP 请求批量提交，候选数越多单次请求延迟越高；`RAG_RERANK_CANDIDATE_COUNT=12` 是"覆盖 RRF 阶段可能的排序误差"和"精排延迟可控"之间的一个经验取值，不是理论最优值。但它只是这个数字的下界——实际保留数是 `max(top_k, RAG_RERANK_CANDIDATE_COUNT)`（§3.4），`top_k` 调大之后精排批量会跟着涨，且没有和 TEI 自身的 `--max-client-batch-size`（默认同样是 16）做过显式协调（§3.5 末尾）。这是"经验取值"没有随着参数空间扩大而被重新审视的一个具体例子。

**为什么本地部署 Embedding 而不是调云端 API？** 全量法规建库需要对约 5.5 万条内容发起大量向量化请求，云端按 token 计费会产生明显成本，同时法律咨询数据也不适合无谓地经过第三方网络出口。本地 Ollama 换来的是零边际成本和数据不出域，代价是要自己管理模型生命周期（见 §6）。

## 5. 易错点

- **把 `top_k` 当作"只影响返回数量"的参数**：实际上它通过 `max(30, top_k*4)` 同时改变了两路候选池宽度，还通过 `max(top_k, RAG_RERANK_CANDIDATE_COUNT)` 影响精排候选数。对比两次实验如果 `top_k` 不同，结果差异可能来自候选池大小或精排候选数变化，而不是被测试的那个变量（检索模式/精排开关）。
- **关闭 Reranker 时以为 Dense 也被关了**：精排和 Dense 是两个独立开关，关闭精排不会关闭 Dense 召回；要做"纯 BM25"对比，必须显式设置 `retrieval_mode=bm25`——并且要知道这个模式没有对应的"dense-only"版本（§3.7）。
- **把"精排服务故障降级"和"精排正常判定不相关"当成同一种异常**：前者是 `RerankerUnavailable` 触发的整批回退，用户仍能拿到未精排的 RRF 结果；后者是精排正常完成、分数低于 `RAG_RERANK_MIN_SCORE` 导致最终列表真的是空的（§3.5）。两者的运维含义完全不同，混为一谈会导致"检索是不是坏了"这个问题被误诊。
- **以为精排冷却期内的请求会重试或排队**：冷却期内到达的请求会在进入网络调用之前就被拒绝，也不会延长或重置冷却窗口（§3.5）；冷却状态是单个进程内的实例属性，多进程部署下各自独立计时，不是全局熔断。
- **以为 `filters` 是通用的元数据过滤器**：目前只支持 `law_name` 一个维度，且是归一化后的双向模糊子串匹配，不是精确匹配、也不支持组合条件（§3.7）；传其他字段会直接抛异常而不是被忽略。
- **以为检索候选只要综合分不错就一定会被采纳**：置信度门只看 Top-1 的加权分数，一旦判定不通过会清空整个结果列表，即使排名靠后的候选各自都已经通过了自己那一路的阈值（§3.8）。
- **修改 Embedding 查询指令文本但没提升 `legal-query-v1` 这个版本号**：查询指令文本本身不参与指纹计算，只有指令**版本号**进指纹；改了指令文字但忘记提升版本号，索引不会重建，会出现"代码变了但检索结果没变"的诡异现象。
- **把"没有引入向量数据库"和"没有做向量检索"混为一谈**：本项目确实做了完整的 Dense 向量检索，只是没有引入 Milvus/Pinecone/Weaviate 这类独立的向量数据库服务，用的是进程内的 FAISS 文件索引——这两件事完全不冲突，面试时说清楚这个区别，比笼统说"用了向量数据库"更准确也更显专业。

## 6. 生产化差距与面试应对

这套实现在算法链路（BM25→Dense→RRF→精排→置信度门→证据闭环）上已经是一套完整、可验证的混合检索方案，但离生产级 RAG 服务还有几处明确差距，面试被追问时应该主动讲清楚，而不是被问到才尴尬地承认：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 向量存储 | FAISS 本地文件索引，进程内加载，`IndexFlatIP` 精确搜索，截至上次建库约 5.5 万个 chunk（[manifest.json](../../data/indexes/law/manifest.json)） | 独立向量数据库服务（Milvus/Pinecone/Weaviate/Qdrant/pgvector），支持分布式分片、在线增量更新、多租户隔离 | "当前规模用进程内 FAISS 足够，如果法规量级或并发规模上升到需要水平扩展、在线增量更新，会评估独立向量数据库服务，索引指纹和 chunk 元数据体系可以直接迁移过去" |
| 索引更新方式 | 离线全量重建 + staging + 双跳原子切换（先备份旧索引再切新索引，失败原路恢复），不支持单条 CRUD | 增量 upsert/delete，通常配合变更数据捕获（CDC）或消息队列驱动索引更新 | "当前法规更新频率低，全量重建可接受；生产场景如果法条频繁更新，需要引入增量索引能力" |
| 检索融合与阈值 | 单机内 RRF，两路等权、无学习排序；BM25/Dense/RRF/精排四层阈值各自独立设定、互不知晓，且和"只看 Top-1"的置信度门叠加生效 | 大规模场景常引入 Learning to Rank（LTR）或专门训练的融合模型，权重和阈值可学习而非固定分散设置 | "RRF 和分层阈值是可解释性优先的首版方案，代价是可能出现'综合分不错但被某一路单独拒收'的边界情况；下一步是收集点击/采纳反馈做 LTR 或统一的置信度模型，而不是继续手调分散的固定阈值" |
| 精排部署 | 单个 TEI 实例，无水平扩展、无排队保护，超时/失败后的冷却状态是进程内内存，不跨副本共享 | 精排服务通常做成独立可扩展的推理服务（多副本 + 负载均衡 + 请求排队/批处理），并有独立的 SLA 监控和跨副本共享的熔断状态 | "当前 TEI 是单实例、有超时+冷却兜底，但冷却是单进程状态；生产环境会加多副本、排队机制和共享的熔断状态存储，避免单点故障导致全部请求退化到 RRF" |
| 阈值/置信度标定 | BM25/Dense/RRF/精排阈值和置信度门权重均为经验值，[evals/config/retrieval-gate-v1.json](../../evals/config/retrieval-gate-v1.json) 显式标注 `calibration_status: provisional` | 生产系统会用持续收集的真实查询日志和标注反馈做在线/离线阈值校准，形成闭环 | "当前阈值靠冻结验证集做过一次校准，配置文件里也明确标了 provisional，还没有接入生产反馈闭环；这是从'能用'到'持续优化'之间的差距" |
| 领域适配 | Jieba 通用词典分词，无法律领域词典 | 生产法律检索系统通常会训练领域词典、领域 Embedding 微调，甚至法律实体识别前置 | "当前用通用分词器，如果要进一步提升精确法律术语召回，下一步是引入领域词典或对 Embedding 模型做领域微调" |

## 7. 动手验证方式

1. 分别用 `RAG_RETRIEVAL_MODE=bm25` 和默认 `hybrid` 跑同一个口语化查询（比如"公司一直不跟我签合同"），对比返回的法条是否相同、排名是否变化。
2. 找一条同时包含法律名称精确匹配和语义改写的查询，观察 `retrieval_scores` 里 `bm25`、`dense`、`rrf` 三个分数分别是多少，哪一路对这条查询贡献更大。
3. 临时把 `RAG_RERANK_CANDIDATE_COUNT` 改小（比如改成 3），重跑一次 `top_k` 较大（比如 8）的查询，观察是否有原本该进入结果的候选因为精排候选池太窄而被漏掉——直观感受"粗排候选池宽度"和"最终质量"之间的关系。
4. 用 `filters={"law_name": "劳动"}` 这种短词跑一次查询，观察它匹配到了哪些法律名称，验证 §3.7 里"归一化后双向子串匹配"到底有多模糊。
5. 把 `RAG_RERANK_MIN_SCORE` 临时调到一个很高的值（比如 `0.9`）重跑一次能召回结果的查询，对比这时候的空结果和"直接停掉 TEI 服务"造成的空结果——分别在响应的 `rerank_applied` 字段上确认这是 §3.5 里说的两条不同路径。

**自测题：**

- 如果 BM25 单独召回 5 个候选、Dense 单独召回 5 个候选，其中恰好有 1 个 chunk 同时被两路命中且都排第一，这个 chunk 的 RRF 分数大约是多少？（提示：套 §3.4 的公式）
- 为什么本项目选择"先按候选池宽度召回、精排后再截断 `top_k`"，而不是"先截断 `top_k`、再精排"？如果反过来做会有什么风险？
- 一次查询的 Top-1 候选只被 BM25 单路命中（Dense 相似度略低于 `RAG_DENSE_MIN_SCORE` 被丢弃），Top-2 候选被两路同时命中、`rerank` 分数也很高。置信度门最终会不会清空整个结果列表？为什么只看 Top-2 更好或更差不能改变这个结论？（提示：重读 §3.8 关于置信度门"只评估 Top-1"的部分）
