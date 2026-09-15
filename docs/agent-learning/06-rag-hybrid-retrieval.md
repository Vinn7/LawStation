# RAG 混合检索与 Reranker（BM25 + Dense + RRF + BGE）

> 面向：知道"RAG 就是检索加生成"，但没有亲手算过 BM25 分数、没有亲手调过混合检索权重、没有踩过 Reranker 延迟坑的后端工程师。
> 目标：先把 BM25/Dense/RRF/精排这几个算法本身用手算例子讲透，再看本项目具体用什么库、什么参数把它们接起来，最后搞清楚"向量数据库"这个词在本项目里该怎么讲才不算夸大。

## 0. 前置知识

不需要提前读其他篇，也不需要提前懂 BM25/Embedding 的具体公式——本篇 §2 会从零开始讲。本篇是检索链路的起点，[04-langgraph-stategraph.md](04-langgraph-stategraph.md) 的 Research 节点会调用这里讲的工具，[05-tool-calling-mcp.md](05-tool-calling-mcp.md) 讲的是"模型怎么决定调用"，本篇讲的是"调用之后到底怎么找到正确法条"。

## 1. 要解决的问题

用户的咨询用口语表达（"公司一直没和我签合同"），法条用正式表述（"未订立书面劳动合同"）；同时用户也会直接提到法律名称和条号（"劳动合同法第八十二条怎么算"），这种精确匹配又是自然语言语义检索的弱项。只选一种检索方式，必然在另一种查询上失分：

- 只用关键词/词法检索（如 BM25）：精确匹配法律名称、条号、专业术语很强，但"公司一直不给我签合同"这种口语化表达很难命中"未订立书面劳动合同"。
- 只用向量语义检索：能理解口语和正式表述的语义接近，但对法律名称、条号这类需要精确复现的字符串反而不敏感，容易被"意思相近但法律实体不对"的候选干扰。

法律场景还有一条更硬的约束：**回答里的每一条法条引用都必须能追溯到真实检索到的证据片段**，检索环节的候选边界直接决定了后面能不能防止模型编造法条。

## 2. 核心机制原理

> 本节完全脱离本项目代码，只讲"BM25/Dense/RRF/精排这几个东西本身在算什么"——带手算例子。看完本节，再进 §3 看本项目具体用什么库、什么参数落地这些概念，会轻松很多。

### 2.1 词法检索是什么：BM25 到底在算什么

BM25（Best Matching 25）是一个**纯统计方法**，源自 1970-90 年代的概率相关性框架研究，不涉及任何训练或神经网络。它只用三个统计量给"查询和某篇文档有多匹配"打分：

- **词频（TF）**：查询里的某个词，在这篇文档里出现了几次。
- **逆文档频率（IDF）**：这个词在**整个语料库**里有多稀有——出现的文档越少，这个词越"有区分度"，权重越高；几乎每篇文档都有的词（"的"、"是"），权重会趋近于 0。
- **文档长度归一化**：避免长文档单纯因为字多、词出现次数天然更多而占便宜。

**手算一个例子**，假设语料库里只有 5 篇极简"文档"（已经分好词）：

```
D1: 劳动 合同 应当 采用 书面 形式                       （6 词）
D2: 用人 单位 未 与 劳动者 订立 书面 劳动合同 应当 支付 双倍 工资  （12 词）
D3: 工资 应当 以 货币 形式 按月 支付                     （7 词）
D4: 劳动者 享有 休息 休假 的 权利                        （6 词）
D5: 用人 单位 应当 依法 建立 职工 名册 备查                （8 词）
```

查询是"书面 劳动合同"。先看两个查询词各自的"稀有程度"（IDF）：

- "书面"：5 篇里有 2 篇（D1、D2）包含它 —— 比较常见，IDF 较低（约 `0.34`）
- "劳动合同"：5 篇里只有 1 篇（D2）包含它 —— 更稀有，IDF 明显更高（约 `1.10`）

**"劳动合同"这个词的 IDF 是"书面"的 3 倍多**——这就是 BM25 的核心直觉：越稀有的词，命中时权重越大。把 TF/IDF/文档长度代入 BM25 公式（`k1=1.5, b=0.75` 是最常用的经验取值），实际算出来的最终得分：

```
D2 ≈ 1.16   （两个查询词都命中，尤其"劳动合同"这个稀有词命中权重很大）
D1 ≈ 0.38   （只命中"书面"这个相对常见的词）
D3/D4/D5 = 0（两个查询词都没出现）
```

D2 的分数是 D1 的 3 倍多，不是因为 D2 词多（更长的文档在归一化后其实是吃亏的），而是因为它命中了那个更稀有、更有区分度的词。**这也是为什么不需要额外维护一份停用词表去手动过滤"的"、"是"这类词——IDF 天然就把它们的权重压到接近于 0。**

BM25 对查询里的每一个词，都是用**同一套公式**去对全部文档打分，再把各词的贡献**累加**成最终得分——"操作"对每个词一视同仁，但因为每个词的 IDF 和词频不同，实际贡献天差地别。

### 2.2 语义检索是什么：Embedding 和向量相似度

BM25 的软肋是"词面不一样、意思一样"的情况——用户说"公司一直没和我签合同"，法条写"未订立书面劳动合同"，两句话几乎没有共同的词，BM25 会判定它们毫不相关。语义检索解决的正是这个问题：

- **Embedding 模型**把一段文本转换成一个高维向量（本项目用的模型是 1024 维），语义越接近的文本，向量在这个高维空间里的位置越接近。
- 检索时把查询也转成向量，去和所有候选文档的向量比"距离"（常用余弦相似度：两个向量夹角越小，越相似），距离近的排前面。

如果把"公司一直没和我签合同""未订立书面劳动合同""劳动者可以享受带薪年假"这三句话画在一个二维平面上（真实向量是 1024 维，这里只是打比方），前两句会落在很接近的位置——尽管它们没有一个字重合；第三句因为讲的是完全不同的话题，会落在很远的地方。**这正是 BM25 做不到、Dense 检索要补的能力。**

**一个常见的误解需要提前破除**：一听到"向量检索"，很容易联想到"要接一个专门的向量数据库服务"（Milvus/Pinecone 这类）。但"把文本变成向量"和"对向量做最近邻搜索"这两件事，和"是否需要一个独立的数据库服务"完全是两个维度——最近邻搜索本身可以用一个**进程内的库**实现（比如本项目用的 FAISS），不用额外起一个数据库服务，向量数据存在内存/本地文件里就够。本项目具体怎么做，§3.3 会展开。

另外一个值得知道的通用做法：一些专门为检索优化过的 Embedding 模型（本项目用的 `qwen3-embedding` 系列就是其中之一）在**查询侧**和**文档侧**用的编码方式并不完全一样——查询会额外套一句"任务指令"再编码，文档不会。这不是本项目的特殊设计，是这类"指令微调 Embedding 模型"的通用用法，§3.3 会讲具体为什么。

### 2.3 为什么两路要并存

行业里成熟的做法是"混合检索"（Hybrid Search）：词法路（BM25/TF-IDF 类算法）和向量路（Dense Embedding + 近似最近邻）各自独立召回一批候选，再融合成一个排序。这不是新概念——Elasticsearch/OpenSearch 的 `rank_features` 混合查询、Vespa 的多阶段排序、Weaviate 和 Qdrant 的原生 Hybrid API，本质都是同一个思路的不同实现。

### 2.4 两路结果怎么合并：加权求和 vs 排名融合

合并两路分数最直觉的做法是加权求和：`score = α · bm25_score + β · cosine_similarity`。问题是 BM25 分数没有固定上界（和词频、文档长度相关，上面的例子里 D2 的分数是 1.16，换一批数据完全可能变成十几），余弦相似度天然落在 `[-1, 1]`，两者不在同一量纲，需要额外做归一化，而归一化方式本身又要针对数据集调参，容易在换一批数据后重新失效。

行业里更常用、更稳的替代方案是 **RRF（Reciprocal Rank Fusion）**：不看分数本身，只看每路里的**排名**，按排名的倒数打分再相加：

```
rrf(候选) = Σ 1 / (常数 + 该候选在这一路里的名次)
```

因为只依赖排名（一个自然的相对顺序），不需要关心两路分数的量纲差异，这是 RRF 相比线性加权最大的优势，也是 Elasticsearch 8.8+、OpenSearch 2.11+ 把 RRF 作为内置混合检索融合方式的原因。

**手算一个例子**（常数取本项目实际用的 `61`）：某次查询里，候选 A 被 BM25 排第 1（名次从 0 开始记，即 `0`）、被 Dense 排第 3（名次 `2`）；候选 B 只被 BM25 排第 2（名次 `1`），Dense 完全没召回它；候选 C 只被 Dense 排第 1（名次 `0`），BM25 没召回它：

```
rrf(A) = 1/(61+0) + 1/(61+2) = 0.01639 + 0.01587 = 0.03226   ← 两路都命中，即使排名不是双第一
rrf(C) = 1/(61+0)                                = 0.01639   ← 只被一路排第一
rrf(B) =              1/(61+1)                   = 0.01613   ← 只被一路排第二
```

最终排序 A > C > B——**候选 A 虽然在任何一路里都不是绝对第一，但因为两路都命中了它，最终反而排在只被单路命中的候选前面**，这正是"两路一致同意"这个信号被 RRF 放大的地方。代价是丢失了"这条候选到底有多好"的绝对强度信息，只保留了相对顺序。

### 2.5 精排（Reranking）解决什么问题

粗排（BM25/Dense/RRF）为了速度，通常用"查询和文档各自独立编码、算一个相似度分数"的方式（表示学习，representation-based），计算便宜但精度有限——两段文本是分开编码的，模型看不到它们之间逐词的交互。精排环节换用 **Cross-Encoder**：把查询和候选文档拼在一起，一起输入模型做联合编码，能捕捉更细粒度的语义交互（比如"未支付加班费"和"未足额支付加班费"这种细微但法律意义完全不同的差别），代价是必须对每个候选单独跑一次模型推理，不能像向量检索一样提前离线计算好、只留查询时做最近邻搜索。所以精排几乎总是放在"粗排先筛出一小批候选，精排只处理这一小批"的两阶段架构里，行业里 Cohere Rerank、Voyage Rerank、开源的 BGE-Reranker/Jina Reranker 都是这个定位。

## 3. 本项目具体实现（函数级）

> 本节所有代码引用都对照当前工作区源码逐行核实过；行号会随代码演进漂移，读到行号对不上时以源文件为准。**§2 已经讲过 BM25/Dense/RRF/精排本身是什么，本节不再重复概念，只讲本项目具体用什么库、什么参数、什么额外的工程考量把这些概念落地。**

### 3.1 法条切分与稳定 ID：`load_chunks`

**功能**：整条检索链路的起点——把原始法规 JSON（键是"法律名称+条号"，值是条文正文）转换成一份统一格式的 chunk 列表。后面 BM25 建索引、Dense 算向量、RRF 融合、精排、证据链回填，全部都是在这份 chunk 列表上操作的，是所有环节共用的基础数据结构。

**要解决的问题**：原始法条长度差异极大——有的条文几十字，有的（比如附则、罚则汇总条款）能到几千字。检索和精排都需要"大小适中"的文本单元：太长，Embedding 模型编码出来的向量会把好几个不同的语义点混在一起，精排 Cross-Encoder 的计算量也会线性增长；太短又会丢失上下文。所以需要切分：短条文原样保留一个 chunk，长条文按固定字数切开，相邻切片之间留一段重叠，避免关键信息恰好跨在两个切片分界线上。切完之后，还要给每个 chunk 一个**稳定、唯一、可追溯**的身份 ID——后面每一环节都靠这个 ID 体系相互指代，ID 不稳定或重复，后面每一层都会连锁出错。

**代码实现**：[load_chunks（engine.py:65-85）](../../mcp_servers/law_rag/engine.py#L65) 把法规 JSON 的每个键解析为法律名称和条号：用正则 `(第[...]条)$` 从键尾部截出条号，剩下的部分是法律名称。真正的切分逻辑在 [split_text（engine.py:47-62）](../../mcp_servers/law_rag/engine.py#L47)：只有超过 `maximum` 字才切分；边界优先在切分点前半段区间里找最靠右的换行/句号/分号，找不到就直接硬切——如果一段条文连续几百字没有这三种标点，切片会从字中间断开，这是当前实现完全没有兜底的边界情况。

`document_id` 是对 JSON 键原文（法律名称+条号拼接后的完整字符串，例如"劳动合同法第八十二条"）做 SHA-1（[engine.py:72](../../mcp_servers/law_rag/engine.py#L72)）——标识的是"一条具体条文"这个原始条目本身，不是整部法律；`chunk_id` 由 `document_id:chunk_index:content_hash` 拼接后再做一次 SHA-1（[engine.py:74-75](../../mcp_servers/law_rag/engine.py#L74)）。这两层 ID 拆开，是为了让"长法条命中了后半部分"这件事精确记录下来：同一条文的多个切片共享 `document_id`，但各自有独立 `chunk_id`。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `index_chunk_max_chars` | `1000` | 超过这个字数的条文才会被切分 |
| `index_chunk_overlap_chars` | `150` | 相邻切片的重叠字数 |

### 3.2 索引指纹与幂等建库

**功能**：把 3.1 切好的 chunk 列表转换成两份可以快速加载、反复查询的索引——BM25 的词频统计表、Dense 的向量矩阵——并持久化到磁盘。BM25 建索引很便宜，进程重启重新算一遍无所谓；Dense 需要对几万个 chunk 逐个调用 Embedding 模型编码，这个过程很慢，必须能被复用。

**要解决的问题**：两个独立的工程问题。**第一，什么时候该重建索引？** Dense 索引里的每个向量，都是在某个特定 Embedding 模型、特定切分参数、特定数据版本下算出来的——这几个前提条件里任何一个变了，旧索引里的向量和新查询产生的向量就不再处于同一个语义空间，必须重建；但前提都没变，就不该每次启动都重新算一遍。所以给这几个前提算一个**指纹**，变了才重建。**第二，建库过程本身可能失败。** 给几万个 chunk 调用 Embedding 服务，中途网络抖动、进程被杀、磁盘写到一半都可能发生，索引不能因此变成一份损坏的半新半旧文件。

**代码实现**：[LawSearchEngine._fingerprint（engine.py:150-164）](../../mcp_servers/law_rag/engine.py#L150) 参与哈希的是 9 个字段：`source_sha256`（数据 SHA-256）、`chunker_version`、`max_chars`、`overlap_chars`、`embedding_provider`、`embedding_model`、`embedding_model_digest`、`embedding_dimension`、`query_instruction_version`。模型 digest 进指纹很关键：即使模型名字没变，本地模型被重新下载或悄悄升级了版本，指纹也会变化并触发重建。

建库 [_build（engine.py:375-521）](../../mcp_servers/law_rag/engine.py#L375) 是原子化、可恢复的：新索引先写到 `{index_dir}/.staging-<fingerprint>/`，每批调用一次 Embedding（批大小取 `index_build_batch_size`/`ollama_embedding_batch_size`/硬上限 20 三者最小值，当前两个配置默认值都是 8）。每批先写临时文件再原子替换，重启时只重新生成缺失或损坏的批次。全部批次生成完毕、通过完整性重读校验后，才把 staging 目录切到正式路径——这一步实际是**两跳**：先把已存在的旧索引备份到 `.law-backup`，再把 staging 换到正式路径；第二步失败会把备份换回来。内存中的 FAISS 实例最后在锁内热切换。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `index_dir` | `./data/indexes` | 索引文件根目录 |
| `index_build_batch_size` / `ollama_embedding_batch_size` | 8 / 8 | 建库时每批调用 Embedding 的 chunk 数 |

### 3.3 双路召回：BM25 + Dense 在本项目里具体怎么接

§2.1/2.2 已经讲过 BM25/Dense 本身在算什么，这里只讲本项目用什么库、什么参数落地这两个概念。

**BM25 路**：用的是 `rank_bm25` 库（0.2.2 版本）的 `BM25Okapi` 类，[engine.py:107](../../mcp_servers/law_rag/engine.py#L107) 在构造引擎时直接对全部 chunk 建索引，用的是库默认的 `k1=1.5, b=0.75`（和 §2.1 手算例子用的参数一致，本项目未做定制调参）。查询发生时，[_lexical（engine.py:572-601）](../../mcp_servers/law_rag/engine.py#L572) 先用 Jieba 对查询分词（`jieba.lcut(text.lower())`，和建库时对文档正文分词用的是同一个 `tokens()` 函数，两边分词逻辑一致），再调用 `self.bm25.get_scores(...)` 打分，按分数降序取前 `pool` 个，过滤掉低于阈值的候选。

**Dense 路**：查询文本套一层检索指令模板 `Instruct: {ollama_query_instruction}\nQuery: {query}`（[embeddings.py:107-109](../../mcp_servers/law_rag/embeddings.py#L107)，这就是 §2.2 提到的"查询侧额外套指令"），文档建库时用 `embed_documents` 不套这层模板，是原文直接编码——查询在检索这一刻明确知道任务是"检索匹配"，文档在建库时却不知道未来会被哪个任务查询，所以指令只加在查询侧。查询送入 Ollama `qwen3-embedding:0.6b` 编码成 1024 维向量，`faiss.normalize_L2` 做归一化后（两边都归一化，内积才在数学上等价于余弦相似度），用 `faiss.IndexFlatIP`（[_faiss_search，engine.py:603-627](../../mcp_servers/law_rag/engine.py#L603)）做内积搜索——`IndexFlatIP` 就是 §2.2 说的"进程内库"而非独立数据库服务，`Flat` 的含义是"不训练、不压缩，每次查询都和全部向量精确算一遍距离"。`self.faiss.search(...)` 本身是阻塞的 C++ 调用，要用 `asyncio.to_thread` 丢到线程池执行，外面还包了一把 `_dense_lock`（防止"正在查询"和"建库完成后热切换索引"两件事互相冲突）。查询 Embedding 是一次实时网络请求（每次查询真的要向 Ollama 服务发一次 HTTP 请求），不是本地瞬时计算。低于阈值的候选不进入融合。

**两路共用的候选池宽度**：由 `pool = min(candidate_count, max(30, top_k * 4))`（[engine.py:681](../../mcp_servers/law_rag/engine.py#L681)）决定——`top_k` 不只影响最终返回数量，还同时影响两路各自召回多宽，做对比实验必须固定 `top_k`。如果这次查询带了 `filters`（见 §3.7），Dense 路的池宽度会被强制改成 `len(self.docs)`（全量扫描），因为 `IndexFlatIP` 不支持按元数据下推过滤，只能先全量排序再在应用层筛选。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `RAG_BM25_MIN_SCORE` | `0.01` | BM25 单候选最低分 |
| `RAG_DENSE_MIN_SCORE` | `0.20` | Dense 单候选最低分 |
| `embedding_model` / `embedding_dimension` | `qwen3-embedding:0.6b` / `1024` | 编码模型和向量维度，必须和建库时一致 |
| `ollama_query_instruction` | "Given a legal consultation query..." | 查询侧套的指令模板文本，只加在查询，不加在文档 |

### 3.4 RRF 融合

§2.4 已经讲过 RRF 的原理和手算例子，本项目的常数和阈值：

```text
rrf(chunk) = Σ 1 / (61 + zero_based_rank_in_source)
```

常数 `61` 在两路累加里都是同一个值（[engine.py:687](../../mcp_servers/law_rag/engine.py#L687)、[engine.py:708](../../mcp_servers/law_rag/engine.py#L708)），和 §2.4 手算例子用的一致。这里的"名次"是**过阈值筛选之后**、在各自候选列表里的顺序，不是全库排名——被阈值挡掉的候选不占用名次编号。融合结果低于 `RAG_RRF_MIN_SCORE` 的会被过滤（[_rrf_order，engine.py:629-652](../../mcp_servers/law_rag/engine.py#L629)）。

融合完成后，会额外保留 `rerank_limit = max(top_k, RAG_RERANK_CANDIDATE_COUNT)`（[engine.py:714-718](../../mcp_servers/law_rag/engine.py#L714)）个候选交给下游精排——**不是写死的 12**：默认是 12，但 `top_k` 传得比 12 大时，精排候选池会跟着涨。精排完成后才把结果截断成最终的 `top_k`。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `RAG_RRF_MIN_SCORE` | `0.01` | 融合后最低分 |
| `RAG_RERANK_CANDIDATE_COUNT` | `12` | 精排候选数下限（实际值是 `max(top_k, 这个值)`） |

### 3.5 TEI + BGE Cross-Encoder 精排

§2.5 讲过精排解决什么问题（Cross-Encoder 联合编码 vs 粗排的独立编码）。本项目工程实现上处理了几件容易被忽略但很重要的事：响应必须严格校验、故障要能优雅降级、降级后要有冷却期。

**代码实现**：[TEIReranker.rerank（reranker.py:523-613）](../../mcp_servers/law_rag/reranker.py#L523) 调用独立部署的 Hugging Face TEI 服务，一次 `POST /rerank` 请求把查询和全部候选一起提交，拿到 Cross-Encoder 分数重新排序。响应校验（[parse_tei_ranks，reranker.py:134-160](../../mcp_servers/law_rag/reranker.py#L134)）不接受任何部分成功：候选数、索引唯一性、分数范围任一不满足都抛异常。

精排有两条完全不同性质的"清空"路径：**服务故障**（候选缺失/响应异常/HTTP 失败/超时，触发 `_degrade()` 抛 `RerankerUnavailable`，`search()` 捕获后直接保留精排前的 RRF 结果，`rerank_applied=False`，是降级不是失败）和**精排判定确实不相关**（TEI 正常返回但分数普遍低于 `RAG_RERANK_MIN_SCORE`，`ordered` 被过滤成空列表，`rerank_applied=True`，是真实完成的判定）。测试 [`test_reranker_failure_falls_back_to_rrf_not_no_match`](../../tests/test_index_manager.py#L438) 覆盖前者，[`test_reranker_threshold_can_return_normal_no_match`](../../tests/test_index_manager.py#L470) 覆盖后者。

冷却机制：故障后设置 `_cooldown_until`（默认 60 秒），冷却期内的请求直接拒绝、不发网络请求、也不重置窗口——这个状态是单进程内存属性，多进程部署下各自独立计时，不是全局熔断。

一个尚未被显式校验的耦合：精排候选数可达 `top_k` 上限 20，但本项目自己拉起 TEI 时，服务自身的 `--max-client-batch-size` 默认也是 16——`top_k` 落在 17~20 区间会在 TEI 侧被拒绝，静默退化成未精排结果。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `rag_rerank_enabled` | `True` | 精排总开关 |
| `RAG_RERANK_MIN_SCORE` | `0.0`（默认不生效） | 精排后最低分 |
| `RAG_RERANK_RETRY_SECONDS` | `60.0` | 故障后的冷却时长 |
| `RAG_RERANK_MAX_CLIENT_BATCH_SIZE` | `16` | TEI 服务自身批量上限（仅本项目自启动 TEI 时生效） |

### 3.6 证据链闭环：检索结果不能直接变成引用

**功能**：确保最终回答里出现的每一条法条引用，都能追溯到这一轮真实检索到的证据，不会被模型凭空编造或张冠李戴——检索只负责"找出候选"，模型自己写正文时依然有编造内容、引用错误 chunk、声称用了证据但其实没用这几种失败可能。这一层不是一道关卡，而是**四道独立的关卡**，各自堵住其中一种失败模式，任何一道单独失效，后面的还能兜住——这也是它被称为"闭环"而不是"一次校验"的原因。

**第一道关：Research 节点权威映射——模型只能"选"，不能"写"**

模型在 `EvidencePacket.evidence_items` 里选中一个 `chunk_id` 之后，[nodes/research.py:141](../../backend/app/agent/graph/nodes/research.py#L141) 立即调用 [`_authoritative_evidence`（graph/evidence.py:83-125）](../../backend/app/agent/graph/evidence.py#L83)：

```python
by_chunk: dict[str, dict] = {}      # chunk_id -> 本轮真实候选
by_document: dict[str, list] = {}   # document_id -> 该法条本轮命中的全部 chunk
for candidate in candidates: ...    # 从真实 ToolMessage 里收集，不是模型说了算

for selected in packet.evidence_items:
    source = by_chunk.get(selected.chunk_id) if selected.chunk_id else None
    if source is None and not selected.chunk_id:
        legacy_matches = by_document.get(selected.document_id, [])
        if len(legacy_matches) == 1:          # 只有唯一匹配才允许旧格式兼容
            source = legacy_matches[0]
    if source is None:
        continue                              # 伪造/歧义/本轮不存在的 ID：静默丢弃
    accepted.append(EvidenceItem(
        document_id=..., chunk_id=..., law_name=..., article_number=..., content=...,
    ))  # law_name/article_number/content 全部来自 source（真实候选），不是 selected（模型输出）
```

**关键在最后一步的字段来源**：新构造的 `EvidenceItem` 里，`law_name`/`article_number`/`content` 全部取自 `source`（本轮 MCP 工具真实返回的候选对象），模型自己在 `selected` 里写的任何正文/法名/条号都被**直接丢弃、从不采用**——模型能做的只有"用 `chunk_id` 指向哪一个真实候选"这一个动作，指向之后具体内容是什么，由服务端重新查真实数据回填。这就是为什么就算模型在这一步"编"了一段听起来很像的法条正文，也不可能进入 `EvidenceItem`：它写的正文根本没有被读取。`chunk_id` 找不到对应候选（伪造 ID、拼错、指向别的会话）时，`source is None`，这条证据被整条丢弃，不会有"部分采信"的中间状态。

**第二道关：兜底 Selector——候选存在但模型没表态时，换一次更受限的调用**

如果模型既没在 `evidence_items` 里接受任何候选，也没在 `rejected_candidates` 里明确拒绝（[research.py:142](../../backend/app/agent/graph/nodes/research.py#L142) 的 `if candidates and not accepted and not raw_packet.rejected_candidates`），说明模型很可能是"看漏了"而不是"确认没有相关的"——这时会追加一次**不带任何工具**的 Evidence Selector 子调用，输入只有候选的 `chunk_id`/`law_name`/`article_number`/`content`，模型只能在这个封闭列表里选 `accepted_chunk_ids`。选完之后 [research.py:163-165](../../backend/app/agent/graph/nodes/research.py#L163) 还要用候选 ID 集合把 Selector 的输出再裁剪一遍（`accepted_ids = {item for item in selection.accepted_chunk_ids if item in valid_ids}`）——防止这次子调用自己也编出一个不存在的 ID；裁剪完的结果重新走一遍第一道关的 `_authoritative_evidence`，不会跳过权威映射这一步。

**第三道关：Review 节点的确定性引用校验——能强制推翻模型自己给出的"通过"结论**

这一道关最容易被低估。[nodes/review.py:47](../../backend/app/agent/graph/nodes/review.py#L47) 调用 [`_citation_errors`（graph/evidence.py:211-239）](../../backend/app/agent/graph/evidence.py#L211)，它做两件事：(1) 草稿 `claims` 里声明引用的每个 ID，是否都在证据包的 `known_chunk_ids` 里，或者满足"该 `document_id` 本轮只对应一个 chunk"这个兼容旧格式的条件；(2) 用正则 `《([^》]+)》\s*(第[...]条)` **扫描回答正文本身**，把每一处"《法律名称》第 N 条"这样的表述提取出来，检查是否真的能在证据包里找到法名/条号都匹配的条目——**这一步和第一步不一样：它不看草稿声明了什么，只看正文实际写了什么**，能拦住"claims 字段里老老实实没多写，但正文里偷偷多编了一条"这种情况。

真正关键的是这两类错误产生之后发生的事——[review.py:62-67](../../backend/app/agent/graph/nodes/review.py#L62)：

```python
deterministic_errors = _citation_errors(...)
deterministic_errors.extend(_no_match_violations(...))     # no_match 时额外检查
deterministic_errors.extend(_fact_boundary_errors(...))    # 是否还在用已被替换的旧事实
if deterministic_errors:
    review.approved = False                # 不管 LLM Reviewer 自己判了 approved=True 还是 False，
    review.next_action = "revise_draft"     # 只要确定性检查发现问题，强制改成不通过、要求改稿
```

也就是说，**Reviewer 这个 LLM Agent 自己给出的 `approved` 结论，从来不是最终结论**——它上面永远盖着一层确定性代码检查，任何一条硬错误命中，都会不由分说地把 `approved` 强制改成 `False`，模型的判断在这里只是"建议"，服务端代码才是最终裁决者。这三类确定性检查（引用归属、`no_match` 幻觉、事实边界）共用同一套"命中就否决"的机制，不是三个各自独立生效的小功能。

**第四道关：Finalize 节点——先决定"这段回答能不能原样输出"，再决定"引用列表里放什么"**

[`FinalizeNode.finalize`（graph/nodes/finalize.py:14-105）](../../backend/app/agent/graph/nodes/finalize.py#L14) 其实做了两件独立的事，现有版本的文档只讲了第二件，第一件同样重要：

- **步骤 3A/3B：整段回答的安全替换**（[finalize.py:33-55](../../backend/app/agent/graph/nodes/finalize.py#L33)）——如果 `retrieval_status == "no_match"`，还要再跑一遍 `_no_match_violations`（正则查有没有偷偷出现法名/条号、有没有披露"未检索到可引用法条"），只要有一条没通过，或者 Reviewer 没批准，整段 `answer` 直接被替换成固定安全模板（`_no_match_safe_answer`），不输出模型写的任何内容；如果不是 `no_match`，但 Reviewer 明确指出了无依据论断、遗漏争议点、引用错误或自相矛盾，也不输出原草稿，只列出"已核验到的材料"这份确定性摘要加一句"建议重新咨询"。这一步发生在 Citation 收集**之前**——先确保正文本身没有越界内容，再考虑给它配哪些引用。
- **步骤 5-7：草稿实际引用和证据包已核验证据的真正交集**（[finalize.py:64-91](../../backend/app/agent/graph/nodes/finalize.py#L64)）——先收集 `counsel_draft.claims` 里实际用到的全部 `chunk_id`/`document_id`（`referenced_ids`），再统计每个 `document_id` 本轮对应几个 chunk（`document_counts`），只遍历 `retrieval_status == "matched"` 的证据条目（`no_match` 恒产出零条引用，呼应上面 3A 的模板替换逻辑），对其中被草稿引用、或满足"该 `document_id` 本轮只有一个 chunk"这一兼容旧格式条件的条目去重（`seen_documents`）后，才构造最终 `Citation`，摘录截断到 240 字。**检索找到但回答没有真正用到的候选，不会出现在引用列表里**——"检索到"和"被引用"是两件独立的事，只有真正被用上、且通过了上面三道关的，才走到这一步。

**四道关各自堵住的失败模式**：

| 关卡 | 堵住什么 |
|---|---|
| ① Research 权威映射 | 模型编造正文、引用伪造/不存在的 chunk_id |
| ② 兜底 Selector | 模型看到候选但没有明确表态，导致证据被遗漏 |
| ③ Review 确定性校验 | 草稿声明之外，正文里偷偷多编的引用；Reviewer 自己误判"通过" |
| ④ Finalize 双重收敛 | 整体回答内容越界（no_match 幻觉/无依据论断）；检索到但没真正用上的候选混进引用列表 |

**相关参数**：这一层没有配置参数——是纯代码逻辑约束，不受任何环境变量控制。

### 3.7 `filters` 参数与 `search()` 的整体边界

**功能**：让调用方能把检索范围限定在某一部具体法律内，而不是永远全库检索。

**代码实现**：[_filtered_indices（engine.py:542-570）](../../mcp_servers/law_rag/engine.py#L542) 只接受 `law_name` 字段，其他字段直接抛异常。匹配用 `normalize_law_name` 去空格/书名号/大小写/"中华人民共和国"前缀后做**双向子串**判断——传"劳动"这种短词也能匹配到"中华人民共和国劳动合同法"，没有精确匹配选项。过滤在检索**之前**生效：BM25 只在过滤后的子集里打分排序，Dense 改为全量扫描后再筛选（见 §3.3）；过滤后候选为空会提前短路返回 `no_match_reason`。

`search()` 另外两个边界：`retrieval_mode` 只有 `bm25`/`hybrid` 两个合法值，**没有 dense-only 模式**；`top_k` 会被无提示地夹到 `[1,20]`。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `rag_retrieval_mode` | `hybrid` | 只有 `bm25`/`hybrid` 两个合法取值 |
| `filters`（调用参数） | `None` | 只支持 `{"law_name": "..."}`，双向模糊子串匹配 |
| `top_k`（调用参数） | `8`，范围 `[1,20]` | 最终返回数量，同时联动候选池宽度和精排候选数 |

### 3.8 阈值分散在四处，以及只看 Top-1 的置信度门

**功能**：在"检索到候选"和"返回给调用方"之间，再做一次整体质量判断——结果整体不够可信时，宁可返回"没查到"，也不把低质量候选硬塞给下游。

**要解决的问题**：前面每一路（BM25/Dense/RRF/精排）都各自有一道阈值，但这些阈值**独立生效**，没有环节站在"综合所有信号"的角度做整体判断——一个候选可能"综合来看还不错"，却在某一路单独判定里被拒收。置信度门是补这个缺口的机制，但它自己的实现方式（只看排名第一的候选）又带来了新的局限。

**代码实现**：在 BM25/Dense/RRF/精排四层独立阈值之上，`search()` 还会跑一次 [`RetrievalConfidenceGate.evaluate`（confidence.py:59-86）](../../mcp_servers/law_rag/confidence.py#L59)（默认开启），不通过就把**整个结果列表清空**（不只是清空 Top-1）。机制：只取 Top-1 候选，把它的 `bm25`/`dense`/`rrf`/`rerank` 分数各自归一化到 `[0,1]`，再加上"两路是否同时命中 Top-1""正文覆盖了多少查询关键词""Top-1 和 Top-2 的分数差距"三个特征，加权求和后和阈值 `0.25` 比较。权重和阈值来自 [evals/config/retrieval-gate-v1.json](../../evals/config/retrieval-gate-v1.json)，标注 `calibration_status: provisional`。

**置信度门只评估 Top-1，但清空的是整个列表**——如果 Top-1 恰好是一个"综合分不上不下"的候选，即使 Top-2、Top-3 单独看质量都不错、已经分别通过了各自阈值，也会被这次基于 Top-1 的单点判定一起清空。

**相关参数**：

| 参数 | 默认值 | 作用 |
|---|---|---|
| `RAG_MATCH_GATE_ENABLED` | `True` | 置信度门总开关 |
| `rag_match_gate_config_path` | `./evals/config/retrieval-gate-v1.json` | 权重和阈值配置文件 |

> 索引/语料库规模：工作区里最近一次建库的 [manifest.json](../../data/indexes/law/manifest.json) 记录 `source_document_count: 55348`、`chunk_count: 55374`、`built_at: 2026-08-21`——这是当前工作区实际落盘的数字，讲述时应带上"截至上次建库"的限定，不是恒定的产品规模承诺。

## 4. 设计取舍

**为什么用 RRF 而不是加权求和？** BM25 分数和余弦相似度不在同一量纲，直接线性加权需要额外归一化和参数标定；RRF 只依赖排名，首版实现更稳定、更容易向别人解释清楚"为什么这条排在前面"。代价是丢失了绝对强度信息——这也是 §3.8 里"阈值分散在多处"这个问题的根源之一：融合阶段本身已经放弃了绝对分数，各路只能各自在自己的量纲里独立设阈值（置信度门算是对这个缺口的一次补救，但它自己又引入了"只看 Top-1"的新局限）。

**为什么用 `IndexFlatIP` 而不是近似索引？** 按当前规模（约 5.5 万个 chunk），精确内积搜索的延迟和内存都还能接受，换来的是不会有 ANN 近似索引本身带来的召回损失。这是用查询延迟换检索质量的确定性，规模继续增长后才需要考虑近似索引。

**为什么精排候选数默认是 12，又不是真正"固定"的？** 候选数越多单次请求延迟越高；`12` 是"覆盖 RRF 阶段可能的排序误差"和"精排延迟可控"之间的一个经验取值。但它只是下界，实际保留数是 `max(top_k, 12)`，`top_k` 调大后精排批量会跟着涨，且没有和 TEI 自身的批量上限做过显式协调（§3.5）。

**为什么本地部署 Embedding 而不是调云端 API？** 全量法规建库需要对约 5.5 万条内容发起大量向量化请求，云端按 token 计费会产生明显成本，同时法律咨询数据也不适合无谓地经过第三方网络出口。本地 Ollama 换来零边际成本和数据不出域，代价是要自己管理模型生命周期（见 §6）。

## 5. 易错点

- **把 `top_k` 当作"只影响返回数量"的参数**：实际上它同时改变了两路候选池宽度和精排候选数。对比两次实验如果 `top_k` 不同，结果差异可能来自这些联动，而不是被测试的那个变量。
- **关闭 Reranker 时以为 Dense 也被关了**：精排和 Dense 是两个独立开关，要做"纯 BM25"对比必须显式设置 `retrieval_mode=bm25`——并且这个模式没有对应的"dense-only"版本。
- **把"精排服务故障降级"和"精排正常判定不相关"当成同一种异常**：前者是整批回退，用户仍能拿到未精排结果；后者是最终列表真的是空的。混为一谈会导致"检索是不是坏了"被误诊。
- **以为精排冷却期内的请求会重试或排队**：冷却期内的请求会在进入网络调用之前就被拒绝，冷却状态是单进程内存属性，不是全局熔断。
- **以为 `filters` 是通用的元数据过滤器**：目前只支持 `law_name` 一个维度，且是模糊匹配，传其他字段会直接抛异常。
- **以为检索候选只要综合分不错就一定会被采纳**：置信度门只看 Top-1，一旦不通过会清空整个结果列表。
- **修改 Embedding 查询指令文本但没提升版本号**：指令文本本身不参与指纹计算，只有版本号进指纹，改了文字忘记提版本号，索引不会重建。
- **把"没有引入向量数据库"和"没有做向量检索"混为一谈**：本项目做了完整的 Dense 向量检索，只是没有引入 Milvus/Pinecone 这类独立向量数据库服务，用的是进程内的 FAISS——两件事不冲突（§2.2 已经讲过这个区分）。

## 6. 生产化差距与面试应对

这套实现在算法链路（BM25→Dense→RRF→精排→置信度门→证据闭环）上已经是一套完整、可验证的混合检索方案，但离生产级 RAG 服务还有几处明确差距，面试被追问时应该主动讲清楚：

| 维度 | 本项目现状 | 生产环境通常怎么做 | 面试怎么答 |
|---|---|---|---|
| 向量存储 | FAISS 本地文件索引，进程内加载，`IndexFlatIP` 精确搜索，截至上次建库约 5.5 万个 chunk | 独立向量数据库服务（Milvus/Pinecone/Weaviate/Qdrant/pgvector），支持分布式分片、在线增量更新、多租户隔离 | "当前规模用进程内 FAISS 足够，如果法规量级或并发规模上升到需要水平扩展、在线增量更新，会评估独立向量数据库服务，索引指纹和 chunk 元数据体系可以直接迁移过去" |
| 索引更新方式 | 离线全量重建 + staging + 双跳原子切换，不支持单条 CRUD | 增量 upsert/delete，通常配合变更数据捕获（CDC）或消息队列驱动索引更新 | "当前法规更新频率低，全量重建可接受；生产场景如果法条频繁更新，需要引入增量索引能力" |
| 检索融合与阈值 | 单机内 RRF，两路等权、无学习排序；四层阈值各自独立设定，且和只看 Top-1 的置信度门叠加生效 | 大规模场景常引入 Learning to Rank（LTR）或专门训练的融合模型 | "RRF 和分层阈值是可解释性优先的首版方案，下一步是收集点击/采纳反馈做 LTR 或统一的置信度模型" |
| 精排部署 | 单个 TEI 实例，无水平扩展，冷却状态是进程内内存，不跨副本共享 | 独立可扩展的推理服务（多副本+负载均衡+排队/批处理），有跨副本共享的熔断状态 | "当前 TEI 是单实例，冷却是单进程状态；生产环境会加多副本、排队机制和共享的熔断状态存储" |
| 阈值/置信度标定 | 均为经验值，配置文件显式标注 `calibration_status: provisional` | 用持续收集的真实查询日志和标注反馈做在线/离线校准 | "当前阈值靠冻结验证集做过一次校准，还没接入生产反馈闭环" |
| 领域适配 | Jieba 通用词典分词，无法律领域词典 | 训练领域词典、领域 Embedding 微调，甚至法律实体识别前置 | "当前用通用分词器，下一步是引入领域词典或对 Embedding 模型做领域微调" |

## 7. 动手验证方式

1. 按 §2.1 的例子，用 `python -c` 手写一个极简 BM25 打分函数（或直接用 `rank_bm25`），验证"稀有词权重更高"这个直觉。
2. 分别用 `RAG_RETRIEVAL_MODE=bm25` 和默认 `hybrid` 跑同一个口语化查询（比如"公司一直不跟我签合同"），对比返回的法条是否相同、排名是否变化。
3. 找一条同时包含法律名称精确匹配和语义改写的查询，观察 `retrieval_scores` 里 `bm25`、`dense`、`rrf` 三个分数分别是多少，哪一路贡献更大。
4. 临时把 `RAG_RERANK_CANDIDATE_COUNT` 改小（比如改成 3），重跑一次 `top_k` 较大的查询，观察是否有候选因为精排候选池太窄而被漏掉。
5. 用 `filters={"law_name": "劳动"}` 这种短词跑一次查询，验证 §3.7 里"双向模糊子串匹配"到底有多模糊。
6. 把 `RAG_RERANK_MIN_SCORE` 临时调高重跑一次查询，对比这时候的空结果和"直接停掉 TEI 服务"造成的空结果，分别在响应的 `rerank_applied` 字段上确认这是 §3.5 说的两条不同路径。

**自测题：**

- 如果 BM25 单独召回 5 个候选、Dense 单独召回 5 个候选，其中恰好有 1 个 chunk 同时被两路命中且都排第一，这个 chunk 的 RRF 分数大约是多少？（提示：套 §2.4/§3.4 的公式，和正文的 A/B/C 例子算法一样，只是排名不同）
- 为什么本项目选择"先按候选池宽度召回、精排后再截断 `top_k`"，而不是"先截断 `top_k`、再精排"？如果反过来做会有什么风险？
- 一次查询的 Top-1 候选只被 BM25 单路命中（Dense 相似度略低于阈值被丢弃），Top-2 候选被两路同时命中、`rerank` 分数也很高。置信度门最终会不会清空整个结果列表？为什么"Top-2 表现更好"这件事本身改变不了这个结论？
- §2.1 的例子里，D2 的分数（≈1.16）远高于 D1（≈0.38），但 D2 比 D1 长一倍（12 词 vs 6 词）——按"文档长度归一化"的逻辑，长文档应该被打压才对，为什么 D2 还是明显赢了？（提示：对比"书面"和"劳动合同"两个词各自的贡献大小）
