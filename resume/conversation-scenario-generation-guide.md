# LawStation 多轮对话样例生成操作手册

## 1. 手册用途

本手册对应 LawStation 的“多轮对话样例生成与一次性执行计划”，用于：

```text
18 类确定性场景蓝图
→ DeepSeek 生成合成用户话术
→ 确定性安全与结构校验
→ 必要时定向修复单个变体
→ 冻结为版本化 JSONL 数据集
```

当前仓库已经完成过一次真实生成并冻结 36 条场景。日常查看和验证不需要再次调用模型。

> 重要边界：生成/冻结命令不会把36条场景发送给LawStation，也不会自动启动LawStation、Ollama或TEI。项目另提供默认关闭的前端“场景观察模式”，但它每次只在用户点击后执行一个步骤，不是自动Runner。“36条已生成”仍不等于“36条端到端测试已通过”。

## 2. 当前已冻结结果

主要产物位于：

```text
evals/conversations/
├── blueprints-v1.json
├── generated-candidates-v1.jsonl
├── generation-validation.json
├── lawstation-dialogue-scenarios-v1.jsonl
└── lawstation-dialogue-scenarios-v1.manifest.json
```

当前冻结版本：

| 项目 | 值 |
|---|---|
| 蓝图数 | 18 |
| 每类变体数 | 2 |
| 冻结场景数 | 36 |
| 生成模型 | `deepseek-v4-flash` |
| 初始模型调用 | 18 |
| 定向修复调用 | 1 |
| 总模型调用 | 19 |
| Dataset SHA256 | `a13e18feef7ba90c153b17f0133fd47d9eb17a555c7e3a76e9b07e08a1b3f27f` |

类别覆盖为：`routing=4`、`rag=6`、`skill=12`、`memory=6`、`concurrency=4`、`durable_run=4`。

## 3. 前置条件

### 3.1 进入项目和 Conda 环境

```bash
cd /Users/Admin1/Files/LawStation
conda activate LawStation
```

也可以不激活环境，统一使用：

```bash
conda run --no-capture-output -n LawStation python <命令>
```

### 3.2 配置检查

只有 `generate` 阶段需要访问 DeepSeek。根目录 `.env` 至少需要：

```dotenv
DEEPSEEK_API_KEY=<真实密钥>
TEST_SCENARIO_GENERATOR_MODEL=
TEST_SCENARIO_GENERATOR_TEMPERATURE=0.7
TEST_SCENARIO_GENERATOR_MAX_CALLS=18
TEST_SCENARIO_VARIANTS_PER_BLUEPRINT=2
TEST_SCENARIO_GENERATOR_SEED=42
```

`TEST_SCENARIO_GENERATOR_MODEL` 为空时复用 `DEEPSEEK_MODEL`。生成器使用非流式 JSON Output，并显式关闭 Thinking；它不会发现或调用 MCP 工具。

`TEST_SCENARIOS_ENABLED=false`时不读取场景文件，后端场景API返回404，前端不显示入口。推荐不修改该默认值，而是用`python run.py --test-scenarios`只为本次进程开启；`--no-test-scenarios`可覆盖`.env`强制关闭。启动参数只提供手工逐步观察，不会在应用启动时自动运行场景。

### 3.3 各阶段外部资源

| 命令 | DeepSeek | LawStation | Ollama/TEI | 主要写入 |
|---|---:|---:|---:|---|
| `prepare` | 否 | 否 | 否 | 蓝图、checkpoint 目录 |
| `generate` | 是 | 否 | 否 | checkpoint、候选和校验结果 |
| `validate` | 否 | 否 | 否 | 重建候选和校验结果 |
| `freeze` | 否 | 否 | 否 | 冻结 JSONL 和 manifest |

## 4. 推荐操作路径

### 4.1 只查看和核验当前冻结数据（推荐）

当前 36 条样例已经冻结，通常只需要运行离线测试：

```bash
conda run --no-capture-output -n LawStation \
  pytest -q tests/test_conversation_scenarios.py
```

查看 manifest：

```bash
python -m json.tool \
  evals/conversations/lawstation-dialogue-scenarios-v1.manifest.json
```

查看前 2 条冻结场景：

```bash
python - <<'PY'
import json
from pathlib import Path

path = Path("evals/conversations/lawstation-dialogue-scenarios-v1.jsonl")
for line in path.read_text("utf-8").splitlines()[:2]:
    print(json.dumps(json.loads(line), ensure_ascii=False, indent=2))
PY
```

这条路径不调用任何外部模型，也不会改写冻结数据。

## 4.2 启动场景观察模式

```bash
conda activate LawStation
python run.py --test-scenarios
```

启动器会先校验JSONL、manifest和SHA256，再启动本地模型与主服务。成功日志会显示Dataset ID、36条场景及“顶部栏 / 左侧栏”入口。若配置或数据无效，会在Ollama、TEI和Uvicorn启动前退出。

进入页面后，每次点击“执行下一步”只消费一个动作。绿色结果必须具有实际终态、调用计数、Run重叠、SSE序号或记忆来源消息等证据；未知断言、Fixture依赖项、Checkpoint不可用和来源消息不足会显示`inconclusive`，而不是误判为通过。

### 4.2 完整生成与冻结流程

只有在蓝图、Prompt 版本、生成模型或样例版本需要更新时，才执行完整流程。

#### 第一步：准备蓝图

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py prepare
```

该阶段：

- 从 `backend/app/evaluation/conversation_scenarios.py::blueprint_definitions()` 读取 18 类确定性模板；
- 从当前 `LAW_DATA_PATH` 加载法规 chunk，仅为需要来源约束的蓝图选择 source；
- 将蓝图写入 `evals/conversations/blueprints-v1.json`；
- 创建 `evals/conversations/.checkpoints/`。

如果已有蓝图文件与当前代码生成结果不一致，命令会停止并拒绝静默覆盖。应先审查差异并创建新数据集版本，不应直接删除旧文件绕过保护。

#### 第二步：生成合成话术

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py generate
```

每个蓝图默认触发一次模型调用并返回两个变体。模型只能生成：

```text
title
messages[]
```

Actor、会话、用户切换、发送、取消、断线重连、预期 SSE 事件、Skill ID 和工具权限均由代码模板固定，不能由模型自由生成。

生成过程中每个蓝图成功后立即原子写入独立 checkpoint：

```text
evals/conversations/.checkpoints/<blueprint-id>.json
```

如果一个候选因法规原文泄漏等规则被拒绝，脚本只定向修复该变体，不重新生成同蓝图中已经通过的变体。

#### 第三步：确定性校验

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py validate
```

校验内容包括：

- Pydantic Schema 和场景版本；
- Actor、Conversation、Action 和 Skill 白名单；
- 动作顺序、消息槽位和预期终态；
- 重复标题、重复问题和重复对话；
- 手机号、身份证、银行卡、邮箱和密钥；
- Prompt Injection 文本；
- 法名、条号和连续法规原文泄漏；
- 单轮最多两个运行时 Skill 及角色/权限边界。

查看校验汇总：

```bash
python -m json.tool evals/conversations/generation-validation.json
```

只有 `candidate_count=36`、`rejected_count=0` 且结构错误为空时，才满足冻结条件。

#### 第四步：冻结数据集

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py freeze
```

`freeze` 会再次执行校验，然后生成：

```text
evals/conversations/lawstation-dialogue-scenarios-v1.jsonl
evals/conversations/lawstation-dialogue-scenarios-v1.manifest.json
```

manifest 会记录模型、Prompt 版本、蓝图数、样本数、随机种子、模型调用数、修复历史、类别覆盖和 SHA256。有效样例不足时会保留诊断结果，但不会生成一个不完整的正式冻结集。

## 5. 限额、断点续跑和定向补生成

### 5.1 限制单次模型调用数

首次调试时可以限制本次调用：

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py generate --max-calls 2
```

`--max-calls` 必须大于 0。它限制本次执行的初始生成和定向修复总调用数，不会删除现有 checkpoint。

### 5.2 中断后继续

直接重新执行：

```bash
conda run --no-capture-output -n LawStation \
  python scripts/generate_conversation_scenarios.py generate
```

脚本会先校验已有 checkpoint 的 `blueprint_id` 和 `blueprint_sha256`，只调用尚未完成的蓝图。蓝图指纹变化时会拒绝复用旧 checkpoint，防止新旧规则混用。

### 5.3 已完成状态下再次执行

当 18 个 checkpoint 全部存在且有效时，再次运行 `generate` 通常不会产生新的初始模型调用；它会根据 checkpoint 重新物化候选并报告剩余数量。

不要为了“重新跑一次”随意删除 checkpoint。若确实要创建新实验，应升级 Prompt 或 Dataset 版本并保留旧 manifest，以便审计和对比。

## 6. 场景文件结构与阅读方法

每行是一个独立场景，例如：

```json
{
  "scenario_id": "casual-chat-01",
  "category": "routing",
  "actors": ["primary"],
  "preconditions": [],
  "steps": [
    {
      "action": "send_message",
      "actor": "primary",
      "conversation": "main",
      "content": "你好，在吗？",
      "expected": {
        "events": ["message_start", "agent_status", "message_end"],
        "terminal": "completed",
        "citations": false
      }
    }
  ]
}
```

常见动作：

```text
send_message
switch_user
switch_conversation
wait_for_completion
cancel_run
disconnect_stream
reconnect_stream
inspect_messages
inspect_memories
```

`expected` 表示未来 Runner 或人工测试应验证的结果，不表示这些断言已经实际通过。

## 7. 常见失败与处理

### `ModuleNotFoundError: langchain_openai`

原因：使用了系统 Python，而不是 Conda `LawStation` 环境。

处理：

```bash
conda activate LawStation
python scripts/generate_conversation_scenarios.py --help
```

### `未配置DEEPSEEK_API_KEY`

只影响 `generate`。在根目录 `.env` 配置密钥；不要把密钥写入命令、日志或数据集。

### `请先执行prepare`

缺少 `blueprints-v1.json`，先运行 `prepare`。

### `既有blueprints-v1.json与当前模板不一致`

代码模板或法规 source 选择结果发生变化。不要覆盖旧版本；先检查 Git diff、`LAW_DATA_PATH`、chunk 参数和随机种子，再决定是否升级数据集版本。

### `checkpoint蓝图指纹已变化`

旧 checkpoint 与当前蓝图不兼容。应保留旧冻结集并以新版本重新生成，不能把不兼容结果拼接到同一数据集中。

### `有效样例不足，保留当前结果但不冻结`

查看：

```bash
python -m json.tool evals/conversations/generation-validation.json
```

修复生成失败或拒绝原因后再次运行 `generate`。脚本会复用已通过 checkpoint，并在额度允许时定向修复失败变体。

## 8. 生成后的验证清单

```bash
conda run --no-capture-output -n LawStation \
  pytest -q tests/test_conversation_scenarios.py

conda run --no-capture-output -n LawStation \
  ruff check backend/app/evaluation/conversation_scenarios.py \
  scripts/generate_conversation_scenarios.py \
  tests/test_conversation_scenarios.py
```

人工核验 manifest：

```text
status=frozen
synthetic=true
human_verified=false
blueprint_count=18
sample_count=36
rejected_count=0
dataset_sha256 非空
```

## 9. 数据使用和简历边界

可以描述：

> 建立 18 类确定性多轮场景蓝图，使用 DeepSeek JSON Output 生成 36 条合成用户话术，并通过 checkpoint、定向修复、敏感信息、Prompt Injection、动作白名单、Skill 权限和法规来源泄漏校验进行版本化冻结。

不能描述：

- “36 条真实用户数据”；
- “律师人工标注”；
- “36 条 Agent 端到端测试全部通过”；
- “法律回答准确率 100%”。

当前可在前端人工逐步观察实际结果，但尚未实现无人值守的自动Scenario Runner，因此仍不能直接汇总36条的自动通过率。

## 10. 前端场景观察操作

1. 在`.env`启用白名单：

```dotenv
TEST_SCENARIOS_ENABLED=true
TEST_SCENARIO_DATA_PATHS=["./evals/conversations/lawstation-dialogue-scenarios-v1.jsonl"]
TEST_SCENARIO_STEP_TIMEOUT_SECONDS=60
```

2. 由用户手工启动LawStation。打开页面后，点击左侧栏“场景观察”，选择数据集、类别和场景。
3. 点击“开始场景”。`primary`绑定当前用户，需要`secondary`时绑定第一个不同用户；系统为每个Actor/会话组合创建`[场景]`专用会话。
4. 每次点击“执行下一步”只执行当前步骤。聊天区展示真实AgentRun状态，右侧面板展示预期/实际。
5. `disconnect_stream`只断开当前浏览器订阅；`reconnect_stream`从上次sequence恢复。含离线Fixture的场景不会注入故障或Skill，相关断言标记“不可判定”。
6. 观察完成后点击“清理场景数据”。如仍有queued/running Run，先停止或等待完成；清理不能删除普通会话或其他用户会话。

注意：场景步骤游标不跨页面刷新持久化；AgentRun本身仍会后台继续，刷新后可在普通会话中观察。

## 11. 一次性执行命令汇总

```bash
cd /Users/Admin1/Files/LawStation
conda activate LawStation

python scripts/generate_conversation_scenarios.py prepare
python scripts/generate_conversation_scenarios.py generate
python scripts/generate_conversation_scenarios.py validate
python scripts/generate_conversation_scenarios.py freeze

pytest -q tests/test_conversation_scenarios.py
```

这组命令只在明确需要重新生成时执行。普通开发和 Review 应优先使用第 4.1 节的离线核验路径，避免无意义消耗模型额度。
