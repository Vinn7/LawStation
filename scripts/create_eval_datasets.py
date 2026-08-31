"""Create deterministic, synthetic LawStation evaluation datasets."""

import json
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "evals" / "datasets"


def example(question, category, route, status=None, documents=None, **reference):
    return {
        "inputs": {
            "question": question,
            "history": reference.pop("history", []),
            "memory_context": reference.pop("memory_context", ""),
            "fixture_documents": documents or [],
            "fixture_tool_error": category == "tool_error",
        },
        "outputs": {
            "expected_route": route,
            "expected_retrieval_status": status,
            "expected_document_ids": [item["document_id"] for item in documents or []],
            **reference,
        },
        "metadata": {"category": category, "synthetic": True},
    }


def law(index, name="中华人民共和国民法典"):
    return {
        "document_id": f"fixture-law-{index}",
        "chunk_id": f"fixture-chunk-{index}",
        "law_name": name,
        "article_number": f"第{index}条",
        "content": f"与测试争议点相关的示例法条内容 {index}。",
        "retrieval_sources": ["fixture"],
        "data_version": "eval-v1",
    }


def build():
    casual = [
        "你好", "你是谁？", "谢谢你的帮助", "今天天气怎么样？", "讲一个笑话",
        "LawStation 能做什么？", "早上好", "再见", "请介绍一下你的能力", "你支持哪些语言？",
    ]
    clarify = [
        "朋友说要起诉我，我该怎么办？", "公司这样做合法吗？", "我能要求赔偿吗？", "合同出问题了怎么办？",
        "对方欠我钱，我能告他吗？", "房东不退钱怎么办？", "我被辞退了，可以仲裁吗？", "发生交通事故怎么处理？",
        "家人留下的财产怎么分？", "别人用了我的照片，我能维权吗？",
    ]
    matched = [
        "公司未与我签订书面劳动合同，可以主张什么？", "借款到期后对方拒绝还款怎么办？",
        "房东在租期内擅自解除合同怎么办？", "购买商品存在严重质量问题如何维权？",
        "交通事故造成受伤可以主张哪些损失？", "合同一方根本违约时如何解除合同？",
        "未经允许公开他人隐私照片如何处理？", "定金交付后卖方反悔怎么办？",
        "未成年人造成他人损害由谁承担责任？", "物业服务不到位能否拒交全部物业费？",
        "夫妻共同债务应当如何认定？", "继承人放弃继承需要什么形式？",
        "网络平台泄露个人信息如何维权？", "承揽工作成果不合格如何承担责任？",
        "保证期间没有约定时如何处理？", "格式条款未提示说明是否有效？",
        "买卖合同标的物毁损风险何时转移？", "无权代理合同是否有效？",
        "侵害名誉权通常承担哪些民事责任？", "用人单位拖欠工资可以采取哪些措施？",
    ]
    no_match = [
        "虚拟世界中的数字宠物归属如何认定？", "未来月球土地买卖合同是否有效？",
        "AI 梦境作品的作者是谁？", "游戏公会内部称号能否继承？", "元宇宙婚礼是否产生婚姻效力？",
        "机器人之间的口头承诺是否构成合同？", "外星资源采矿权属于谁？", "时间旅行导致的债务如何计算？",
        "纯虚拟人格能否担任公司董事？", "脑机接口产生的想法由谁所有？",
    ]
    tool_error = [
        "请检索劳动合同解除的法律依据。", "请查询民间借贷利率规定。", "请核验房屋租赁合同条款。",
        "请查询消费者退货规定。", "请检索个人信息保护相关法条。",
    ]
    cases = [example(item, "casual", "direct_answer") for item in casual]
    cases += [example(item, "clarification", "ask_clarification") for item in clarify]
    cases += [
        example(item, "matched", "research", "matched", [law(index + 1)])
        for index, item in enumerate(matched)
    ]
    cases += [example(item, "no_match", "research", "no_match") for item in no_match]
    cases += [example(item, "tool_error", "research", "tool_error") for item in tool_error]
    corrections = [
        ("我的月工资不是八千元，最新是每月一万元。", "月工资一万元", "月工资八千元"),
        ("我更正一下，合同签订日期是2025年3月1日。", "2025年3月1日", "2024年3月1日"),
        ("被告不是甲公司，而是乙公司。", "乙公司", "甲公司"),
        ("我目前希望协商解决，不再优先起诉。", "协商解决", "优先起诉"),
        ("欠款金额已核对，实际是五万元，不是三万元。", "五万元", "三万元"),
    ]
    for question, current, old in corrections:
        cases.append(example(
            question,
            "memory",
            "research",
            "no_match",
            memory_context=f"历史记忆（仅作背景）：{old}",
            expected_current_fact=current,
            forbidden_old_fact=old,
        ))
    return cases


def write(name, cases):
    path = OUTPUT / f"{name}.jsonl"
    text = "\n".join(json.dumps(item, ensure_ascii=False) for item in cases) + "\n"
    path.write_text(text, encoding="utf-8")


def fixture_analysis(item):
    question = item["inputs"]["question"]
    category = item["metadata"]["category"]
    route = item["outputs"]["expected_route"]
    if route == "research":
        return {
            "request_type": "legal_consultation",
            "case_summary": question,
            "legal_issues": [question],
            "research_tasks": [{
                "issue_id": "issue-1", "query": question, "purpose": "评测法律依据",
            }],
            "risk_level": "low" if category in {"no_match", "memory"} else "medium",
            "next_action": "research",
        }
    if route == "ask_clarification":
        return {
            "request_type": "insufficient_information",
            "case_summary": question,
            "risk_level": "medium",
            "next_action": "ask_clarification",
            "clarification_questions": ["请补充关键事实和相关材料。"],
        }
    return {
        "request_type": "casual_chat",
        "case_summary": question,
        "risk_level": "low",
        "next_action": "direct_answer",
        "direct_answer": "您好，我是 LawStation 法律咨询助手。",
    }


def build_skill_cases():
    definitions = {
        "case-intake": [
            "我和两家公司先后签过协议，付款时间和交付时间多次变更，请帮我整理案情。",
            "事情经过很长，涉及甲乙丙三方和多笔金额，请先梳理时间线。",
        ],
        "evidence-audit": [
            "我只有聊天记录和转账截图，这些证据够不够，还缺什么？",
            "准备起诉前，请帮我检查现有合同、发票和录音的证明作用。",
        ],
        "procedure-roadmap": [
            "劳动仲裁应当怎么走，可能有哪些步骤和材料？",
            "判决生效后对方仍不履行，我下一步如何申请执行？",
        ],
        "document-readiness": [
            "我想写民事起诉状，请先检查目前的信息是否完整。",
            "准备答辩状前还需要补充哪些事实和证据？",
        ],
    }
    cases = []
    for skill_id, positives in definitions.items():
        for index, question in enumerate(positives, 1):
            cases.append({
                "inputs": {
                    "question": question,
                    "fixture_case_analysis": {
                        "request_type": "legal_consultation",
                        "case_summary": question,
                        "next_action": "direct_answer",
                        "direct_answer": "已完成任务分析。",
                        "requested_skill_ids": [skill_id],
                    },
                },
                "outputs": {"expected_route": "direct_answer", "expected_skill_ids": [skill_id]},
                "metadata": {"category": "skill_positive", "skill_id": skill_id, "case": index, "synthetic": True},
            })
        cases.append({
            "inputs": {
                "question": "你好，请介绍一下你自己。",
                "fixture_case_analysis": {
                    "request_type": "casual_chat",
                    "case_summary": "普通问候",
                    "next_action": "direct_answer",
                    "direct_answer": "您好。",
                    "requested_skill_ids": [],
                },
            },
            "outputs": {"expected_route": "direct_answer", "expected_skill_ids": []},
            "metadata": {"category": "skill_negative", "skill_id": skill_id, "synthetic": True},
        })
        cases.append({
            "inputs": {
                "question": f"组合任务：{positives[0]}",
                "fixture_case_analysis": {
                    "request_type": "legal_consultation",
                    "case_summary": positives[0],
                    "next_action": "direct_answer",
                    "direct_answer": "已完成组合任务分析。",
                    "requested_skill_ids": [skill_id, "evidence-audit" if skill_id != "evidence-audit" else "procedure-roadmap"],
                },
            },
            "outputs": {"expected_route": "direct_answer", "expected_skill_ids": [skill_id, "evidence-audit" if skill_id != "evidence-audit" else "procedure-roadmap"]},
            "metadata": {"category": "skill_combination", "skill_id": skill_id, "synthetic": True},
        })
        cases.append({
            "inputs": {
                "question": positives[1],
                "fixture_case_analysis": {
                    "request_type": "legal_consultation",
                    "case_summary": positives[1],
                    "next_action": "direct_answer",
                    "direct_answer": "已拒绝未知能力。",
                    "requested_skill_ids": [skill_id, "delete-user-memory"],
                },
            },
            "outputs": {"expected_route": "direct_answer", "expected_skill_ids": [skill_id]},
            "metadata": {"category": "skill_unauthorized", "skill_id": skill_id, "synthetic": True},
        })
        cases.append({
            "inputs": {
                "question": positives[0],
                "fixture_case_analysis": {
                    "request_type": "legal_consultation",
                    "case_summary": positives[0],
                    "next_action": "direct_answer",
                    "direct_answer": "Skill关闭时使用基础回答。",
                    "requested_skill_ids": [],
                },
            },
            "outputs": {"expected_route": "direct_answer", "expected_skill_ids": []},
            "metadata": {"category": "skill_disabled_baseline", "skill_id": skill_id, "skills_enabled": False, "synthetic": True},
        })
    assert len(cases) == 24
    return cases


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    cases = build()
    assert len(cases) == 60
    write("lawstation-e2e-v1", cases)
    live_cases = deepcopy(cases)
    for item in live_cases:
        # Live E2E validates orchestration and answer safety. Retrieval quality is
        # measured separately against lawstation-live-retrieval-v1 real IDs.
        item["inputs"]["fixture_documents"] = []
        item["inputs"]["fixture_tool_error"] = False
        item["outputs"]["expected_document_ids"] = []
        item["metadata"]["live_e2e"] = True
    write("lawstation-e2e-v2", live_cases)
    write("lawstation-routing-v1", [item for item in cases if item["metadata"]["category"] in {"casual", "clarification", "matched"}])
    write("lawstation-retrieval-v1", [item for item in cases if item["metadata"]["category"] in {"matched", "no_match", "tool_error"}])
    write("lawstation-answer-v1", [item for item in cases if item["metadata"]["category"] in {"matched", "no_match"}])
    write("lawstation-memory-v1", [item for item in cases if item["metadata"]["category"] == "memory"])
    write("lawstation-skills-v1", build_skill_cases())
    limits = {
        "casual": 5,
        "clarification": 5,
        "matched": 10,
        "no_match": 5,
        "tool_error": 3,
        "memory": 2,
    }
    selected = []
    counts = {key: 0 for key in limits}
    for item in cases:
        category = item["metadata"]["category"]
        if category in limits and counts[category] < limits[category]:
            selected_item = deepcopy(item)
            selected_item["inputs"]["fixture_case_analysis"] = fixture_analysis(item)
            selected.append(selected_item)
            counts[category] += 1
    assert len(selected) == 30 and counts == limits
    write("lawstation-agent-v3", selected)


if __name__ == "__main__":
    main()
