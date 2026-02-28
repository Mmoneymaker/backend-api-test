"""
LangSmith Agent 性能评估
评估 Agent 常见的性能指标
"""
import asyncio
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any

# 将项目根目录加入 Python 路径
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from langsmith import Client, traceable
from langsmith.run_helpers import get_current_run_tree

from app.agent.manus import Manus


# ============================================================
# 工具函数
# ============================================================

def _last_assistant_content(agent: Manus) -> str:
    """取最后一条 assistant 消息的 content"""
    for msg in reversed(agent.messages):
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            return (msg.content or "").strip()
    return ""


def _count_tool_calls(agent: Manus) -> int:
    """统计工具调用次数"""
    count = 0
    for msg in agent.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            count += len(msg.tool_calls)
    return count


def _get_used_tools(agent: Manus) -> list:
    """获取使用的工具列表"""
    tools = set()
    for msg in agent.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                if hasattr(tc, "function") and tc.function:
                    tools.add(tc.function.name)
    return list(tools)


# ============================================================
# Step 1: 定义应用
# ============================================================

@traceable(name="manus-agent", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def manus_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
    query = inputs.get("query", "")
    start_time = time.time()

    agent = await Manus.create()
    try:
        await agent.run(query)

        # 统计信息
        elapsed_time = time.time() - start_time
        prompt_tokens = getattr(agent.llm, "total_input_tokens", 0) if getattr(agent, "llm", None) else 0
        completion_tokens = getattr(agent.llm, "total_completion_tokens", 0) if getattr(agent, "llm", None) else 0
        tool_calls_count = _count_tool_calls(agent)
        used_tools = _get_used_tools(agent)
        output = _last_assistant_content(agent)

        # 添加元数据到 LangSmith
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "metrics": {
                    "elapsed_time": elapsed_time,
                    "tool_calls_count": tool_calls_count,
                    "used_tools": used_tools
                }
            })

        return {
            "output": output if output else "No assistant reply",
            "elapsed_time": elapsed_time,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "tool_calls_count": tool_calls_count,
            "used_tools": used_tools,
        }
    finally:
        await agent.cleanup()


# ============================================================
# Step 2: 数据集
# ============================================================

DATASET_NAME = "manus-performance-dataset"


def create_dataset():
    """创建性能评估数据集"""
    ls_client = Client()

    if ls_client.has_dataset(dataset_name=DATASET_NAME):
        print(f"数据集已存在，使用: {DATASET_NAME}")
        return DATASET_NAME

    # 常见 Agent 测试任务
    examples = [
        # 简单问答
        {
            "inputs": {"query": "法国首都是什么？"},
            "outputs": {"expected": "巴黎"},
        },
        {
            "inputs": {"query": "1+1等于多少？"},
            "outputs": {"expected": "2"},
        },
        # 需要工具使用的任务
        {
            "inputs": {"query": "帮我计算 25 的平方根"},
            "outputs": {"expected": "5"},
        },
        # 多步推理
        {
            "inputs": {"query": "如果今天是周一，后天是星期几？"},
            "outputs": {"expected": "周三"},
        },
        # 编程任务
        {
            "inputs": {"query": "写一个 Python 函数判断素数"},
            "outputs": {"expected": "def"},
        },
        # 搜索任务（需要联网）
        {
            "inputs": {"query": "当前美国总统是谁？"},
            "outputs": {"expected": ""},  # 开放答案
        },
    ]

    dataset = ls_client.create_dataset(dataset_name=DATASET_NAME)
    ls_client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"数据集创建完成: {dataset.name}")
    return dataset.name


# ============================================================
# Step 3: 评估器
# ============================================================

def answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估回答是否包含期望关键词"""
    output = outputs.get("output", "")
    expected = reference_outputs.get("expected", "")

    # 如果没有期望答案，跳过评分
    if not expected:
        return {"key": "answer_correct", "score": None, "reason": "No expected answer"}

    is_correct = expected in output
    return {
        "key": "answer_correct",
        "score": 1.0 if is_correct else 0.0,
        "reason": f"Output contains expected: {is_correct}"
    }


def has_content(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估是否有有效输出"""
    output = outputs.get("output", "")
    has_output = len(output) > 0 and output != "No assistant reply"

    return {
        "key": "has_content",
        "score": 1.0 if has_output else 0.0,
        "reason": f"Has valid output: {has_output}, length: {len(output)}"
    }


def efficiency_score(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估效率 - 基于步数/工具调用"""
    tool_calls = outputs.get("tool_calls_count", 0)

    # 简单任务应该在合理步数内完成
    score = 1.0 if tool_calls <= 3 else 0.5
    return {
        "key": "efficiency",
        "score": score,
        "reason": f"Tool calls: {tool_calls}"
    }


def token_usage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估 Token 消耗"""
    total_tokens = outputs.get("total_tokens", 0)

    # 合理范围内
    if total_tokens < 5000:
        score = 1.0
        reason = "Token usage: normal"
    elif total_tokens < 10000:
        score = 0.7
        reason = "Token usage: moderate"
    else:
        score = 0.5
        reason = "Token usage: high"

    return {
        "key": "token_usage",
        "score": score,
        "reason": f"{reason}, total: {total_tokens}"
    }


def tool_usage(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估工具使用是否合理"""
    output = outputs.get("output", "")
    used_tools = outputs.get("used_tools", [])

    # 检查是否使用了 Terminate（说明任务完成）
    has_terminate = "Terminate" in used_tools or "terminate" in used_tools

    return {
        "key": "tool_usage",
        "score": 1.0 if has_terminate else 0.5,
        "reason": f"Used tools: {used_tools}"
    }


# ============================================================
# Step 4: 运行评估
# ============================================================

async def run_evaluation(dataset_name: str = None):
    """运行评估"""
    ls_client = Client()

    if dataset_name is None:
        dataset_name = create_dataset()

    print(f"\n开始评估数据集: {dataset_name}")
    print("=" * 50)

    results = await ls_client.aevaluate(
        manus_application,
        data=dataset_name,
        evaluators=[
            answer_correct,
            has_content,
            efficiency_score,
            token_usage,
            tool_usage,
        ],
        experiment_prefix="manus-performance",
        description="Manus Agent 性能评估",
        max_concurrency=2,
    )

    print("\n评估完成!")
    print(f"结果: {results}")
    return results


async def test_single_query():
    """测试单个查询"""
    query = "法国首都是什么？"
    print(f"\n测试查询: {query}")
    print("=" * 50)

    result = await manus_application({"query": query})
    print(f"结果: {result}")
    return result


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="LangSmith Agent 性能评估")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test", "create-dataset"],
        default="eval",
        help="运行模式"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称"
    )

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_dataset()
    elif args.mode == "test":
        asyncio.run(test_single_query())
    elif args.mode == "eval":
        asyncio.run(run_evaluation(args.dataset))


if __name__ == "__main__":
    main()
