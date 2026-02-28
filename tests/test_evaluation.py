"""
LangSmith 评估测试
根据 LangSmith 官方评估体系创建
"""
import asyncio
import os
import sys
from pathlib import Path
from typing import Dict, Any

# 将项目根目录加入 Python 路径，便于从 tests/ 或任意目录运行
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from langsmith import Client, traceable
from langsmith.evaluation import evaluate
from langsmith.evaluation import aevaluate
from langsmith.run_helpers import get_current_run_tree

from app.agent.manus import Manus


# Step 1: 定义应用 (使用 @traceable 装饰)
def _last_assistant_content(agent: Manus) -> str:
    """取最后一条 assistant 消息的 content，即模型给出的「回答」正文（如 1+1等于2、法国首都是巴黎）。"""
    for msg in reversed(agent.messages):
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            return (msg.content or "").strip()
    return ""


@traceable(name="manus-agent", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def manus_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
    query = inputs.get("query", "")
    agent = await Manus.create()
    try:
        await agent.run(query)

        # 1. 获取 Token 统计（确保键名符合标准）
        prompt_tokens = getattr(agent.llm, "total_input_tokens", 0) if getattr(agent, "llm", None) else 0
        completion_tokens = getattr(agent.llm, "total_completion_tokens", 0) if getattr(agent, "llm", None) else 0

        # 2. 获取当前运行树并注入标准键名
        run_tree = get_current_run_tree()
        if run_tree:
            # 使用 LangSmith 识别的标准字段：prompt_tokens 和 completion_tokens
            run_tree.add_metadata({
                "results": {
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": prompt_tokens + completion_tokens
                    }
                },
                # 直接在 metadata 根层级也放一份作为备份
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens
            })

        output = _last_assistant_content(agent)
        return {
            "output": output if output else "No assistant reply",
            # 这里返回的是给评估器看的，可以保留原样
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
        }
    finally:
        await agent.cleanup()


# Step 2: 创建数据集
DATASET_NAME = "manus-test-dataset"


def create_dataset():
    """创建测试数据集；若同名数据集已存在则直接使用。"""
    ls_client = Client()

    # 若已存在则直接返回名称，避免 409 Conflict
    if ls_client.has_dataset(dataset_name=DATASET_NAME):
        print(f"数据集已存在，使用: {DATASET_NAME}")
        return DATASET_NAME

    # 定义测试用例
    examples = [
        {
            "inputs": {"query": "法国首都是什么？"},
            "outputs": {"expected": "巴黎"},
        },
        {
            "inputs": {"query": "1+1等于多少？"},
            "outputs": {"expected": "2"},
        },
        {
            "inputs": {"query": "Python中如何定义一个函数？"},
            "outputs": {"expected": "def"},
        },
        {
            "inputs": {"query": "水的化学式是什么？"},
            "outputs": {"expected": "H2O"},
        },
        {
            "inputs": {"query": "谁写了《西游记》？"},
            "outputs": {"expected": "吴承恩"},
        },
    ]

    dataset = ls_client.create_dataset(dataset_name=DATASET_NAME)
    ls_client.create_examples(
        dataset_id=dataset.id,
        examples=examples,
    )

    print(f"数据集创建完成: {dataset.name}")
    return dataset.name


# Step 3: 定义评估器
def answer_correct(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    评估回答是否正确
    检查输出中是否包含期望的关键词
    """
    output = outputs.get("output", "")
    expected = reference_outputs.get("expected", "")

    # 简单的包含检查
    is_correct = expected in output

    return {
        "key": "answer_correct",
        "score": 1.0 if is_correct else 0.0,
        "reason": f"Output contains expected: {is_correct}"
    }


def has_content(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    评估输出是否有内容
    """
    output = outputs.get("output", "")
    has_output = len(output) > 0

    return {
        "key": "has_content",
        "score": 1.0 if has_output else 0.0,
        "reason": f"Output has content: {has_output}, length: {len(output)}"
    }


# Step 4: 运行评估
async def run_evaluation(dataset_name: str = None):
    """运行评估"""
    ls_client = Client()

    # 如果没有指定数据集名称，则创建一个
    if dataset_name is None:
        dataset_name = create_dataset()

    print(f"\n开始评估数据集: {dataset_name}")
    print("=" * 50)

    # 运行评估
    results = await ls_client.aevaluate(
        manus_application,
        data=dataset_name,
        evaluators=[answer_correct, has_content],
        experiment_prefix="manus-v1",
        description="Manus Agent 基础评估测试",
        max_concurrency=2,
    )

    print("\n评估完成!")
    print(f"结果: {results}")

    return results


# 异步版本
async def run_evaluation_async(dataset_name: str = None):
    """异步运行评估"""
    ls_client = Client()

    # 如果没有指定数据集名称，则创建一个
    if dataset_name is None:
        dataset_name = create_dataset()

    print(f"\n开始评估数据集: {dataset_name}")
    print("=" * 50)

    # 异步运行评估
    results = await ls_client.aevaluate(
        manus_application,
        data=dataset_name,
        evaluators=[answer_correct, has_content],
        experiment_prefix="manus-v1-async",
        description="Manus Agent 基础评估测试 (异步)",
        max_concurrency=2,
    )

    print("\n评估完成!")
    print(f"结果: {results}")

    return results


# 测试单个查询
async def test_single_query():
    """测试单个查询"""
    query = "法国首都是什么？"

    print(f"\n测试查询: {query}")
    print("=" * 50)

    result = await manus_application({"query": query})
    print(f"结果: {result}")

    return result


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="LangSmith 评估测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test", "create-dataset"],
        default="eval",
        help="运行模式: eval=运行评估, test=测试单个查询, create-dataset=创建数据集"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称（用于评估模式）"
    )

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_dataset()
    elif args.mode == "test":
        asyncio.run(test_single_query())
    elif args.mode == "eval":
        if args.dataset:
            asyncio.run(run_evaluation(args.dataset))
        else:
            asyncio.run(run_evaluation())


if __name__ == "__main__":
    main()
