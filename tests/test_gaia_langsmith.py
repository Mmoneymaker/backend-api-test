"""
GAIA Benchmark LangSmith 评估测试
使用 GAIA 数据集测试 Manus Agent 的性能
"""
import asyncio
import os
import sys
import json
from pathlib import Path
from typing import Dict, Any, List

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
# GAIA 数据加载
# ============================================================

GAIA_CACHE_PATH = "/Users/xujunjie/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/682dd723ee1e1697e00360edccf2366dc8418dd9"


def prepare_gaia_examples(level: str = "level1") -> List[Dict]:
    """加载 GAIA 数据集并转换为 LangSmith 格式"""
    import pandas as pd

    parquet_path = os.path.join(GAIA_CACHE_PATH, f"2023/validation/metadata.{level}.parquet")

    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"GAIA 数据文件不存在: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    print(f"读取 GAIA {level} 数据: {len(df)} 条")

    examples = []
    for _, row in df.iterrows():
        raw_file_path = row.get("file_path", "")
        full_file_path = ""

        if raw_file_path and str(raw_file_path).strip() != "":
            full_file_path = os.path.join(GAIA_CACHE_PATH, raw_file_path)

        example = {
            "inputs": {
                "query": row["Question"],
                "file_path": full_file_path,
                "file_name": row.get("file_name", ""),
                "level": row.get("Level", ""),
                "goal": row.get("Goal", ""),
            },
            "outputs": {
                "reference": row["Final answer"]
            }
        }
        examples.append(example)

    return examples


# ============================================================
# 工具函数
# ============================================================

def _get_tool_calls(agent: Manus) -> List[Dict]:
    """获取所有工具调用记录"""
    tool_calls = []
    for msg in agent.messages:
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls.append({
                    "name": tc.function.name,
                    "arguments": tc.function.arguments
                })
    return tool_calls


def _last_assistant_content(agent: Manus) -> str:
    """取最后一条 assistant 消息的 content"""
    for msg in reversed(agent.messages):
        if getattr(msg, "role", None) == "assistant" and getattr(msg, "content", None):
            return (msg.content or "").strip()
    return ""


def _get_thinking_process(agent: Manus) -> str:
    """获取 Agent 的思考过程"""
    thoughts = []
    for msg in agent.messages:
        if hasattr(msg, "role") and msg.role == "assistant":
            content = getattr(msg, "content", None)
            if content:
                thoughts.append(content)
    return "\n---\n".join(thoughts)


# ============================================================
# Step 1: 定义应用
# ============================================================

@traceable(name="gaia-manus-agent", project_name=os.getenv("LANGSMITH_PROJECT", "manus-gaia-evaluation"))
async def gaia_manus_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    GAIA Benchmark 测试应用
    返回完整的执行信息供评估
    """
    query = inputs.get("query", "")
    file_path = inputs.get("file_path", "")
    file_name = inputs.get("file_name", "")

    # 构建完整的 prompt，包含文件信息
    full_query = query
    if file_path and os.path.exists(file_path):
        full_query = f"{query}\n\n请查看附件文件: {file_path}"
    elif file_name:
        full_query = f"{query}\n\n附件文件名: {file_name}"

    agent = await Manus.create()
    try:
        await agent.run(full_query)

        # 获取完整信息
        tool_calls = _get_tool_calls(agent)
        tool_names = [tc["name"] for tc in tool_calls]
        tool_calls_count = len(tool_calls)
        output = _last_assistant_content(agent)
        thinking = _get_thinking_process(agent)

        # 获取 Token 使用
        prompt_tokens = getattr(agent.llm, "total_input_tokens", 0) if getattr(agent, "llm", None) else 0
        completion_tokens = getattr(agent.llm, "total_completion_tokens", 0) if getattr(agent, "llm", None) else 0

        # 添加元数据到 LangSmith
        run_tree = get_current_run_tree()
        if run_tree:
            run_tree.add_metadata({
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens
                },
                "tool_calls": {
                    "count": tool_calls_count,
                    "names": tool_names,
                    "details": tool_calls
                }
            })

        return {
            "query": query,
            "output": output if output else "No output",
            "thinking": thinking if thinking else "No thinking recorded",
            "tool_calls": tool_calls,
            "tool_names": tool_names,
            "tool_calls_count": tool_calls_count,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "has_file_attachment": bool(file_path),
        }
    finally:
        await agent.cleanup()


# ============================================================
# Step 2: 创建数据集
# ============================================================

def create_gaia_dataset(level: str = "level1", dataset_name: str = None) -> str:
    """创建 GAIA 数据集"""
    import pandas as pd

    ls_client = Client()

    if dataset_name is None:
        dataset_name = f"GAIA_Manus_{level}"

    # 检查数据集是否已存在
    if ls_client.has_dataset(dataset_name=dataset_name):
        print(f"数据集 '{dataset_name}' 已存在，跳过创建")
        return dataset_name

    # 加载数据
    examples = prepare_gaia_examples(level)
    print(f"创建数据集 '{dataset_name}'，包含 {len(examples)} 个测试用例")

    # 创建数据集
    dataset = ls_client.create_dataset(
        dataset_name=dataset_name,
        description=f"GAIA Benchmark - {level} - Manus Agent Evaluation"
    )
    ls_client.create_examples(dataset_id=dataset.id, examples=examples)

    print(f"数据集创建完成: {dataset.name}")
    return dataset.name


# ============================================================
# Step 3: 评估器
# ============================================================

def answer_contains_reference(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估回答是否包含标准答案"""
    output = outputs.get("output", "").lower()
    reference = reference_outputs.get("reference", "").lower()

    # 简单检查答案是否在输出中
    contains_answer = reference in output

    return {
        "key": "answer_contains_reference",
        "score": 1.0 if contains_answer else 0.0,
        "reason": f"Reference in output: {contains_answer}, reference: {reference}"
    }


def has_tool_calls(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估是否有工具调用"""
    tool_calls_count = outputs.get("tool_calls_count", 0)
    has_calls = tool_calls_count > 0

    return {
        "key": "has_tool_calls",
        "score": 1.0 if has_calls else 0.0,
        "reason": f"Tool calls count: {tool_calls_count}"
    }


def tool_call_efficiency(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估工具调用效率"""
    tool_calls_count = outputs.get("tool_calls_count", 0)

    if tool_calls_count == 0:
        score = 0.0
        reason = "No tool calls"
    elif tool_calls_count <= 3:
        score = 1.0
        reason = "Efficient"
    elif tool_calls_count <= 6:
        score = 0.7
        reason = "Moderate"
    else:
        score = 0.5
        reason = "Could use fewer tools"

    return {
        "key": "tool_call_efficiency",
        "score": score,
        "reason": f"{reason}, count: {tool_calls_count}"
    }


def has_output(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估是否有有效输出"""
    output = outputs.get("output", "")
    has_output = len(output) > 0 and output != "No output"

    return {
        "key": "has_output",
        "score": 1.0 if has_output else 0.0,
        "reason": f"Has output: {has_output}, length: {len(output)}"
    }


# ============================================================
# Step 4: LLM 评估器 (单次调用，多维度评估)
# ============================================================

def create_llm_judge_evaluator():
    """创建基于 LLM 的评估器 - 单次调用评估多个维度"""
    from openai import AsyncOpenAI

    api_key = os.getenv("LLM_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("LLM_JUDGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4o")

    if not api_key:
        print("警告: 未配置 LLM_JUDGE_API_KEY，跳过 LLM 评估")
        return None

    print(f"LLM Judge 配置: model={judge_model}, base_url={base_url}")

    judge_client = AsyncOpenAI(api_key=api_key, base_url=base_url)

    @traceable(name="GAIA-LLM-Judge")
    async def llm_judge_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """
        LLM 评估器 - 一次调用评估三个维度:
        1. task_completion: 任务完成度
        2. tool_usage: 工具使用合理性
        3. answer_quality: 回答质量
        """
        query = inputs.get("query", "")
        output = outputs.get("output", "")
        thinking = outputs.get("thinking", "")
        reference = reference_outputs.get("reference", "")
        tool_names = outputs.get("tool_names", [])
        tool_calls_count = outputs.get("tool_calls_count", 0)
        tool_calls = outputs.get("tool_calls", [])

        evaluation_prompt = f"""你是一个严格的评估专家。请从以下三个维度评估 Agent 在 GAIA Benchmark 上的表现：

## 用户问题
{query}

## 标准答案
{reference}

## Agent 使用的工具
{tool_names} (共 {tool_calls_count} 次调用)

## 工具调用详情
{json.dumps(tool_calls[:3], ensure_ascii=False, indent=2) if tool_calls else "无"}

## Agent 的思考过程
{thinking[:1500]}

## Agent 的最终回答
{output}

## 评估维度

请从以下三个维度进行评分（每个维度 0-1 分）：

1. **任务完成度 (task_completion)**: Agent 是否正确完成了用户的问题？
   - 1分：完全正确完成了任务，答案准确
   - 0.5分：部分完成或有轻微错误
   - 0分：未完成或错误严重

2. **工具使用合理性 (tool_usage)**: Agent 是否合理使用了工具？
   - 1分：工具选择合理，调用恰当，完美解决了问题
   - 0.5分：工具有使用但不够高效或有些冗余
   - 0分：工具使用不当、过度使用或完全没用到需要的工具

3. **回答质量 (answer_quality)**: Agent 的最终回答是否有帮助、准确？
   - 1分：回答准确、清晰、有帮助，格式良好
   - 0.5分：回答一般，有些帮助但不够完整或清晰
   - 0分：回答无帮助、不准确、混乱或无意义

请以 JSON 格式返回评估结果：
{{
    "task_completion": <0-1之间的分数>,
    "tool_usage": <0-1之间的分数>,
    "answer_quality": <0-1之间的分数>,
    "overall_score": <上述三项的平均分>,
    "reason": "<简要说明评估理由>"
}}

注意：只返回 JSON，不要有其他内容。"""

        try:
            response = await judge_client.chat.completions.create(
                model=judge_model,
                messages=[
                    {"role": "system", "content": "你是一个严格的评估专家。"},
                    {"role": "user", "content": evaluation_prompt}
                ],
                temperature=0.3,
                max_tokens=500
            )

            result_text = response.choices[0].message.content.strip()
            print(f"LLM Judge 原始输出: {result_text[:200]}...")

            # 提取 JSON
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)

            # 提取各维度分数
            task_completion = float(result.get("task_completion", 0.5))
            tool_usage = float(result.get("tool_usage", 0.5))
            answer_quality = float(result.get("answer_quality", 0.5))
            overall = float(result.get("overall_score", (task_completion + tool_usage + answer_quality) / 3))

            reason = result.get("reason", "")
            detailed_reason = f"Task: {task_completion}, Tool: {tool_usage}, Quality: {answer_quality}. {reason}"

            return {
                "key": "llm_judge_overall",
                "score": overall,
                "reason": detailed_reason
            }

        except Exception as e:
            return {
                "key": "llm_judge_overall",
                "score": 0.5,
                "reason": f"LLM evaluation failed: {str(e)}"
            }

    return llm_judge_evaluator


# ============================================================
# Step 5: 运行评估
# ============================================================

async def run_evaluation(
    level: str = "level1",
    dataset_name: str = None,
    use_llm_judge: bool = False,
    max_examples: int = None
):
    """运行 GAIA 评估"""
    ls_client = Client()

    # 创建或获取数据集
    if dataset_name is None:
        dataset_name = f"GAIA_Manus_{level}"

    if not ls_client.has_dataset(dataset_name=dataset_name):
        dataset_name = create_gaia_dataset(level, dataset_name)

    # 获取数据集
    dataset = ls_client.read_dataset(dataset_name=dataset_name)
    examples = list(ls_client.list_examples(dataset_id=dataset.id))

    # 如果设置了 max_examples，限制测试数量
    if max_examples and max_examples < len(examples):
        examples = examples[:max_examples]
        print(f"限制测试数量: {max_examples}")

    print(f"\n开始评估: {dataset_name}")
    print(f"测试用例数量: {len(examples)}")
    print("=" * 50)

    # 准备评估器
    evaluators = [
        # answer_contains_reference,
        has_tool_calls,
        tool_call_efficiency,
        has_output,
    ]

    # 添加 LLM 评估器（单次调用，多维度评估）
    if use_llm_judge:
        llm_judge = create_llm_judge_evaluator()
        if llm_judge:
            evaluators.append(llm_judge)
            print("已添加 LLM 评估器 (task_completion, tool_usage, answer_quality)")

    # 运行评估
    results = await ls_client.aevaluate(
        gaia_manus_application,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix=f"manus-gaia-{level}",
        description=f"GAIA Benchmark - {level} - Manus Agent",
        max_concurrency=2,
    )

    print("\n评估完成!")
    print(f"结果: {results}")

    return results


# ============================================================
# 测试单个查询
# ============================================================

async def test_single_example():
    """测试单个 GAIA 示例"""
    # 加载一个示例
    examples = prepare_gaia_examples("level1")

    if not examples:
        print("没有找到 GAIA 数据")
        return

    example = examples[0]
    print(f"\n测试查询: {example['inputs']['query']}")
    print(f"标准答案: {example['outputs']['reference']}")
    print("=" * 50)

    result = await gaia_manus_application(example['inputs'])
    print(f"\n输出: {result.get('output', '')[:500]}...")
    print(f"工具调用: {result.get('tool_names', [])}")
    print(f"调用次数: {result.get('tool_calls_count', 0)}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="GAIA Benchmark LangSmith 评估测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test", "create-dataset"],
        default="eval",
        help="运行模式"
    )
    parser.add_argument(
        "--level",
        type=str,
        default="level1",
        choices=["level1", "level2", "level3"],
        help="GAIA 难度级别"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称"
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="最大测试数量"
    )
    parser.add_argument(
        "--llm-judge",
        action="store_true",
        help="使用 LLM 评估器"
    )

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_gaia_dataset(args.level, args.dataset)
    elif args.mode == "test":
        asyncio.run(test_single_example())
    elif args.mode == "eval":
        asyncio.run(run_evaluation(
            level=args.level,
            dataset_name=args.dataset,
            use_llm_judge=args.llm_judge,
            max_examples=args.max_examples
        ))


if __name__ == "__main__":
    main()
