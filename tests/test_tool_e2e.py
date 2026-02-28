"""
OpenManus Agent 工具调用测试
端到端测试 + LLM 评估 (LLM as Judge)
"""
import asyncio
import os
import sys
import time
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
    """获取 Agent 的思考过程

    思考过程就是 messages 中所有 assistant 角色的 content
    每次调用 LLM 后的回复就是思考过程
    """
    thoughts = []
    for msg in agent.messages:
        # 获取 assistant 角色的消息内容（即 LLM 的思考/回复）
        if hasattr(msg, "role") and msg.role == "assistant":
            content = getattr(msg, "content", None)
            if content:
                thoughts.append(content)
    return "\n---\n".join(thoughts)


# ============================================================
# Step 1: 定义应用 (带完整信息返回)
# ============================================================

@traceable(name="agent-tool-test", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def agent_tool_application(inputs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Agent 工具调用测试应用
    返回完整的执行信息供 LLM 评估
    """
    query = inputs.get("query", "")

    agent = await Manus.create()
    try:
        await agent.run(query)

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
        }
    finally:
        await agent.cleanup()


# ============================================================
# Step 2: 数据集 - 不硬编码 expected_tool
# ============================================================

DATASET_NAME = "manus-tool-llm-judge-dataset"


def create_dataset():
    """创建数据集 - 覆盖 Manus 所有工具"""
    ls_client = Client()

    if ls_client.has_dataset(dataset_name=DATASET_NAME):
        print(f"数据集已存在，使用: {DATASET_NAME}")
        return DATASET_NAME

    # ============================================================
    # 测试用例 - 覆盖 Manus 的所有工具:
    # 1. PythonExecute - Python 代码执行
    # 2. BrowserUseTool - 浏览器自动化
    # 3. StrReplaceEditor - 文件操作 (view, create, str_replace, insert)
    # 4. AskHuman - 询问人类
    # 5. Terminate - 终止对话
    # ============================================================

    examples = [
        # ========== 1. PythonExecute 工具测试 ==========
        # 基础计算
        {
            "inputs": {"query": "请帮我计算 123 * 456 的结果", "expected_tool": "python_execute"},
            "outputs": {"expected_answer_contains": "56088"},
        },
        {
            "inputs": {"query": "计算 1 到 100 的总和是多少？", "expected_tool": "python_execute"},
            "outputs": {"expected_answer_contains": "5050"},
        },
        {
            "inputs": {"query": "计算 2 的 10 次方是多少？", "expected_tool": "python_execute"},
            "outputs": {"expected_answer_contains": "1024"},
        },

        # 数据处理
        {
            "inputs": {"query": "用 Python 找出 1 到 100 之间的所有素数", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },
        {
            "inputs": {"query": "写一个 Python 函数来计算斐波那契数列的第 n 项", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_written"},
        },
        {
            "inputs": {"query": "用 Python 对列表 [5, 3, 8, 1, 9] 进行排序并找出最大值", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },

        # 字符串处理
        {
            "inputs": {"query": "用 Python 实现反转字符串的功能，比如输入 'hello' 输出 'olleh'", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },
        {
            "inputs": {"query": "用 Python 统计字符串 'hello world' 中每个字符出现的次数", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },

        # 文件操作结合 Python
        {
            "inputs": {"query": "先用 Python 读取 workspace/example.txt 文件内容，然后计算其中包含多少个字符", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },

        # ========== 2. StrReplaceEditor 工具测试 ==========
        # 查看文件
        {
            "inputs": {"query": "查看 workspace/example.txt 文件的内容", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_viewed"},
        },
        {
            "inputs": {"query": "列出 workspace 目录下有哪些文件", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "directory_listed"},
        },

        # 创建文件
        {
            "inputs": {"query": "在 workspace 目录下创建一个名为 test_file.txt 的文件，内容为 'Hello World'", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_created"},
        },
        {
            "inputs": {"query": "创建一个 Python 文件 workspace/test_script.py，内容是一个打印 hello 的程序", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_created"},
        },
        {
            "inputs": {"query": "在 workspace 目录下创建一个 JSON 文件 config.json，包含 name 和 age 字段", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_created"},
        },

        # 编辑文件 (str_replace)
        {
            "inputs": {"query": "修改 workspace/example.txt 文件，把 'sample' 改成 'modified'", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_modified"},
        },
        {
            "inputs": {"query": "把 workspace/test_file.txt 文件中的 'Hello' 改成 'Hi'", "expected_tool": "str_replace_editor"},
            "outputs": {"expected_action": "file_modified"},
        },

        # ========== 3. BrowserUseTool 工具测试 ==========
        # 注意: 需要安装 playwright 浏览器才能测试
        # 网页搜索
        {
            "inputs": {"query": "搜索最新的 Python 教程", "expected_tool": "browser_use", "requires_browser": True},
            "outputs": {"expected_action": "web_search"},
        },
        {
            "inputs": {"query": "帮我搜索 OpenManus 项目的相关信息", "expected_tool": "browser_use", "requires_browser": True},
            "outputs": {"expected_action": "web_search"},
        },

        # 访问 URL
        {
            "inputs": {"query": "访问 https://www.example.com 并告诉我页面内容是什么", "expected_tool": "browser_use", "requires_browser": True},
            "outputs": {"expected_action": "page_visited"},
        },

        # 内容提取
        {
            "inputs": {"query": "打开百度首页，提取页面上所有的链接文字", "expected_tool": "browser_use", "requires_browser": True},
            "outputs": {"expected_action": "content_extracted"},
        },

        # ========== 4. AskHuman 工具测试 ==========
        # 需要用户输入的任务
        {
            "inputs": {"query": "请问我需要给你什么信息才能帮你写一个完整的程序？", "expected_tool": "ask_human"},
            "outputs": {"expected_action": "question_asked"},
        },
        {
            "inputs": {"query": "你想让我帮你完成什么具体任务？请告诉我", "expected_tool": "ask_human"},
            "outputs": {"expected_action": "question_asked"},
        },

        # ========== 5. 多工具组合测试 ==========
        # Python + 文件操作
        {
            "inputs": {"query": "创建一个 CSV 文件包含 3 行数据，然后用 Python 读取并计算总和", "expected_tool": "multi_tools"},
            "outputs": {"expected_action": "multi_tool_used"},
        },
        {
            "inputs": {"query": "先查看 workspace 目录结构，然后创建一个 Python 脚本来列出该目录下所有文件", "expected_tool": "multi_tools"},
            "outputs": {"expected_action": "multi_tool_used"},
        },

        # 浏览器 + Python
        {
            "inputs": {"query": "搜索一个 Python 库的文档，然后把关键信息保存到文件", "expected_tool": "multi_tools", "requires_browser": True},
            "outputs": {"expected_action": "multi_tool_used"},
        },

        # ========== 6. 简单问答 (不需要工具) ==========
        # {
        #     "inputs": {"query": "法国首都是什么？", "expected_tool": "none"},
        #     "outputs": {"expected_answer_contains": "巴黎"},
        # },
        # {
        #     "inputs": {"query": "1+1等于多少？", "expected_tool": "none"},
        #     "outputs": {"expected_answer_contains": "2"},
        # },
        # {
        #     "inputs": {"query": "谁写了《西游记》？", "expected_tool": "none"},
        #     "outputs": {"expected_answer_contains": "吴承恩"},
        # },
        # {
        #     "inputs": {"query": "水的化学式是什么？", "expected_tool": "none"},
        #     "outputs": {"expected_answer_contains": "H2O"},
        # },
        {
            "inputs": {"query": "太阳系有几个行星？", "expected_tool": "none"},
            "outputs": {"expected_answer_contains": "8"},
        },

        # ========== 7. 知识问答 (可能需要工具) ==========
        {
            "inputs": {"query": "用 Python 写一个判断素数的函数", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_written"},
        },
        {
            "inputs": {"query": "写一个 Python 类，实现一个简单的栈数据结构", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_written"},
        },
        {
            "inputs": {"query": "用 Python 实现快速排序算法", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_written"},
        },

        # ========== 8. 复杂任务测试 ==========
        {
            "inputs": {"query": "创建一个完整的 TODO 应用，包含添加、删除、显示功能，保存到 Python 文件", "expected_tool": "multi_tools"},
            "outputs": {"expected_action": "multi_tool_used"},
        },
        {
            "inputs": {"query": "生成一个包含 10 个随机数的列表，计算平均值、中位数和标准差", "expected_tool": "python_execute"},
            "outputs": {"expected_action": "code_executed"},
        },
        {
            "inputs": {"query": "把 1 到 9 的数字写入文件，每个数字占一行，然后用 Python 读取并计算总和", "expected_tool": "multi_tools"},
            "outputs": {"expected_action": "multi_tool_used"},
        },

        # ========== 9. Terminate 工具测试 ==========
        # 这类任务应该在回答后主动终止
        {
            "inputs": {"query": "再见", "expected_tool": "terminate"},
            "outputs": {"expected_action": "conversation_ended"},
        },
        {
            "inputs": {"query": "好了，我的任务完成了", "expected_tool": "terminate"},
            "outputs": {"expected_action": "conversation_ended"},
        },
    ]

    dataset = ls_client.create_dataset(dataset_name=DATASET_NAME)
    ls_client.create_examples(dataset_id=dataset.id, examples=examples)
    print(f"数据集创建完成: {dataset.name}, 包含 {len(examples)} 个测试用例")
    return dataset.name


# ============================================================
# Step 3: LLM 评估器 (LLM as Judge)
# ============================================================

def create_llm_judge_evaluator():
    """
    创建一个基于 LLM 的评估器
    使用更强的模型来评估 Agent 的表现

    支持的环境变量:
    - LLM_JUDGE_API_KEY: 评估用的 API Key
    - LLM_JUDGE_BASE_URL: 评估用的 API 端点
    - LLM_JUDGE_MODEL: 评估用的模型 (默认 gpt-4o)
    """
    from openai import AsyncOpenAI

    # 优先使用专门的评估 API Key，如果没有就用默认的
    api_key = os.getenv("LLM_JUDGE_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("LLM_JUDGE_BASE_URL", os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    judge_model = os.getenv("LLM_JUDGE_MODEL", "gpt-4o")

    if not api_key:
        raise ValueError("需要配置 LLM_JUDGE_API_KEY 环境变量来使用 LLM 评估器")

    print(f"LLM Judge 配置: model={judge_model}, base_url={base_url}")

    judge_client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    @traceable(name="LLM-Judge-Decision")
    async def llm_judge_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
        """
        LLM 评估器：判断 Agent 是否正确完成了任务
        """
        query = inputs.get("query", "")
        output = outputs.get("output", "")
        thinking = outputs.get("thinking", "")
        print(f"思考过程: {thinking}")
        print(f"输出: {output}")
        print(f"用户问题: {query}")
        tool_names = outputs.get("tool_names", [])
        tool_calls_count = outputs.get("tool_calls_count", 0)

        # 构建评估 prompt
        evaluation_prompt = f"""你是一个严格的评估专家。请根据以下信息评估 Agent 的表现：

## 用户问题
{query}

## Agent 的思考过程
{thinking}

## Agent 使用的工具
{tool_names}

## Agent 的最终回答
{output}

## 评估标准

请从以下几个维度进行评分（0-1分）：

1. **任务完成度 (task_completion)**: Agent 是否正确完成了用户的问题？
   - 1分：完全正确完成了任务
   - 0.5分：部分完成或有轻微错误
   - 0分：未完成或错误严重

2. **工具使用合理性 (tool_usage)**: Agent 是否合理使用了工具？
   - 1分：工具选择合理，调用恰当
   - 0.5分：工具使用一般
   - 0分：工具使用不当或过度使用

3. **回答质量 (answer_quality)**: Agent 的最终回答是否有帮助、准确？
   - 1分：回答准确、有帮助
   - 0.5分：回答一般
   - 0分：回答无帮助或不准确

请以 JSON 格式返回评估结果：
{{
    "task_completion": <0-1之间的分数>,
    "tool_usage": <0-1之间的分数>,
    "answer_quality": <0-1之间的分数>,
    "reason": "<简要说明评估理由>",
    "overall_score": <上述三项的平均分>
}}

注意：只返回 JSON，不要有其他内容。"""

        try:
            # 调用更强的模型进行评估
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
            print(f"LLM Judge 评估结果: {result_text}")
            # 解析 JSON 结果
            # 尝试提取 JSON 部分
            if "```json" in result_text:
                result_text = result_text.split("```json")[1].split("```")[0]
            elif "```" in result_text:
                result_text = result_text.split("```")[1].split("```")[0]

            result = json.loads(result_text)

            # 返回 LangSmith 评估结果格式
            overall = result.get("overall_score", 0.5)

            return {
                "key": "llm_judge_overall",
                "score": overall,
                "reason": f"Task: {result.get('task_completion')}, Tool: {result.get('tool_usage')}, Quality: {result.get('answer_quality')}. {result.get('reason', '')}"
            }

        except Exception as e:
            return {
                "key": "llm_judge_overall",
                "score": 0.5,
                "reason": f"LLM evaluation failed: {str(e)}"
            }

    return llm_judge_evaluator


# ============================================================
# Step 4: 其他基础评估器
# ============================================================
@traceable(name="LLM-Has-Toolcalls")
def has_tool_calls(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估器：是否有工具调用"""
    tool_calls_count = outputs.get("tool_calls_count", 0)
    has_calls = tool_calls_count > 0

    return {
        "key": "has_tool_calls",
        "score": 1.0 if has_calls else 0.0,
        "reason": f"Tool calls count: {tool_calls_count}"
    }


def tool_call_efficiency(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估器：工具调用效率"""
    tool_calls_count = outputs.get("tool_calls_count", 0)

    if tool_calls_count == 0:
        score = 0.0
        reason = "No tool calls"
    elif tool_calls_count <= 2:
        score = 1.0
        reason = "Efficient tool usage"
    elif tool_calls_count <= 4:
        score = 0.7
        reason = "Moderate tool usage"
    else:
        score = 0.5
        reason = "Could use fewer tools"

    return {
        "key": "tool_call_efficiency",
        "score": score,
        "reason": f"{reason}, count: {tool_calls_count}"
    }


def has_output(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估器：是否有有效输出"""
    output = outputs.get("output", "")
    has_output = len(output) > 0 and output != "No output"

    return {
        "key": "has_output",
        "score": 1.0 if has_output else 0.0,
        "reason": f"Has output: {has_output}, length: {len(output)}"
    }


def token_efficiency(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """评估器：Token 使用效率"""
    total_tokens = outputs.get("total_tokens", 0)
    tool_calls_count = outputs.get("tool_calls_count", 1)

    tokens_per_call = total_tokens / tool_calls_count if tool_calls_count > 0 else total_tokens

    if tokens_per_call < 2000:
        score = 1.0
    elif tokens_per_call < 4000:
        score = 0.7
    else:
        score = 0.5

    return {
        "key": "token_efficiency",
        "score": score,
        "reason": f"Tokens per call: {tokens_per_call:.0f}"
    }


# ============================================================
# Step 5: 运行评估
# ============================================================

async def run_evaluation(dataset_name: str = None, use_llm_judge: bool = True):
    """运行评估"""
    ls_client = Client()

    if dataset_name is None:
        dataset_name = create_dataset()

    print(f"\n开始评估数据集: {dataset_name}")
    print("=" * 50)

    # 准备评估器列表
    evaluators = [
        has_tool_calls,
        tool_call_efficiency,
        has_output,
        token_efficiency,
    ]

    # 添加 LLM 评估器（如果启用）
    if use_llm_judge:
        print("添加 LLM 评估器 (GPT-4o)...")
        llm_judge = create_llm_judge_evaluator()
        evaluators.append(llm_judge)

    results = await ls_client.aevaluate(
        agent_tool_application,
        data=dataset_name,
        evaluators=evaluators,
        experiment_prefix="manus-llm-judge",
        description="Manus Agent LLM 评估测试",
        max_concurrency=2,
    )

    print("\n评估完成!")
    print(f"结果: {results}")
    return results


async def test_single_query():
    """测试单个查询"""
    test_queries = [
        "请帮我计算 123 * 456 的结果",
        "法国首都是什么？",
        "用 Python 写一个判断素数的函数",
    ]

    for query in test_queries:
        print(f"\n{'=' * 50}")
        print(f"测试查询: {query}")
        print(f"{'=' * 50}")

        result = await agent_tool_application({"query": query})
        print(f"输出: {result.get('output', '')[:150]}...")
        print(f"工具调用: {result.get('tool_names', [])}")
        print(f"调用次数: {result.get('tool_calls_count', 0)}")


async def test_llm_judge():
    """单独测试 LLM 评估器"""
    print("\n测试 LLM 评估器...")

    # 先运行一个查询
    query = "法国首都是什么？"
    result = await agent_tool_application({"query": query})

    print(f"\n查询: {query}")
    print(f"输出: {result.get('output', '')}")
    print(f"工具: {result.get('tool_names', [])}")

    # 创建评估器并测试
    evaluator = create_llm_judge_evaluator()

    # 模拟评估器输入
    inputs = {"query": query}
    outputs = result
    reference_outputs = {}

    eval_result = await evaluator(inputs, outputs, reference_outputs)
    print(f"\nLLM 评估结果: {eval_result}")


# ============================================================
# 主函数
# ============================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(description="OpenManus LLM 评估测试")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["eval", "test", "create-dataset", "llm-judge-test"],
        default="eval",
        help="运行模式"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="数据集名称"
    )
    parser.add_argument(
        "--no-llm-judge",
        action="store_true",
        help="不使用 LLM 评估器"
    )

    args = parser.parse_args()

    if args.mode == "create-dataset":
        create_dataset()
    elif args.mode == "test":
        asyncio.run(test_single_query())
    elif args.mode == "llm-judge-test":
        asyncio.run(test_llm_judge())
    elif args.mode == "eval":
        asyncio.run(run_evaluation(args.dataset, use_llm_judge=not args.no_llm_judge))


if __name__ == "__main__":
    main()
