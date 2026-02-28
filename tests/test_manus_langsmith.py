"""
LangSmith 测试脚本 - 简单方式
使用 @traceable 装饰器追踪 Manus Agent 的执行
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

from langsmith import traceable
from langsmith.run_helpers import get_current_run_tree

from app.agent.manus import Manus


# 使用 @traceable 装饰器自动追踪函数（与 .env 中 LANGSMITH_PROJECT 一致）
@traceable(name="manus-agent", project_name="manus-evaluation")
async def run_manus(query: str) -> Dict[str, Any]:
    """运行 Manus Agent 并返回结果"""
    agent = await Manus.create()
    try:
        result = await agent.run(query)
        input_tokens = getattr(agent.llm, "total_input_tokens", 0) if getattr(agent, "llm", None) else 0
        output_tokens = getattr(agent.llm, "total_completion_tokens", 0) if getattr(agent, "llm", None) else 0
        run_tree = get_current_run_tree()
        if run_tree is not None:
            try:
                run_tree.add_metadata({"input_tokens": input_tokens, "output_tokens": output_tokens})
            except Exception:
                pass
        return {
            "query": query,
            "result": result,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }
    finally:
        await agent.cleanup()


async def test_single_query(query: str):
    """测试单个查询"""
    print(f"\n{'='*50}")
    print(f"测试查询: {query}")
    print(f"{'='*50}\n")

    result = await run_manus(query)

    print(f"\n{'='*50}")
    print(f"结果: {result}")
    print(f"{'='*50}\n")

    return result


async def main():
    """运行测试"""
    # 确保 LangSmith tracing 已启用
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    if not os.environ.get("LANGSMITH_API_KEY"):
        print("警告: 未设置 LANGSMITH_API_KEY，trace 不会上报到 LangSmith。请在 .env 中配置。")

    # 测试查询 - 你可以修改这些测试用例
    test_queries = [
        "为什么HashMap不是线程安全的，将他的原因总结到docs目录下的一个.md文件中",
        # 可以添加更多测试用例...
    ]

    for query in test_queries:
        await test_single_query(query)


if __name__ == "__main__":
    asyncio.run(main())
