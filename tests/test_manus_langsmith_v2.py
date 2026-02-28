"""
LangSmith 测试脚本 - 手动追踪方式
使用 langsmith.Client 手动创建 traces，可以追踪 LLM 调用
"""
import asyncio
import os
from typing import Dict, Any, Optional
from contextlib import asynccontextmanager

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv()

from langsmith import Client
from langsmith.run_trees import RunTree
from app.agent.manus import Manus


class LangSmithTracer:
    """手动 LangSmith 追踪器"""

    def __init__(self, project_name: str = "manus-evaluation"):
        self.client = Client()
        self.project_name = project_name
        self.current_run: Optional[RunTree] = None

    @asynccontextmanager
    async def trace(self, name: str, inputs: Dict, metadata: Dict = None):
        """创建一个 trace 上下文"""
        self.current_run = self.client.create_run(
            name=name,
            run_type="agent",
            inputs=inputs,
            metadata=metadata or {},
            project_name=self.project_name,
        )
        try:
            yield self.current_run
        except Exception as e:
            self.client.end_run(
                run_id=self.current_run.id,
                error=str(e)
            )
            raise
        finally:
            self.current_run = None

    def log_llm(self, messages: list, response: str, model: str):
        """记录 LLM 调用"""
        if not self.current_run:
            return

        self.client.create_run(
            name=model,
            run_type="llm",
            inputs={"messages": messages},
            outputs={"response": response},
            parent_run_id=self.current_run.id,
            project_name=self.project_name,
        )

    def log_tool(self, tool_name: str, arguments: Dict, result: Any):
        """记录工具调用"""
        if not self.current_run:
            return

        self.client.create_run(
            name=tool_name,
            run_type="tool",
            inputs=arguments,
            outputs={"result": str(result)[:500]},  # 限制长度
            parent_run_id=self.current_run.id,
            project_name=self.project_name,
        )


# 全局追踪器
tracer = LangSmithTracer(project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))


async def run_manus_with_tracing(query: str) -> Dict[str, Any]:
    """运行 Manus Agent 并追踪执行过程"""
    async with tracer.trace("manus-agent", inputs={"query": query}) as run:
        agent = await Manus.create()
        try:
            # 注意：这里只是一个简单的追踪
            # 实际的 LLM 追踪需要在 LLM 调用处添加
            result = await agent.run(query)
            return {
                "query": query,
                "result": result,
            }
        finally:
            await agent.cleanup()


async def test_single_query(query: str):
    """测试单个查询"""
    print(f"\n{'='*50}")
    print(f"测试查询: {query}")
    print(f"{'='*50}\n")

    result = await run_manus_with_tracing(query)

    print(f"\n{'='*50}")
    print(f"结果: {result}")
    print(f"{'='*50}\n")

    return result


async def main():
    """运行测试"""
    # 测试查询
    test_queries = [
        "法国首都在哪",
    ]

    for query in test_queries:
        await test_single_query(query)


if __name__ == "__main__":
    asyncio.run(main())
