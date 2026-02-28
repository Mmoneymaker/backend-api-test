"""
测试 LLM 工具调用 - 查看 LLM 返回的每个字段
"""
import asyncio
import json

from pydantic import BaseModel

# 导入 OpenManus 的 LLM 和相关模块
import sys
from pathlib import Path

_project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_project_root))

from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

from app.llm import LLM
from app.schema import Message, ToolChoice,ToolCall
from app.tool.base import BaseTool, ToolResult


# 1. 定义一个简单的工具
class Calculate(BaseTool):
    """一个简单的计算器工具"""

    name: str = "calculate"
    description: str = "用于执行简单的数学计算，如加法、减法、乘法、除法"
    parameters: dict = {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": ["add", "subtract", "multiply", "divide"],
                "description": "运算类型：add(加), subtract(减), multiply(乘), divide(除)"
            },
            "a": {"type": "number", "description": "第一个数字"},
            "b": {"type": "number", "description": "第二个数字"}
        },
        "required": ["operation", "a", "b"]
    }

    async def execute(self, operation: str, a: float, b: float) -> ToolResult:
        """执行计算"""
        try:
            if operation == "add":
                result = a + b
            elif operation == "subtract":
                result = a - b
            elif operation == "multiply":
                result = a * b
            elif operation == "divide":
                if b == 0:
                    return ToolResult(error="除数不能为零")
                result = a / b
            else:
                return ToolResult(error=f"未知运算: {operation}")

            return ToolResult(output=f"{a} {operation} {b} = {result}")
        except Exception as e:
            return ToolResult(error=str(e))


# 2. 创建 LLM 实例和工具
llm = LLM(config_name="default")
calculator = Calculate()

# 3. 定义工具参数
tools = [calculator.to_param()]


# 4. 测试函数
async def test_llm_tool_call():
    """测试 LLM 工具调用"""



    # 准备消息 - 让 LLM 使用工具计算 123 + 456
    messages = [
        Message.user_message("请帮我计算 123 + 456 等于多少？")
    ]

    print("=" * 60)
    print("发送的消息:")
    print("=" * 60)
    for msg in messages:
        print(f"role: {msg.role}")
        print(f"content: {msg.content}")
    print()

    # 调用 LLM
    print("=" * 60)
    print("调用 LLM API...")
    print("=" * 60)

    response = await llm.ask_tool(
        messages=messages,
        tools=tools,
        tool_choice=ToolChoice.AUTO
    )

    # 5. 打印 LLM 返回的每个字段
    print()
    print("=" * 60)
    print("LLM 返回的完整结果:")
    print("=" * 60)

    if response is None:
        print("LLM 返回为空")
        return

    # 打印所有属性
    print(f"\n--- response 对象类型: {type(response)} ---")
    print(f"\n--- response 所有属性 ---")
    for attr in dir(response):
        if not attr.startswith('_'):
            try:
                value = getattr(response, attr)
                if not callable(value):
                    print(f"\n{attr}:")
                    print(f"  类型: {type(value)}")
                    print(f"  值: {value}")
            except Exception as e:
                print(f"{attr}: 无法获取 - {e}")

    # 重点字段
    print("\n" + "=" * 60)
    print("关键字段详情:")
    print("=" * 60)

    print(f"\n1. role: {response.role}")

    print(f"\n2. content: {response.content}")


    print(f"\n3. tool_calls:")
    if response.tool_calls:
        for i, tc in enumerate(response.tool_calls):
            print(f"\n   --- ToolCall {i+1} ---")
            print(f"   id: {tc.id}")
            print(f"   type: {tc.type}")
            print(f"   function.name: {tc.function.name}")
            print(f"   function.arguments: {tc.function.arguments}")
            # 尝试解析 arguments 为 JSON
            try:
                args_dict = json.loads(tc.function.arguments)
                print(f"   function.arguments (解析后): {args_dict}")
            except:
                pass
    else:
        print("   None")

    tool_calls = tool_calls = (
            response.tool_calls if response and response.tool_calls else []
        )
    calculator=Calculate()
    results = []

    for command in tool_calls:
        args=json.loads(command.function.arguments)

        print(type(args))
        print(f"{args}")
        response=await calculator.execute(**args)

        print(f"🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟🌟工具的结果是:{response}")

        results.append(response)




    print(f"\n4. function_call (旧版，已废弃): {response.function_call}")

    print(f"\n5. refusal: {response.refusal}")

    print(f"\n6. annotations: {response.annotations}")



    # 打印原始响应（如果有）
    print("\n" + "=" * 60)
    print("原始 response 对象:")
    print("=" * 60)
    print(response)


# 运行测试
if __name__ == "__main__":
    asyncio.run(test_llm_tool_call())
