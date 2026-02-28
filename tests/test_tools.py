"""
OpenManus 工具测试
直接测试各个工具的调用，并使用 LangSmith 追踪
"""
import asyncio
import os
import sys
from pathlib import Path

# 将项目根目录加入 Python 路径
_project_root = Path(__file__).resolve().parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# 加载 .env 文件中的环境变量
from dotenv import load_dotenv

load_dotenv(_project_root / ".env")

# LangSmith 追踪
from langsmith import traceable

from app.tool import ToolCollection
from app.tool.python_execute import PythonExecute
from app.tool.str_replace_editor import StrReplaceEditor
from app.tool.terminate import Terminate
from app.tool.ask_human import AskHuman


# ============================================================
# 工具测试 (使用 @traceable 装饰)
# ============================================================

@traceable(name="test-python-execute", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def test_python_execute():
    """测试 PythonExecute 工具"""
    print("\n" + "=" * 50)
    print("测试: PythonExecute")
    print("=" * 50)

    tool = PythonExecute()

    # 测试 1: 简单计算
    result = await tool.execute(code="print(1 + 1)")
    print(f"1 + 1 = {result}")

    # 测试 2: 字符串操作
    result = await tool.execute(code='print("Hello, World!".upper())')
    print(f"String upper: {result}")

    # 测试 3: 列表操作
    result = await tool.execute(code="print([1, 2, 3, 4, 5])")
    print(f"List: {result}")

    # 测试 4: 错误处理
    result = await tool.execute(code="print(undefined_var)")
    print(f"Error: {result}")

    return result


@traceable(name="test-str-replace-editor", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def test_str_replace_editor():
    """测试 StrReplaceEditor 工具"""
    print("\n" + "=" * 50)
    print("测试: StrReplaceEditor")
    print("=" * 50)

    tool = StrReplaceEditor()
    workspace = _project_root / "workspace"

    # 测试 1: 创建文件
    test_file = workspace / "test_tool.txt"
    result = await tool.execute(
        command="create",
        file_path=str(test_file),
        text="Hello from OpenManus tool test!"
    )
    print(f"Create file: {result}")

    # 测试 2: 读取文件
    result = await tool.execute(
        command="read",
        file_path=str(test_file)
    )
    print(f"Read file: {result}")

    # 测试 3: 编辑文件
    result = await tool.execute(
        command="str_replace",
        file_path=str(test_file),
        old_str="Hello",
        new_str="Hi"
    )
    print(f"Edit file: {result}")

    # 测试 4: 再次读取
    result = await tool.execute(
        command="read",
        file_path=str(test_file)
    )
    print(f"Read after edit: {result}")

    # 清理
    if test_file.exists():
        test_file.unlink()

    return result


@traceable(name="test-terminate", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def test_terminate():
    """测试 Terminate 工具"""
    print("\n" + "=" * 50)
    print("测试: Terminate")
    print("=" * 50)

    tool = Terminate()

    # 测试: 正常终止
    result = await tool.execute(status="success")
    print(f"Terminate: {result}")

    return result


@traceable(name="test-tool-collection", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def test_tool_collection():
    """测试 ToolCollection"""
    print("\n" + "=" * 50)
    print("测试: ToolCollection")
    print("=" * 50)

    # 创建工具集合
    tools = ToolCollection(
        PythonExecute(),
        StrReplaceEditor(),
        Terminate(),
        AskHuman(),
    )

    # 打印所有工具
    print("注册的工具:")
    for t in tools.tools:
        print(f"  - {t.name}: {t.description[:50]}...")

    # 获取工具Schema
    schema = tools.to_params()
    print(f"\n工具数量: {len(schema)}")

    return schema


@traceable(name="test-browser", project_name=os.getenv("LANGSMITH_PROJECT", "manus-evaluation"))
async def test_browser():
    """测试浏览器工具（如果有）"""
    print("\n" + "=" * 50)
    print("测试: BrowserUseTool")
    print("=" * 50)

    try:
        from app.tool.browser_use_tool import BrowserUseTool

        tool = BrowserUseTool()
        print(f"Browser tool: {tool.name}")
        print(f"Description: {tool.description[:100]}...")

        # 注意: 浏览器工具需要实际的浏览器环境，这里只测试初始化
        return tool
    except Exception as e:
        print(f"Browser tool test skipped: {e}")
        return None


# ============================================================
# 主函数
# ============================================================

async def main():
    """运行所有工具测试"""
    print("=" * 50)
    print("OpenManus 工具测试 (LangSmith 追踪)")
    print("=" * 50)

    # 确保 LangSmith 追踪已启用
    print(f"LangSmith Project: {os.getenv('LANGSMITH_PROJECT', 'manus-evaluation')}")
    print()

    # 测试各个工具
    await test_python_execute()
    await test_str_replace_editor()
    await test_terminate()
    await test_tool_collection()
    await test_browser()

    print("\n" + "=" * 50)
    print("所有测试完成! 查看 LangSmith 获取追踪结果")
    print("=" * 50)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OpenManus 工具测试")
    parser.add_argument(
        "--tool",
        type=str,
        choices=["python", "editor", "terminate", "collection", "browser", "all"],
        default="all",
        help="要测试的工具"
    )

    args = parser.parse_args()

    if args.tool == "python":
        asyncio.run(test_python_execute())
    elif args.tool == "editor":
        asyncio.run(test_str_replace_editor())
    elif args.tool == "terminate":
        asyncio.run(test_terminate())
    elif args.tool == "collection":
        asyncio.run(test_tool_collection())
    elif args.tool == "browser":
        asyncio.run(test_browser())
    else:
        asyncio.run(main())
