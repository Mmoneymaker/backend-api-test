# Manus Agent 适配 LangSmith 测试指南

## 概述

LangSmith 是 LangChain 生态的可观测性平台，提供以下核心功能：
- **Tracing**: 追踪 LLM 调用链，记录完整的执行过程
- **Datasets**: 创建和管理测试数据集
- **Evaluation**: 自动评估 Agent 性能

由于 Manus 不是基于 LangChain 构建的，需要通过以下方式适配。

---

## 第一步：环境准备

### 1.1 安装依赖

```bash
pip install langsmith
```

### 1.2 配置 LangSmith 环境变量

在 `.env` 文件中添加：

```bash
# LangSmith 配置
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_api_key_here
LANGSMITH_PROJECT=manus-evaluation  # 项目名称
```

### 1.3 获取 API Key

1. 访问 [LangSmith](https://smith.langchain.com/)
2. 注册/登录账号
3. 进入 Settings → API Keys
4. 创建新的 API Key

---

## 第二步：创建测试数据集

### 2.1 方式一：通过代码创建

```python
# tests/data/create_dataset.py
from langsmith import Client

client = Client()

# 定义测试用例
examples = [
    {
        "input": "请帮我计算 1+1 等于多少",
        "expected": "2"
    },
    {
        "input": "写一个 Python 函数来判断一个数是否为素数",
        "expected": "包含 is_prime 或类似函数定义"
    },
    {
        "input": "搜索关于 LangChain 的最新资讯",
        "expected": "返回搜索结果"
    },
]

# 创建数据集
dataset = client.create_dataset(
    dataset_name="manus-test-dataset",
    description="Manus Agent 测试数据集"
)

# 添加示例
for example in examples:
    client.create_example(
        inputs={"query": example["input"]},
        outputs={"expected": example["expected"]},
        dataset_id=dataset.id
    )

print(f"数据集创建完成: {dataset.id}")
```

### 2.2 方式二：通过 LangSmith Web UI 创建

1. 登录 LangSmith
2. 切换到 Datasets 标签
3. 点击 "Create Dataset"
4. 手动添加测试用例

---

## 第三步：编写测试脚本

### 3.1 方式一：直接使用 LangSmith SDK（推荐）

```python
# tests/test_manus_with_langsmith.py
import asyncio
import os
from typing import Dict, Any

from langsmith import traceable
from langsmith.run_trees import RunTree

# 确保环境变量已设置
os.environ.setdefault("LANGSMITH_TRACING", "true")

from app.agent.manus import Manus


# 使用 @traceable 装饰器自动追踪函数
@traceable(name="manus-agent", project_name="manus-evaluation")
async def run_manus(query: str) -> Dict[str, Any]:
    """运行 Manus Agent 并返回结果"""
    agent = await Manus.create()
    try:
        result = await agent.run(query)
        return {
            "query": query,
            "result": result,
            "messages": [msg.dict() for msg in agent.messages]
        }
    finally:
        await agent.cleanup()


async def test_single_query():
    """测试单个查询"""
    query = "请计算 5 的阶乘"

    # 方式1: 直接使用装饰器
    result = await run_manus(query)
    print(f"Result: {result}")

    # 方式2: 手动创建 trace
    with traceable(name="manual-trace") as rt:
        rt.metadata = {"query": query}
        agent = await Manus.create()
        result = await agent.run(query)
        rt.end(outputs={"result": result})


if __name__ == "__main__":
    asyncio.run(test_single_query())
```

### 3.2 方式二：自定义 Tracer 类（更精细的控制）

```python
# app/tracing/langsmith_tracer.py
from typing import Dict, Any, Optional, List
from langsmith import Client
from langsmith.run_trees import RunTree
from langsmith.schemas import Run

from app.schema import Message


class LangSmithTracer:
    """自定义 LangSmith 追踪器"""

    def __init__(self, project_name: str = "manus-evaluation"):
        self.client = Client()
        self.project_name = project_name
        self.current_run: Optional[RunTree] = None

    def start_trace(self, query: str, metadata: Dict = None) -> RunTree:
        """开始一个新的 trace"""
        self.current_run = self.client.create_run(
            name="manus-agent",
            run_type="agent",
            inputs={"query": query},
            metadata=metadata or {}
        )
        return self.current_run

    def log_tool_call(self, tool_name: str, arguments: Dict, result: Any):
        """记录工具调用"""
        if not self.current_run:
            return

        self.client.create_run(
            name=tool_name,
            run_type="tool",
            inputs=arguments,
            outputs={"result": str(result)},
            parent_run_id=self.current_run.id
        )

    def log_llm_call(self, messages: List[Message], response: str):
        """记录 LLM 调用"""
        if not self.current_run:
            return

        self.client.create_run(
            name="openai-llm",
            run_type="llm",
            inputs={"messages": [msg.dict() for msg in messages]},
            outputs={"response": response},
            parent_run_id=self.current_run.id
        )

    def end_trace(self, output: str, error: Optional[str] = None):
        """结束 trace"""
        if not self.current_run:
            return

        self.client.end_run(
            run_id=self.current_run.id,
            outputs={"result": output},
            error=error
        )
```

### 3.3 集成到 Manus Agent

```python
# app/agent/manus_with_tracing.py
from typing import Optional
from app.agent.manus import Manus
from app.tracing.langsmith_tracer import LangSmithTracer


class ManusWithTracing(Manus):
    """带 LangSmith 追踪的 Manus Agent"""

    tracer: Optional[LangSmithTracer] = None

    async def think(self) -> bool:
        """重写 think 方法，添加追踪"""
        # 记录 LLM 调用前状态
        if self.tracer:
            self.tracer.log_llm_call(
                messages=self.memory.messages,
                response="Thinking..."
            )

        result = await super().think()

        return result

    async def execute_tool(self, command):
        """重写工具执行方法，添加追踪"""
        tool_name = command.function.name
        args = command.function.arguments

        if self.tracer:
            self.tracer.log_tool_call(
                tool_name=tool_name,
                arguments=args,
                result="Executing..."
            )

        result = await super().execute_tool(command)

        if self.tracer:
            self.tracer.log_tool_call(
                tool_name=tool_name,
                arguments=args,
                result=result
            )

        return result

    async def run(self, request: str = None):
        """重写 run 方法，添加完整的 trace"""
        if self.tracer:
            self.tracer.start_trace(
                query=request,
                metadata={
                    "agent_name": self.name,
                    "max_steps": self.max_steps
                }
            )

        try:
            result = await super().run(request)

            if self.tracer:
                self.tracer.end_trace(output=result)

            return result
        except Exception as e:
            if self.tracer:
                self.tracer.end_trace(output="", error=str(e))
            raise
```

---

## 第四步：运行评估

### 4.1 创建评估器

```python
# tests/evaluators/response_evaluator.py
from typing import Optional
from langsmith.evaluation import EvaluationResult, run_evaluator


@run_evaluator
def evaluate_response(run: dict, example: dict) -> EvaluationResult:
    """评估 Agent 响应质量"""

    # 从 run 中获取输入和输出
    query = run.get("inputs", {}).get("query", "")
    result = run.get("outputs", {}).get("result", "")
    expected = example.get("outputs", {}).get("expected", "")

    # 简单的评估逻辑
    score = 0.0
    reason = ""

    if expected in result:
        score = 1.0
        reason = "Response contains expected output"
    elif any(keyword in result.lower() for keyword in ["error", "fail"]):
        score = 0.0
        reason = "Response contains error"
    else:
        score = 0.5
        reason = "Partial match"

    return EvaluationResult(
        key="response_quality",
        score=score,
        reason=reason
    )


@run_evaluator
def evaluate_step_count(run: dict, example: dict) -> EvaluationResult:
    """评估 Agent 执行的步数"""

    # 假设 trace 中记录了 step 数量
    steps = run.get("metrics", {}).get("step_count", 0)

    # 期望步数应该合理（不太多也不太少）
    if steps <= 3:
        score = 1.0
    elif steps <= 10:
        score = 0.8
    else:
        score = 0.5

    return EvaluationResult(
        key="efficiency",
        score=score,
        reason=f"Completed in {steps} steps"
    )
```

### 4.2 运行批量评估

```python
# tests/run_evaluation.py
from langsmith import Client
from langsmith.evaluation import evaluate

from tests.test_manus_with_langsmith import run_manus
from tests.evaluators.response_evaluator import evaluate_response, evaluate_step_count


def main():
    client = Client()

    # 获取数据集
    dataset = client.read_dataset(dataset_name="manus-test-dataset")

    # 运行评估
    results = evaluate(
        run_manus,
        data=dataset,
        evaluators=[evaluate_response, evaluate_step_count],
        project_name="manus-evaluation",
    )

    # 查看结果
    print(f"评估完成！")
    print(f"平均分数: {results.mean_score}")


if __name__ == "__main__":
    main()
```

---

## 第五步：查看结果

### 5.1 在 Web UI 查看 Traces

1. 登录 [LangSmith](https://smith.langchain.com/)
2. 选择项目 "manus-evaluation"
3. 查看 Traces 列表
4. 点击任意 trace 查看详细调用链

### 5.2 关键指标

LangSmith 会自动收集以下指标：
- **Latency**: 响应时间
- **Token Usage**: Token 消耗
- **Cost**: 调用成本
- **Step Count**: 执行步数
- **Error Rate**: 错误率

---

## 第六步：进阶功能

### 6.1 自定义反馈收集

```python
# 在 trace 中添加反馈
from langsmith import feedback

# 方式1: 基于 run ID
feedback(
    project_name="manus-evaluation",
    run_id="run_id_here",
    key="user-feedback",
    score=1.0,  # 0-1 之间
    comment="Great result!"
)

# 方式2: 在 trace 内部
@traceable
def my_function():
    # ... 执行逻辑
    pass
```

### 6.2 比较不同版本

```python
# tests/compare_versions.py
from langsmith import Client

client = Client()

# 获取不同版本的运行记录
runs_v1 = client.list_runs(
    project_name="manus-evaluation",
    filter='eq(metadata.version, "v1")'
)

runs_v2 = client.list_runs(
    project_name="manus-evaluation",
    filter='eq(metadata.version, "v2")'
)

# 计算平均分数
avg_v1 = sum(r.score for r in runs_v1) / len(runs_v1)
avg_v2 = sum(r.score for r in runs_v2) / len(runs_v2)

print(f"V1 平均分: {avg_v1}")
print(f"V2 平均分: {avg_v2}")
```

### 6.3 添加自定义元数据

```python
from langsmith import traceable

@traceable(
    name="manus-agent",
    metadata={
        "agent_version": "1.0.0",
        "model": "gpt-4",
        "temperature": 0.7
    }
)
async def run_manus_with_metadata(query: str):
    agent = await Manus.create()
    result = await agent.run(query)
    return result
```

---

## 总结

| 步骤 | 内容 | 难度 |
|------|------|------|
| 1 | 环境准备、安装依赖 | 简单 |
| 2 | 创建测试数据集 | 简单 |
| 3 | 编写测试脚本（选择集成方式） | 中等 |
| 4 | 运行评估 | 简单 |
| 5 | 查看结果和分析 | 简单 |
| 6 | 进阶功能 | 困难 |

**推荐集成方式**: 方式一（使用 `@traceable` 装饰器）最简单，方式二（自定义 Tracer）提供更精细的控制。

---

## 参考资源

- [LangSmith 官方文档](https://docs.smith.langchain.com/)
- [LangSmith Python SDK](https://pypi.org/project/langsmith/)
- [LangChain Python API](https://api.python.langchain.com/en/latest/)
