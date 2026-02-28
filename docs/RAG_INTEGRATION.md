# OpenManus RAG 集成方案

## 一、RAG 集成的三种方式

### 方式 1：作为 Agent 的 Tool（推荐）

```
┌─────────────────────────────────────────────────────────────────┐
│  Agent (Manus)                                                  │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  available_tools:                                              │
│  ├── PythonExecute      ← 现有                                  │
│  ├── BrowserUseTool    ← 现有                                  │
│  ├── StrReplaceEditor ← 现有                                    │
│  ├── RAGTool          ← 新增                                   │
│  └── Terminate        ← 现有                                  │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

**实现方式：**
```python
class RAGTool(BaseTool):
    name = "rag_search"
    description = "从知识库中检索相关信息"

    async def execute(self, query: str, top_k: int = 5) -> ToolResult:
        # 1. 向量检索
        results = await self.retriever.search(query, top_k=top_k)
        # 2. 构造上下文
        context = self.format_context(results)
        return ToolResult(output=context)
```

**优点：**
- Agent 可以自主决定何时调用 RAG
- 与其他工具统一管理
- 适合"需要时查询知识库"的场景

**缺点：**
- 需要在 prompt 中告诉 Agent 什么时候用 RAG
- 可能被 Agent 忽略

---

### 方式 2：作为独立接口（适合批处理）

```python
# 独立 RAG 接口
class RAGService:
    async def query(self, query: str) -> str:
        """直接返回检索结果"""

    async def query_with_context(self, query: str, agent: Manus):
        """将检索结果注入 Agent 的 context"""
        results = await self.retriever.search(query)
        context = self.format_context(results)
        agent.messages.append(SystemMessage(content=context))
```

**优点：**
- 精确控制检索时机
- 适合预处理场景

**缺点：**
- 需要修改 Agent 内部逻辑
- 与其他工具调用方式不一致

---

### 方式 3：混合模式（最佳实践）

```python
class HybridRAGTool(BaseTool):
    """支持两种模式"""

    async def execute(self, mode: str, query: str = None, file_path: str = None):
        if mode == "search":
            # 手动检索
            return await self.search(query)
        elif mode == "auto":
            # 自动检索（基于问题）
            return await self.auto_retrieve(query)
```

**优点：**
- 灵活可控
- 兼容多种场景

---

## 二、RAG Tool 的设计

### 基础功能

```python
class RAGTool(BaseTool):
    """RAG 检索工具"""

    name = "rag_search"
    description = """
    从知识库中检索相关信息。
    适用于：回答基于文档的问题、查找特定信息、总结文档内容等场景。

    参数：
    - query: 检索关键词或问题
    - top_k: 返回结果数量（默认5）
    - filter: 可选的元数据过滤条件
    """

    parameters = {
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "检索 query"},
            "top_k": {"type": "integer", "default": 5, "description": "返回数量"},
            "filter": {"type": "object", "description": "元数据过滤"}
        },
        "required": ["query"]
    }

    async def execute(self, query: str, top_k: int = 5, filter: dict = None):
        # 1. 向量化 query
        embedding = await self.embeddings.embed(query)

        # 2. 向量检索
        results = await self.vectorstore.search(
            embedding=embedding,
            top_k=top_k,
            filter=filter
        )

        # 3. 构造上下文
        context = self.format_results(results)

        return ToolResult(output=context)
```

### 上下文构造

```python
def format_results(self, results: List[Document]) -> str:
    """将检索结果格式化为上下文"""

    context_parts = ["以下是检索到的相关文档：\n"]

    for i, doc in enumerate(results, 1):
        context_parts.append(f"""
---
文档 {i} (来源: {doc.metadata.get('source', '未知')})
---
{doc.content}
""")

    return "\n".join(context_parts)
```

---

## 三、LangSmith 测试方法

### 1. 数据集设计

```python
examples = [
    # 知识库相关问题
    {
        "inputs": {
            "query": "公司的年假政策是什么？",
            "expected_source": "employee_handbook.pdf"
        },
        "outputs": {
            "expected_content_keywords": ["年假", "15天", "工作满一年"]
        }
    },

    # 需要多文档检索的问题
    {
        "inputs": {
            "query": "这个项目使用了哪些技术栈？",
            "expected_sources": ["README.md", "tech_spec.md"]
        },
        "outputs": {
            "expected_content_keywords": ["React", "Python", "PostgreSQL"]
        }
    },

    # 开放性问题（需要综合多文档）
    {
        "inputs": {
            "query": "总结这个项目的架构设计",
            "expected_action": "multi_doc_summarization"
        },
        "outputs": {}
    }
]
```

### 2. 评估指标

| 指标 | 说明 | 评估方式 |
|------|------|----------|
| **相关性 (Relevance)** | 检索结果是否与问题相关 | LLM 判断 |
| **完整性 (Completeness)** | 是否包含回答问题的所有信息 | LLM 判断 |
| **准确性 (Accuracy)** | 检索内容是否正确 | 关键词匹配 |
| **来源正确性 (Source)** | 是否从正确文档检索 | 来源匹配 |
| **上下文质量** | 上下文是否便于 LLM 理解 | LLM 判断 |

### 3. LLM 评估器示例

```python
def rag_quality_evaluator(inputs: dict, outputs: dict, reference_outputs: dict) -> dict:
    """
    RAG 质量评估器
    """
    query = inputs.get("query", "")
    output = outputs.get("output", "")
    expected_keywords = reference_outputs.get("expected_content_keywords", [])

    # 构建评估 prompt
    evaluation_prompt = f"""
    你是一个 RAG 系统评估专家。请评估以下检索结果的质量：

    ## 用户问题
    {query}

    ## 检索结果
    {output}

    ## 评估标准

    1. **相关性 (0-1)**: 检索结果是否与问题相关？

    2. **完整性 (0-1)**: 结果是否足够回答问题？

    3. **准确性 (0-1)**: 信息是否准确可靠？

    请返回 JSON：
    {{
        "relevance": <0-1>,
        "completeness": <0-1>,
        "accuracy": <0-1>,
        "overall": <平均分>,
        "reason": "<评估理由>"
    }}
    """

    # 调用 LLM 评估
    # ...

    return {
        "key": "rag_quality",
        "score": overall,
        "reason": reason
    }
```

---

## 四、性能指标在 Prompt 中的体现

### 1. System Prompt 设计

```python
RAG_SYSTEM_PROMPT = """
你是一个智能助手，可以通过 RAG 工具从知识库中检索信息来回答问题。

## 可用工具

1. **rag_search**: 从知识库中检索相关信息
   - 用法: rag_search(query="问题", top_k=5)
   - 适用: 回答基于文档的问题、查找特定信息

2. **python_execute**: 执行 Python 代码

3. **terminate**: 完成任务

## 决策指南

当用户提问时，根据以下规则决定是否使用 RAG：

| 问题类型 | 示例 | 是否检索 |
|----------|------|----------|
| 事实性问题 | "公司年假政策是什么？" | ✅ 需要 |
| 知识性问题 | "Python 如何定义函数？" | ❌ 不需要 |
| 项目相关 | "这个项目用的什么技术？"| ✅ 需要 |
| 计算问题 | "1+1 等于多少？" | ❌ 不需要 |
| 开放性问题 | "总结 XXX 文档" | ✅ 需要 |

## 回答质量要求

1. **引用来源**: 回答中必须提及信息来源
   - 正确: "根据《员工手册》第三条，年假为 15 天..."
   - 错误: "年假是 15 天"

2. **仅使用检索到的信息**: 不要编造未检索到的内容

3. **明确标注不确定**: 如果检索结果不足以回答，请明确说明
"""

# 注册到 Agent
system_prompt = RAG_SYSTEM_PROMPT
```

### 2. 检索质量问题检测 Prompt

```python
# 在 Agent think 阶段追加的检查
POST_THINK_PROMPT = """
基于以上思考，检查以下问题：

1. 是否有需要检索但未检索的问题？
2. 检索结果是否足够回答问题？
3. 是否有不确定的信息需要确认？

如果有问题，返回需要执行的额外操作。
"""
```

### 3. 答案验证 Prompt

```python
VALIDATION_PROMPT = """
你刚准备了一个回答，请验证：

## 用户问题
{query}

## 你准备回答
{answer}

## 检索到的上下文
{context}

## 验证项

1. 回答是否基于检索到的上下文？
2. 是否有遗漏的重要信息？
3. 是否有未经证实的猜测？

返回验证结果和必要的修改。
"""
```

---

## 五、实施建议

### 1. 推荐的集成方式

```
┌─────────────────────────────────────────────────────────────┐
│                    推荐: Tool 模式                           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  - 统一工具接口                                              │
│  - Agent 自主决策                                            │
│  - 易于测试和扩展                                            │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2. 实施步骤

1. **定义 RAGTool 类**
   - 向量检索功能
   - 上下文格式化
   - 元数据处理

2. **集成到 Manus**
   - 添加到 available_tools
   - 修改 system_prompt

3. **创建测试**
   - 数据集：问题 + 期望来源
   - 评估器：相关性、完整性、准确性

4. **监控指标**
   - 检索命中率
   - 答案准确率
   - Token 消耗

---

## 六、测试命令示例

```bash
# 1. 创建 RAG 测试数据集
python tests/test_rag.py --mode create-dataset

# 2. 运行评估
python tests/test_rag.py --mode eval

# 3. 查看 LangSmith 结果
# 访问 https://smith.langchain.com/ 查看 traces 和评估结果
```

---

## 总结

| 集成方式 | 适用场景 | 推荐度 |
|----------|----------|--------|
| Tool 模式 | Agent 需要动态检索 | ⭐⭐⭐⭐⭐ |
| 独立接口 | 预处理/批处理 | ⭐⭐⭐ |
| 混合模式 | 复杂系统 | ⭐⭐⭐⭐ |

**核心思路**：将 RAG 作为 Agent 的"知识库工具"，通过 prompt 指导 Agent 在合适时机调用，并通过 LangSmith 评估检索质量和回答质量。
