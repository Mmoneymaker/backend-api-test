# OpenManus Tool Calling 实现机制总结

在面试中回答 Tool Calling 的实现原理时，可以从**整体架构**、**具体工具实现**和**执行流程**三个层面来阐述。

---

## 一、整体架构概览

OpenManus 采用分层设计，核心组件如下：

```
┌─────────────────────────────────────────────────────────┐
│                    Agent Layer                          │
│  (ToolCallAgent, ReActAgent)                           │
├─────────────────────────────────────────────────────────┤
│                 Tool Collection                         │
│  (工具注册与执行管理)                                    │
├─────────────────────────────────────────────────────────┤
│                   Tool Base                             │
│  (BaseTool - 所有工具的基类)                            │
├─────────────────────────────────────────────────────────┤
│              Concrete Tools                             │
│  (WebSearch, Crawl4ai, FileOperators, etc.)            │
└─────────────────────────────────────────────────────────┘
```

### 核心文件

| 层级 | 文件路径 | 功能 |
|------|----------|------|
| 基础定义 | `app/tool/base.py` | BaseTool 基类、ToolResult 结果类 |
| 工具集合 | `app/tool/tool_collection.py` | 工具注册与执行管理 |
| Schema 定义 | `app/schema.py` | Message、ToolCall 等数据结构 |
| LLM 调用 | `app/llm.py` | ask_tool 方法实现 tool calling |
| Agent | `app/agent/toolcall.py` | 工具调用 Agent 实现 |

---

## 二、Web 搜索工具实现

### 核心文件

- **Web 搜索入口**: `app/tool/web_search.py`
- **搜索引擎基类**: `app/tool/search/base.py`
- **具体实现**: `google_search.py`, `duckduckgo_search.py`, `bing_search.py`, `baidu_search.py`

### 实现架构

```python
# 1. 搜索引擎基类
class WebSearchEngine(BaseModel):
    def perform_search(self, query, num_results, *args, **kwargs) -> List[SearchItem]:
        raise NotImplementedError

# 2. 具体搜索引擎实现（以 Google 为例）
class GoogleSearchEngine(WebSearchEngine):
    def perform_search(self, query, num_results=10, *args, **kwargs):
        raw_results = search(query, num_results=num_results, advanced=True)
        return [SearchItem(title=item.title, url=item.url, description=item.description)
                for item in raw_results]

# 3. Web 搜索工具
class WebSearch(BaseTool):
    name: str = "web_search"
    _search_engine: dict = {
        "google": GoogleSearchEngine(),
        "baidu": BaiduSearchEngine(),
        "duckduckgo": DuckDuckGoSearchEngine(),
        "bing": BingSearchEngine(),
    }

    async def execute(self, query, num_results=5, fetch_content=False, ...):
        # 支持多引擎 fallback
        for engine_name in engine_order:
            results = await self._perform_search_with_engine(engine, query, ...)
            if results:
                return SearchResponse(results=results, ...)
```

### 特性

- **多引擎 fallback**: 按配置顺序尝试，失败后自动切换
- **重试机制**: 使用 tenacity 库实现指数退避重试
- **内容抓取**: 可选 `fetch_content` 参数，获取搜索结果的完整内容
- **结果模型化**: SearchResult、SearchMetadata、SearchResponse 结构化返回

---

## 三、网页抓取工具实现

### 核心文件

`app/tool/crawl4ai.py`

### 实现方式

```python
class Crawl4aiTool(BaseTool):
    name: str = "crawl4ai"
    description: str = "Web crawler that extracts clean, AI-ready content..."

    async def execute(self, urls, timeout=30, bypass_cache=False, word_count_threshold=10):
        from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig

        # 浏览器配置
        browser_config = BrowserConfig(
            headless=True,
            browser_type="chromium",
            java_script_enabled=True,
        )

        # 爬取配置
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS if bypass_cache else CacheMode.ENABLED,
            word_count_threshold=word_count_threshold,
            process_iframes=True,
        )

        async with AsyncWebCrawler(config=browser_config) as crawler:
            for url in valid_urls:
                result = await crawler.arun(url=url, config=run_config)
                if result.success:
                    markdown = result.markdown
                    # 提取 links、images 数量
```

### 特性

- **无头浏览器**: 使用 Chromium 处理 JavaScript 渲染的页面
- **Markdown 输出**: 直接生成 AI 友好的 markdown 格式
- **缓存支持**: 内置缓存机制，可选 bypass
- **多 URL 处理**: 支持批量抓取，返回详细统计信息

---

## 四、文件操作工具实现

### 核心文件

- **通用文件操作接口**: `app/tool/file_operators.py`
- **编辑器工具**: `app/tool/str_replace_editor.py`
- **Sandbox 文件工具**: `app/tool/sandbox/sb_files_tool.py`

### 文件操作接口设计

```python
# file_operators.py
@runtime_checkable
class FileOperator(Protocol):
    async def read_file(self, path: PathLike) -> str: ...
    async def write_file(self, path: PathLike, content: str) -> None: ...
    async def is_directory(self, path: PathLike) -> bool: ...
    async def exists(self, path: PathLike) -> bool: ...
    async def run_command(self, cmd: str, timeout: float) -> Tuple[int, str, str]: ...

# 本地环境实现
class LocalFileOperator(FileOperator):
    async def read_file(self, path):
        return Path(path).read_text(encoding="utf-8")

    async def write_file(self, path, content):
        Path(path).write_text(content, encoding="utf-8")
```

### StrReplaceEditor 工具

```python
class StrReplaceEditor(BaseTool):
    name: str = "str_replace_editor"
    parameters: dict = {
        "command": {"enum": ["view", "create", "str_replace", "insert", "undo_edit"]},
        "path": {"type": "string"},
        "file_text": {...},
        "old_str": {...},
        "new_str": {...},
        "insert_line": {...},
    }

    async def execute(self, command, path, ...):
        if command == "view":
            return await self.view(path, view_range, operator)
        elif command == "create":
            await operator.write_file(path, file_text)
        elif command == "str_replace":
            new_content = content.replace(old_str, new_str)
            await operator.write_file(path, new_content)
```

---

## 五、Tool Calling 执行流程

### Agent 层面 (`toolcall.py`)

```python
class ToolCallAgent(ReActAgent):
    available_tools: ToolCollection

    async def think(self) -> bool:
        # 1. 调用 LLM 的 ask_tool 方法
        response = await self.llm.ask_tool(
            messages=self.messages,
            tools=self.available_tools.to_params(),  # 工具 schema
            tool_choice=self.tool_choices,
        )
        # 2. 解析 tool_calls
        self.tool_calls = response.tool_calls
        return bool(self.tool_calls)

    async def act(self) -> str:
        # 3. 遍历执行工具
        for command in self.tool_calls:
            result = await self.execute_tool(command)
            # 4. 添加 tool 结果到 memory
            tool_msg = Message.tool_message(content=result, name=command.function.name, ...)
            self.memory.add_message(tool_msg)

    async def execute_tool(self, command: ToolCall) -> str:
        # 5. 从 ToolCollection 获取工具并执行
        tool = self.available_tools.get_tool(name)
        result = await tool.execute(**args)
        return str(result)
```

### LLM 层面 (`llm.py`)

```python
async def ask_tool(self, messages, tools, tool_choice, ...):
    # 1. 格式化 messages 和 tools
    params = {
        "model": self.model,
        "messages": formatted_messages,
        "tools": tools,  # [{"type": "function", "function": {...}}]
        "tool_choice": tool_choice,
    }

    # 2. 调用 OpenAI API
    response = await self.client.chat.completions.create(**params)

    # 3. 返回包含 tool_calls 的响应
    return response.choices[0].message
```

### 工具 Schema 格式

```python
# ToolCollection.to_params() 生成
{
    "type": "function",
    "function": {
        "name": "web_search",
        "description": "Search the web for real-time information...",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "(required) The search query..."},
                "num_results": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        }
    }
}
```

---

## 六、面试回答要点总结

### 1. 整体设计思路

- **分层架构**: Agent → ToolCollection → BaseTool → 具体工具
- **标准化接口**: 所有工具继承 BaseTool，实现 `execute` 方法
- **与大模型集成**: 通过 OpenAI Function Calling 格式交互

### 2. Web 搜索实现要点

- **多引擎 fallback**: 失败时自动切换搜索引擎
- **重试机制**: 使用指数退避策略
- **可选内容抓取**: 获取搜索结果详情

### 3. 网页抓取实现要点

- **无头浏览器**: 使用 Chromium 处理 JS 渲染
- **Markdown 输出**: AI 友好的内容格式
- **缓存机制**: 提升性能

### 4. 文件操作实现要点

- **Protocol 接口**: 定义统一的文件操作抽象
- **多环境支持**: 本地文件和 Sandbox 文件操作分离
- **精确编辑**: 支持字符串级别的替换

### 5. 扩展性

- 继承 BaseTool 并实现 `execute` 方法即可添加新工具
- ToolCollection 自动管理工具注册和调用
