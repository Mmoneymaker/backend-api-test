# 浏览器无头模式配置说明

## 什么是无头模式 (Headless Mode)?

**无头模式**是指浏览器在**没有图形用户界面 (GUI)** 的情况下运行：

| 模式 | 特点 |
|------|------|
| **有头模式 (默认)** | 打开真实浏览器窗口，用户可以看到浏览器操作 |
| **无头模式** | 浏览器在后台运行，无窗口，不影响用户操作其他应用 |

- 浏览器进程仍然存在，只是**不显示窗口**
- 所有浏览器功能（JavaScript、网络请求等）都能正常工作
- 性能比有头模式更好（不需要渲染页面）

---

## 需要修改的文件

### 1. `app/tool/browser_use_tool.py`

**位置**: 第 144 行

**当前代码**:
```python
browser_config_kwargs = {"headless": False, "disable_security": True}
```

**问题**: 硬编码 `headless=False`，没有读取配置文件的值

**修改方案**: 改为从 `config.browser_config` 读取 `headless` 值（如果存在的话）

---

### 2. `app/config.py`

**位置**: 第 70 行

**当前代码**:
```python
class BrowserSettings(BaseModel):
    headless: bool = Field(False, description="Whether to run browser in headless mode")
```

**说明**: 配置类已定义，默认值是 `False`（有头模式）

**无需修改**: 如果需要无头模式，只需在配置文件中将 `headless` 设为 `True`

---

### 3. 配置文件 (`.env` 或新建配置)

需要在配置文件中添加 browser 相关配置：

```
# 方式 A: 在 .env 中添加
BROWSER_HEADLESS=true
```

或者

```yaml
# 方式 B: 如果有 config.yaml
browser:
  headless: true
  disable_security: true
```

---

## 修改优先级

| 优先级 | 修改内容 | 改动量 |
|--------|----------|--------|
| **P0 (最简)** | 只修改 `browser_use_tool.py` 第 144 行，改为 `True` | 1 行 |
| **P1 (推荐)** | 在 `.env` 中添加配置 + 修改 `browser_use_tool.py` 读取配置 | 3-5 行 |
| **P2 (完善)** | 通过配置文件支持完整的 BrowserSettings | 改动较大 |

---

## 推荐方案 (P1)

1. 修改 `app/tool/browser_use_tool.py` 第 144 行，改为从 config 读取
2. 在 `.env` 文件中添加 `BROWSER_HEADLESS=true`

这样可以实现：
- ✅ 默默在后台运行，不影响日常使用
- ✅ 可通过配置切换是否有头模式
- ✅ 改动最小
