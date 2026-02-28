import pandas as pd
import os
import json  # 用于格式化嵌套字典/JSON字段

# ===================== 核心显示设置（增强版，适配parquet数据） =====================
# 1. 取消列宽限制，完整显示长文本/嵌套字段
pd.set_option('display.max_colwidth', None)
# 2. 显示所有列，不折叠（parquet数据列数可能较多）
pd.set_option('display.max_columns', None)
# 3. 取消终端行宽限制，避免列强制换行
pd.set_option('display.width', None)
# 4. 确保前N行完整显示，不截断
pd.set_option('display.max_rows', None)
# 5. 禁止DataFrame自动换行，保证列对齐
pd.set_option('display.expand_frame_repr', False)
# 6. 禁止科学计数法（如果有数字字段，避免显示成1e8这种形式）
pd.set_option('display.float_format', lambda x: f"{x}")

# ===================== 数据读取与处理 =====================
# GAIA数据集路径
GAIA_CACHE_PATH = "/Users/xujunjie/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/682dd723ee1e1697e00360edccf2366dc8418dd9"
parquet_path = os.path.join(GAIA_CACHE_PATH, f"2023/validation/metadata.level1.parquet")

# 检查文件是否存在（保留你的逻辑，补充提示）
if not os.path.exists(parquet_path):
    raise FileNotFoundError(
        f"GAIA 数据文件不存在: {parquet_path}\n"
        "请检查：1. 路径是否正确 2. GAIA数据集是否已下载 3. snapshot版本号是否匹配"
    )

# 读取parquet文件（parquet支持嵌套字段，这里确保完整读取）
df = pd.read_parquet(parquet_path, engine='pyarrow')  # 指定engine，提升兼容性

# 可选：格式化嵌套字典/JSON字段（GAIA数据的Annotator Metadata等字段是嵌套结构）
def format_nested_field(cell):
    """将嵌套字典/JSON字符串格式化为规整的缩进形式"""
    if isinstance(cell, (dict, list)):
        return json.dumps(cell, indent=2, ensure_ascii=False)
    if isinstance(cell, str):
        # 处理字符串形式的JSON/字典
        try:
            return json.dumps(json.loads(cell), indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, ValueError):
            pass  # 不是JSON字符串，直接返回
    return cell

# 遍历所有列，格式化嵌套字段（按需启用，注释掉则用原始格式）
# for col in df.columns:
#     df[col] = df[col].apply(format_nested_field)

# ===================== 打印前两行完整数据 =====================
print("=== GAIA数据集 level1 前两行完整数据 ===")
# 用to_string()强制完整输出，避免终端自动截断
print(df.head(2).to_string())

# （可选）如果想单独查看某一列的完整内容，比如Question列
# print("\n=== 单独查看Question列前两行 ===")
# print(df['Question'].head(2).to_string())
