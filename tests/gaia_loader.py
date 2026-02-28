import os
import pandas as pd
import json

# 设置路径
GAIA_CACHE_PATH = "/Users/xujunjie/.cache/huggingface/hub/datasets--gaia-benchmark--GAIA/snapshots/682dd723ee1e1697e00360edccf2366dc8418dd9"

def prepare_gaia_for_langsmith(level="level1"):
    print(f"🔍 正在尝试读取 GAIA {level} 数据...")

    # 构造路径
    parquet_path = os.path.join(GAIA_CACHE_PATH, f"2023/validation/metadata.{level}.parquet")

    if not os.path.exists(parquet_path):
        print(f"❌ 错误：在路径下找不到文件: {parquet_path}")
        return []

    # 读取数据
    df = pd.read_parquet(parquet_path)
    print(f"✅ 成功读取 Parquet 文件，共有 {len(df)} 条原始数据。")

    examples = []
    for i, row in df.iterrows():
        raw_file_path = row.get("file_path", "")
        full_file_path = ""

        # 处理文件路径
        if raw_file_path and str(raw_file_path).strip() != "":
            full_file_path = os.path.join(GAIA_CACHE_PATH, raw_file_path)

        # 构造 Example
        example = {
            "inputs": {
                "query": row["Question"],
                "file_path": full_file_path,
                "file_name": row["file_name"],
                "level": row["Level"]
            },
            "outputs": {
                "reference": row["Final answer"]
            }
        }
        examples.append(example)

        # 只打印前 2 条数据作为样例查看
        if i < 2:
            print(f"\n--- 样例数据 {i+1} ---")
            print(f"❓ 问题: {row['Question'][:50]}...")
            print(f"📂 附件文件名: {row['file_name'] if row['file_name'] else '无'}")
            print(f"📍 附件绝对路径: {full_file_path if full_file_path else '无'}")
            print(f"🎯 标准答案: {row['Final answer']}")

            if full_file_path and not os.path.exists(full_file_path):
                print(f"⚠️ 警告：该附件在本地路径不存在，Agent 运行时可能会报错！")

    print(f"\n🚀 转换完成！总计生成 {len(examples)} 个测试用例。")
    return examples

from langsmith import Client

# ... 保持你之前的 prepare_gaia_for_langsmith 函数不变 ...

if __name__ == "__main__":
    # 1. 解析数据
    examples = prepare_gaia_for_langsmith("level1")

    if examples:
        # 2. 初始化 LangSmith 客户端
        ls_client = Client()

        DATASET_NAME = "GAIA_Validation_Level1"

        # 3. 创建数据集（如果已存在则跳过）
        if ls_client.has_dataset(dataset_name=DATASET_NAME):
            print(f"⚠️ 数据集 '{DATASET_NAME}' 已存在，正在跳过创建...")
        else:
            dataset = ls_client.create_dataset(
                dataset_name=DATASET_NAME,
                description="GAIA Benchmark Validation Set - Level 1"
            )
            # 4. 批量上传
            ls_client.create_examples(dataset_id=dataset.id, examples=examples)
            print(f"🎉 成功上传 {len(examples)} 道题目到 LangSmith 数据集: {DATASET_NAME}")
