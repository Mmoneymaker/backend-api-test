import os
from huggingface_hub import login
login(token="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")
from datasets import load_dataset
from huggingface_hub import snapshot_download

data_dir = snapshot_download(repo_id="gaia-benchmark/GAIA", repo_type="dataset")
print(f"数据实际存储在: {data_dir}")
dataset = load_dataset(data_dir, "2023_level1", split="validation")
for example in dataset:
    question = example["Question"]
    file_path = os.path.join(data_dir, example["file_path"])

