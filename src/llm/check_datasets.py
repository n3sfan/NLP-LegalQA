import os
HF_TOKEN =  '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'
os.environ['HF_TOKEN'] =HF_TOKEN

os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id="google/gemma-4-E2B-it",
    local_dir="./gemma-4-E2B-it",
    local_dir_use_symlinks=False,
)
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk

# 1. Download the specific split
ds_thangvip = load_dataset("thangvip/vietnamese-legal-qa", split="train")

# 2. Save the dataset to a local directory
ds_thangvip.save_to_disk("./vietnamese_legal_qa_local")