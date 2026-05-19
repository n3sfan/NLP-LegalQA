from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk

# 1. Download the specific split
ds_thangvip = load_dataset("QuangTran276/new_reasoning", split="train")

# 2. Save the dataset to a local directory
ds_thangvip.save_to_disk("./vietnamese_legal_qa_local")