import os
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
from transformers import (
    TrainingArguments,
)
from unsloth import FastLanguageModel
from trl import SFTTrainer
import ast
from pathlib import Path

def load_template(filename: str) -> str:
    p = Path(filename)
    if not p.exists():
        p = Path("src/llm") / filename
    if not p.exists():
        p = Path(__file__).parent / filename
    if not p.exists():
        raise FileNotFoundError(f"Template not found: {filename}")
    return p.read_text(encoding="utf-8")

def preprocess_thangvip(dataset):
    new_data = []
    for example in dataset:
        qa_pairs = example['generated_qa_pairs']
        # If it's a string, parse it. If it's already a list, use it.
        if isinstance(qa_pairs, str):
            qa_pairs = ast.literal_eval(qa_pairs)
        
        law_text = example['article_content']
        for pair in qa_pairs:
            new_data.append({
                "law_text": law_text,
                "question": pair['question'],
                "answer": pair['answer']
            })

    if not new_data:
        raise ValueError("Preprocessing resulted in an empty dataset. Check the input data format.")
        
    return Dataset.from_list(new_data)

def preprocess_vlsp(dataset):
    new_data = []
    for example in dataset:
        new_data.append({
            "law_text": "",  # Syllogism dataset doesn't have separate law_text
            "question": example['question'],
            "answer": example['answer']
        })
    return Dataset.from_list(new_data)

class Config:
    model_id = "/kaggle/working/gemma-4-E2B-it"
    output_dir = "./gemma-legal-qa-qlora"
    epochs = 1
    batch_size = 2
    gradient_accumulation_steps = 1
    limit = None  # Set to a number for testing
    template = "./prompt_qa_0shot.md"

def main():
    # 1. Load Data
    print("Loading datasets...")
    # ds_thangvip = load_dataset("thangvip/vietnamese-legal-qa", split="train")
    ds_thangvip = load_from_disk("./vietnamese_legal_qa_local")

    if Config.limit:
        ds_thangvip = ds_thangvip.select(range(min(Config.limit, len(ds_thangvip))))

    print("Preprocessing datasets...")
    ds_thangvip_proc = preprocess_thangvip(ds_thangvip)
    
    full_dataset = concatenate_datasets([ds_thangvip_proc])
    full_dataset = full_dataset.shuffle(seed=42)

    # 2. Load Template
    template_str = load_template(Config.template)
    
    def formatting_func(example):
        # Format the template with law_text and question
        # Then append the assistant response
        prompt = template_str.format(
            law_text=example['law_text'],
            question=example['question']
        )
        text = f"{prompt}\n<|im_start|>assistant\n{example['answer']} <|im_end|>"
        return [text]

    # 3. Model & Tokenizer
    print(f"Loading model {Config.model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = Config.model_id,
        max_seq_length = 2048,
        dtype = None,
        load_in_4bit = True,
        trust_remote_code = True,
    )

    # 4. PEFT Config
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=full_dataset,
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=Config.batch_size,
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=Config.epochs,
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            fp16=True,
            logging_steps=10,
            output_dir=Config.output_dir,
            optim="paged_adamw_8bit",
            save_strategy="epoch",
        ),
        formatting_func=formatting_func,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {Config.output_dir}...")
    trainer.save_model(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)


if __name__ == "__main__":
    main()
