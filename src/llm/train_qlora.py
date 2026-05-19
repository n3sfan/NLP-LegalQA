import os
if os.path.exists("/kaggle/working"):
    os.chdir("/kaggle/working")
import sys
# os.environ["HF_HUB_OFFLINE"] = "1"
# os.environ["TRANSFORMERS_OFFLINE"] = "1"

import torch
from datasets import load_dataset, concatenate_datasets, Dataset, load_from_disk
import gc
from transformers import (
    TrainingArguments,
    AutoTokenizer,
    EarlyStoppingCallback,
    TrainerCallback,
)
from unsloth import FastLanguageModel
from trl import SFTTrainer
import ast
from pathlib import Path

def load_template(filename: str) -> str:
    p = Path(filename)
    # if not p.exists():
    #     p = Path("src/llm") / filename
    # if not p.exists():
    #     p = Path(__file__).parent / filename
    # if not p.exists():
    #     raise FileNotFoundError(f"Template not found: {filename}")
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
    output_dir = "/kaggle/gemma-legal-qa-qlora"
    data_dir = "/kaggle/vietnamese_legal_qa_local"
    
    reasoning = True
    max_seq_length = 2048
    epochs = 1
    batch_size = 32
    gradient_accumulation_steps = 2
    limit = None  # Set to a number for testing
    template = "./prompts/prompt_qa_0shot_finetune.md"


tokenizer = None

def convert_gemma4_channels(messages):
    """Replace thought tags with Gemma 4 channel tokens in message content."""
    converted = []
    for msg in messages:
        new_msg = {"role": msg["role"]}
        if "content" in msg:
            content = msg["content"]
            # if "<think>" in content and "</think>" not in content:
            #     content = content + "</think>"
            content = content.replace("<think>", "<|channel>thought")
            content = content.replace("</think>", "<channel|>")
            new_msg["content"] = content
        converted.append(new_msg)
    return converted

class VRAMCleanupCallback(TrainerCallback):
    """Callback to clean up VRAM cache during train/eval transitions."""
    def on_step_end(self, args, state, control, **kwargs):
        # Clear VRAM when transitioning from Train -> Validate
        if control.should_evaluate:
            gc.collect()
            torch.cuda.empty_cache()
            # print("\n[VRAM Cleanup] CUDA cache cleared before starting validation (Train -> Validate)...")

    def on_evaluate(self, args, state, control, **kwargs):
        # Clear VRAM when transitioning from Validate -> Train
        gc.collect()
        torch.cuda.empty_cache()
        # print("\n[VRAM Cleanup] CUDA cache cleared after completing validation (Validate -> Train)...")


def format_dataset(batch):
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained(Config.model_id)
    
    formatted_texts = []
    for messages in batch["messages"]:
        if len(messages) > 0 and isinstance(messages[0], list):
            for msg_list in messages:
                msg_list = convert_gemma4_channels(msg_list)
                text = tokenizer.apply_chat_template(msg_list, tokenize=False, enable_thinking=Config.reasoning, add_generation_prompt=False)
                formatted_texts.append(text)
        else:
            msg_list = convert_gemma4_channels(messages)
            text = tokenizer.apply_chat_template(msg_list, tokenize=False, enable_thinking=Config.reasoning, add_generation_prompt=False)
            formatted_texts.append(text)
    return {"text": formatted_texts}

def main():
    # 1. Load Data
    print("Loading datasets...")
    # ds_thangvip = load_dataset("thangvip/vietnamese-legal-qa", split="train")
    ds_thangvip = load_from_disk(Config.data_dir)

    print("Filtering dataset...")
    def filter_system_content(example):
        messages = example.get("messages", [])
        if len(messages) > 0 and messages[0].get("role") == "system":
            return "Hãy trả lời câu hỏi pháp lý với lý luận chi tiết và có cấu trúc." in messages[0].get("content", "")
        return False
    ds_thangvip = ds_thangvip.filter(filter_system_content, num_proc=os.cpu_count() or 4)

    if Config.limit:
        ds_thangvip = ds_thangvip.select(range(min(Config.limit, len(ds_thangvip))))

    print("Preprocessing and tokenizing datasets in parallel...")
    full_dataset = ds_thangvip.shuffle(seed=42)
    full_dataset = full_dataset.map(
        format_dataset,
        batched=True,
        remove_columns=full_dataset.column_names,
        num_proc=os.cpu_count() or 4
    )

    print("Splitting dataset into train, validation, and test splits (8:1:1)...")
    # Split 80% train, 20% validation + test
    train_test_split1 = full_dataset.train_test_split(test_size=0.2, seed=42)
    train_dataset = train_test_split1["train"]
    temp_dataset = train_test_split1["test"]
    
    # Split the 20% temp dataset into 50% validation and 50% test (10% of total each)
    train_test_split2 = temp_dataset.train_test_split(test_size=0.5, seed=42)
    eval_dataset = train_test_split2["train"]
    test_dataset = train_test_split2["test"]
   
    print(f"Dataset sizes - Train: {len(train_dataset)}, Validation: {len(eval_dataset)}, Test: {len(test_dataset)}")
    
    # Save test dataset to disk for offline evaluation
    test_dataset_path = os.path.join(Config.output_dir, "test_dataset")
    print(f"Saving test dataset to {test_dataset_path}...")
    try:
        os.makedirs(Config.output_dir, exist_ok=True)
        test_dataset.save_to_disk(test_dataset_path)
    except Exception as e:
        print(f"Could not save test dataset: {e}")

    # 3. Model & Tokenizer
    print(f"Loading model {Config.model_id}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = Config.model_id,
        max_seq_length = Config.max_seq_length,
        dtype = None,
        load_in_4bit = True,     # MoE QLoRA not recommended, dense 31B is fine
        # load_in_16bit = True,     # bf16/16-bit LoRA
        trust_remote_code = True,
    )

    # 2. Load Template
    try:
        template_str = load_template(Config.template)
    except Exception:
        template_str = ""

    # 4. PEFT Config
    model = FastLanguageModel.get_peft_model(
        model,
        r = 32,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha = 64,
        lora_dropout = 0,
        bias = "none",
        use_cache=True,
        use_gradient_checkpointing = "unsloth",
        random_state = 3407,
        use_rslora = False,
        loftq_config = None,
    )

    # 5. Trainer
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        max_seq_length=Config.max_seq_length,
        args=TrainingArguments(
            per_device_train_batch_size=Config.batch_size,
            per_device_eval_batch_size=128,  # Safe and highly parallelized for RTX 6000 Blackwell without OOM
            eval_accumulation_steps=10,    # Periodically offloads logits to CPU; CRITICAL to prevent logit-gathering OOM
            dataloader_num_workers=os.cpu_count() or 4,  # Prevents CPU bottleneck during rapid evaluation
            gradient_accumulation_steps=Config.gradient_accumulation_steps,
            warmup_steps=5,
            num_train_epochs=Config.epochs,
            learning_rate=2e-5,
            lr_scheduler_type="linear",
            fp16=False,
            bf16=True,
            logging_steps=100,
            output_dir=Config.output_dir,
            optim="paged_adamw_8bit",
            
            save_strategy="steps",
            save_steps=100,
            save_total_limit=5,
            eval_strategy="steps",
            eval_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
        ),
        dataset_text_field="text",
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=3),
            VRAMCleanupCallback(),
        ],
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating best model on held-out test set...")
    try:
        test_metrics = trainer.evaluate(eval_dataset=test_dataset, metric_key_prefix="test")
        print(f"Test Set Evaluation Results: {test_metrics}")
    except Exception as e:
        print(f"Failed to evaluate on test set: {e}")

    print(f"Saving model to {Config.output_dir}...")
    trainer.save_model(Config.output_dir)
    tokenizer.save_pretrained(Config.output_dir)
    
    # model.save_pretrained_gguf(
    #     "/kaggle/working/gemma-4-E2B-it", 
    #     tokenizer, 
    #     quantization_method = "q8_0"
    # )
    return model, tokenizer

model, tokenizer = main()
