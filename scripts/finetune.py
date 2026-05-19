import os
import pandas as pd
from datasets import Dataset
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template, train_on_responses_only
from trl import SFTTrainer, SFTConfig

def main():
    print("Loading and preprocessing dataset...")
    # 1. Load dataset
    df = pd.read_parquet('data/finetune/legal-qa.parquet')
    
    # Filter for traffic-related documents
    traffic_docs = [
        'Dự thảo Luật Bảo đảm trật tự, an toàn giao thông đường bộ lần 5',
        'Dự thảo Luật Giao thông đường bộ (sửa đổi) lần 1',
        'Dự thảo Luật Giao thông đường bộ (sửa đổi) lần 3',
        'Dự thảo Luật Giao thông đường bộ (sửa đổi) lần 4',
        'Dự thảo Luật Trật tự, an toàn giao thông đường bộ',
        'Dự thảo Luật Trật tự, an toàn giao thông đường bộ (Dự thảo 4)',
        'Dự thảo Luật Đường bộ',
        'Dự thảo Luật Đường bộ (T5/2024)',
        'Luật Giao thông đường bộ số 23/2008/QH12 của Quốc hội',
        'Luật Trật tự, an toàn giao thông đường bộ của Quốc hội, số 36/2024/QH15',
        'Luật của Quốc hội số 26/2001/QH10 về Giao thông đường bộ',
        'Luật Đường bộ của Quốc hội, số 35/2024/QH15'
    ]
    if 'doc_name' in df.columns:
        df = df[df['doc_name'].isin(traffic_docs)]
        print(f"Filtered dataset to {len(df)} traffic-related documents.")
    else:
        print("Warning: 'doc_name' column not found, skipping filter.")

    # 2. Explode the QA pairs into conversations
    records = []
    for _, row in df.iterrows():
        context = row.get('article_content', '')
        qa_pairs = row.get('generated_qa_pairs', [])
        
        if hasattr(qa_pairs, 'tolist'):
            qa_pairs = qa_pairs.tolist()

        if isinstance(qa_pairs, (list, tuple)):
            for qa in qa_pairs:
                if isinstance(qa, dict) and 'question' in qa and 'answer' in qa:
                    records.append({
                        "conversations": [
                            {"role": "user", "content": f"Dựa vào ngữ cảnh pháp lý sau:\n{context}\n\nCâu hỏi: {qa['question']}"},
                            {"role": "assistant", "content": qa['answer']}
                        ]
                    })

    # Convert to HuggingFace Dataset
    hf_dataset = Dataset.from_pandas(pd.DataFrame(records))
    print(f"Total training examples: {len(hf_dataset)}")

    print("Loading Unsloth model...")
    # 3. Load Unsloth model
    model, tokenizer = FastModel.from_pretrained(
        model_name = "unsloth/gemma-4-E4B-it",
        max_seq_length = 8192,
        load_in_4bit = True,
        full_finetuning = False,
    )

    # 4. Apply LoRA parameters
    print("Applying LoRA parameters...")
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers = False,
        finetune_language_layers = True,
        finetune_attention_modules = True,
        finetune_mlp_modules = True,
        r = 16,
        lora_alpha = 16,
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )

    # 5. Apply Gemma 4 Chat Template
    print("Formatting dataset with Gemma 4 template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "gemma-4",
    )

    def formatting_prompts_func(examples):
        convos = examples["conversations"]
        texts = [
            tokenizer.apply_chat_template(
                convo,
                tokenize=False,
                add_generation_prompt=False
            ).removeprefix("<bos>")
            for convo in convos
        ]
        return {"text": texts}

    hf_dataset = hf_dataset.map(formatting_prompts_func, batched=True)

    # 6. Setup SFTTrainer
    print("Setting up Trainer...")
    trainer = SFTTrainer(
        model = model,
        processing_class = tokenizer,
        train_dataset = hf_dataset,
        eval_dataset = None,
        args = SFTConfig(
            dataset_text_field = "text",
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 4,
            warmup_steps = 5,
            # max_steps = 60, # Uncomment to run a quick test
            num_train_epochs = 1, # Full training run
            learning_rate = 2e-4,
            logging_steps = 1,
            save_strategy = "steps",
            save_steps = 100,
            save_total_limit = 2,
            optim = "adamw_8bit",
            weight_decay = 0.001,
            lr_scheduler_type = "linear",
            seed = 3407,
            output_dir = "outputs",
            report_to = "none", # Set to "wandb" if you use Weights and Biases
        ),
    )

    # 7. Train on responses only (ignores loss on user instructions)
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|turn>user\n",
        response_part = "<|turn>model\n",
    )

    # 8. Start training
    print("Starting training...")
    
    # Check if there's a checkpoint to resume from
    import os
    resume_from_checkpoint = True if os.path.exists("outputs") and any(f.startswith("checkpoint") for f in os.listdir("outputs")) else False
    if resume_from_checkpoint:
        print("Resuming from latest checkpoint found in outputs/...")
        
    trainer_stats = trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # 9. Save the finetuned model (LoRA adapters)
    output_model_dir = "models/gemma-4-E4B-legal-qa-lora"
    print(f"Training complete. Saving LoRA adapters to {output_model_dir}...")
    model.save_pretrained(output_model_dir)
    tokenizer.save_pretrained(output_model_dir)
    
    print("Done!")

if __name__ == "__main__":
    main()
