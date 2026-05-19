import argparse
import sys
from unsloth import FastModel
from unsloth.chat_templates import get_chat_template
from transformers import TextStreamer

def main():
    parser = argparse.ArgumentParser(description="Test inference on the fine-tuned Gemma 4 model.")
    parser.add_argument("--context", type=str, required=True, help="The legal context to ground the answer.")
    parser.add_argument("--question", type=str, required=True, help="The question to ask.")
    parser.add_argument("--model-dir", type=str, default="models/gemma-4-E4B-legal-qa-lora", help="Path to the saved LoRA adapters.")
    args = parser.parse_args()

    print(f"Loading model and LoRA from {args.model_dir}...")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_dir,
            max_seq_length=2048,
            load_in_4bit=True,
        )
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)

    print("Enabling native 2x faster inference...")
    FastModel.for_inference(model)

    print("Applying Gemma 4 Chat Template...")
    tokenizer = get_chat_template(
        tokenizer,
        chat_template="gemma-4",
    )

    # Format the input using the exact prompt structure from finetune.py
    user_content = f"Dựa vào ngữ cảnh pháp lý sau:\n{args.context}\n\nCâu hỏi: {args.question}"
    
    messages = [
        {"role": "user", "content": [{"type": "text", "text": user_content}]}
    ]

    print("\nPreparing generation inputs...")
    # Tokenize the messages. enable_thinking=False aligns with training.
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        enable_thinking=False,
        tokenize=True,
        return_dict=True,
        return_tensors="pt"
    ).to("cuda")

    text_streamer = TextStreamer(tokenizer, skip_prompt=True)

    print("\n--- Model Output ---")
    _ = model.generate(
        **inputs,
        streamer=text_streamer,
        max_new_tokens=1024,
        use_cache=True,
        temperature=1.0, 
        top_p=0.95, 
        top_k=64
    )
    print("\n--------------------")

if __name__ == "__main__":
    main()
