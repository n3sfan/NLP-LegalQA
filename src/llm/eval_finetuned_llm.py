"""
Finetuned LLM Evaluation Script (Offline Mode)

Evaluates the generated answers against ground truth using traditional metrics (BLEU, ROUGE, METEOR)
and LLM-as-a-judge scoring via OpenRouter (google/gemini-2.5-flash).
Generation is done locally using Unsloth and the finetuned model.
This script runs in offline mode, loading context and question details from a pre-built JSONL payload file.

Usage:
    # 1. Generate answers (offline from payload)
    uv run python scripts/eval_finetuned_llm.py \
        --step-generate \
        --results eval_results/QA_Part2/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/row_results_full_pipeline.csv \
        --payload-dir offline_payloads/ \
        --output eval_results/QA_Part2/eval_finetuned_results.csv \
        --model-dir models/gemma-4-E4B-legal-qa-lora
        
    # 2. Run judge (offline from payload)
    uv run python scripts/eval_finetuned_llm.py \
        --step-judge \
        --results eval_results/QA_Part2/eval_results_pipeline_all_ablations_geminiflash_rerank_top_30/row_results_full_pipeline.csv \
        --payload-dir offline_payloads/ \
        --output eval_results/QA_Part2/eval_finetuned_results.csv
"""

import argparse
import json
import os
import sys
import time
import ssl
from pathlib import Path
from datetime import datetime

import pandas as pd
from dotenv import load_dotenv

import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk.tokenize import word_tokenize
from rouge_score import rouge_scorer
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Ensure src is in path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from llm.eval_qa_utils import get_payload_path

_QA_SYSTEM_PROMPT = """
Bạn là một chuyên gia pháp luật Việt Nam. Hãy trả lời câu hỏi pháp lý với lý luận chi tiết và có cấu trúc.
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên các văn bản pháp luật được cung cấp.

Nguyên tắc bắt buộc:
1. TRUNG THÀNH TUYỆT ĐỐI VỚI NGỮ CẢNH: CHỈ dựa vào phần "[Văn bản pháp luật]" được cung cấp. Tuyệt đối không sử dụng kiến thức có sẵn của bạn để tự suy diễn hay trả lời.
2. XỬ LÝ DỮ LIỆU THIẾU: Nếu các văn bản pháp luật được cung cấp HOÀN TOÀN KHÔNG chứa quy định liên quan đến hành vi mà người dùng hỏi, Bạn phải đối chiếu hành vi người dùng hỏi với văn bản luật dựa trên BẢN CHẤT NGỮ NGHĨA, không chỉ khớp từ khóa (Ví dụ: "vượt đèn đỏ" tương đương "không chấp hành hiệu lệnh của đèn tín hiệu"). Chỉ khi HOÀN TOÀN KHÔNG có nội dung nào liên quan về mặt ngữ nghĩa, BẮT BUỘC trả lời: "Dựa trên dữ liệu pháp luật hiện tại, tôi chưa tìm thấy đủ thông tin để trả lời chính xác câu hỏi này." LƯU Ý: Nếu văn bản CÓ chứa các quy định liên quan (dù người dùng không nêu rõ mức độ cụ thể), hãy liệt kê TẤT CẢ các mức phạt/trường hợp có trong ngữ cảnh, phân theo mức vi phạm (km/h, nồng độ cồn, v.v.).
3. CHÍNH XÁC THUẬT NGỮ: Giữ nguyên thuật ngữ pháp lý, các mốc định lượng (độ tuổi, nồng độ cồn, km/h) và mức phạt tiền/tù giam như trong văn bản.
4. XỬ LÝ VĂN BẢN CHỒNG CHÉO: Mỗi đoạn văn bản sẽ bắt đầu bằng header dạng [Văn bản: xxx — Hiệu lực: yyyy-mm-dd]. Sử dụng header này để định nguồn văn bản và ngày hiệu lực. Ưu tiên văn bản có ngày hiệu lực gần nhất VÀ đã có hiệu lực tại thời điểm [Ngày hiện tại]. Văn bản chưa có hiệu lực (ngày hiệu lực > ngày hiện tại) thì ghi chú rõ.
5. VĂN PHONG: Trả lời với thái độ chuyên nghiệp, khách quan, mang tính tư vấn pháp lý.
6. ĐÚNG ĐỐI TƯỢNG VÀ TỪ ĐỒNG NGHĨA: Chỉ trả lời mức phạt cho phương tiện người dùng hỏi. Bạn PHẢI tự động liên kết các từ gọi thông thường với thuật ngữ pháp lý tương ứng: "xe máy" = xe mô tô/xe gắn máy; "xe hơi" = xe ô tô. Nếu luật quy định chung cho nhóm lớn (ví dụ: "phương tiện giao thông cơ giới đường bộ") mà phương tiện người dùng hỏi thuộc nhóm đó, bạn vẫn phải sử dụng điều khoản đó để trả lời. Không liệt kê lan man các loại phương tiện khác. Lưu ý: Xe chuyên dụng khác xe mô tô/ xe gắn máy.
7. QUY ĐỊNH ĐÃ BỊ BÃI BỎ: Nếu một điều khoản có ghi chú [ĐÃ BỊ BÃI BỎ] hoặc [ĐÃ BỊ THAY THẾ], KHÔNG được trích dẫn điều khoản đó. Thay vào đó, sử dụng điều khoản thay thế (nếu có trong ngữ cảnh).
8. TRẢ LỜI ĐÚNG DẠNG CÂU HỎI VÀ NGỮ CẢNH TÌNH HUỐNG:
   - Nếu câu hỏi mô tả tình huống cụ thể có nhân vật (VD: "Anh A vượt đèn đỏ...", "Chị B uống rượu lái xe..."), phải sử dụng tên nhân vật đó trong câu trả lời (VD: "Anh A sẽ bị phạt..."), KHÔNG được bỏ qua ngữ cảnh để trả lời chung chung.
   - Luôn bám sát trọng tâm câu hỏi của người dùng. Đọc kỹ câu hỏi trước khi trả lời, người dùng có thể hỏi câu hỏi chứa nhiều hành vi vi phạm khác nhau, yêu cầu phải trả lời tất cả các ý liên quan đến vi phạm giao thông.
9. Một số lưu ý về thứ tự các điểm: Các điểm cần được trình bày theo đúng thứ tự như sau: a, b, c, d, đ, e,... (tức là đ trước e). Khi trích dẫn, không cần liệt kê các điểm nếu như tất cả các điểm được bao gồm.
10. Nên sử dụng các câu gốc từ văn bản pháp luật, hạn chế viết lại hoặc gộp các phần tử. Mỗi phần tử văn bản (Điều, Khoản, Điểm) liệt kê trên một dòng riêng.
Cấu trúc câu trả lời chuẩn:
- Căn cứ pháp lý: BẮT BUỘC trích dẫn đầy đủ và chính xác Điểm, Khoản, Điều, và TÊN VĂN BẢN (số hiệu Nghị định/Luật) chứa điều khoản đó. Tuyệt đối không được viết trích dẫn mà thiếu tên văn bản (VD ĐÚNG: "Căn cứ theo Điểm a, Khoản 3, Điều 6 Nghị định 100/2019/NĐ-CP"; VD SAI: "Căn cứ theo Điểm a, Khoản 3, Điều 6"). Nếu có nhiều văn bản, phải ghi rõ văn bản mới nhất.
- Kết luận trực tiếp: Trả lời thẳng vào trọng tâm (Có bị phạt không? Mức phạt khoảng bao nhiêu?).
- Chi tiết chế tài (nếu có): Mức phạt tiền, phạt tù (nếu có).
- Hình phạt bổ sung (nếu có): Tước giấy phép lái xe (bao nhiêu tháng), tạm giữ phương tiện (bao nhiêu ngày).
Nếu nhiều văn bản cùng quy định một hành vi:
1. Chọn văn bản còn hiệu lực tại thời điểm hiện tại.
2. Nếu có nhiều bản cùng hiệu lực, chọn bản có ngày hiệu lực mới hơn.
3. Nếu là văn bản sửa đổi/hợp nhất, ưu tiên điều khoản đã được cập nhật.
"""

_QA_USER_PROMPT = """[Ngày hiện tại]: {current_date}

[Văn bản pháp luật]:
{context}

[Câu hỏi của người dùng]:
{query}
{rewritten_section}"""

_JUDGE_SYSTEM_PROMPT = """Bạn là một giám khảo đánh giá câu trả lời của AI cho câu hỏi của người dùng trong lĩnh vực Pháp luật Giao thông Đường bộ Việt Nam."""

_JUDGE_USER_PROMPT = """[Câu hỏi]: {query}
[Các điều luật đã được trích xuất và cung cấp cho AI trả lời]:
{context}

[Câu trả lời ground truth]: {ground_truth}
[Câu trả lời của AI]: {generated_answer}

Hãy đánh giá câu trả lời của AI dựa trên các tiêu chí nghiêm ngặt sau, so sánh đối chiếu với câu trả lời ground truth:

1. Chính xác pháp lý (2 điểm): AI có trả lời đúng bản chất pháp lý không? Các mốc định lượng (độ tuổi, nồng độ cồn, tốc độ,...) và mức phạt (hành chính, hình sự,...) có khớp với câu trả lời ground truth không?
2. Trích dẫn chính xác (2 điểm): AI có trích dẫn đúng các điều luật như trong câu trả lời ground truth hay không? Có nêu rõ Điều, Khoản, Điểm, thuộc văn bản nào hay không?
3. Tính đầy đủ (2 điểm): AI có liệt kê đầy đủ các hình phạt chính và hình phạt bổ sung (tước quyền sử dụng giấy phép lái xe, tạm giữ phương tiện,...) như trong ground truth hay không?
4. Không bịa đặt (2 điểm): AI có bịa ra nội dung điều luật không?
5. Cấu trúc & Xử lý tình huống (2 điểm): AI có trả lời trực tiếp vào trọng tâm câu hỏi của người dùng không (ví dụ câu hỏi là dạng Có/ Không thì phải trả lời Có/ Không trước rồi mới giải thích)? Nếu câu hỏi có nhân vật cụ thể (Anh A, Chị B), AI có xưng hô đúng ngữ cảnh không hay chỉ trả lời chung chung nội dung các điều luật?

Dựa trên các tiêu chí trên, hãy chấm điểm từng tiêu chí theo thang điểm đã quy định bên trên (có thể chấm điểm lẻ đến 0.5, vị dụ: 0, 0.5, 1, 1.5, 2).

Trả về kết quả dưới định dạng JSON hợp lệ (không chứa markdown, không giải thích thêm). LƯU Ý: Không sử dụng dấu xuống dòng (newline) bên trong chuỗi JSON (đặc biệt là phần "reasoning"), hãy viết trên một dòng:
{{
  "reasoning": "<phân tích chi tiết từng tiêu chí và giải thích lý do chấm điểm>",
  "scores": {{
    "legal_accuracy": <điểm>,
    "correct_citation": <điểm>,
    "completeness": <điểm>,
    "hallucination_citation": <điểm>,
    "structure": <điểm>
  }}
}}
"""

load_dotenv()

# Attempt to download required NLTK resources
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")

try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab")

try:
    from nltk.translate.meteor_score import meteor_score
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
except ImportError:
    pass

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def calculate_metrics(ref: str, hyp: str) -> dict:
    """Calculate BLEU, ROUGE-L, and METEOR scores."""
    if not ref or not hyp:
        return {"bleu": 0.0, "rougeL": 0.0, "meteor": 0.0}

    ref_tokens = word_tokenize(ref.lower())
    hyp_tokens = word_tokenize(hyp.lower())
    
    # BLEU
    smoothie = SmoothingFunction().method1
    bleu = sentence_bleu([ref_tokens], hyp_tokens, smoothing_function=smoothie)
    
    # ROUGE
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = scorer.score(ref, hyp)
    rouge_l = rouge_scores['rougeL'].fmeasure
    
    # METEOR
    try:
        from nltk.translate.meteor_score import meteor_score
        meteor = meteor_score([ref_tokens], hyp_tokens)
    except Exception:
        meteor = 0.0

    return {
        "bleu": round(bleu, 4),
        "rougeL": round(rouge_l, 4),
        "meteor": round(meteor, 4),
    }

def clean_json_response(text: str) -> str:
    """Clean markdown formatting from LLM JSON output."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()

# ─────────────────────────────────────────────────────────────────────────────
# Main Evaluation Logic
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Finetuned LLM Answers (Offline Mode)")
    parser.add_argument("--results", default=None, help="Path to retrieval results CSV (used to find payload if --payload-file is not specified)")
    parser.add_argument("--payload-file", default=None, help="Direct path to the offline JSONL payload file")
    parser.add_argument("--ground-truth", default=None, help="Optional path to QA dataset CSV (ground truth, fallback if needed)")
    parser.add_argument("--payload-dir", default="offline_payloads/", help="Directory where offline payloads are stored")
    parser.add_argument("--output", default="eval_finetuned_results.csv", help="Output CSV path")
    
    parser.add_argument("--step-generate", action="store_true", help="Run generation step with Unsloth")
    parser.add_argument("--step-judge", action="store_true", help="Run LLM-as-a-judge scoring step")
    parser.add_argument("--model-dir", type=str, default="models/gemma-4-E4B-legal-qa-lora", help="Path to the saved LoRA adapters.")
    
    parser.add_argument("--sleep", type=float, default=2.0, help="Seconds to sleep between LLM calls")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of rows to evaluate (for testing)")
    return parser.parse_args()


def run_generate(args, payloads):
    import unsloth # Must be imported before other libraries like transformers
    from unsloth import FastModel
    from unsloth.chat_templates import get_chat_template

    gen_out_path = Path(args.output).with_suffix(".generated.csv")
    gen_out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    records = []
    
    if gen_out_path.exists():
        try:
            existing_df = pd.read_csv(gen_out_path)
            processed_ids = set(str(qid) for qid in existing_df["id"].tolist())
            records = existing_df.to_dict("records")
            print(f"Resuming generation from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"Could not read existing generation output: {e}")

    print(f"Loading model and LoRA from {args.model_dir}...")
    try:
        model, tokenizer = FastModel.from_pretrained(
            model_name=args.model_dir,
            max_seq_length=32000,
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

    current_date = datetime.now().strftime("%Y-%m-%d")

    for idx, payload in enumerate(payloads, 1):
        row_id = str(payload["id"])
        if row_id in processed_ids:
            continue
            
        question = payload.get("question", "")
        ground_truth = str(payload.get("expert_answer", ""))
        context = payload.get("law_text", "")
        
        if not question or pd.isna(question):
            print(f"  Row {row_id}: skipped (no question)")
            continue
            
        print(f"\n[Generate] {idx}/{len(payloads)} QID {row_id} Q: {question[:80]}...")
            
        # Format the user content
        user_content = _QA_SYSTEM_PROMPT + "\n\n" + _QA_USER_PROMPT.format(
            current_date=current_date,
            context=context,
            query=question,
            rewritten_section=""
        )
        
        messages = [{
            "role": "user",
            "content": [{
                "type": "text",
                "text": user_content
            }]
        }]

        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=False,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        ).to("cuda")

        t0 = time.time()
        outputs = model.generate(
            **inputs,
            max_new_tokens=10000,
            use_cache=True,
            temperature=0.7, 
            top_p=0.95, 
            top_k=64
        )
        gen_time = time.time() - t0

        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        generated_answer = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

        # Sanitize surrogates
        generated_answer = generated_answer.encode("utf-8", errors="replace").decode("utf-8")

        # Save row results mimicking eval_qa_offline style
        record = {
            "id": row_id,
            "question": question,
            "model": args.model_dir,
            "generated_answer": generated_answer,
            "expert_answer": ground_truth,
            "latency_ms": round(gen_time * 1000, 1),
            "law_text_preview": context[:200] + "..." if context else ""
        }
        
        records.append(record)
        processed_ids.add(row_id)
        
        # Save incrementally
        write_header = not gen_out_path.exists()
        pd.DataFrame([record]).to_csv(gen_out_path, mode='a', header=write_header, index=False)
        
    print(f"\nGeneration complete. {len(processed_ids)} rows processed. Results saved to {gen_out_path}")


def run_judge(args, payloads):
    gen_out_path = Path(args.output).with_suffix(".generated.csv")
    if not gen_out_path.exists():
        print(f"Error: Generated results not found at {gen_out_path}. Run --step-generate first.")
        sys.exit(1)
        
    eval_df = pd.read_csv(gen_out_path)
    
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    processed_ids = set()
    records = []
    
    if out_path.exists():
        try:
            existing_df = pd.read_csv(out_path)
            processed_ids = set(str(qid) for qid in existing_df["id"].tolist())
            records = existing_df.to_dict("records")
            print(f"Resuming judge from {len(processed_ids)} already processed questions.")
        except Exception as e:
            print(f"Could not read existing judge output: {e}")

    # Build context lookup map from payloads
    context_map = {str(p["id"]): p.get("law_text", "") for p in payloads}

    # Initialize components
    print("\nInitializing components for judge...")
    
    judge_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model="google/gemini-2.5-flash",
        temperature=0.0,
        max_tokens=1024,
    )
    judge_chain = (
        ChatPromptTemplate.from_messages([
            ("system", _JUDGE_SYSTEM_PROMPT),
            ("human", _JUDGE_USER_PROMPT),
        ])
        | judge_llm
        | StrOutputParser()
    )

    print("Starting judge evaluation...")
    skipped = 0

    for idx, row in eval_df.iterrows():
        row_id = str(row["id"])
        if row_id in processed_ids:
            continue
            
        question = row.get("question", "")
        ground_truth = str(row.get("expert_answer", ""))
        generated_answer = str(row.get("generated_answer", ""))
        latency_ms = row.get("latency_ms", 0.0)
        
        if not question or pd.isna(question):
            skipped += 1
            continue
            
        if not ground_truth or pd.isna(ground_truth) or ground_truth == "nan":
            print(f"  Row {row_id}: skipped (no ground truth answer)")
            skipped += 1
            continue
            
        print(f"\n[Judge] {idx+1}/{len(eval_df)} QID {row_id} Q: {question[:80]}...")
        
        # Get context from pre-loaded payload mapping
        context = context_map.get(row_id, "")
            
        # 2. Traditional Metrics
        metrics = calculate_metrics(ground_truth, generated_answer)
        
        # 3. LLM-as-a-judge Scoring
        t1 = time.time()
        judge_result = {}
        judge_output = ""
        retries = 3
        while retries > 0:
            try:
                judge_output = judge_chain.invoke({
                    "query": question,
                    "context": context,
                    "ground_truth": ground_truth,
                    "generated_answer": generated_answer,
                })
                judge_output = clean_json_response(judge_output)
                judge_result = json.loads(judge_output)
                
                # Verify structure
                _ = judge_result["scores"]["legal_accuracy"]
                break
            except Exception as e:
                print(f"  [WARNING] Judge failed: {e}.")
                print(f"  [DEBUG] Raw output: {judge_output}")
                retries -= 1
                time.sleep(args.sleep * 2)
        
        judge_time = time.time() - t1
        
        if not judge_result:
            print("  [ERROR] Judge failed permanently for this row.")
            judge_result = {
                "reasoning": "Failed to parse judge output.",
                "scores": {"legal_accuracy": 0, "correct_citation": 0, "completeness": 0, "hallucination_citation": 0, "structure": 0}
            }
            
        scores = judge_result.get("scores", {})
        overall_score = (
            scores.get("legal_accuracy", 0) +
            scores.get("correct_citation", 0) +
            scores.get("completeness", 0) +
            scores.get("hallucination_citation", 0) +
            scores.get("structure", 0)
        )
        print(f"  BLEU: {metrics['bleu']:.4f} | ROUGE-L: {metrics['rougeL']:.4f} | Judge: {overall_score}/10")

        record = {
            "id": row_id,
            "question": question,
            "ground_truth": ground_truth,
            "generated_answer": generated_answer,
            "bleu": metrics["bleu"],
            "rougeL": metrics["rougeL"],
            "meteor": metrics["meteor"],
            "judge_legal_accuracy": scores.get("legal_accuracy", 0),
            "judge_correct_citation": scores.get("correct_citation", 0),
            "judge_completeness": scores.get("completeness", 0),
            "judge_hallucination_citation": scores.get("hallucination_citation", 0),
            "judge_structure": scores.get("structure", 0),
            "judge_overall_score": overall_score,
            "judge_reasoning": judge_result.get("reasoning", ""),
            "latency_ms": latency_ms,
            "judge_time": round(judge_time, 2),
        }
        
        records.append(record)
        processed_ids.add(row_id)
        
        # Save incrementally
        write_header = not out_path.exists()
        pd.DataFrame([record]).to_csv(out_path, mode='a', header=write_header, index=False)
        
        time.sleep(args.sleep)
        
    print(f"\nEvaluation complete. {len(processed_ids)} rows processed. Results saved to {out_path}")
    
    if records:
        # Save final sorted CSV
        pd.DataFrame(records).to_csv(out_path, index=False)
        
        # Calculate summary metrics
        df = pd.DataFrame(records)
        summary = {
            "avg_bleu": df["bleu"].mean(),
            "avg_rougeL": df["rougeL"].mean(),
            "avg_meteor": df["meteor"].mean(),
            "avg_judge_legal_accuracy": df["judge_legal_accuracy"].mean(),
            "avg_judge_correct_citation": df["judge_correct_citation"].mean(),
            "avg_judge_completeness": df["judge_completeness"].mean(),
            "avg_judge_hallucination_citation": df["judge_hallucination_citation"].mean(),
            "avg_judge_structure": df["judge_structure"].mean(),
            "avg_judge_overall": df["judge_overall_score"].mean(),
        }
        
        print("\n--- Summary ---")
        for k, v in summary.items():
            print(f"  {k}: {v:.4f}")


def main():
    args = parse_args()
    
    if not args.step_generate and not args.step_judge:
        print("Please specify either --step-generate or --step-judge (or both).")
        sys.exit(1)

    if args.payload_file:
        payload_path = Path(args.payload_file)
    else:
        if not args.results:
            print("Error: Either --results or --payload-file must be specified.")
            sys.exit(1)
        # Resolve payload path using results and payload directory
        payload_path = get_payload_path(args.results, args.payload_dir)
        
    if not payload_path.exists():
        print(f"Error: Payload file not found at {payload_path}.")
        print("Please run payload generation (eval_qa_online.py) first to build context data offline.")
        sys.exit(1)
        
    print(f"Loading offline payloads from: {payload_path}")
    payloads = []
    with open(payload_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                payloads.append(json.loads(line))
                
    if args.limit is not None:
        payloads = payloads[:args.limit]
        
    print(f"Loaded {len(payloads)} offline payloads.")
    
    if args.step_generate:
        print("\n=== Running Generation Step ===")
        run_generate(args, payloads)
        
    if args.step_judge:
        print("\n=== Running Judge Step ===")
        run_judge(args, payloads)

if __name__ == "__main__":
    main()
