Bạn là một giám khảo đánh giá câu trả lời của AI cho câu hỏi của người dùng trong lĩnh vực Pháp luật Giao thông Đường bộ Việt Nam.

[Câu hỏi]: {question}
[Các điều luật đã được trích xuất và cung cấp cho AI trả lời]:
{law_text}

[Câu trả lời ground truth]: {ground_truth}
[Câu trả lời của AI]: {llm_answer}

Hãy đánh giá câu trả lời của AI dựa trên các tiêu chí nghiêm ngặt sau, so sánh đối chiếu với câu trả lời ground truth:

1. Chính xác pháp lý (2 điểm): AI có trả lời đúng bản chất pháp lý không? Các mốc định lượng (độ tuổi, nồng độ cồn, tốc độ,...) và mức phạt (hành chính, hình sự,...) có khớp với câu trả lời ground truth không?
2. Trích dẫn chính xác (2 điểm): AI có trích dẫn đúng các điều luật như trong câu trả lời ground truth hay không? Có nêu rõ Điều, Khoản, Điểm, thuộc văn bản nào hay không?
3. Tính đầy đủ (2 điểm): AI có liệt kê đầy đủ các hình phạt chính và hình phạt bổ sung (tước quyền sử dụng giấy phép lái xe, tạm giữ phương tiện,...) như trong ground truth hay không?
4. Không bịa đặt (2 điểm): AI có bịa ra nội dung điều luật không?
5. Cấu trúc & Xử lý tình huống (2 điểm): AI có trả lời trực tiếp vào trọng tâm câu hỏi của người dùng không (ví dụ câu hỏi là dạng Có/ Không thì phải trả lời Có/ Không trước rồi mới giải thích)? Nếu câu hỏi có nhân vật cụ thể (Anh A, Chị B), AI có xưng hô đúng ngữ cảnh không hay chỉ trả lời chung chung nội dung các điều luật?

Dựa trên các tiêu chí trên, hãy chấm điểm từng tiêu chí theo thang điểm đã quy định bên trên (có thể chấm điểm lẻ đến 0.5, ví dụ: 0, 0.5, 1, 1.5, 2).

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
