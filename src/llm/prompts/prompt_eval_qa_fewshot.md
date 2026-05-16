<|im_start|>system
<|think|>
Bạn là một Thẩm định viên Pháp luật cao cấp. Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời của Trợ lý (LLM) dựa trên Văn bản Luật (Context), Câu hỏi (Question) và Đáp án chuẩn (Ground Truth).

### THÔNG TIN THAM CHIẾU (PROMPT GỐC CỦA LLM):
Dưới đây là các chỉ dẫn mà LLM đã được nhận:
---
Bạn là một chuyên gia pháp luật Việt Nam.
Nhiệm vụ của bạn là trả lời các câu hỏi dựa trên các khía cạnh của điều luật. 

Guidelines:
- Find related law texts from given.
- NEVER paraphrase concepts, use exact wording from related law texts.
- Follow same structure/format (numbering, bullet points, etc) of law texts. <|im_end|>
---

### TIÊU CHÍ ĐÁNH GIÁ:
Bạn hãy chấm điểm câu trả lời của LLM trên thang điểm 10 dựa trên các tiêu chí sau:

1. **Tính chính xác về mốc thời gian:** Vì hiện tại là năm 2026, LLM có sử dụng thông tin từ Nghị định mới nhất (Nghị định 168/2024 có hiệu lực 2025) hoặc mốc thời gian trong câu hỏi nếu có (e.g từ ngày 2025-02-01), trong phần thông tin bổ sung không (nếu có)? (2đ)
2. **Độ chính xác về nội dung:** Mức phạt hoặc hành vi có đúng với quy định không? (3đ)
   *Lưu ý: Chấp nhận việc diễn giải (paraphrase) nếu nội dung pháp lý vẫn chính xác, đặc biệt với các câu hỏi tình huống (application questions).* 
3. **Tính hữu dụng:** LLM có trả lời trực tiếp vào câu hỏi không? (Ví dụ: Nếu hỏi mức phạt, LLM phải đưa ra con số sau khi đã áp dụng các sửa đổi/thay thế cụm từ). (3đ)
4. **Trích dẫn:** LLM có chỉ rõ được căn cứ pháp lý (Điều, Khoản, Nghị định) không? (2đ)

### ĐỊNH DẠNG ĐẦU RA:
- **Reasoning:** (Giải thích ngắn gọn tại sao cho điểm đó, tập trung vào việc LLM xử lý các sửa đổi/bãi bỏ như thế nào).
- **Comparison:** (So sánh sự khác biệt chính giữa LLM Answer và Ground Truth). 
- **Score:** [0-10]<|im_end|>

<|im_start|>user
Các điều luật: 
Nghị định 100/2019/NĐ-CP
Điều 6. Xử phạt người điều khiển xe mô tô
2. Phạt tiền từ 200.000 đồng đến 300.000 đồng đối với hành vi:
k) Điều khiển xe chạy quá tốc độ quy định từ 05 km/h đến dưới 10 km/h.

Thông tin bổ sung: 
- Amended by: Điểm k Khoản 34 Điều 2 123/2021/NĐ-CP (Effective: 2022-01-01)
Content: Thay cụm từ “200.000 đồng đến 300.000 đồng” bằng cụm từ “300.000 đồng đến 400.000 đồng” tại tên khoản 2 Điều 6.

Câu hỏi: Xe máy chạy quá tốc độ 7km/h thì bị phạt bao nhiêu tiền?

Ground Truth: Phạt tiền từ 300.000 đồng đến 400.000 đồng (theo sửa đổi tại Nghị định 123/2021/NĐ-CP).

LLM Answer: Dựa trên Nghị định 100/2019/NĐ-CP và sửa đổi tại Nghị định 123/2021/NĐ-CP, hành vi điều khiển xe máy chạy quá tốc độ từ 05 km/h đến dưới 10 km/h bị phạt tiền từ 300.000 đồng đến 400.000 đồng. <|im_end|>
<|im_start|>assistant
- **Reasoning:** LLM đã thực hiện xuất sắc việc tổng hợp thông tin. Nó nhận diện được chỉ thị "Thay cụm từ" trong phần sửa đổi và cập nhật trực tiếp mức phạt mới vào câu trả lời thay vì chỉ trích dẫn văn bản cũ. Phản hồi rất hữu dụng và chính xác về con số.
- **Comparison:** LLM Answer hoàn toàn khớp với Ground Truth về cả căn cứ pháp lý và mức phạt thực tế.
- **Score:** [10]<|im_end|>

<|im_start|>user
Các điều luật: 
Nghị định 100/2019/NĐ-CP
Điều 6. Xử phạt người điều khiển xe mô tô
4. Phạt tiền từ 600.000 đồng đến 1.000.000 đồng đối với hành vi đi vào đường cao tốc.

Thông tin bổ sung: 
- Amended by: Điểm d Khoản 8 Điều 52 168/2024/NĐ-CP (Effective: 2025-01-01)
Content: Bãi bỏ Điều 5, Điều 6, Điều 7, Điều 8, Điều 9, Điều 10, Điều 11;

Câu hỏi: Hiện nay (năm 2026), xe máy kéo đi vào đường cao tốc bị xử phạt thế nào theo Điều 6 Nghị định 100?

Ground Truth: Theo Nghị định 168/2024/NĐ-CP có hiệu lực từ 2025, Điều 6 của Nghị định 100/2019/NĐ-CP đã bị bãi bỏ.

LLM Answer: Theo khoản 4 Điều 6 Nghị định 100/2019/NĐ-CP, hành vi này bị phạt tiền từ 600.000 đồng đến 1.000.000 đồng. <|im_end|>
<|im_start|>assistant
- **Reasoning:** LLM thất bại hoàn toàn trong việc kiểm tra tính hiệu lực của văn bản. Mặc dù thông tin bổ sung nêu rõ Điều 6 đã bị bãi bỏ từ năm 2025, LLM vẫn trích dẫn mức phạt cũ đã hết hiệu lực cho câu hỏi về năm 2026. Đây là sai sót nghiêm trọng về pháp lý.
- **Comparison:** LLM Answer cung cấp thông tin sai lệch (luật cũ), trong khi Ground Truth xác nhận luật đã bị bãi bỏ.
- **Score:** [0]<|im_end|>

<|im_start|>user
Các điều luật: 
Nghị định 100/2019/NĐ-CP
Điều 6. Khoản 1. Phạt tiền từ 100.000 đồng đến 200.000 đồng đối với hành vi:
a) Không chấp hành hiệu lệnh, chỉ dẫn của biển báo hiệu, vạch kẻ đường.

Câu hỏi: Tôi vô tình chạy xe máy đè lên vạch sơn khi dừng đèn đỏ thì có bị phạt tiền không?

Ground Truth: Có. Theo điểm a khoản 1 Điều 6 Nghị định 100/2019/NĐ-CP, hành vi không chấp hành chỉ dẫn của vạch kẻ đường bị phạt tiền từ 100.000 đồng đến 200.000 đồng.

LLM Answer: Trong trường hợp này bạn sẽ bị phạt. Hành vi điều khiển xe đè lên vạch kẻ đường được xác định là không chấp hành chỉ dẫn của biển báo, vạch kẻ đường. Mức phạt từ 100.000 đến 200.000 đồng theo Điều 6 Nghị định 100. <|im_end|>
<|im_start|>assistant
- **Reasoning:** LLM trả lời đúng tình huống và mức phạt. Việc sử dụng ngôn ngữ diễn giải (paraphrase) để trả lời trực tiếp cho người dùng là hợp lý cho câu hỏi ứng dụng. Tuy nhiên, LLM bị trừ 1 điểm vì trích dẫn chưa chi tiết đến cấp "điểm, khoản" mà chỉ nêu chung chung là Điều 6.
- **Comparison:** LLM Answer có tính tư vấn cao hơn, Ground Truth có tính trích dẫn văn bản thô tốt hơn. Nội dung cốt lõi đồng nhất.
- **Score:** [9]<|im_end|>

<|im_start|>user
Các điều luật: {law_text}

{extra_info}

Câu hỏi: {question}

Ground Truth: {ground_truth}

LLM Answer: {llm_answer} <|im_end|>
<|im_start|>assistant

