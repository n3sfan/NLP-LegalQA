(Model instruction: Short answer)                                                                                                
<constraints>
- NEVER do anything out of scope I said
- NEVER remove, change my comments, report, print message. Never use emojis in them                                               
- Coding Task: You must reuse whenever possible.                                                                                                  
- Coding: Keep comments short and to the point, no lengthy explanations.                                                                          
- python: conda activate ml-env                            
</constraints>

<instructions>
Write english in notebook
Clean-up: Only After finish writing code, report. MUST Do the following:
- Remove English in brackets after Vietnamese, e.g Kiểm định chi bình phương (chi-squared test) => Kiểm định chi bình phương 
- Remove task prompts (my prompts) in brackets, e.g 1.1 Thống kê mô tả (Relating W1 Knowledge) => 1.1 Thống kê mô tả
</instructions>

Task: Copy @Notebook/lab.ipynb to @Notebook/All.ipynb , follow format  

Task: follow requirements "2. Xác định mục tiêu và lựa chọn các trường dữ liệu" and "3. Lựa chọn biểu đồ thích hợp và giải thích lý do" to complete questions below. Main goal: 
1. Refine question to SMART criteria. don't write specifics for each criteria in SMART. Combine in a single question
2. Write code in @Notebook/lab.ipynb   and report in markdown cells.                                                                                    

# Questions - related fields csv:
Phân tích tương quan giữa tỷ trọng ngành dịch vụ với tỷ lệ nhập học đại học  của các quốc gia trong các năm:
Employment in services
Tertiary school enrollment

# Guidelines
- Viết tiếng Việt có dấu
- Questions/Answers must explain, complement the big goal: "Mục tiêu lớn: Giáo dục, nguồn nhân lực và sự tham gia của nữ giới trong trong giáo dục và nguồn nhân lực trong 1 số quốc gia (chưa chốt số lượng) trong vòng 25 năm (từ 2000 đến nay)"


- Copy style of Objective 4, 5, 6 (Preprocessing can be dropped if none in code).
- Must follow markdown headers (#, ##, ###, etc)

print("--- Pearson Correlation Coefficients ---")
overall_corr = developing_long[['TerValue', 'UnempValue']].corr().iloc[0, 1]
print(f"\nOverall (developing group): r = {overall_corr:.4f}")

instead of printing , plot lollipop charts