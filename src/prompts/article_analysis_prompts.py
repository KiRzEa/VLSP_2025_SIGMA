CHECK_ARTICLE_RELEVANCY_PROMPT = """
Bạn là một trợ lý pháp lý giao thông. \
Dựa trên thông tin sau, hãy đánh giá xem điều luật dưới đây có liên quan đến câu hỏi trắc nghiệm, các lựa chọn hay các phân tích về ảnh hay không.

### Câu hỏi:
{question}

### Các lựa chọn trả lời:
{choices}

### Điều luật:
{article}

### Yêu cầu:
Trả lời dưới dạng JSON đúng định dạng sau, không viết thêm bất kỳ dòng nào khác ngoài JSON:

{{
  "reason": "<giải thích vì sao điều luật này có hoặc không liên quan đến câu hỏi>",
  "relevant": True hoặc False
}}

### Gợi ý:
- Nếu điều luật giúp loại trừ hoặc lựa chọn được đáp án đúng → relevant = True.
- Nếu điều luật không có thông tin nào giúp trả lời câu hỏi → relevant = False.
"""