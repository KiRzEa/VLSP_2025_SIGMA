CHECK_ARTICLE_RELEVANCY_PROMPT = """
Bạn là một trợ lý pháp lý giao thông. \
Dựa trên thông tin sau, hãy đánh giá xem điều luật dưới đây có liên quan đến câu hỏi trắc nghiệm, các lựa chọn hay các phân tích về ảnh hay không.

### Câu hỏi:
{question}

### Các lựa chọn trả lời:
{choices}

### Thông tin nhận diện từ biển báo giao thông:
{sign_interpretation}

### Ngữ cảnh trong hình ảnh:
{scene_text}

### Nội dung điều luật:
{article_text}

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

ARTICLE_FILTER_INFORMATION_PROMPT = """\
Bạn là một trợ lý pháp lý giao thông Việt Nam.

Dựa trên thông tin dưới đây, hãy đọc, phân tích và **tóm tắt** nội dung chính của điều luật, sao cho phù hợp với ngữ cảnh câu hỏi, lựa chọn trả lời và thông tin phân tích biển báo.

### Câu hỏi:
{question}

### Các lựa chọn trả lời:
{choices}

### Thông tin nhận diện từ biển báo giao thông:
{sign_interpretation}

### Ngữ cảnh trong hình ảnh:
{scene_text}

### Nội dung điều luật:
{article_text}

### Yêu cầu:
- Nếu điều luật không liên quan tới ngữ cảnh, vẫn tóm tắt nội dung chính của nó.
- Giữ nguyên ý chính của điều luật nhưng có thể rút gọn, bỏ các chi tiết không cần thiết.
- Chỉ trả lời **duy nhất** bằng JSON thuần, có thể phân tích bằng `json.loads()`.
- Đảm bảo định dạng JSON thuần, không thêm văn bản, giải thích hoặc ký tự thừa như sau:

```json
{{
  "text": "nội dung tóm tắt ngắn gọn, súc tích, tập trung vào điểm chính liên quan đến ngữ cảnh trên"
}}
```
"""

ARTICLE_RETRIEVAL_QUERY_TEMPLATE = """\
Câu hỏi: {question}

Các lựa chọn:
{choices}

Thông tin từ nhận diện biển báo giao thông:
{sign_interpretation}

Ngữ cảnh trong hình ảnh:
{scene_text}
"""