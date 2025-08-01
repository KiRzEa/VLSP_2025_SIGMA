SIGN_ATTRIBUTES_EXTRACTION_PROMPT = """\
Bạn là một chuyên gia phân tích hình ảnh giao thông.
Phân tích đặc điểm của biển báo giao thông trong ảnh dưới đây. \
Hãy mô tả chi tiết các đặc điểm có thể giúp phân biệt và truy xuất thông tin từ một tập biển báo có sẵn:

1. Hình dạng tổng thể của biển báo là gì?
2. Màu nền chính của biển là gì? Có viền không? Nếu có, màu viền là gì?
3. Biển có chứa hình ảnh biểu tượng gì không? (ví dụ: mũi tên, xe, người, v.v.)
4. Biển có chứa chữ hay số không? Nếu có, hãy ghi rõ nội dung chữ/số đó.
5. Biển có gạch chéo đỏ hay không?
6. Có biển phụ kèm theo không? Nếu có, hãy mô tả nội dung và hình dạng của biển phụ.

Yêu cầu:
- Chỉ trả lời dưới dạng **JSON**, không kèm theo giải thích hoặc văn bản dư thừa.
- Mỗi trường cần điền càng rõ càng tốt, nếu không thấy thì để `null`.
- Đảm bảo đúng định dạng JSON như ví dụ bên dưới.

### Cần trả lời theo cấu trúc JSON như sau:
```json
[
  {{
    "id": "Biển số của biển báo hoặc null nếu không rõ",
    "shape": "Hình dạng tổng thể của biển (ví dụ: Hình tròn, Hình vuông, Hình tam giác, ...)",
    "background_color": "Màu nền chính (ví dụ: Xanh dương, Trắng, Vàng, ...)",
    "border": "Màu viền của biển (ví dụ: Đỏ), null nếu không có",
    "icon": "Miêu tả biểu tượng hoặc hình ảnh bên trong biển báo (ví dụ: Mũi tên hướng lên, Xe ô tô màu đen, ...)",
    "text": "Chữ hoặc số trên biển, nếu có (ví dụ: '60', 'CẤM RẼ TRÁI'), nếu không có thì null",
    "diagonal_line": "Có gạch chéo đỏ không? Nếu có mô tả vị trí và màu, nếu không thì null",
    "sub_sign": "Có biển phụ không? Nếu có mô tả hình dạng và nội dung, nếu không thì null"
  }}
]
```
"""

SIGN_SEMANTIC_MATCHING_PROMPT = """\
Bạn là một chuyên gia giao thông được giao nhiệm vụ đối chiếu và so sánh các biển báo giao thông.

Dưới đây là hai bộ thông tin về biển báo:
- Một là biển báo cần nhận diện (ảnh query, trích xuất từ thực tế).
- Hai là biển báo từ cơ sở dữ liệu đã được trích xuất sẵn thông tin.

Hãy đánh giá mức độ tương đồng giữa chúng, dựa trên các tiêu chí:
1. **Hình dạng** (shape) có giống nhau không?
2. **Màu nền** (background_color) và **viền** (border) có trùng hoặc tương tự không?
3. Có **biểu tượng** giống nhau không? (ví dụ: mũi tên, xe, người, ...)
4. **Nội dung chữ hoặc số** trên biển có giống hoặc tương đồng không?
5. Có cùng **gạch chéo đỏ** hoặc đặc điểm nổi bật không?
6. Có biển phụ kèm theo giống nhau không?

### Yêu cầu:
- Hãy trả về đánh giá dưới dạng JSON, gồm:
[
  {{
    "image_id": "tên của ảnh đầu vào (ví dụ: image022.png)",
    "match_score": "điểm số từ 0 đến 10 thể hiện mức độ phù hợp",
    "reason": "giải thích ngắn gọn vì sao lại cho điểm đó",
    "is_match": True nếu hai biển có khả năng giống nhau cao (điểm ≥ 4), False nếu không.
  }},
  ...
]

### Dữ liệu:

**Query image attributes**:
```json
{query_attributes}
```
**Candidate image attributes**:
```json
{candidate_attributes}
```
"""