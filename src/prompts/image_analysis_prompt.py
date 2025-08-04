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
- Hai là **danh sách biển báo trong cơ sở dữ liệu** (gồm nhiều ứng viên), đã được trích xuất thông tin mô tả chi tiết.

Hãy đánh giá mức độ tương đồng giữa chúng, dựa trên các tiêu chí:
1. **Hình dạng** (`shape`) có giống nhau không? (*ưu tiên cao*)
2. **Màu nền** (`background_color`) và **viền** (`border`) có trùng hoặc tương tự không?
3. Có **biểu tượng** (`icon`) giống hoặc tương đương không? (*ưu tiên cao*)
4. **Nội dung chữ hoặc số** (`text`) trên biển có giống hoặc tương đồng không?
5. Có cùng **gạch chéo đỏ** (`diagonal_line`) hoặc đặc điểm nổi bật không?
6. Có **biển phụ** (`sub_sign`) giống nhau không?

Chỉ so sánh những trường không phải `null`. Nếu một trường bị thiếu trong cả query hoặc candidate thì **bỏ qua tiêu chí đó**.

Nếu có nhiều ứng viên tương tự nhau, **ưu tiên các yếu tố quan trọng** như `shape`, `icon`, và `text`.

### Yêu cầu:

- **Chỉ trả về kết quả dưới dạng JSON thuần**, có thể phân tích bằng `json.loads()` (không có mô tả hay giải thích thêm).
- Chỉ trả về **một ứng viên phù hợp nhất** (best match) từ tất cả các ứng viên.
- Nếu có nhiều ứng viên có điểm tương đồng như nhau, **ưu tiên các tiêu chí quyết định** như `shape`, `icon`, và `text` để chọn ra ứng viên phù hợp nhất.

### Format bắt buộc:
```json
{{
  "image_id": "id của ảnh ứng viên (ví dụ: image022.png)",
  "id": "Biển số của biển báo (từ candidate)",
  "reason": "Giải thích ngắn gọn vì sao đây là lựa chọn phù hợp nhất (dựa trên các tiêu chí)",
  "attributes": {{
    "id": "...",
    "shape": "...",
    "background_color": "...",
    "border": "...",
    "icon": "...",
    "text": "...",
    "diagonal_line": "...",
    "sub_sign": "..."
  }}
}}
```

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

INTERPRET_SIGN_MEANING_PROMPT= """\
Bạn là chuyên gia về luật giao thông Việt Nam, có nhiệm vụ phân tích và giải thích ý nghĩa của các biển báo giao thông theo **Bộ luật giao thông đường bộ Việt Nam**.

### Nhiệm vụ:

1. **Phân loại và giải nghĩa từng biển báo riêng lẻ**:
   - Dựa trên các thuộc tính (hình dạng, màu sắc, biểu tượng, chữ viết, ...) để xác định:
     - **Loại biển báo**: ví dụ *biển cấm*, *biển cảnh báo*, *biển chỉ dẫn*, *biển hiệu lệnh*, v.v.
     - **Nội dung/ý nghĩa chính xác của từng biển báo** theo đúng quy định Việt Nam.

2. **Phân tích mối quan hệ giữa các biển báo nếu có**:
   - Nếu có **biển phụ** (thông qua trường `"sub_sign"`), hãy xác định nó **bổ nghĩa cho biển báo nào** và **ý nghĩa khi kết hợp lại là gì**.
   - Nếu có **các biển báo chính khác nhau nhưng mang thông điệp liên quan** (ví dụ biển cấm + biển hiệu lệnh về tốc độ), hãy mô tả **ý nghĩa tổ hợp**.

### Yêu cầu:

- **Chỉ trả về kết quả dưới dạng JSON thuần**, có thể phân tích bằng `json.loads()` (không có mô tả hay giải thích thêm).
   
### Format bắt buộc (JSON):

```json
{{
  "individual_signs": [
    {{
      "type": "Loại biển báo giao thông, xác định theo quy chuẩn Việt Nam. Ví dụ: 'Biển cấm', 'Biển cảnh báo', 'Biển chỉ dẫn', 'Biển hiệu lệnh', 'Biển phụ'.",
      "meaning": "Nội dung hoặc thông điệp chính xác của biển báo đó, mô tả hành vi mà người tham gia giao thông cần lưu ý, cấm, tuân theo hoặc cảnh báo."
    }},
    ...
  ],
  "combinations": [
    {{
      "combined_signs": "Danh sách các biển báo có mối liên hệ, thường gồm 1 biển chính và 1 hoặc nhiều biển phụ. Chuỗi mô tả tên hoặc nội dung dễ hiểu của từng biển.",
      "meaning": "Ý nghĩa của việc kết hợp các biển báo này lại với nhau. Ví dụ như biển tốc độ kết hợp với biển phụ áp dụng cho xe tải, cho ra thông điệp đầy đủ là 'Chỉ xe tải mới bị giới hạn tốc độ'."
    }},
    ...
  ]
}}
```

### Dữ liệu đầu vào:
```json
{query_descriptions}
```
"""

SCENE_DESCRIPTION_PROMPT = """\
Bạn là chuyên gia phân tích hình ảnh trong lĩnh vực giao thông, có nhiệm vụ **mô tả ngữ cảnh giao thông tổng thể của một bức ảnh** thực tế, dựa trên:
- Hình ảnh thực tế chứa các biển báo giao thông
- Danh sách các biển báo đã được nhận diện, phân loại và giải nghĩa theo luật giao thông Việt Nam

### Mục tiêu:
1. **Xác định ngữ cảnh tổng thể của bức ảnh**: Đây là đoạn mô tả cảnh vật, tình huống giao thông, các quy định hiện hành trong khung cảnh này. Hãy mô tả như thể bạn đang hướng dẫn một người lái xe chuẩn bị đi qua khu vực đó.
2. **Liên kết các biển báo với ngữ cảnh**:
   - Giải thích ngắn gọn **tác động của các biển báo** đến hành vi người lái xe.
   - Ví dụ: nếu có biển cấm rẽ trái + biển phụ áp dụng cho xe tải → nghĩa là “xe tải không được rẽ trái tại đoạn này”.

### Đầu ra yêu cầu:
- Trả về kết quả dưới dạng JSON thuần có cấu trúc như sau:

```json
{{
  "scene_text": "Mô tả bằng tiếng Việt, dài 2-5 câu, phản ánh ngữ cảnh giao thông tổng thể của bức ảnh, bao gồm vị trí giả định (nếu có), loại khu vực (đường nội đô, quốc lộ, ngã ba, khu dân cư, v.v.), hành vi người lái cần chú ý, và quy định được áp dụng qua các biển báo."
}}
```

### Dữ liệu đầu vào:
- Danh sách các biển báo đã phân tích:
{sign_interpretion}
"""