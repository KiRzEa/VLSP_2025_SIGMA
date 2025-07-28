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