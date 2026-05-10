import requests

url = 'http://127.0.0.1:8000/summarize'
payload = {
    "text": "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự.",
    "extractive_sentences": 3,
    "max_abstractive_length": 80,
}

resp = requests.post(url, json=payload, timeout=120)
print('Status:', resp.status_code)
print(resp.text)
