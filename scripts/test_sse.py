import sys
import json

try:
    import requests
except Exception as e:
    print("requests library not found. Install with: pip install requests")
    sys.exit(1)

URL = "http://127.0.0.1:8000/summarize/compare/stream"
PAYLOAD = {
    "text": (
        "Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình "
        "leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay "
        "lập tức và mở hành lang nhân đạo cho người dân vùng chiến sự."
    ),
    "algorithms": ["textrank", "lsa", "lexrank"],
    "extractive_sentences": 3,
}

print(f"POST -> {URL} (algorithms={PAYLOAD['algorithms']})")

try:
    with requests.post(URL, json=PAYLOAD, stream=True, timeout=60) as r:
        print('Status:', r.status_code)
        if r.status_code != 200:
            print('Response not OK:', r.text)
            sys.exit(1)
        for raw in r.iter_lines(decode_unicode=True):
            if not raw:
                continue
            line = raw.strip()
            # SSE format: data: {...}
            if line.startswith('data:'):
                try:
                    payload = line[len('data:'):].strip()
                    obj = json.loads(payload)
                    print('EVENT:', json.dumps(obj, ensure_ascii=False))
                    if obj.get('event') == 'finished':
                        print('Received finished -> exiting')
                        break
                except Exception as e:
                    print('INVALID JSON:', line, e)
            else:
                print('LINE:', line)
except Exception as e:
    print('Request error:', e)
    sys.exit(1)
