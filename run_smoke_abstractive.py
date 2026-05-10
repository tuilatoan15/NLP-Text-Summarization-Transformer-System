from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

sample = '''
Hội đồng Bảo an Liên Hợp Quốc đã họp khẩn cấp để thảo luận về tình hình
leo thang căng thẳng ở Trung Đông. Nhiều quốc gia kêu gọi ngừng bắn ngay lập
tức và mở hành lang nhân đạo cho người dân vùng chiến sự. Đại diện Mỹ phát biểu
rằng Washington ủng hộ giải pháp hai nhà nước nhưng nhấn mạnh quyền tự vệ hợp
pháp. Nga và Trung Quốc phản đối dự thảo nghị quyết, cho rằng văn kiện còn
thiếu cân bằng. Cuộc khủng hoảng nhân đạo ngày càng nghiêm trọng khi hàng nghìn
thường dân phải di tản. Các tổ chức phi chính phủ kêu gọi cộng đồng quốc tế
hành động khẩn cấp để bảo vệ dân thường.
'''

model_path = "models/vit5-finetuned"
print(f"Loading tokenizer from {model_path}...")
tokenizer = AutoTokenizer.from_pretrained(model_path)
print(f"Loading model from {model_path}...")
model = AutoModelForSeq2SeqLM.from_pretrained(model_path)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Device used:', device)
model.to(device)

inputs = tokenizer(sample, return_tensors='pt', truncation=True, max_length=512)
input_ids = inputs['input_ids'].to(device)
attention_mask = inputs.get('attention_mask')
if attention_mask is not None:
    attention_mask = attention_mask.to(device)

print('Generating... (this may take a while on CPU)')
outputs = model.generate(input_ids, max_length=80, min_length=20, num_beams=2, no_repeat_ngram_size=3, early_stopping=True)
print('Raw output tensor:', outputs)
try:
    ids = outputs[0].tolist()
except Exception:
    ids = None
print('Output token ids:', ids)
summary = tokenizer.decode(outputs[0], skip_special_tokens=False)
print('\n=== RAW DECODE (with special tokens) ===')
print(summary)
summary_clean = tokenizer.decode(outputs[0], skip_special_tokens=True)
print('\n=== SUMMARY ===')
print(summary_clean)

# Debug: try using tokenizer from Hugging Face hub (VietAI/vit5-base) to check token mapping
try:
    print('\n--- Debug: testing tokenizer from VietAI/vit5-base ---')
    tokenizer_hub = AutoTokenizer.from_pretrained('VietAI/vit5-base')
    inputs_hub = tokenizer_hub(sample, return_tensors='pt', truncation=True, max_length=512)
    in_ids = inputs_hub['input_ids'].to(device)
    outputs_hub = model.generate(in_ids, max_length=80, min_length=20, num_beams=2, no_repeat_ngram_size=3, early_stopping=True)
    summary_hub = tokenizer_hub.decode(outputs_hub[0], skip_special_tokens=True)
    print('SUMMARY (hub tokenizer):')
    print(summary_hub)
except Exception as e:
    print('Debug hub tokenizer failed:', e)
