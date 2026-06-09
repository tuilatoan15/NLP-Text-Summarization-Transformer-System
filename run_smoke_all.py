import os
import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Set UTF-8 encoding for Windows console (avoid Vietnamese encoding errors)
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

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

models_to_test = {
    "ViT5": "models/vit5-finetuned",
    "mT5": "models/mt5-finetuned",
    "BARTPho": "models/bartpho-finetuned"
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[*] Python version: {sys.version}")
print(f"[*] PyTorch version: {torch.__version__}")
print(f"[*] Device used: {device}")

for name, model_path in models_to_test.items():
    print("\n" + "="*50)
    print(f"[*] Testing {name} at {model_path}")
    print("="*50)
    
    if not os.path.exists(model_path):
        print(f"[-] Directory {model_path} does not exist. Skipping.")
        continue
        
    try:
        print(f"[*] Loading tokenizer from {model_path}...")
        # Load tokenizer with use_fast=False if it fails, or by default let transformers load
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        print(f"[*] Loading model from {model_path}...")
        model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
        model.to(device)
        
        print("[*] Tokenizing sample text...")
        inputs = tokenizer(sample, return_tensors='pt', truncation=True, max_length=512)
        input_ids = inputs['input_ids'].to(device)
        attention_mask = inputs.get('attention_mask')
        if attention_mask is not None:
            attention_mask = attention_mask.to(device)
            
        print("[*] Generating summary...")
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_length=120,
            min_length=20,
            num_beams=2,
            no_repeat_ngram_size=3,
            early_stopping=True
        )
        
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"[+] {name} SUMMARY:")
        print(summary)
        print(f"[+] Output length: {len(summary.split())} words.")
    except Exception as e:
        print(f"[-] Error testing {name}: {str(e)}")
