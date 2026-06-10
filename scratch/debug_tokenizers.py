import sys
import io
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Fix windows console output encoding
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

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

def test_model(name, path, prefix=""):
    print("\n" + "="*60)
    print(f"Testing {name} from {path}")
    print("="*60)
    
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForSeq2SeqLM.from_pretrained(path)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    # Prepend prefix if needed
    text = prefix + sample if prefix else sample
    inputs = tokenizer(text, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    
    print("Encoded input token IDs:", input_ids[0][:15].tolist(), "...")
    
    outputs = model.generate(
        input_ids,
        max_length=120,
        min_length=20,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )
    
    print("Output token IDs:", outputs[0].tolist())
    raw_decode = tokenizer.decode(outputs[0], skip_special_tokens=False)
    clean_decode = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    print("\n--- RAW DECODE ---")
    print(repr(raw_decode))
    print("\n--- CLEAN DECODE ---")
    print(clean_decode)

print("Starting debugger...")
test_model("ViT5 (Local)", "models/vit5-finetuned", prefix="summarize: ")
test_model("mT5 (Local)", "models/mt5-finetuned", prefix="summarize: ")
test_model("BARTPho (Local)", "models/bartpho-finetuned")

# Also test VietAI/vit5-base from hub with local vit5 model to see if tokenizer files got corrupted or mismatched
try:
    print("\n\n" + "#"*60)
    print("Testing local ViT5 model with VietAI/vit5-base Hub Tokenizer")
    print("#"*60)
    tokenizer_hub = AutoTokenizer.from_pretrained("VietAI/vit5-base")
    model = AutoModelForSeq2SeqLM.from_pretrained("models/vit5-finetuned")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    inputs = tokenizer_hub("summarize: " + sample, return_tensors='pt')
    input_ids = inputs['input_ids'].to(device)
    outputs = model.generate(
        input_ids,
        max_length=120,
        min_length=20,
        num_beams=4,
        no_repeat_ngram_size=3,
        repetition_penalty=1.2,
        early_stopping=True
    )
    print("Hub decode:", repr(tokenizer_hub.decode(outputs[0], skip_special_tokens=False)))
except Exception as e:
    print("Hub test failed:", e)
