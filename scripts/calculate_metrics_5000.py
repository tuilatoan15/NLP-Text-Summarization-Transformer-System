#!/usr/bin/env python3
"""
scripts/calculate_metrics_5000.py
Tính toán các chỉ số chất lượng (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum, BERTScore P/R/F1)
và thời gian suy luận (Inference Time) cho 5000 mẫu thử đã chạy với 15 thuật toán.
"""

import os
import sys
import json
import csv
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from statistics import mean
import torch

# Thiết lập sys.path để import các module từ dự án
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Sử dụng rouge_score từ dự án để đồng bộ tiền xử lý và tokenization
from evaluation.metrics import compute_rouge

# Danh sách 15 cấu hình thuật toán cần đánh giá
ALL_CONFIGS = [
    "textrank", "lexrank", "lsa", "vit5", "mt5", "bartpho",
    "textrank_vit5", "lexrank_vit5", "lsa_vit5",
    "textrank_mt5", "lexrank_mt5", "lsa_mt5",
    "textrank_bartpho", "lexrank_bartpho", "lsa_bartpho"
]

def calculate_rouge_single(args):
    """Tính toán ROUGE cho một mẫu đơn lẻ trên một worker CPU."""
    pred, ref = args
    if not pred or not ref:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0, "rougeLsum": 0.0}
    # Sử dụng hàm chuẩn của dự án (đã có loại bỏ stopword, chuẩn hóa văn bản)
    return compute_rouge(pred, ref)

def main():
    json_path = PROJECT_ROOT / "storage/results/benchmark_5000_real.json"
    output_csv = PROJECT_ROOT / "storage/results/benchmark_results_5000.csv"
    output_md = PROJECT_ROOT / "storage/results/benchmark_results_5000.md"
    
    if not json_path.exists():
        print(f"Lỗi: Không tìm thấy file dữ liệu tại {json_path}")
        print("Vui lòng chạy benchmark trước hoặc kiểm tra lại đường dẫn.")
        sys.exit(1)
        
    print(f"Đang đọc dữ liệu từ: {json_path.name}...")
    t0 = time.time()
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"Đã đọc xong file trong {time.time() - t0:.2f} giây.")
    
    samples = data.get("samples", [])
    print(f"Số lượng mẫu thử: {len(samples)}")
    
    # Gom dữ liệu tóm tắt của 15 thuật toán
    model_data = {cfg: [] for cfg in ALL_CONFIGS}
    
    for s in samples:
        ref = s.get("summary", "")
        models = s.get("models", {})
        for cfg in ALL_CONFIGS:
            if cfg in models:
                pred = models[cfg].get("summary", "")
                latency = models[cfg].get("latency", 0.0)
                # Kiểm tra fallback trong trường metrics nếu latency ngoài bị rỗng
                if not latency:
                    latency = models[cfg].get("metrics", {}).get("latency", 0.0)
                model_data[cfg].append({
                    "pred": pred,
                    "ref": ref,
                    "latency": latency
                })
                
    # Lọc ra các cấu hình thực sự có dữ liệu
    configs_to_run = [cfg for cfg, val in model_data.items() if len(val) > 0]
    print(f"Các thuật toán phát hiện được trong file dữ liệu: {configs_to_run}")
    
    # -------------------------------------------------------------
    # BƯỚC 1: TÍNH TOÁN CÁC CHỈ SỐ ROUGE SONG SONG TRÊN CPU
    # -------------------------------------------------------------
    print("\n--- BƯỚC 1: TÍNH TOÁN CÁC CHỈ SỐ ROUGE-1/2/L/Lsum ---")
    rouge_results = {}
    
    for cfg in configs_to_run:
        runs = model_data[cfg]
        print(f"Đang tính ROUGE cho mô hình {cfg.upper()} ({len(runs)} mẫu)...")
        
        pairs = [(r["pred"], r["ref"]) for r in runs]
        
        t_start = time.time()
        # Chạy song song đa nhân CPU để tối đa hóa hiệu năng
        with ProcessPoolExecutor() as executor:
            scores = list(executor.map(calculate_rouge_single, pairs, chunksize=100))
            
        elapsed = time.time() - t_start
        print(f"-> Hoàn tất ROUGE cho {cfg.upper()} trong {elapsed:.2f} giây.")
        rouge_results[cfg] = scores

    # -------------------------------------------------------------
    # BƯỚC 2: TÍNH TOÁN BERTScore TRÊN GPU/CPU SỬ DỤNG BATCHING
    # -------------------------------------------------------------
    print("\n--- BƯỚC 2: TÍNH TOÁN BERTScore (Precision, Recall, F1) ---")
    bertscore_results = {}
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Sử dụng thiết bị: {device.upper()}")
    
    try:
        from bert_score import score as bert_score_fn
        
        # Gom toàn bộ các cặp của tất cả các mô hình để tính BERTScore hàng loạt nhằm tối ưu hóa GPU
        all_pairs = []
        for cfg in configs_to_run:
            for idx, r in enumerate(model_data[cfg]):
                all_pairs.append({
                    "cfg": cfg,
                    "idx": idx,
                    "pred": r["pred"],
                    "ref": r["ref"]
                })
                
        print(f"Tổng số cặp cần tính điểm BERTScore: {len(all_pairs)}")
        cands = [p["pred"] for p in all_pairs]
        refs = [p["ref"] for p in all_pairs]
        
        # Thiết lập batch size nhỏ (32) để tránh tràn VRAM 4GB của GPU laptop (RTX 3050 Ti)
        batch_size = 32 if device == "cuda" else 64
        all_p, all_r, all_f1 = [], [], []
        
        t_start = time.time()
        for i in range(0, len(cands), batch_size):
            c_batch = cands[i : i + batch_size]
            r_batch = refs[i : i + batch_size]
            
            with torch.no_grad():
                p, r, f1 = bert_score_fn(
                    c_batch, r_batch,
                    lang="vi",
                    model_type="bert-base-multilingual-cased",
                    verbose=False,
                    device=device,
                    batch_size=batch_size
                )
                all_p.extend([float(v) for v in p])
                all_r.extend([float(v) for v in r])
                all_f1.extend([float(v) for v in f1])
                
            if (i // batch_size + 1) % 50 == 0 or i + batch_size >= len(cands):
                print(f"Tiến độ BERTScore: {min(i + batch_size, len(cands))}/{len(cands)}")
                
        elapsed = time.time() - t_start
        print(f"-> Hoàn tất tính toán BERTScore trong {elapsed/60:.2f} phút.")
        
        # Phân phối kết quả ngược lại cho từng mô hình
        for idx, p_item in enumerate(all_pairs):
            cfg = p_item["cfg"]
            if cfg not in bertscore_results:
                bertscore_results[cfg] = []
            bertscore_results[cfg].append({
                "precision": all_p[idx],
                "recall": all_r[idx],
                "f1": all_f1[idx]
            })
            
    except Exception as e:
        print(f"Lỗi trong quá trình tính toán BERTScore: {e}")
        print("Sẽ gán giá trị mặc định là 0.0.")
        for cfg in configs_to_run:
            bertscore_results[cfg] = [{"precision": 0.0, "recall": 0.0, "f1": 0.0} for _ in range(len(model_data[cfg]))]

    # -------------------------------------------------------------
    # BƯỚC 3: TỔNG HỢP VÀ KẾT XUẤT BẢNG KẾT QUẢ
    # -------------------------------------------------------------
    print("\n--- BƯỚC 3: TỔNG HỢP VÀ XUẤT KẾT QUẢ ---")
    leaderboard = []
    
    for cfg in configs_to_run:
        runs = model_data[cfg]
        r_scores = rouge_results[cfg]
        b_scores = bertscore_results[cfg]
        
        avg_r1 = mean([s["rouge1"] for s in r_scores])
        avg_r2 = mean([s["rouge2"] for s in r_scores])
        avg_rl = mean([s["rougeL"] for s in r_scores])
        avg_rlsum = mean([s["rougeLsum"] for s in r_scores])
        
        avg_p = mean([s["precision"] for s in b_scores])
        avg_r = mean([s["recall"] for s in b_scores])
        avg_f1 = mean([s["f1"] for s in b_scores])
        
        avg_time = mean([r["latency"] for r in runs])
        
        leaderboard.append({
            "cfg": cfg,
            "name": cfg.upper().replace("_", " ➔ "),
            "r1": round(avg_r1, 4),
            "r2": round(avg_r2, 4),
            "rl": round(avg_rl, 4),
            "rlsum": round(avg_rlsum, 4),
            "bert_p": round(avg_p, 4),
            "bert_r": round(avg_r, 4),
            "bert_f1": round(avg_f1, 4),
            "time": round(avg_time, 4)
        })
        
    # Sắp xếp thứ tự hiển thị chuẩn theo danh sách ALL_CONFIGS ban đầu
    leaderboard.sort(key=lambda x: ALL_CONFIGS.index(x["cfg"]) if x["cfg"] in ALL_CONFIGS else 99)
    
    # Tạo bảng Markdown kết quả
    headers_md = ["Phương pháp", "R1", "R2", "RL", "RLSum", "BERT P", "BERT R", "BERT F1", "Time (s)"]
    md_lines = []
    md_lines.append("# Báo cáo kết quả đánh giá chất lượng và hiệu năng trên 5000 mẫu thử")
    md_lines.append(f"Ngày tính toán: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md_lines.append("")
    md_lines.append("| " + " | ".join(headers_md) + " |")
    md_lines.append("| " + " | ".join([":---" if i == 0 else ":---:" for i in range(len(headers_md))]) + " |")
    
    for row in leaderboard:
        md_lines.append(
            f"| {row['name']} | {row['r1']:.4f} | {row['r2']:.4f} | {row['rl']:.4f} | {row['rlsum']:.4f} | "
            f"{row['bert_p']:.4f} | {row['bert_r']:.4f} | {row['bert_f1']:.4f} | {row['time']:.4f} |"
        )
        
    md_content = "\n".join(md_lines)
    
    # Lưu file Markdown
    output_md.parent.mkdir(parents=True, exist_ok=True)
    with open(output_md, "w", encoding="utf-8") as f:
        f.write(md_content)
    print(f"Đã lưu bảng báo cáo Markdown tại: {output_md}")
    
    # Lưu file CSV
    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(headers_md)
        for row in leaderboard:
            writer.writerow([
                row['name'], row['r1'], row['r2'], row['rl'], row['rlsum'],
                row['bert_p'], row['bert_r'], row['bert_f1'], row['time']
            ])
    print(f"Đã lưu bảng tính toán CSV tại: {output_csv}")
    
    print("\n" + "="*80)
    print("HOÀN TẤT TÍNH TOÁN!")
    print("="*80)
    print(md_content)
    print("="*80)

if __name__ == "__main__":
    main()
