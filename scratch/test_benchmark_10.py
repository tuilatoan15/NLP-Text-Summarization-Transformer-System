import sys
import time
import random
import gc
from pathlib import Path
import torch

# Add project root to python path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from summarizers.extractive.extractive_summarizer import summarize_extractive_algorithm
from summarizers.abstractive.abstractive_summarizer import abstractive_summarize_key, get_summarizer
from pipeline.hybrid_summarizer import HybridSummarizer
from src.utils import count_words, logger, clear_gpu_cache

# Cấu hình 15 thuật toán
EXTRACTIVE_CONFIGS = ["textrank", "lexrank", "lsa"]
VIT5_CONFIGS = ["vit5", "textrank_vit5", "lexrank_vit5", "lsa_vit5"]
MT5_CONFIGS = ["mt5", "textrank_mt5", "lexrank_mt5", "lsa_mt5"]
BARTPHO_CONFIGS = ["bartpho", "textrank_bartpho", "lexrank_bartpho", "lsa_bartpho"]

ALL_CONFIGS = EXTRACTIVE_CONFIGS + VIT5_CONFIGS + MT5_CONFIGS + BARTPHO_CONFIGS

def load_10_samples() -> list[dict]:
    from datasets import load_dataset
    print("[*] Loading 'nam194/vietnews' test split...")
    dataset = load_dataset("nam194/vietnews", split="test")
    
    valid_samples = []
    for item in dataset:
        article = item.get("article", "").strip()
        summary = item.get("abstract", "").strip() or item.get("title", "").strip()
        if article and len(article.split()) >= 30 and summary:
            valid_samples.append({
                "article": article,
                "summary": summary,
                "title": item.get("title", "Không có tiêu đề")
            })
            
    print(f"[+] Found {len(valid_samples)} valid samples. Selecting 10 random samples...")
    random.seed(42)
    sampled = random.sample(valid_samples, 10)
    for idx, s in enumerate(sampled):
        s["id"] = f"test_sample_{idx+1:02d}"
    return sampled

def run_model_inference(model_key: str, text: str) -> tuple[str, float]:
    t0 = time.perf_counter()
    try:
        if model_key in ["textrank", "lexrank", "lsa"]:
            # Extractive
            res = summarize_extractive_algorithm(text, algorithm=model_key, sentence_count=3)
            summary = res.get("summary", "") if isinstance(res, dict) else str(res)
        elif model_key in ["vit5", "mt5", "bartpho"]:
            # Abstractive
            summary = abstractive_summarize_key(model_key, text, max_output_length=150)
        elif "_" in model_key:
            # Hybrid
            ext_algo, abs_algo = model_key.split("_")
            hybrid = HybridSummarizer(abstractive_model_key=abs_algo)
            summary = hybrid.summarize(text, extractive_algo=ext_algo, max_target_tokens=150)
        else:
            summary = ""
    except Exception as e:
        print(f"[-] Inference failed for {model_key}: {e}")
        summary = ""
    elapsed = time.perf_counter() - t0
    return summary, elapsed

def unload_model(key: str):
    from ai_models.model_loader import _registry
    if key in _registry._loaded:
        print(f"[*] Unloading model [{key}] from GPU to free VRAM...")
        loaded = _registry._loaded.pop(key)
        loaded.model.to("cpu")
        del loaded.model
        del loaded.tokenizer
        del loaded
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[+] Unloaded model [{key}] successfully.")

def main():
    # 1. Load samples
    samples = load_10_samples()
    
    # Database to store measurements
    # Structure: {config_key: [latencies]}
    results = {cfg: [] for cfg in ALL_CONFIGS}
    
    print("\n" + "="*60)
    print("STARTING 10-SAMPLE TRIAL BENCHMARK FOR 15 CONFIGURATIONS")
    print("="*60)
    
    # 2. Run Extractive on CPU
    print("\n[+] Group 1: Running CPU Extractive models...")
    for cfg in EXTRACTIVE_CONFIGS:
        print(f"  -> Testing {cfg}...")
        for sample in samples:
            _, elapsed = run_model_inference(cfg, sample["article"])
            results[cfg].append(elapsed)
            
    # 3. Run ViT5 configurations
    print("\n[+] Group 2: Running ViT5 configurations...")
    # Preload ViT5
    _ = get_summarizer("vit5")
    for cfg in VIT5_CONFIGS:
        print(f"  -> Testing {cfg}...")
        for sample in samples:
            _, elapsed = run_model_inference(cfg, sample["article"])
            results[cfg].append(elapsed)
    unload_model("vit5")
    
    # 4. Run mT5 configurations
    print("\n[+] Group 3: Running mT5 configurations...")
    # Preload mT5
    _ = get_summarizer("mt5")
    for cfg in MT5_CONFIGS:
        print(f"  -> Testing {cfg}...")
        for sample in samples:
            _, elapsed = run_model_inference(cfg, sample["article"])
            results[cfg].append(elapsed)
    unload_model("mt5")
    
    # 5. Run BARTPho configurations
    print("\n[+] Group 4: Running BARTPho configurations...")
    # Preload BARTPho
    _ = get_summarizer("bartpho")
    for cfg in BARTPHO_CONFIGS:
        print(f"  -> Testing {cfg}...")
        for sample in samples:
            _, elapsed = run_model_inference(cfg, sample["article"])
            results[cfg].append(elapsed)
    unload_model("bartpho")
    
    # 6. Print Report
    print("\n" + "="*60)
    print("TRIAL RESULTS & MEASUREMENTS SUMMARY")
    print("="*60)
    print(f"{'Configuration':<25} | {'Avg Latency (s)':<18} | {'Total Time for 10 (s)':<22}")
    print("-" * 72)
    
    group_latencies = {"extractive": [], "abstractive": [], "hybrid": []}
    
    for cfg in ALL_CONFIGS:
        times = results[cfg]
        avg = sum(times) / len(times)
        total = sum(times)
        print(f"{cfg:<25} | {avg:<18.4f} | {total:<22.4f}")
        
        # Categorize
        if cfg in EXTRACTIVE_CONFIGS:
            group_latencies["extractive"].append(avg)
        elif cfg in ["vit5", "mt5", "bartpho"]:
            group_latencies["abstractive"].append(avg)
        else:
            group_latencies["hybrid"].append(avg)
            
    print("\n" + "="*60)
    print("GROUP STATISTICS")
    print("="*60)
    for group_name, averages in group_latencies.items():
        group_avg = sum(averages) / len(averages)
        print(f"Group: {group_name.upper():<12} | Avg Latency: {group_avg:.4f} s/sample")
        
    # Save statistics to JSON for easy loading in report script
    import json
    out_path = Path("storage/results/benchmark_10_trial.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\n[+] Saved raw measurements to {out_path}")

if __name__ == "__main__":
    main()
