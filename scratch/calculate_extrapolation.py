import json
from pathlib import Path

trial_path = Path("C:/Users/ASUS/Desktop/NLP-Text-Summarization-Transformer-System/storage/results/benchmark_10_trial.json")

if not trial_path.exists():
    print(f"Error: {trial_path} does not exist!")
    exit(1)

with open(trial_path, "r", encoding="utf-8") as f:
    results = json.load(f)

# Categories
extractive = ["textrank", "lexrank", "lsa"]
abstractive = ["vit5", "mt5", "bartpho"]
hybrid = [
    "textrank_vit5", "lexrank_vit5", "lsa_vit5",
    "textrank_mt5", "lexrank_mt5", "lsa_mt5",
    "textrank_bartpho", "lexrank_bartpho", "lsa_bartpho"
]

all_configs = extractive + abstractive + hybrid

# Sum of all latencies per sample
sample_totals = []
for i in range(10):
    sample_sum = sum(results[cfg][i] for cfg in all_configs)
    sample_totals.append(sample_sum)

avg_total_per_sample = sum(sample_totals) / 10

# Calculate averages per config
averages = {}
for cfg in all_configs:
    averages[cfg] = sum(results[cfg]) / 10

print("=== DETAILED CALCULATIONS FOR 10,000 SAMPLES ===")
print(f"Average total time for 1 sample (all 15 configs): {avg_total_per_sample:.4f} seconds")
print(f"Total sequential time for 10,000 samples: {10000 * avg_total_per_sample:.2f} seconds")
print(f"  In hours: {10000 * avg_total_per_sample / 3600:.2f} hours")
print(f"  In days: {10000 * avg_total_per_sample / 86400:.2f} days")

print("\n=== TIMINGS BY CONFIGURATION (10,000 SAMPLES) ===")
print(f"{'Config':<20} | {'Latency/Sample (s)':<20} | {'Time for 10k (s)':<18} | {'In Hours':<10}")
print("-" * 75)
for cfg in all_configs:
    avg = averages[cfg]
    total_10k = avg * 10000
    hours_10k = total_10k / 3600
    print(f"{cfg:<20} | {avg:<20.4f} | {total_10k:<18.2f} | {hours_10k:.2f}h")

# Sum by groups
sum_extractive = sum(averages[cfg] for cfg in extractive)
sum_abstractive = sum(averages[cfg] for cfg in abstractive)
sum_hybrid = sum(averages[cfg] for cfg in hybrid)

print("\n=== TIMINGS BY GROUP (10,000 SAMPLES) ===")
print(f"Extractive (3 configs): {sum_extractive:.4f} s/sample -> {sum_extractive * 10000 / 3600:.2f} hours total")
print(f"Abstractive (3 configs): {sum_abstractive:.4f} s/sample -> {sum_abstractive * 10000 / 3600:.2f} hours total")
print(f"Hybrid (9 configs): {sum_hybrid:.4f} s/sample -> {sum_hybrid * 10000 / 3600:.2f} hours total")

# Scenario: What if we parallelize Extractive and GPU-optimize Abstractive/Hybrid?
# Extractive can easily run in parallel (CPU 6-core factor ~4.5x speedup)
# GPU models run sequentially but model preloading saves ~0s per sample.
# What if we only run a subset? (e.g. only 1 best config vs all 15)
# If only 1 best config (e.g. textrank_bartpho at 0.6736 s/sample):
# 10k samples * 0.6736s = 6736s = 1.87 hours!
