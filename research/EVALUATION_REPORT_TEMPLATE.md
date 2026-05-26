# Evaluation Report Template

## Document

- ID: `{document_id}`
- Title: `{title}`
- Words: `{word_count}` | Chunks: `{chunk_count}`

## Algorithm Comparison

| Algorithm | Group | ROUGE-L | BERTScore | Semantic | Latency (s) | Consistency |
|---|---|---:|---:|---:|---:|---:|
| TextRank | extractive | | | | | |
| TF-IDF | extractive | | | | | |
| ViT5 | abstractive | | | | | |

## Citation Grounding

- Coverage: `{grounding_coverage}`
- High-risk sentences: list from `hallucination.audit_summary`

## Export

```bash
python scripts/benchmark_document_intelligence.py --document-id DOC_ID
```
