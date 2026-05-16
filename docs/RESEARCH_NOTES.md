# Vietnamese Summarization Research Notes

## Scope

The project is now scoped to algorithm comparison for a graduation thesis:

- Extractive: TextRank, LexRank, LSA Summarizer
- Abstractive: ViT5, mT5, BARTPho
- Metrics: ROUGE-1, ROUGE-2, ROUGE-L, BLEU, BERTScore, semantic similarity, compression ratio, processing time

## ViT5 Failure Analysis

The malformed output pattern such as `Zn hồ[n hồ] hồ! hồ>` usually comes from a combination of:

- Loading a local checkpoint with a tokenizer that was not saved from the same base model.
- Using a fast tokenizer artifact that does not match the SentencePiece vocabulary expected by ViT5.
- Decoding labels without replacing `-100` by `pad_token_id` during evaluation.
- Using `decode()` on a single tensor path inconsistently instead of `batch_decode()`.
- Training on noisy HTML/PDF text or duplicated/garbled samples.
- Not validating generated text before returning it to the UI.

The refactor fixes those points by using `use_fast=False` for ViT5-style SentencePiece tokenizers, checking tokenizer/model vocab size, using `text_target` for label preprocessing, decoding with `batch_decode(..., clean_up_tokenization_spaces=False)`, cleaning VNExpress records before training, and rejecting/falling back from bad generations.

## Why TextRank Often Wins In The Dashboard

When the reference is missing, the system uses the source article as the evaluation reference. Extractive methods reuse original sentences, so n-gram metrics such as ROUGE and BLEU naturally reward them. TextRank also selects central sentences from the document graph, which is strong for news articles where key facts are repeated across lead and body paragraphs. This does not mean TextRank is always better semantically; it means it is favored by source-as-reference evaluation.

## Recommended Vietnamese Model

For Vietnamese abstractive summarization, the best thesis recommendation is:

1. Fine-tuned ViT5 on cleaned VNExpress as the main model.
2. BARTPho as the strongest Vietnamese-specific comparison model.
3. mT5 as the multilingual baseline.

Do not judge pretrained ViT5/mT5/BARTPho directly against TextRank without fine-tuning; pretrained seq2seq models are not reliable summarizers out of the box.

## Standard Structure

```text
api/                 FastAPI research API
configs/             Model and training configuration
frontend/            React + Tailwind + Recharts dashboard
scripts/preprocess.py
scripts/train.py
scripts/evaluate.py
scripts/inference.py
src/                 Core AI modules
train/               Dataset loader and backward-compatible training entrypoint
tests/               Unit and API tests
storage/             Uploads and reports
```
