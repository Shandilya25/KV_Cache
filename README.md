# KV Cache From Scratch

Adding a **KV cache** to attention in PyTorch, step by step. This repo covers three kinds of attention:

- Multi-Head Attention (MHA)
- Multi-Query Attention (MQA)
- Grouped-Query Attention (GQA)

## What is a KV cache?

A model like GPT writes text **one token at a time**. To pick the next token, every new token needs to look at all the tokens that came before it.

Without a cache, the model recomputes the **keys (K)** and **values (V)** for the *whole* sentence at every step. That's a lot of repeated work.

With a cache:

1. Compute K and V for the new token only.
2. Save them in the cache.
3. Run attention between the new token's query and **all** the K and V stored in the cache.

The old K and V never change, so there's no need to compute them again. That makes generation much faster.

```
Step 1:  "The cat"        -> cache holds K,V for [The, cat]
Step 2:  + "sat"          -> compute K,V for "sat" only, add to cache
Step 3:  + "on"           -> compute K,V for "on" only, add to cache
```

## MHA vs MQA vs GQA

The cache can get big. These three types of attention differ in **how many K/V heads** they keep, and so in how big the cache gets.

| Type | Query heads | K/V heads | Cache size |
|---|---|---|---|
| **MHA** | 8 | 8 (one per query head) | biggest |
| **GQA** | 8 | 2 (each shared by a group of query heads) | medium |
| **MQA** | 8 | 1 (shared by all query heads) | smallest |

- **MHA**: every head has its own K and V. Best quality, most memory.
- **MQA**: all heads share one K and one V. Least memory, can lose a bit of quality.
- **GQA**: the middle ground. Heads are split into groups, and each group shares one K and V. Models like Llama 2 and 3 use this.

## Files

| File | What's in it |
|---|---|
| `config.py` | Settings: batch size, max sequence length (`seq_len`) and so on. The cache size comes from here. |
| `model.py` | Embedding, positional encoding, layer norm, feed-forward, and **Multi-Head Attention with a KV cache**. |
| `mqa_attention.py` | The same building blocks with **Multi-Query Attention** (1 shared K/V head). |
| `gqa_attention.py` | The same building blocks with **Grouped-Query Attention** (K/V shared within groups). |

## How the cache works in the code

- `key_cache` and `value_cache` are created once with `register_buffer`, so they move to the GPU with the model, aren't trained, and aren't saved in checkpoints.
- `cache_len` remembers how many tokens are already stored.
- Each call to `forward(..., use_cache=True)`:
  - writes the new K and V into slots `start:end`
  - reads back everything from `0:end`
  - slices the causal mask to `mask[:, :, start:end, :end]`, so the rows are the new tokens and the columns are all the tokens so far
- `reset_cache()` empties the cache. Call it before generating a new sentence.

## Example (`model.py`)

Run this from inside this folder, because the files import `config.py`.

```python
import torch
from model import MultiheadAttention

attn = MultiheadAttention(embedding_dim=64, n_heads=8, dropout=0.0).eval()
x = torch.randn(1, 5, 64)        # 1 sentence, 5 tokens

with torch.no_grad():
    attn.reset_cache()
    out = attn(x[:, :3], x[:, :3], x[:, :3], use_cache=True)     # prompt: first 3 tokens
    out = attn(x[:, 3:4], x[:, 3:4], x[:, 3:4], use_cache=True)  # then 1 new token at a time
```

## Status

Work in progress.

- [x] Multi-Head Attention with KV cache (`model.py`). Checked: the cached output matches normal attention.
- [ ] Multi-Query Attention (`mqa_attention.py`). Still being fixed.
- [ ] Grouped-Query Attention (`gqa_attention.py`). Still being fixed.
- [ ] Put the attention into a full model and generate text with it.

## Tech stack

- Python
- PyTorch

## References

- Vaswani et al., *Attention Is All You Need* (2017). [Paper](https://arxiv.org/abs/1706.03762)
- Shazeer, *Fast Transformer Decoding: One Write-Head is All You Need* (2019), which introduced MQA. [Paper](https://arxiv.org/abs/1911.02150)
- Ainslie et al., *GQA: Training Generalized Multi-Query Transformer Models* (2023). [Paper](https://arxiv.org/abs/2305.13245)
