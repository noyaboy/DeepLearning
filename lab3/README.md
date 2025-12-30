# Lab 3: Sequence-to-Sequence Translation

Implement a Transformer model for Chinese-English translation.

## Objectives

- Understand Transformer architecture
- Implement multi-head attention mechanism
- Build encoder-decoder model for machine translation

## Implementation

### Architecture (`network.py`)

#### Components
- **Positional Encoding**: Sinusoidal position embeddings
- **Multi-Head Attention**: Scaled dot-product attention with multiple heads
- **Feed-Forward Network**: Two linear layers with ReLU activation
- **Encoder**: Stack of encoder layers with self-attention
- **Decoder**: Stack of decoder layers with self and cross-attention

### Data

| File | Description |
|------|-------------|
| `translation_train_data.json` | Training data (Chinese-English pairs) |
| `translation_test_data.json` | Test data |

### Utilities (`utils.py`)

- Tokenizers for Chinese and English
- Special tokens: PAD, BOS, EOS
- Device configuration

## Files

| File | Description |
|------|-------------|
| `network.py` | Transformer model implementation |
| `utils.py` | Tokenizers and utilities |
| `run.py` | Training/inference script |
| `Lab3.ipynb` | Main notebook |
| `314580042_report.pdf` | Lab report |
| `2025_DL_Lab03_v2.pdf` | Lab specification |

## Usage

```python
from network import Seq2SeqTransformer
from utils import tokenizer_chinese, tokenizer_english

# Initialize model
model = Seq2SeqTransformer(...)

# Tokenize input
src_tokens = tokenizer_chinese(chinese_text)

# Generate translation
output = model.generate(src_tokens)
translation = tokenizer_english.decode(output)
```
