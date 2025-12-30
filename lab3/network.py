import torch
import torch.nn as nn
import math
from utils import *

### ==================================================
### (1) TO-DO: Model Definition
### ==================================================
### Base transformer layers in "Attention Is All You Need"
###   TransformerEncoderLayer
###   TransformerDecoderLayer
###   Positional encoding and input embedding
###   Note that you may need masks when implementing attention mechanism
###     Padding mask: prevent input from attending to padding tokens
###     Causal mask: prevent decoder input from attending to future input

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Linear projections for Q, K, V
        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        # Output projection
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def scaled_dot_product_attention(self, Q, K, V, mask=None):
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Apply softmax
        attention = torch.softmax(scores, dim=-1)
        attention = self.dropout(attention)

        # Compute output
        output = torch.matmul(attention, V)

        return output, attention

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # Linear projections and reshape to (batch_size, num_heads, seq_len, d_k)
        Q = self.W_q(query).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        K = self.W_k(key).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.W_v(value).view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Apply attention
        attn_output, _ = self.scaled_dot_product_attention(Q, K, V, mask)

        # Concatenate heads and apply final linear projection
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        output = self.W_o(attn_output)
        output = self.dropout(output)  # Apply dropout to output projection

        return output


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, src_mask=None):
        # Self-attention with residual connection and layer norm
        attn_output = self.self_attn(x, x, x, src_mask)
        x = self.norm1(x + self.dropout1(attn_output))

        # Feedforward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout2(ff_output))

        return x


class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-head self-attention (with causal mask)
        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Multi-head cross-attention (attend to encoder output)
        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        # Feedforward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

    def forward(self, x, enc_output, tgt_mask=None, src_mask=None):
        # Self-attention with residual connection and layer norm
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout1(self_attn_output))

        # Cross-attention with residual connection and layer norm
        cross_attn_output = self.cross_attn(x, enc_output, enc_output, src_mask)
        x = self.norm2(x + self.dropout2(cross_attn_output))

        # Feedforward with residual connection and layer norm
        ff_output = self.feed_forward(x)
        x = self.norm3(x + self.dropout3(ff_output))

        return x


class Transformer(nn.Module):
    def __init__(self, d_model, num_heads, num_encoder_layers, num_decoder_layers, d_ff, dropout=0.1):
        super().__init__()

        # Encoder stack
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_encoder_layers)
        ])

        # Decoder stack
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, num_heads, d_ff, dropout)
            for _ in range(num_decoder_layers)
        ])

    def generate_square_subsequent_mask(self, sz, device):
        mask = torch.tril(torch.ones(sz, sz, device=device))
        return mask.unsqueeze(0).unsqueeze(0)  # (1, 1, sz, sz)

    def create_padding_mask(self, seq):
        # seq: (batch_size, seq_len)
        mask = (seq != PAD_IDX).unsqueeze(1).unsqueeze(2)  # (batch_size, 1, 1, seq_len)
        return mask

    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Encode
        enc_output = self.encode(src, src_mask)

        # Decode
        dec_output = self.decode(tgt, enc_output, tgt_mask, src_mask)

        return dec_output

    def encode(self, src, src_mask=None):
        output = src
        for layer in self.encoder_layers:
            output = layer(output, src_mask)
        return output

    def decode(self, tgt, enc_output, tgt_mask=None, src_mask=None):
        output = tgt
        for layer in self.decoder_layers:
            output = layer(output, enc_output, tgt_mask, src_mask)
        return output


class PositionalEncoding(nn.Module):
    def __init__(self, emb_size, dropout=0.1, maxlen=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(maxlen, emb_size)
        position = torch.arange(0, maxlen, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, emb_size, 2).float() * (-math.log(10000.0) / emb_size))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Add batch dimension: (1, maxlen, emb_size)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size, emb_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens):
        # Scale embeddings by sqrt(emb_size) as in the paper
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)


# Seq2Seq Network (Top-level module)
# Hint: the masks should be carefully applied
class Seq2SeqNetwork(nn.Module):
    def __init__(self, 
                 num_encoder_layers,
                 num_decoder_layers,
                 emb_size,
                 nhead,
                 src_vocab_size,
                 tgt_vocab_size,
                 dim_feedforward,
                 dropout=0.1, 
                 device='cpu'):
        super().__init__()
        self.device=device
        self.transformer = Transformer(
            d_model=emb_size,
            num_heads=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            d_ff=dim_feedforward,
            dropout=dropout
        )
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self, src, tgt):
        # src shape: (batch_size, src_length)
        # tgt shape: (batch_size, tgt_length)
        # src_emb shape: (batch_size, src_length, emb_size)
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(tgt))

        # Create masks
        src_mask = self.transformer.create_padding_mask(src).to(self.device)
        tgt_padding_mask = self.transformer.create_padding_mask(tgt).to(self.device)
        tgt_causal_mask = self.transformer.generate_square_subsequent_mask(tgt.size(1), self.device)

        # Combine causal and padding masks for decoder (both must be 1 to attend)
        tgt_mask = tgt_causal_mask * tgt_padding_mask

        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask)
        return self.generator(outs)


### ==================================================
### (2) TO-DO: Inference Function
### ==================================================
### Finish the translate function
### Input: Chinese string
### Output: English string

def translate(model: torch.nn.Module, src_sentence: str, input_tokenizer, output_tokenizer,
              beam_width=3, length_penalty=None, max_len=128):
    """
    Length-Adaptive Beam Search

    Automatically adjusts length penalty based on source sentence length:
    - Short sentences (5 tokens): penalty=0.55 -> allows longer translations
    - Medium sentences (15 tokens): penalty=0.65 -> balanced
    - Long sentences (30+ tokens): penalty=0.80 -> prevents over-generation
    - Details are described in the report

    Formula: length_penalty = 0.5 + 0.01 * min(src_len, 30)

    This adaptive approach improves BLEU by matching target length to source length,
    avoiding the common issues of fixed penalties:
    - Fixed low penalty -> all translations too long
    - Fixed high penalty -> all translations too short

    Args:
        model: Seq2Seq Transformer model
        src_sentence: Source Chinese sentence (string)
        input_tokenizer: Chinese tokenizer
        output_tokenizer: English tokenizer
        beam_width: Number of beams to maintain (default: 3)
        length_penalty: If None (default), computed adaptively; if provided, uses fixed value
        max_len: Maximum generation length (default: 128)

    Returns:
        Translated English sentence (string)
    """
    model.eval()

    # Tokenize and prepare source sentence
    src_tokens = input_tokenizer.encode(src_sentence)
    src = torch.tensor(src_tokens).view(1, -1).to(model.device)

    # ADAPTIVE LENGTH PENALTY: Adjust based on source length
    if length_penalty is None:
        src_len = len(src_tokens)
        # Linear scaling: penalty increases with source length
        # Short sentences (len=5): penalty=0.55 → allows longer relative translations
        # Medium sentences (len=15): penalty=0.65 → balanced
        # Long sentences (len=30+): penalty=0.80 → prevents overly long translations
        length_penalty = 0.5 + 0.01 * min(src_len, 30)
        # Clamp to reasonable range
        length_penalty = max(0.5, min(0.8, length_penalty))

    # Encode source sentence once
    src_emb = model.positional_encoding(model.src_tok_emb(src))
    src_mask = model.transformer.create_padding_mask(src).to(model.device)
    memory = model.transformer.encode(src_emb, src_mask)

    # Initialize beam: (score, tokens)
    beams = [(0.0, [BOS_IDX])]

    with torch.no_grad():
        for i in range(max_len):
            candidates = []

            for score, tokens in beams:
                # If beam already ended, keep it as is
                if tokens[-1] == EOS_IDX:
                    candidates.append((score, tokens))
                    continue

                # Decode current sequence
                tgt = torch.tensor([tokens]).to(model.device)
                tgt_emb = model.positional_encoding(model.tgt_tok_emb(tgt))
                tgt_mask = model.transformer.generate_square_subsequent_mask(tgt.size(1), model.device)
                output = model.transformer.decode(tgt_emb, memory, tgt_mask, src_mask)
                logits = model.generator(output[:, -1, :])
                log_probs = torch.log_softmax(logits, dim=-1)

                # Get top beam_width tokens
                top_probs, top_indices = torch.topk(log_probs[0], beam_width)

                for prob, idx in zip(top_probs, top_indices):
                    new_score = score + prob.item()
                    new_tokens = tokens + [idx.item()]
                    candidates.append((new_score, new_tokens))

            # Select top beam_width candidates with length normalization
            candidates = sorted(candidates, key=lambda x: x[0] / (len(x[1]) ** length_penalty), reverse=True)
            beams = candidates[:beam_width]

            # Stop if all beams ended
            if all(tokens[-1] == EOS_IDX for _, tokens in beams):
                break

    # Return best beam
    best_tokens = max(beams, key=lambda x: x[0] / (len(x[1]) ** length_penalty))[1]
    return output_tokenizer.decode(best_tokens, skip_special_tokens=True)


### ==================================================
### (3) TO-DO: load model
### ==================================================
### You can modify the hyper parameter below
def load_model(MODEL_PATH=None):
    EMB_SIZE = 256
    NHEAD = 8
    FFN_HID_DIM = 1024
    NUM_ENCODER_LAYERS = 4
    NUM_DECODER_LAYERS = 4
    DROPOUT = 0.15
    SRC_VOCAB_SIZE = tokenizer_chinese().vocab_size
    TGT_VOCAB_SIZE = tokenizer_english().vocab_size

    model = Seq2SeqNetwork(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM, dropout=DROPOUT, device=DEVICE)
    if MODEL_PATH is not None:
        model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    return model
