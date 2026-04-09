"""
Seq2Seq models for SCAN compositional generalization.

SGS encoder + autoregressive decoder vs Transformer baselines.
Includes vanilla Transformer and Transformer with Relative Positional Encoding (RPE).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import gaussian_kernel_diag
from .rendering import render


class SGSSeq2Seq(nn.Module):
    """
    SGS encoder with autoregressive decoder for SCAN.

    Encoder: SGS rendering (Gaussian kernel + transmittance + multi-pass)
    Decoder: Simple GRU conditioned on rendered meaning
    """

    def __init__(self, in_vocab_size, out_vocab_size, d_model=128, n_passes=2):
        super().__init__()
        self.d_model = d_model

        # Encoder: Gaussian primitives in splatting space
        self.enc_embed_mu = nn.Embedding(in_vocab_size, d_model)
        self.enc_embed_feat = nn.Embedding(in_vocab_size, d_model)
        self.enc_log_var = nn.Parameter(torch.zeros(in_vocab_size, d_model))
        self.enc_raw_alpha = nn.Parameter(torch.zeros(in_vocab_size))
        self.log_tau = nn.Parameter(torch.tensor(float(d_model)).log())
        self.n_passes = n_passes

        # Per-pass update (encoder)
        if n_passes > 1:
            self.enc_mu_update = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model * 2, d_model), nn.Tanh())
                for _ in range(n_passes - 1)
            ])
            self.enc_alpha_gate = nn.ModuleList([
                nn.Sequential(nn.Linear(d_model * 2, 1), nn.Sigmoid())
                for _ in range(n_passes - 1)
            ])
            self.enc_ffn = nn.ModuleList([
                nn.Sequential(
                    nn.LayerNorm(d_model * 2),
                    nn.Linear(d_model * 2, d_model * 2), nn.GELU(),
                    nn.Linear(d_model * 2, d_model),
                )
                for _ in range(n_passes - 1)
            ])

        # Positional embedding for encoder
        self.enc_pos = nn.Embedding(128, d_model)
        nn.init.normal_(self.enc_pos.weight, std=0.01)

        # Decoder: GRU + output projection
        self.dec_embed = nn.Embedding(out_vocab_size, d_model)
        self.dec_gru = nn.GRU(d_model, d_model, num_layers=2, batch_first=True, dropout=0.1)
        self.dec_out = nn.Linear(d_model, out_vocab_size)

    @property
    def tau(self):
        return self.log_tau.exp()

    def encode(self, src_ids, src_mask):
        """Encode source sequence via SGS rendering."""
        batch_size, seq_len = src_ids.shape

        mu = self.enc_embed_mu(src_ids)
        features = self.enc_embed_feat(src_ids)
        log_var = self.enc_log_var[src_ids]
        alpha = torch.sigmoid(self.enc_raw_alpha[src_ids])

        # Positional modulation
        positions = torch.arange(seq_len, device=src_ids.device).clamp(max=127)
        mu = mu + self.enc_pos(positions).unsqueeze(0)
        alpha = alpha * src_mask.float()

        # Multi-pass rendering
        for p in range(self.n_passes):
            masked_mu = mu * src_mask.float().unsqueeze(-1)
            lengths = src_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            query = masked_mu.sum(dim=1) / lengths

            K = gaussian_kernel_diag(query, mu, log_var, self.tau)
            K = K * src_mask.float()
            meaning, _ = render(features, alpha, K)

            if p < self.n_passes - 1:
                context = meaning.unsqueeze(1).expand_as(features)
                combined = torch.cat([features, context], dim=-1)
                mu = mu + self.enc_mu_update[p](combined)
                alpha = alpha * self.enc_alpha_gate[p](combined).squeeze(-1)
                features = features + self.enc_ffn[p](combined)
                alpha = alpha * src_mask.float()

        return meaning  # [batch, d_model]

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask):
        """
        Teacher-forced forward pass.

        Args:
            src_ids: [batch, src_len]
            src_mask: [batch, src_len]
            tgt_ids: [batch, tgt_len] — includes <BOS>, target tokens, <EOS>
            tgt_mask: [batch, tgt_len]

        Returns:
            logits: [batch, tgt_len-1, out_vocab_size] — predict each token from previous
        """
        # Encode source
        enc_out = self.encode(src_ids, src_mask)  # [batch, d_model]

        # Decode: teacher forcing
        tgt_embed = self.dec_embed(tgt_ids[:, :-1])  # [batch, tgt_len-1, d_model]

        # Init hidden state from encoder output
        h0 = enc_out.unsqueeze(0).expand(2, -1, -1).contiguous()  # [2, batch, d_model]
        dec_out, _ = self.dec_gru(tgt_embed, h0)  # [batch, tgt_len-1, d_model]

        logits = self.dec_out(dec_out)  # [batch, tgt_len-1, out_vocab_size]
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_mask, max_len=100, bos_id=1, eos_id=2):
        """Greedy autoregressive decoding."""
        batch_size = src_ids.shape[0]
        enc_out = self.encode(src_ids, src_mask)

        h = enc_out.unsqueeze(0).expand(2, -1, -1).contiguous()
        input_id = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        outputs = []
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)

        for _ in range(max_len):
            embed = self.dec_embed(input_id)
            out, h = self.dec_gru(embed, h)
            logits = self.dec_out(out[:, -1, :])
            next_id = logits.argmax(dim=-1)  # [batch]
            outputs.append(next_id)
            input_id = next_id.unsqueeze(1)
            finished = finished | (next_id == eos_id)
            if finished.all():
                break

        if outputs:
            return torch.stack(outputs, dim=1)  # [batch, decode_len]
        return torch.zeros(batch_size, 0, dtype=torch.long, device=src_ids.device)


class TransformerSeq2Seq(nn.Module):
    """
    Standard Transformer encoder-decoder baseline for SCAN.
    Matched parameter count with SGS.
    """

    def __init__(self, in_vocab_size, out_vocab_size, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.d_model = d_model

        self.enc_embed = nn.Embedding(in_vocab_size, d_model)
        self.dec_embed = nn.Embedding(out_vocab_size, d_model)
        self.enc_pos = nn.Embedding(128, d_model)
        self.dec_pos = nn.Embedding(128, d_model)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, d_model * 4,
                                                    dropout=0.1, batch_first=True)
        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead, d_model * 4,
                                                    dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers)
        self.out_proj = nn.Linear(d_model, out_vocab_size)

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask):
        src_len = src_ids.shape[1]
        tgt_len = tgt_ids.shape[1] - 1

        # Encoder
        src_pos = torch.arange(src_len, device=src_ids.device).clamp(max=127)
        src_embed = self.enc_embed(src_ids) + self.enc_pos(src_pos).unsqueeze(0)
        src_key_padding_mask = ~src_mask.bool()
        memory = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask)

        # Decoder (teacher forcing)
        tgt_pos = torch.arange(tgt_len, device=tgt_ids.device).clamp(max=127)
        tgt_embed = self.dec_embed(tgt_ids[:, :-1]) + self.dec_pos(tgt_pos).unsqueeze(0)
        tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=tgt_ids.device)
        dec_out = self.decoder(tgt_embed, memory, tgt_mask=tgt_causal_mask,
                               memory_key_padding_mask=src_key_padding_mask)

        logits = self.out_proj(dec_out)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_mask, max_len=100, bos_id=1, eos_id=2):
        batch_size = src_ids.shape[0]
        src_len = src_ids.shape[1]

        src_pos = torch.arange(src_len, device=src_ids.device).clamp(max=127)
        src_embed = self.enc_embed(src_ids) + self.enc_pos(src_pos).unsqueeze(0)
        src_key_padding_mask = ~src_mask.bool()
        memory = self.encoder(src_embed, src_key_padding_mask=src_key_padding_mask)

        input_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=src_ids.device)
        finished = torch.zeros(batch_size, dtype=torch.bool, device=src_ids.device)
        outputs = []

        for step in range(max_len):
            tgt_len = input_ids.shape[1]
            tgt_pos = torch.arange(tgt_len, device=src_ids.device).clamp(max=127)
            tgt_embed = self.dec_embed(input_ids) + self.dec_pos(tgt_pos).unsqueeze(0)
            tgt_causal_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len, device=src_ids.device)
            dec_out = self.decoder(tgt_embed, memory, tgt_mask=tgt_causal_mask,
                                   memory_key_padding_mask=src_key_padding_mask)
            logits = self.out_proj(dec_out[:, -1, :])
            next_id = logits.argmax(dim=-1)
            outputs.append(next_id)
            input_ids = torch.cat([input_ids, next_id.unsqueeze(1)], dim=1)
            finished = finished | (next_id == eos_id)
            if finished.all():
                break

        if outputs:
            return torch.stack(outputs, dim=1)
        return torch.zeros(batch_size, 0, dtype=torch.long, device=src_ids.device)


# ═══════════════════════════════════════════════════════════
# Transformer with Relative Positional Encoding (RPE)
# ═══════════════════════════════════════════════════════════

class RPEMultiHeadAttention(nn.Module):
    """Multi-head attention with learned relative positional bias per head."""

    def __init__(self, d_model, nhead, dropout=0.1, max_rel=128):
        super().__init__()
        self.nhead = nhead
        self.head_dim = d_model // nhead
        self.d_model = d_model
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)
        self.max_rel = max_rel
        self.rel_bias = nn.Embedding(2 * max_rel + 1, nhead)

    def forward(self, query, key, value, attn_mask=None, key_padding_mask=None):
        B, T, _ = query.shape
        S = key.shape[1]
        q = self.q(query).view(B, T, self.nhead, self.head_dim).transpose(1, 2)
        k = self.k(key).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        v = self.v(value).view(B, S, self.nhead, self.head_dim).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        rel = (torch.arange(T, device=query.device).unsqueeze(1) -
               torch.arange(S, device=query.device).unsqueeze(0))
        rel = rel.clamp(-self.max_rel, self.max_rel) + self.max_rel
        scores = scores + self.rel_bias(rel).permute(2, 0, 1).unsqueeze(0)
        if attn_mask is not None:
            scores = scores + attn_mask.unsqueeze(0).unsqueeze(0)
        if key_padding_mask is not None:
            scores = scores.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
        w = self.drop(F.softmax(scores, dim=-1))
        out = (w @ v).transpose(1, 2).contiguous().view(B, T, self.d_model)
        return self.o(out)


class RPEEncLayer(nn.Module):
    def __init__(self, d, nh, ff, dp=0.1):
        super().__init__()
        self.attn = RPEMultiHeadAttention(d, nh, dp)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dp), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.dp = nn.Dropout(dp)

    def forward(self, x, kpm=None):
        x = self.n1(x + self.dp(self.attn(x, x, x, key_padding_mask=kpm)))
        return self.n2(x + self.dp(self.ffn(x)))


class RPEDecLayer(nn.Module):
    def __init__(self, d, nh, ff, dp=0.1):
        super().__init__()
        self.sa = RPEMultiHeadAttention(d, nh, dp)
        self.ca = RPEMultiHeadAttention(d, nh, dp)
        self.ffn = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Dropout(dp), nn.Linear(ff, d))
        self.n1 = nn.LayerNorm(d)
        self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.dp = nn.Dropout(dp)

    def forward(self, tgt, mem, tgt_mask=None, mem_kpm=None):
        tgt = self.n1(tgt + self.dp(self.sa(tgt, tgt, tgt, attn_mask=tgt_mask)))
        tgt = self.n2(tgt + self.dp(self.ca(tgt, mem, mem, key_padding_mask=mem_kpm)))
        return self.n3(tgt + self.dp(self.ffn(tgt)))


class TransformerRPESeq2Seq(nn.Module):
    """Transformer + Relative Positional Encoding for SCAN. No absolute pos embed."""

    def __init__(self, in_vocab_size, out_vocab_size, d_model=128, nhead=4,
                 num_encoder_layers=2, num_decoder_layers=2):
        super().__init__()
        self.d_model = d_model
        self.enc_emb = nn.Embedding(in_vocab_size, d_model)
        self.dec_emb = nn.Embedding(out_vocab_size, d_model)
        self.enc = nn.ModuleList([RPEEncLayer(d_model, nhead, d_model * 4) for _ in range(num_encoder_layers)])
        self.dec = nn.ModuleList([RPEDecLayer(d_model, nhead, d_model * 4) for _ in range(num_decoder_layers)])
        self.proj = nn.Linear(d_model, out_vocab_size)

    def _enc(self, src_ids, src_mask):
        h = self.enc_emb(src_ids)
        kpm = ~src_mask.bool()
        for layer in self.enc:
            h = layer(h, kpm=kpm)
        return h, kpm

    def forward(self, src_ids, src_mask, tgt_ids, tgt_mask):
        tl = tgt_ids.shape[1] - 1
        mem, kpm = self._enc(src_ids, src_mask)
        h = self.dec_emb(tgt_ids[:, :-1])
        cm = torch.triu(torch.full((tl, tl), float('-inf'), device=src_ids.device), diagonal=1)
        for layer in self.dec:
            h = layer(h, mem, tgt_mask=cm, mem_kpm=kpm)
        return self.proj(h)

    @torch.no_grad()
    def greedy_decode(self, src_ids, src_mask, max_len=100, bos_id=1, eos_id=2):
        B = src_ids.shape[0]
        mem, kpm = self._enc(src_ids, src_mask)
        ids = torch.full((B, 1), bos_id, dtype=torch.long, device=src_ids.device)
        done = torch.zeros(B, dtype=torch.bool, device=src_ids.device)
        outs = []
        for _ in range(max_len):
            tl = ids.shape[1]
            h = self.dec_emb(ids)
            cm = torch.triu(torch.full((tl, tl), float('-inf'), device=src_ids.device), diagonal=1)
            for layer in self.dec:
                h = layer(h, mem, tgt_mask=cm, mem_kpm=kpm)
            nxt = self.proj(h[:, -1, :]).argmax(-1)
            outs.append(nxt)
            ids = torch.cat([ids, nxt.unsqueeze(1)], dim=1)
            done = done | (nxt == eos_id)
            if done.all():
                break
        if outs:
            return torch.stack(outs, dim=1)
        return torch.zeros(B, 0, dtype=torch.long, device=src_ids.device)
