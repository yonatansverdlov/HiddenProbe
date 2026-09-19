"""Response-only predictor (spec §8, phase-2): tokenizer + cross-response Transformer + scalar head.
Q-parameterized (Q up to 128); readout in {cls, multi=concat[CLS,mean,max], pma=K learned pooling seeds}.
Tokens = 3 hidden channels * Q + Q logits + 1 CLS = 4Q+1. Predictor sees ONLY responses + fixed semantic IDs.
"""
from __future__ import annotations
import torch
import torch.nn as nn

W = 192
DSTATE = 32
H_CH = 3
N_BLOCKS = 3
POOL_HEADS = 6
PRED_HEADS = 6


class ResponseTokenizer(nn.Module):
    def __init__(self, n_probes: int, nmax: int, n_classes: int, n_routes: int, w: int = W):
        super().__init__(); self.w = w
        self.hid_proj = nn.Linear(DSTATE, w)
        self.logit_proj = nn.Linear(n_classes, w)
        self.token_pos_emb = nn.Parameter(torch.zeros(nmax, w))
        self.pool_queries = nn.Parameter(torch.zeros(2, w))
        self.pool_mha = nn.MultiheadAttention(w, POOL_HEADS, batch_first=True)
        self.pool_out = nn.Linear(2 * w, w)
        self.stat_proj = nn.Linear(2, w)
        self.tok_ln = nn.LayerNorm(w)
        self.probe_emb = nn.Parameter(torch.zeros(n_probes, w))
        self.channel_emb = nn.Parameter(torch.zeros(H_CH + 1, w))
        self.route_emb = nn.Parameter(torch.zeros(n_routes, w))
        for p in (self.token_pos_emb, self.pool_queries, self.probe_emb, self.channel_emb, self.route_emb):
            nn.init.normal_(p, std=0.02)

    def _pool(self, tok):
        B = tok.shape[0]
        q = self.pool_queries.unsqueeze(0).expand(B, -1, -1)
        pooled, _ = self.pool_mha(q, tok, tok, need_weights=False)
        return self.pool_out(pooled.reshape(B, 2 * self.w))

    def tokenize_hidden(self, states, channel, probe_ids, route_ids):
        n = states.shape[-2]
        summ = self._pool(self.hid_proj(states) + self.token_pos_emb[:n])
        mean = states.mean(dim=(-1, -2)); log_rms = 0.5 * torch.log((states ** 2).mean(dim=(-1, -2)) + 1e-8)
        summ = self.tok_ln(summ + self.stat_proj(torch.stack([mean, log_rms], dim=-1)))
        return summ + self.probe_emb[probe_ids] + self.channel_emb[channel] + self.route_emb[route_ids]

    def tokenize_logits(self, logits, probe_ids, route_ids):
        return self.tok_ln(self.logit_proj(logits)) + self.probe_emb[probe_ids] + self.channel_emb[H_CH] + self.route_emb[route_ids]


class CrossResponsePredictor(nn.Module):
    def __init__(self, ffn: int, readout: str = "cls", pma_seeds: int = 1, w: int = W, dropout: float = 0.1):
        super().__init__(); self.readout = readout; self.pma_seeds = pma_seeds
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(w, PRED_HEADS, ffn, dropout=dropout, activation="gelu",
                                       batch_first=True, norm_first=True) for _ in range(N_BLOCKS)])
        self.final_norm = nn.LayerNorm(w)
        if readout == "pma":
            self.pma_seed = nn.Parameter(torch.zeros(1, pma_seeds, w)); nn.init.normal_(self.pma_seed, std=0.02)
            self.pma = nn.MultiheadAttention(w, PRED_HEADS, batch_first=True); ro = pma_seeds * w
        elif readout == "multi":
            self.cls = nn.Parameter(torch.zeros(1, 1, w)); nn.init.normal_(self.cls, std=0.02); ro = 3 * w
        else:
            self.cls = nn.Parameter(torch.zeros(1, 1, w)); nn.init.normal_(self.cls, std=0.02); ro = w
        self.head = nn.Sequential(nn.Linear(ro, w), nn.GELU(), nn.Linear(w, 1))

    def forward(self, tokens):
        B = tokens.shape[0]
        if self.readout == "pma":
            x = self.final_norm(self._run(tokens))
            pooled, _ = self.pma(self.pma_seed.expand(B, -1, -1), x, x)
            return self.head(pooled.reshape(B, -1)).squeeze(-1)
        x = self._run(torch.cat([self.cls.expand(B, -1, -1), tokens], dim=1))
        x = self.final_norm(x)
        if self.readout == "multi":
            return self.head(torch.cat([x[:, 0], x.mean(1), x.max(1).values], dim=-1)).squeeze(-1)
        return self.head(x[:, 0]).squeeze(-1)

    def _run(self, x):
        for b in self.blocks:
            x = b(x)
        return x
