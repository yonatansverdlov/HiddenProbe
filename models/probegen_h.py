"""ProbeGen-H: the official Kahana ProbeGen output spine + a compact late hidden-activation adapter.

Architecture (per the Track-B design review):

        probe latents z_p in R^32  --G(deep_linear_5)-->  probe images X
                                              |
                                       frozen target CNN f_theta
                                    /                              \
                        ordered logits O=[o_1..o_P]        hidden activation maps H
                                    |                               |
                     Kahana MLP body (build_mlp_head[:-1])   late spatial encoder
                                    |                               |
                                  q_K in R^256                    z_H in R^{d_h}
                                    \                               /
                                     z_cross = (A q_K) (x) (B z_H)   [low rank]
                                    \        |                     /
                                     FusionBody([q_K, z_H, z_cross]) -> h_feat
                                                |
                     y_hat = FinalLinear([q_K, h_feat])
                     init: q_K block = Kahana final classifier weights ; h_feat block = 0
                     => at init y_hat == y_K exactly ; hidden_mode="off" == Kahana exactly.

This is NOT a residual-correction model: there is no `score_K + alpha*delta`. The prediction is a
single fusion F([q_K, z_H, z_cross]); the Kahana-preserving init merely makes F reduce to the Kahana
head at step 0, after which all parameters (generator, latents, Kahana MLP, hidden encoder, fusion)
train jointly end-to-end. No scalar gate. No early branch. Target CNN parameters stay frozen.

Backbone is REUSED unchanged from the official code:
  - probegen_utils.DeepLinearGenerator  (5x ConvTranspose2d, deep-linear, width 16, latent 32)
  - probegen_utils.build_mlp_head        (6-layer MLP incl. output, width 256)
Hidden encoder is REUSED from the validated late model:
  - models.lr_dipt_capture.{capture_features, pad_batch, META_DIM}
  - models.lr_dipt_spatial.{SpatialTokenizer, LayerMetadataEncoder, LateBranch, tokenize_maps}
"""
from __future__ import annotations
import torch
import torch.nn as nn

from models.probegen_utils import DeepLinearGenerator, build_mlp_head, build_probe_source
from models.lr_dipt_capture import capture_features, pad_batch, META_DIM
from models.lr_dipt_spatial import SpatialTokenizer, LayerMetadataEncoder, LateBranch, tokenize_maps
from models.lr_dipt_lowrank import count_params


class ProbeGenH(nn.Module):
    def __init__(self,
                 n_out_probes: int = 128,        # OUTPUT probes -> logits -> q_K  (Kahana query budget)
                 n_hidden_probes: int = 128,     # HIDDEN probes -> activations -> z_H (extra queries)
                 n_classes: int = 10,
                 gen_latent_z: int = 32,
                 generator_width: int = 16,
                 gen_n_layers: int = 5,          # deep_linear_5 for Wild Park
                 mixer_hidden: int = 256,        # Kahana classifier width
                 mixer_n_layers: int = 6,        # 6 Linear layers incl. output
                 hidden_dim: int = 128,          # hidden branch width d (tokenizer/encoder)
                 z_hidden_dim: int = 128,        # z_H dimensionality
                 interaction_rank: int = 32,     # rank of z_cross
                 fusion_hidden: int = 256,       # FusionBody width
                 fusion_out: int = 128,          # h_feat width
                 use_hidden_statistics: bool = False,
                 hidden_mode: str = "on",        # "on" | "off"
                 probe_sharing: str = "shared",  # "shared" (PRIMARY: Q unique inputs, one forward gives
                                                 #   BOTH logits+hidden) | "separate" (ablation: X_out != X_hidden)
                 spatial_grid: int = 4,
                 models_c_in: int = 3,           # target-CNN input channels (3=CIFAR/WP, 1=grayscale SVHN-GS)
                 gen_type: str = "deep_linear_5", # probe generator (deep_linear_5=WP canonical, deep_linear_6=GS/Kahana)
                 hidden_agg: str = "neuron_collapse",  # hidden aggregation order: "neuron_collapse" (recipe ii,
                                                       #   DEFAULT, = current model) | "neuron_profile" (recipe i)
                 probe_mixer: str = "none"):           # recipe-(i) only: "none" | "attn" | "tokenmix"
        super().__init__()
        assert hidden_mode in ("on", "off")
        assert probe_sharing in ("shared", "separate")
        self.models_c_in = models_c_in
        self.hidden_mode = hidden_mode
        self.probe_sharing = probe_sharing
        # SHARED: the same Q probes feed both paths -> Q unique target-CNN queries. Do NOT make a 2nd bank.
        if probe_sharing == "shared":
            n_hidden_probes = n_out_probes
        self.n_out_probes = n_out_probes
        self.n_hidden_probes = n_hidden_probes
        self.n_classes = n_classes
        self.gen_latent_z = gen_latent_z
        self.spatial_grid = spatial_grid
        self.use_hidden_statistics = use_hidden_statistics

        # ---- Kahana backbone (official modules) --------------------------------------------------
        # ONE shared deep-linear generator; the output probe bank (also the hidden bank when shared).
        # CANONICAL Kahana probe source — the SAME factory official ProbeGen.py + ST gen use:
        # ProbeSource(latents=randn(Q,32,1,1) std 1.0, DeepLinearGenerator(3,32,16,5)), both trainable.
        self.gen_type = gen_type
        self.probe_source = build_probe_source(n_tokens=n_out_probes, models_c_in=models_c_in,
                                               gen_type=gen_type, gen_latent_z=gen_latent_z,
                                               generator_width=generator_width)
        # ordered concat of P_out*10 logits -> 6-layer MLP -> 1.  Split into body (=> q_K) + final (=> y_K).
        self.kahana_mlp = build_mlp_head(input_dim=n_out_probes * n_classes, hidden_dim=mixer_hidden,
                                         output_dim=1, n_layers=mixer_n_layers)
        self.d_qK = mixer_hidden

        # ---- Hidden adapter (validated late encoder), only wired when hidden_mode=="on" ----------
        if hidden_mode == "on":
            hd = hidden_dim
            if probe_sharing == "separate":       # ONLY separate mode gets a 2nd probe bank
                self.hidden_probe_source = build_probe_source(n_tokens=n_hidden_probes, models_c_in=models_c_in,
                                                              gen_type=gen_type, gen_latent_z=gen_latent_z,
                                                              generator_width=generator_width)
            self.meta_enc = LayerMetadataEncoder(META_DIM, d_meta=32)
            self.tokenizer = SpatialTokenizer(d_spatial=32, d_hidden=hd, d_meta=32, G=spatial_grid)
            self.late_branch = LateBranch(d=hd, P=n_hidden_probes, attn_rank=16, z_dim=z_hidden_dim,
                                          glob_dim=28, use_stats=use_hidden_statistics,
                                          hidden_agg=hidden_agg, probe_mixer=probe_mixer)
            # low-rank interaction z_cross = (A q_K) (x) (B z_H)
            self.inter_A = nn.Linear(self.d_qK, interaction_rank, bias=False)
            self.inter_B = nn.Linear(z_hidden_dim, interaction_rank, bias=False)
            # fusion: nonlinear body over [q_K, z_H, z_cross] -> h_feat ; final linear over [q_K, h_feat]
            d_body_in = self.d_qK + z_hidden_dim + interaction_rank
            self.fusion_body = nn.Sequential(
                nn.Linear(d_body_in, fusion_hidden), nn.ReLU(),
                nn.Linear(fusion_hidden, fusion_out))
            self.fusion_final = nn.Linear(self.d_qK + fusion_out, 1)
            self._init_kahana_preserving()

    # ------------------------------------------------------------------------------------------------
    def _init_kahana_preserving(self):
        """Make y_hat == y_K at initialization: final layer's q_K block = Kahana final weights,
        h_feat block = 0, bias = Kahana final bias. Then delta from hidden is exactly 0 at init."""
        kah_final = self.kahana_mlp[-1]                       # Linear(mixer_hidden, 1)
        with torch.no_grad():
            self.fusion_final.weight.zero_()
            self.fusion_final.weight[:, :self.d_qK].copy_(kah_final.weight)   # q_K block = Kahana
            self.fusion_final.weight[:, self.d_qK:].zero_()                    # h_feat block = 0
            self.fusion_final.bias.copy_(kah_final.bias)

    # ------------------------------------------------------------------------------------------------
    def generate_out_probes(self) -> torch.Tensor:
        return self.probe_source()                           # generator(latents) -> [P_out, 3, H, W]

    def generate_hidden_probes(self) -> torch.Tensor:
        return self.hidden_probe_source()                    # separate mode only -> [P_hidden, 3, H, W]

    def n_target_queries(self) -> dict:
        """UNIQUE target-CNN inputs per CNN = the efficiency metric of record.
        SHARED: one forward per probe gives both logits+hidden -> Q = n_out_probes.
        SEPARATE: X_out and X_hidden are distinct -> Q = n_out_probes + n_hidden_probes."""
        hp = self.n_hidden_probes if self.hidden_mode == "on" else 0
        if self.hidden_mode == "off" or self.probe_sharing == "shared":
            unique = self.n_out_probes           # off: output only; shared: same bank feeds both
        else:
            unique = self.n_out_probes + hp      # separate ablation
        return {"probe_sharing": self.probe_sharing, "out_probes": self.n_out_probes,
                "hidden_probes": hp, "unique_query_count": unique,
                "target_forward_input_count": unique, "total_queries": unique}

    def _kahana_qK_yK(self, ordered_logits: torch.Tensor):
        """ordered_logits [B, P, 10] -> q_K [B, d_qK], y_K [B]."""
        O = ordered_logits.reshape(ordered_logits.shape[0], -1)   # ordered concat P*10
        q_K = self.kahana_mlp[:-1](O)                              # penultimate representation
        y_K = self.kahana_mlp[-1](q_K).squeeze(-1)                # Kahana output-only prediction
        return q_K, y_K

    def _encode_hidden(self, recs_list, glob, cmask, lmask, rep, probe_mask=None):
        """recs already padded into batch tensors -> z_H [B, z_hidden_dim].
        probe_mask [B, P] (True=active) masks dropped probes in the probe-pool softmax and the
        probe-mean stats (active-count normalized). None = all probes active (UNCHANGED behavior)."""
        memb = self.meta_enc(recs_list["meta"])
        H = tokenize_maps(self.tokenizer, recs_list["cells"], glob, memb, cmask)
        return self.late_branch(H, glob, cmask, lmask, rep, probe_mask=probe_mask)

    # ------------------------------------------------------------------------------------------------
    def forward(self, nets, device="cuda", return_features: bool = False,
                zero_hidden: bool = False, shuffle_hidden: bool = False, hidden_perm=None,
                probe_mask=None):
        """nets: list of FROZEN target CNNs (ragged). Generates shared probes, runs each net,
        builds the Kahana output prediction and (if on) the hidden-augmented prediction.
        Diagnostics (§10): zero_hidden -> z_H:=0 ; shuffle_hidden -> permute z_H across the batch
        (q_K and labels stay attached to the original CNN).
        probe_dropout: probe_mask [B, P] bool (True=keep). When given, dropped probes' logit blocks
        contribute ZERO to the Kahana path and are excluded (-inf softmax / active-count mean) from the
        hidden path. probe_mask is None at eval and by default -> IDENTICAL to the pre-dropout model."""
        X_out = self.generate_out_probes()
        B = len(nets)

        def _mask_logits(lg):   # zero the dropped probes' 10-logit blocks (Kahana path only)
            return lg if probe_mask is None else lg * probe_mask[:, :, None].to(lg.dtype)

        if self.hidden_mode == "off":
            # Exact Kahana(P_out): only output-probe logits, no capture, no hidden modules.
            logits = torch.stack([net(X_out) for net in nets])                # [B, P_out, 10]
            q_K, y_K = self._kahana_qK_yK(_mask_logits(logits))
            out = {"prediction": y_K, "y_K": y_K, "q_K": q_K, "ordered_outputs": logits}
            return out if return_features else y_K

        if self.probe_sharing == "shared":
            # PRIMARY: ONE forward on the SAME Q probes yields BOTH logits and hidden (Q unique inputs).
            caps = [capture_features(net, X_out, G=self.spatial_grid,
                                     stats=self.use_hidden_statistics) for net in nets]
            logits = torch.stack([c[0] for c in caps])                        # [B, Q, 10]
            recs_l = [c[1] for c in caps]
            P_hid = self.n_out_probes
        else:
            # SEPARATE ablation: distinct output vs hidden probes -> P_out + P_hidden unique inputs.
            logits = torch.stack([net(X_out) for net in nets])                # [B, P_out, 10]
            X_hidden = self.generate_hidden_probes()
            recs_l = [capture_features(net, X_hidden, G=self.spatial_grid,
                                       stats=self.use_hidden_statistics)[1] for net in nets]
            P_hid = self.n_hidden_probes
        batch = pad_batch(recs_l, META_DIM, P_hid, G=self.spatial_grid,
                          device=device, dtype=X_out.dtype)
        q_K, y_K = self._kahana_qK_yK(_mask_logits(logits))
        z_H = self._encode_hidden(batch, batch["glob"], batch["cmask"], batch["lmask"], batch.get("rep"),
                                  probe_mask=probe_mask)
        if zero_hidden:
            z_H = torch.zeros_like(z_H)                                        # ablate hidden signal
        elif shuffle_hidden:
            perm = hidden_perm if hidden_perm is not None else torch.randperm(B, device=z_H.device)
            z_H = z_H[perm]                                                    # mismatch hidden with its CNN
        z_cross = self.inter_A(q_K) * self.inter_B(z_H)                        # [B, interaction_rank]
        h_feat = self.fusion_body(torch.cat([q_K, z_H, z_cross], dim=-1))     # [B, fusion_out]
        y_hat = self.fusion_final(torch.cat([q_K, h_feat], dim=-1)).squeeze(-1)
        out = {"prediction": y_hat, "y_K": y_K, "q_K": q_K, "z_H": z_H,
               "z_cross": z_cross, "ordered_outputs": logits}
        return out if return_features else y_hat

    # ------------------------------------------------------------------------------------------------
    def param_report(self) -> dict:
        """Decomposed trainable-parameter counts (Kahana vs hidden)."""
        r = {
            "kahana_out_latents": self.probe_source.input.numel(),
            "kahana_generator": count_params(self.probe_source.generator),
            "kahana_output_mlp": count_params(self.kahana_mlp),
        }
        if self.hidden_mode == "on":
            r["hidden_latents"] = self.hidden_probe_source.input.numel() if self.probe_sharing == "separate" else 0
            r["hidden_tokenizer"] = count_params(self.meta_enc) + count_params(self.tokenizer)
            r["hidden_encoder"] = count_params(self.late_branch)
            r["interaction"] = count_params(self.inter_A) + count_params(self.inter_B)
            r["fusion"] = count_params(self.fusion_body) + count_params(self.fusion_final)
        r["kahana_total"] = r["kahana_out_latents"] + r["kahana_generator"] + r["kahana_output_mlp"]
        r["hidden_total"] = sum(r.get(k, 0) for k in ("hidden_latents", "hidden_tokenizer",
                                                      "hidden_encoder", "interaction", "fusion"))
        r["total_trainable"] = sum(p.numel() for p in self.parameters() if p.requires_grad)
        r.update({f"queries_{k}": v for k, v in self.n_target_queries().items() if isinstance(v, int)})
        return r
