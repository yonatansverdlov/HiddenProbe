"""PAT-CNN accuracy-regression metrics (PAT_CNN_SPEC ruling 4).

The model predicts y_hat in LOGIT space; the label is y = logit(clip(acc, 1e-3, 1-1e-3)).
  train loss        = MSE(y_hat, y)                       [logit-space MSE]
  headline error    = MSE(sigmoid(y_hat), sigmoid(y))     [ACCURACY-space MSE]
  metric of record  = Kendall tau-b between y_hat and acc [rank]
All three reported on val+test each eval, per-seed; mean±std at finals.
"""
import torch

try:
    from scipy.stats import kendalltau as _kendalltau
except Exception:  # pragma: no cover
    _kendalltau = None

CLIP_LO, CLIP_HI = 1e-3, 1.0 - 1e-3


def acc_to_logit(acc: torch.Tensor) -> torch.Tensor:
    a = acc.clamp(CLIP_LO, CLIP_HI)
    return torch.log(a) - torch.log1p(-a)


def logit_space_mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((y_hat - y) ** 2).item()


def acc_space_mse(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((torch.sigmoid(y_hat) - torch.sigmoid(y)) ** 2).item()


def acc_space_mae(y_hat: torch.Tensor, y: torch.Tensor) -> float:
    return torch.mean((torch.sigmoid(y_hat) - torch.sigmoid(y)).abs()).item()


def kendall_tau_b(y_hat: torch.Tensor, acc: torch.Tensor) -> float:
    yh = y_hat.detach().flatten().cpu().numpy()
    a = acc.detach().flatten().cpu().numpy()
    if _kendalltau is None:
        raise RuntimeError("scipy required for Kendall tau")
    tau, _ = _kendalltau(yh, a, variant="b")
    return float(tau)


def all_metrics(y_hat: torch.Tensor, acc: torch.Tensor, space: str = "logit") -> dict:
    """y_hat: model score; acc: true accuracy in [0,1].
    space='logit': y_hat is a logit (Kahana official regresses on RAW accuracy -> use space='raw').
    space='raw':   y_hat is a DIRECT accuracy prediction (acc_mse/acc_mae computed without sigmoid).
    tau is rank-based so it is identical either way."""
    acc = acc.to(y_hat.dtype).to(y_hat.device)
    if space == "raw":
        d = y_hat - acc
        return {
            "logit_mse": float("nan"),
            "acc_mse": torch.mean(d ** 2).item(),
            "acc_mae": torch.mean(d.abs()).item(),
            "tau_b": kendall_tau_b(y_hat, acc),
        }
    y = acc_to_logit(acc)
    return {
        "logit_mse": logit_space_mse(y_hat, y),
        "acc_mse": acc_space_mse(y_hat, y),
        "acc_mae": acc_space_mae(y_hat, y),
        "tau_b": kendall_tau_b(y_hat, acc),
    }
