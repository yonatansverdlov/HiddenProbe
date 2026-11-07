"""Shared hidden-layer selection for INR-based HiddenProbe experiments.

Indices are 0-based and refer to the hidden Linear layers of the *target*
network (not the output Linear). The default `None` means all hidden layers
if hidden features are enabled, otherwise no hidden layers.
"""
from __future__ import annotations

import argparse
from collections.abc import Iterable


def parse_bool_argument(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(
        "Expected a boolean: true/false, 1/0, yes/no, or on/off."
    )


def resolve_hidden_layer_ids(
    requested_ids: None | int | str | Iterable[int | str],
    total_hidden_layers: int,
    include_hidden_features: bool,
) -> tuple[int, ...]:
    """Resolve optional zero-based IDs without changing the target architecture.

    Examples for two hidden layers:
      None + include=True   -> (0, 1)
      None + include=False  -> ()
      [0] + include=True    -> (0,)
      [1] + include=True    -> (1,)
      [0, 1] + include=True -> (0, 1)

    CLI input accepts `--hidden_layer_ids 0 1`, `--hidden_layer_ids 0,1`,
    and `--hidden_layer_ids "[0, 1]"`. Duplicate and out-of-range IDs fail.
    """
    if total_hidden_layers < 0:
        raise ValueError("total_hidden_layers must be nonnegative")

    if requested_ids is None:
        return tuple(range(total_hidden_layers)) if include_hidden_features else ()

    if isinstance(requested_ids, (str, int)):
        requested_ids = [requested_ids]

    tokens = []
    for element in requested_ids:
        value = str(element).strip()
        if value.startswith("[") and value.endswith("]"):
            value = value[1:-1]
        tokens.extend(part.strip() for part in value.split(",") if part.strip())

    if len(tokens) == 1 and tokens[0].lower() in {"none", "null"}:
        return tuple(range(total_hidden_layers)) if include_hidden_features else ()

    if not include_hidden_features:
        raise ValueError(
            "Explicit --hidden_layer_ids requires --include_hidden_features=true; "
            "use --hidden_layer_ids None (or omit it) for output-only mode."
        )

    if not tokens:
        raise ValueError("Explicit --hidden_layer_ids must contain at least one index.")

    try:
        ids = tuple(int(token) for token in tokens)
    except ValueError as exc:
        raise ValueError(
            f"Invalid --hidden_layer_ids={tokens!r}; expected integer indices."
        ) from exc

    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate IDs in --hidden_layer_ids={ids}.")
    if any(index < 0 or index >= total_hidden_layers for index in ids):
        raise ValueError(
            f"Invalid hidden layer IDs {ids}: expected indices "
            f"from 0 to {total_hidden_layers - 1}."
        )

    return ids
