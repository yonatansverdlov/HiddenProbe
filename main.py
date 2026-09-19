"""HiddenProbe -- thin dispatcher over the three task trainers in models/ (no training logic here).

    python main.py cnn_zoo     <cnn-zoo trainer arguments...>            -> models/cnn_zoo_trainer.py
    python main.py transformer {train,count,cache,smoke,manifest,verify} <arguments...>
                                                                         -> models/transformer/train.py
    python main.py inr --method {hiddenprobe,probegen} <trainer arguments...>
                                                                         -> models/inr_hiddenprobe_trainer.py
                                                                            models/inr_probegen_trainer.py

The task must be the first argument (for `inr`, `--method` the second); everything after it is handed to the
selected trainer module unchanged (the module runs as __main__ with the remaining argv).
"""
import runpy
import sys

TASKS = {
    "cnn_zoo": "models.cnn_zoo_trainer",          # CNN-zoo accuracy regression (CIFAR10-GS/WP, SVHN, MNIST-GS, FMNIST-GS)
    "transformer": "models.transformer.train",    # transformer accuracy regression (MNIST-/AGNews-Transformers)
}
INR_METHODS = {
    "hiddenprobe": "models.inr_hiddenprobe_trainer",  # learned probes + hidden-response set-transformer head
    "probegen": "models.inr_probegen_trainer",        # ProbeGen baseline (output-only, --aggregator mlp)
}

USAGE = """usage: python main.py <task> [task arguments...]

tasks:
  cnn_zoo      CNN-zoo accuracy regression        python main.py cnn_zoo <trainer args>      (--help for the flags)
  transformer  transformer accuracy regression    python main.py transformer {train,count,cache,smoke,manifest,verify} <args>
  inr          CIFAR-10 INR classification        python main.py inr --method {hiddenprobe,probegen} <trainer args>

Example scripts for every task live in scripts/HiddenProbe and scripts/ProbeGen."""


def _usage(msg=None):
    if msg:
        print(f"main.py: {msg}", file=sys.stderr)
    print(USAGE, file=sys.stderr)
    sys.exit(2)


def main():
    argv = sys.argv[1:]
    if not argv or argv[0] in ("-h", "--help"):
        _usage()
    task, rest = argv[0], argv[1:]
    if task in TASKS:
        module = TASKS[task]
    elif task == "inr":
        if rest and rest[0] == "--method":
            if len(rest) < 2:
                _usage("--method needs a value")
            method, rest = rest[1], rest[2:]
        elif rest and rest[0].startswith("--method="):
            method, rest = rest[0].split("=", 1)[1], rest[1:]
        else:
            _usage("inr: --method {hiddenprobe,probegen} must follow the task")
        if method not in INR_METHODS:
            _usage(f"unknown inr method '{method}'")
        module = INR_METHODS[method]
    else:
        _usage(f"unknown task '{task}'")
    sys.argv = [sys.argv[0]] + rest
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
