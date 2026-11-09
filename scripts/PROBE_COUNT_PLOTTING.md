# Paper-style probe-count plots (Figure 3)

The plotting script reads only **existing benchmark summary CSV files**.
No training or data downloads are performed.

Colors are **exactly** the ones in the paper's vector Figure 3:
ProbeGen: **#D55E00**, dashed orange line with circle markers;
HiddenProbe: **#0072B2**, solid blue line with square markers.
The figures retain serif labels, light-gray horizontal grids, equally
spaced N=16,32,64,128,256, one shared legend, and Figure 3 panel labels.

Unlike the current seed-0 paper plot, the new plot uses mean over the
completed 3 or 5 seeds and **±SEM** error bars by default.

## Plot the current 3-seed experiments

From the repository root:

~~~bash
git pull
python scripts/plot_probe_count.py
~~~

This automatically finds the method-specific files such as

~~~text
checkpoints/probe_count_3seeds/hiddenprobe/mnist_transformer/summary.csv
checkpoints/probe_count_3seeds/probegen/mnist_transformer/summary.csv
~~~

and the corresponding FMNIST-INR and SVHN files, if both methods exist.

To make the MNIST-Transformers panel by itself:

~~~bash
python scripts/plot_probe_count.py --datasets mnist_transformer
~~~

Outputs are under **figs/** in both vector PDF and 300-dpi PNG:

- figs/probe_count_3seeds.pdf (combined panels for available datasets)
- figs/probe_count_3seeds_mnist_transformer.pdf (individual)
- figs/probe_count_3seeds_fmnist_inr.pdf (individual, if available)
- figs/probe_count_3seeds_svhn.pdf (individual, if available)

## Plot the 5-seed experiment or change format

~~~bash
python scripts/plot_probe_count.py --input-root checkpoints/probe_count_5seeds
python scripts/plot_probe_count.py --layout combined
python scripts/plot_probe_count.py --layout individual
python scripts/plot_probe_count.py --uncertainty none
python scripts/plot_probe_count.py --uncertainty sd
~~~

The five-seed version understands the combined
checkpoints/probe_count_5seeds/DATASET/summary.csv format.
The no-error-bars flag reproduces the original visual convention.
The parser checks the expected metric (accuracy or Kendall tau),
duplicate probe counts, valid values, and consistent seed counts.
Missing probes are never interpolated.

## Dependencies and tests

~~~bash
python -m pip install matplotlib
python -m unittest discover -s tests -p 'test_plot_probe_count.py'
~~~
