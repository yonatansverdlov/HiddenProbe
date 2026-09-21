import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import numpy as np


# ============================================================
# Helpers
# ============================================================

def rounded_box(ax, x, y, w, h, text,
                facecolor,
                edgecolor="black",
                fontsize=16,
                textcolor="white",
                lw=1.2,
                radius=0.07,
                zorder=3):

    box = FancyBboxPatch(
        (x, y),
        w, h,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        linewidth=lw,
        edgecolor=edgecolor,
        facecolor=facecolor,
        zorder=zorder
    )
    ax.add_patch(box)

    if text:
        ax.text(
            x + w / 2,
            y + h / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            color=textcolor,
            zorder=zorder + 1
        )

    return box


def arrow(ax, start, end,
          color="black",
          lw=1.7,
          mutation_scale=18,
          zorder=5):

    a = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=mutation_scale,
        linewidth=lw,
        color=color,
        zorder=zorder
    )
    ax.add_patch(a)


# ============================================================
# Figure
# ============================================================

fig, ax = plt.subplots(figsize=(20, 9))

ax.set_xlim(0, 20.5)
ax.set_ylim(1.2, 10)

ax.axis("off")


# ============================================================
# Colors
# ============================================================

probe_color = "#0DA5D9"

network_color = "#1475E8"
connection_blue = "#2383EF"

teal = "#16B5BA"
blue = "#3989ED"
purple = "#9258D7"
orange = "#F27843"

pink = "#F09AAF"


# ============================================================
# PROBES
# ============================================================

probe_y = 8.65

probe_w = 1.15
probe_h = 0.85

offset = 0.5

p1_x = 0.95 + offset
p2_x = 2.55 + offset
p3_x = 4.15 + offset

# Small dots between p3 and pN
probe_dots_x = 6.15

# Final probe
pN_x = 6.5


# p1
rounded_box(
    ax,
    p1_x, probe_y,
    probe_w, probe_h,
    r"$p_1$",
    probe_color,
    fontsize=21
)


# p2
rounded_box(
    ax,
    p2_x, probe_y,
    probe_w, probe_h,
    r"$p_2$",
    probe_color,
    fontsize=21
)


# p3
rounded_box(
    ax,
    p3_x, probe_y,
    probe_w, probe_h,
    r"$p_3$",
    probe_color,
    fontsize=21
)


# ...
ax.text(
    probe_dots_x,
    probe_y + probe_h / 2,
    r"$\cdots$",
    fontsize=15,
    ha="center",
    va="center"
)


# pN
rounded_box(
    ax,
    pN_x, probe_y,
    probe_w, probe_h,
    r"$p_N$",
    probe_color,
    fontsize=21
)


# ============================================================
# Arrow probes -> network
# ============================================================

# Slightly after p2
arrow_x = p2_x + probe_w + 0.18

arrow(
    ax,
    (arrow_x, 8.45),
    (arrow_x, 7.75),
    color="#333333",
    lw=1.7
)


# ============================================================
# ROTATED NEURAL NETWORK
# ============================================================

neuron_xs = np.linspace(1.75, 7.0, 5)

layer_ys = [
    7.10,
    5.70,
    4.30,
    2.90
]

radius = 0.23


# ============================================================
# Dense connections
# ============================================================

for layer in range(len(layer_ys) - 1):

    y1 = layer_ys[layer]
    y2 = layer_ys[layer + 1]

    for x1 in neuron_xs:
        for x2 in neuron_xs:

            ax.plot(
                [x1, x2],
                [y1, y2],
                color=connection_blue,
                linewidth=1.0,
                alpha=0.82,
                zorder=1
            )


# ============================================================
# Neurons
# ============================================================

for y in layer_ys:
    for x in neuron_xs:

        ax.add_patch(
            Circle(
                (x, y),
                radius,
                facecolor=network_color,
                edgecolor="black",
                linewidth=1.25,
                zorder=3
            )
        )


# ============================================================
# NETWORK OUTPUT
# ============================================================

output_x = np.mean(neuron_xs)

arrow(
    ax,
    (output_x, layer_ys[-1] - 0.35),
    (output_x, layer_ys[-1] - 1.05),
    color="#333333",
    lw=1.7
)

ax.text(
    output_x,
    layer_ys[-1] - 1.30,
    "Output",
    fontsize=14,
    ha="center",
    va="center",
    color="black"
)


# ============================================================
# FEATURES
# ============================================================

feature_start_x = 9.1

feature_w = 1.55
feature_h = 0.68

feature_colors = [
    teal,
    blue,
    purple,
    orange
]


feature_xs = [
    feature_start_x,
    feature_start_x + 1.80,
    feature_start_x + 3.60,
    feature_start_x + 4.35
]

dots_x = feature_start_x + 3.82


# ============================================================
# Feature rows
# ============================================================

rows = [

    [
        r"$x_1(p_1,\theta)$",
        r"$x_1(p_2,\theta)$",
        r"$x_1(p_N,\theta)$"
    ],

    [
        r"$x_2(p_1,\theta)$",
        r"$x_2(p_2,\theta)$",
        r"$x_2(p_N,\theta)$"
    ],

    [
        r"$x_3(p_1,\theta)$",
        r"$x_3(p_2,\theta)$",
        r"$x_3(p_N,\theta)$"
    ],

    [
        r"$x_L(p_1,\theta)$",
        r"$x_L(p_2,\theta)$",
        r"$x_L(p_N,\theta)$"
    ]
]


# ============================================================
# Draw feature rows
# ============================================================

for i, y in enumerate(layer_ys):

    color = feature_colors[i]

    # Network layer -> corresponding features
    arrow(
        ax,
        (7.45, y),
        (8.85, y),
        color=color,
        lw=2.0,
        mutation_scale=17
    )


    # p1
    rounded_box(
        ax,
        feature_xs[0],
        y - feature_h / 2,
        feature_w,
        feature_h,
        rows[i][0],
        color,
        edgecolor="#555555",
        fontsize=14,
        lw=0.7
    )


    # p2
    rounded_box(
        ax,
        feature_xs[1],
        y - feature_h / 2,
        feature_w,
        feature_h,
        rows[i][1],
        color,
        edgecolor="#555555",
        fontsize=14,
        lw=0.7
    )


    # ...
    ax.text(
        dots_x,
        y,
        r"$\cdots$",
        fontsize=24,
        ha="center",
        va="center"
    )


    # pN
    rounded_box(
        ax,
        feature_xs[3],
        y - feature_h / 2,
        feature_w,
        feature_h,
        rows[i][2],
        color,
        edgecolor="#555555",
        fontsize=14,
        lw=0.7
    )


# ============================================================
# CLASSIFIER
# ============================================================

classifier_x = 18.2
classifier_y = 3.55

classifier_w = 1.75
classifier_h = 2.8


# Feature representation -> MLP
arrow(
    ax,
    (15.30, 5.00),
    (18.00, 5.00),
    color="#333333",
    lw=1.7
)


rounded_box(
    ax,
    classifier_x,
    classifier_y,
    classifier_w,
    classifier_h,
    "",
    pink,
    edgecolor="black",
    lw=1.4
)


ax.text(
    classifier_x + classifier_w / 2,
    classifier_y + classifier_h / 2 + 0.22,
    "Classifier",
    fontsize=18,
    ha="center",
    va="center",
    color="black"
)


ax.text(
    classifier_x + classifier_w / 2,
    classifier_y + classifier_h / 2 - 0.30,
    r"$\phi_{\mathrm{MLP}}$",
    fontsize=18,
    ha="center",
    va="center",
    color="black"
)


# ============================================================
# Save
# ============================================================

plt.tight_layout()

plt.savefig(
    "hiddenprobe_rotated_v5.png",
    dpi=300,
    bbox_inches="tight"
)

plt.savefig(
    "hiddenprobe_rotated_v5.pdf",
    bbox_inches="tight"
)

plt.show()