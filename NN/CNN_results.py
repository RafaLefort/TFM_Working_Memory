import pickle

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.legend_handler import HandlerTuple
import seaborn as sns

with open("results_bsl.pkl", "rb") as f:
    results_bsl = pickle.load(f)

with open("results_sens.pkl", "rb") as f:
    results_sens = pickle.load(f)

with open("results_delay.pkl", "rb") as f:
    results_delay = pickle.load(f)

sns.set(style="whitegrid", font_scale=1.2)

sns.set(style="whitegrid", font_scale=1.3)

# =========================
# Data
# =========================

acc_bsl = [res["accuracy"] * 100 for res in results_bsl.values()]
acc_sens = [res["accuracy"] * 100 for res in results_sens.values()]
acc_delay = [res["accuracy"] * 100 for res in results_delay.values()]

data = acc_bsl + acc_sens + acc_delay
labels = (["Baseline"] * len(acc_bsl) +
          ["Sensory"] * len(acc_sens) +
          ["Delay"] * len(acc_delay))

# =========================
# Plot
# =========================

plt.figure(figsize=(8,6))

palette = ["#9e9e9e", "#2ca02c", "#9c27b0"]  # softer colors

sns.violinplot(x=labels, y=data, hue=labels,
               legend=False, inner=None,
               palette=palette, linewidth=1.5)

sns.stripplot(x=labels, y=data,
              color="black", size=4,
              jitter=0.15, alpha=0.6)

# =========================
# Means
# =========================

means = [np.mean(acc_bsl), np.mean(acc_sens), np.mean(acc_delay)]

for i, (accs, m, color) in enumerate(zip([acc_bsl, acc_sens, acc_delay], means, palette)):
    # thinner mean line
    plt.hlines(m, i-0.15, i+0.15, colors="black", linewidth=2)

    # move text clearly outside
    plt.text(i + 0.42, m,
             f"{m:.0f}%",
             color=color,
             fontsize=13,
             fontweight='bold',
             va='center')

# =========================
# Chance level
# =========================

plt.axhline(33.33, linestyle="--", color="black", linewidth=1)

# =========================
# Legend
# =========================

palette = ["#9e9e9e", "#2ca02c", "#9c27b0"]
dot_baseline = Line2D([0], [0], marker='o', color=palette[0], linestyle='None', markersize=6)
dot_sensory = Line2D([0], [0], marker='o', color=palette[1], linestyle='None', markersize=6)
dot_delay = Line2D([0], [0], marker='o', color=palette[2], linestyle='None', markersize=6)

legend_elements = [Line2D([0], [0], color='black', lw=2, label='Mean accuracy'),
    (dot_baseline, dot_sensory, dot_delay),
    Line2D([0], [0], color='black', linestyle='--', lw=1, label='Chance level')]

plt.legend(handles=legend_elements,
    labels=['Mean accuracy', 'Individual data points', 'Chance level'],
    handler_map={tuple: HandlerTuple(ndivide=None)},
    loc='upper left',
    frameon=False)

# =========================
# Labels
# =========================

plt.ylabel("Classification accuracy (%)")
plt.title("Classification of time-averaged patterns")

plt.ylim(20, 80)

sns.despine()
plt.tight_layout()
plt.show()