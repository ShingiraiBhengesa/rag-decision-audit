import json
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import os

matplotlib.rcParams["font.family"] = "DejaVu Sans"
matplotlib.rcParams["font.size"] = 12
matplotlib.rcParams["axes.spines.top"] = False
matplotlib.rcParams["axes.spines.right"] = False
matplotlib.rcParams["figure.dpi"] = 300

os.makedirs("/home/sbhengesa/adaptive-rag/analysis/figures", exist_ok=True)
os.makedirs("/home/sbhengesa/combined_analysis/figures", exist_ok=True)

datasets = ["MuSiQue", "HotpotQA", "2WikiMHop"]
adaptive = [21.4, 45.6, 51.2]
no_ret = [4.0, 18.4, 25.6]
single = [15.8, 39.6, 46.2]
multi = [19.2, 49.4, 62.6]

route_data = {"MuSiQue": {"A": 9, "B": 83, "C": 408, "total": 500}, "HotpotQA": {"A": 52, "B": 137, "C": 311, "total": 500}, "2WikiMHop": {"A": 144, "B": 66, "C": 290, "total": 500}}
route_acc = {"MuSiQue": {"A": 44.4, "B": 34.9, "C": 18.1}, "HotpotQA": {"A": 46.2, "B": 43.8, "C": 46.3}, "2WikiMHop": {"A": 48.6, "B": 25.8, "C": 58.3}}
epochs = [15, 20, 25, 30, 35]
val_acc = [47.6, 47.5, 48.3, 50.0, 49.6]

# FIG 1
fig, ax = plt.subplots(figsize=(10, 6))
x = np.arange(len(datasets))
width = 0.2
bars1 = ax.bar(x - 1.5*width, adaptive, width, label="Adaptive (classifier)", color="#2563EB", edgecolor="white", linewidth=0.5)
bars2 = ax.bar(x - 0.5*width, no_ret, width, label="No retrieval (A)", color="#DC2626", edgecolor="white", linewidth=0.5)
bars3 = ax.bar(x + 0.5*width, single, width, label="Single-step (B)", color="#F59E0B", edgecolor="white", linewidth=0.5)
bars4 = ax.bar(x + 1.5*width, multi, width, label="Multi-step (C)", color="#059669", edgecolor="white", linewidth=0.5)
for bars in [bars1, bars2, bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f"{height:.1f}", xy=(bar.get_x() + bar.get_width()/2, height), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=8, fontweight="bold")
ax.set_ylabel("Accuracy (%)", fontsize=13)
ax.set_title("Adaptive-RAG: accuracy across retrieval strategies", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=12)
ax.set_ylim(0, 75)
ax.legend(loc="upper left", frameon=True, edgecolor="#E5E7EB", fontsize=10)
ax.grid(axis="y", alpha=0.3, linestyle="--")
plt.tight_layout()
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig1_accuracy_comparison.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig1_accuracy_comparison.pdf", bbox_inches="tight")
plt.close()
print("Fig 1 saved")

# FIG 2
fig, ax = plt.subplots(figsize=(10, 5))
datasets_rev = list(reversed(datasets))
a_pct = [route_data[d]["A"]/route_data[d]["total"]*100 for d in datasets_rev]
b_pct = [route_data[d]["B"]/route_data[d]["total"]*100 for d in datasets_rev]
c_pct = [route_data[d]["C"]/route_data[d]["total"]*100 for d in datasets_rev]
y = np.arange(len(datasets_rev))
ax.barh(y, a_pct, height=0.5, label="Route A (no retrieval)", color="#DC2626", edgecolor="white", linewidth=0.5)
ax.barh(y, b_pct, height=0.5, left=a_pct, label="Route B (single-step)", color="#F59E0B", edgecolor="white", linewidth=0.5)
ax.barh(y, c_pct, height=0.5, left=[a+b for a,b in zip(a_pct, b_pct)], label="Route C (multi-step)", color="#059669", edgecolor="white", linewidth=0.5)
for i in range(len(datasets_rev)):
    if a_pct[i] > 5:
        ax.text(a_pct[i]/2, i, f"{a_pct[i]:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    if b_pct[i] > 5:
        ax.text(a_pct[i] + b_pct[i]/2, i, f"{b_pct[i]:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
    ax.text(a_pct[i] + b_pct[i] + c_pct[i]/2, i, f"{c_pct[i]:.0f}%", ha="center", va="center", fontsize=10, fontweight="bold", color="white")
ax.set_yticks(y)
ax.set_yticklabels(datasets_rev, fontsize=12)
ax.set_xlabel("Percentage of queries (%)", fontsize=13)
ax.set_title("Adaptive-RAG classifier routing distribution", fontsize=15, fontweight="bold", pad=15)
ax.set_xlim(0, 105)
ax.legend(loc="lower right", frameon=True, edgecolor="#E5E7EB", fontsize=10)
plt.tight_layout()
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig2_route_distribution.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig2_route_distribution.pdf", bbox_inches="tight")
plt.close()
print("Fig 2 saved")

# FIG 3
fig, ax = plt.subplots(figsize=(10, 5))
best_single = [max(no_ret[i], single[i], multi[i]) for i in range(len(datasets))]
best_labels = []
for i in range(len(datasets)):
    vals = {"A": no_ret[i], "B": single[i], "C": multi[i]}
    best_labels.append(max(vals, key=vals.get))
gaps = [adaptive[i] - best_single[i] for i in range(len(datasets))]
colors = ["#059669" if g >= 0 else "#DC2626" for g in gaps]
bars = ax.bar(datasets, gaps, color=colors, width=0.5, edgecolor="white", linewidth=0.5)
for bar, gap, bl in zip(bars, gaps, best_labels):
    label_text = f"{gap:+.1f} pp vs {bl}"
    ax.annotate(label_text, xy=(bar.get_x() + bar.get_width()/2, gap), xytext=(0, 8 if gap >= 0 else -20), textcoords="offset points", ha="center", va="bottom" if gap >= 0 else "top", fontsize=11, fontweight="bold", color="#059669" if gap >= 0 else "#DC2626")
ax.axhline(y=0, color="#9CA3AF", linewidth=0.8)
ax.set_ylabel("Accuracy gap (percentage points)", fontsize=13)
ax.set_title("Adaptive routing vs best single strategy", fontsize=15, fontweight="bold", pad=15)
ax.grid(axis="y", alpha=0.3, linestyle="--")
ax.set_ylim(min(gaps) - 5, max(gaps) + 5)
plt.tight_layout()
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig3_adaptive_vs_best.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig3_adaptive_vs_best.pdf", bbox_inches="tight")
plt.close()
print("Fig 3 saved")

# FIG 4
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
route_colors = {"A": "#DC2626", "B": "#F59E0B", "C": "#059669"}
for idx, ds in enumerate(datasets):
    ax = axes[idx]
    routes = ["A", "B", "C"]
    accs = [route_acc[ds][r] for r in routes]
    counts = [route_data[ds][r] for r in routes]
    bars = ax.bar(routes, accs, color=[route_colors[r] for r in routes], width=0.6, edgecolor="white", linewidth=0.5)
    for bar, acc, cnt in zip(bars, accs, counts):
        label_text = f"{acc:.1f}% (n={cnt})"
        ax.annotate(label_text, xy=(bar.get_x() + bar.get_width()/2, acc), xytext=(0,4), textcoords="offset points", ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_title(f"{ds}", fontsize=13, fontweight="bold", pad=10)
    ax.set_ylim(0, max(accs) + 15)
    ax.set_ylabel("Accuracy (%)" if idx == 0 else "", fontsize=12)
    ax.set_xlabel("Route", fontsize=12)
    ax.grid(axis="y", alpha=0.2, linestyle="--")
fig.suptitle("Accuracy per route assigned by classifier", fontsize=15, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig4_per_route_accuracy.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig4_per_route_accuracy.pdf", bbox_inches="tight")
plt.close()
print("Fig 4 saved")

# FIG 5
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, val_acc, "o-", color="#2563EB", linewidth=2, markersize=8)
best_idx = val_acc.index(max(val_acc))
ax.scatter([epochs[best_idx]], [val_acc[best_idx]], s=150, color="#059669", zorder=5, edgecolors="white", linewidth=2)
ax.annotate(f"Best: {val_acc[best_idx]:.1f}% (epoch {epochs[best_idx]})", xy=(epochs[best_idx], val_acc[best_idx]), xytext=(15, -15), textcoords="offset points", fontsize=11, fontweight="bold", color="#059669")
ax.set_xlabel("Training epochs", fontsize=13)
ax.set_ylabel("Validation accuracy (%)", fontsize=13)
ax.set_title("Classifier training: validation accuracy vs epochs", fontsize=15, fontweight="bold", pad=15)
ax.set_xticks(epochs)
ax.grid(alpha=0.3, linestyle="--")
ax.set_ylim(45, 52)
plt.tight_layout()
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig5_training_curve.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/adaptive-rag/analysis/figures/fig5_training_curve.pdf", bbox_inches="tight")
plt.close()
print("Fig 5 saved")

# COMBINED FIG 1
sr_datasets = ["PopQA", "TriviaQA", "PubHealth", "ARC-C"]
sr_adaptive = [54.9, 66.2, 72.1, 67.2]
sr_noret = [28.7, 50.3, 70.1, 67.4]
ar_datasets = ["MuSiQue", "HotpotQA", "2WikiMHop"]
ar_adaptive_c = [21.4, 45.6, 51.2]
ar_noret_c = [4.0, 18.4, 25.6]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
sr_gains = [sr_adaptive[i] - sr_noret[i] for i in range(4)]
sr_colors = ["#059669" if g > 0.5 else ("#DC2626" if g < -0.5 else "#F59E0B") for g in sr_gains]
bars1 = ax1.bar(sr_datasets, sr_gains, color=sr_colors, width=0.5, edgecolor="white", linewidth=0.5)
for bar, gain in zip(bars1, sr_gains):
    ax1.annotate(f"{gain:+.1f}", xy=(bar.get_x() + bar.get_width()/2, gain), xytext=(0, 6 if gain >= 0 else -14), textcoords="offset points", ha="center", va="bottom" if gain >= 0 else "top", fontsize=11, fontweight="bold")
ax1.axhline(y=0, color="#9CA3AF", linewidth=0.8)
ax1.set_title("Self-RAG: retrieval gain vs no-retrieval", fontsize=14, fontweight="bold", pad=15)
ax1.set_ylabel("Accuracy change (pp)", fontsize=12)
ax1.grid(axis="y", alpha=0.3, linestyle="--")
ax1.set_ylim(-5, 30)

ar_best = [max(ar_noret_c[i], single[i], multi[i]) for i in range(3)]
ar_gaps = [ar_adaptive_c[i] - ar_best[i] for i in range(3)]
ar_colors = ["#059669" if g >= 0 else "#DC2626" for g in ar_gaps]
bars2 = ax2.bar(ar_datasets, ar_gaps, color=ar_colors, width=0.5, edgecolor="white", linewidth=0.5)
for bar, gap in zip(bars2, ar_gaps):
    ax2.annotate(f"{gap:+.1f}", xy=(bar.get_x() + bar.get_width()/2, gap), xytext=(0, 6 if gap >= 0 else -14), textcoords="offset points", ha="center", va="bottom" if gap >= 0 else "top", fontsize=11, fontweight="bold")
ax2.axhline(y=0, color="#9CA3AF", linewidth=0.8)
ax2.set_title("Adaptive-RAG: adaptive vs best strategy", fontsize=14, fontweight="bold", pad=15)
ax2.set_ylabel("Accuracy change (pp)", fontsize=12)
ax2.grid(axis="y", alpha=0.3, linestyle="--")
ax2.set_ylim(-15, 30)

fig.suptitle("Retrieval decision effectiveness: Self-RAG vs Adaptive-RAG", fontsize=16, fontweight="bold", y=1.04)
plt.tight_layout()
plt.savefig("/home/sbhengesa/combined_analysis/figures/combined_fig1_decision_comparison.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/combined_analysis/figures/combined_fig1_decision_comparison.pdf", bbox_inches="tight")
plt.close()
print("Combined Fig 1 saved")

# COMBINED FIG 2
from matplotlib.lines import Line2D
fig, ax = plt.subplots(figsize=(12, 7))
sr_ret_pct = [100.0, 100.0, 94.4, 100.0]
for i, ds in enumerate(sr_datasets):
    gain = sr_adaptive[i] - sr_noret[i]
    ax.scatter(sr_ret_pct[i], gain, s=200, c="#2563EB", alpha=0.8, edgecolors="white", linewidth=2, zorder=5, marker="o")
    ax.annotate(f"SR:{ds}", (sr_ret_pct[i], gain), textcoords="offset points", xytext=(-15, 10), fontsize=10, fontweight="bold", color="#2563EB")

ar_ret_pct_vals = [(route_data[d]["B"] + route_data[d]["C"])/route_data[d]["total"]*100 for d in ["MuSiQue", "HotpotQA", "2WikiMHop"]]
for i, ds in enumerate(ar_datasets):
    gain = ar_adaptive_c[i] - ar_noret_c[i]
    ax.scatter(ar_ret_pct_vals[i], gain, s=200, c="#7C3AED", alpha=0.8, edgecolors="white", linewidth=2, zorder=5, marker="s")
    ax.annotate(f"AR:{ds}", (ar_ret_pct_vals[i], gain), textcoords="offset points", xytext=(-15, 10), fontsize=10, fontweight="bold", color="#7C3AED")

ax.axhline(y=0, color="#9CA3AF", linewidth=0.8, linestyle="--")
legend_elements = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#2563EB", markersize=12, label="Self-RAG"), Line2D([0], [0], marker="s", color="w", markerfacecolor="#7C3AED", markersize=12, label="Adaptive-RAG")]
ax.legend(handles=legend_elements, loc="upper left", fontsize=12, frameon=True, edgecolor="#E5E7EB")
ax.set_xlabel("Retrieval frequency (%)", fontsize=13)
ax.set_ylabel("Accuracy gain from retrieval (pp)", fontsize=13)
ax.set_title("Retrieval behavior: Self-RAG vs Adaptive-RAG", fontsize=15, fontweight="bold", pad=15)
ax.set_xlim(65, 105)
ax.grid(alpha=0.2)
plt.tight_layout()
plt.savefig("/home/sbhengesa/combined_analysis/figures/combined_fig2_retrieval_spectrum.png", bbox_inches="tight")
plt.savefig("/home/sbhengesa/combined_analysis/figures/combined_fig2_retrieval_spectrum.pdf", bbox_inches="tight")
plt.close()
print("Combined Fig 2 saved")

print("ALL DONE")
