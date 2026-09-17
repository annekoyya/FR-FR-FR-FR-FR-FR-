import pandas as pd
import matplotlib.pyplot as plt

# Load the raw ROC data
roc = pd.read_csv("results/roc_curve.csv")
print("ROC Columns found:", roc.columns.tolist())

plt.figure(figsize=(6, 6), dpi=300)
plt.plot(roc["fpr"], roc["tpr"], color="blue", linewidth=2, label="Tri-Modal Fused Model")
plt.plot([0, 1], [0, 1], color="gray", linestyle="--", linewidth=1.5, label="Random Chance")

plt.xlabel("False Positive Rate (FPR)", fontsize=12)
plt.ylabel("True Positive Rate (TPR)", fontsize=12)
plt.title("ROC Curve - Tri-Modal Quishing Detection", fontsize=14)
plt.legend(loc="lower right", fontsize=11)
plt.grid(True, linestyle=":", alpha=0.7)

plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])

plt.tight_layout()
plt.savefig("results/figure_11_roc_curve.png", dpi=300)
print("Saved: results/figure_11_roc_curve.png")
