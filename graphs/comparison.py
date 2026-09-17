import pandas as pd
import matplotlib.pyplot as plt

baseline = pd.read_csv("results/baseline_results_200_to_50k.csv").rename(columns={"samples": "n"})
trimodal = pd.read_csv("results/trimodal_results_200_to_50k.csv").rename(columns={"samples": "n"})

plt.figure()
plt.plot(baseline["n"], baseline["auc"], marker="o", label="Baseline (QRiS, structural-only)")
plt.plot(trimodal["n"], trimodal["auc"], marker="o", label="Proposed (tri-modal fused)")
plt.xscale("log")
plt.xlabel("Sample size (log scale)"); plt.ylabel("AUC")
plt.title("AUC vs. Dataset Size: Baseline vs. Tri-Modal")
plt.legend(); plt.savefig("results/baseline_vs_trimodal_scaling.png", dpi=150, bbox_inches="tight")