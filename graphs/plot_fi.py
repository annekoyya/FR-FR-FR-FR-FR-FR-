import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load feature importance
fi = pd.read_csv("results/feature_importance.csv")
print("Columns found:", fi.columns.tolist())

# Auto-detect the importance column (try common names)
imp_col = None
for candidate in ["importance", "gain", "score", "value", "weight"]:
    if candidate in fi.columns:
        imp_col = candidate
        break

if imp_col is None:
    # Fallback: use the second column (first is usually the feature name)
    imp_col = fi.columns[1]
    print(f"Could not auto-detect; falling back to column '{imp_col}'")

print(f"Using column: '{imp_col}'")

# Sort and take top 15
fi = fi.sort_values(by=imp_col, ascending=True).tail(15)

plt.figure(figsize=(8, 6), dpi=300)
sns.barplot(x=imp_col, y=fi.columns[0], data=fi, palette="viridis")

plt.xlabel(f"{imp_col.replace('_', ' ').title()} Importance", fontsize=12)
plt.ylabel("Feature", fontsize=12)
plt.title("Top 15 Most Important Features (XGBoost Gain)", fontsize=14)

plt.tight_layout()
plt.savefig("results/figure_feature_importance.png", dpi=300)
print("Saved: results/figure_feature_importance.png")
