import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the matrix, using the first column as the row index
cm = pd.read_csv("results/confusion_matrix.csv", index_col=0)
print("Confusion Matrix Data:")
print(cm)

# Extract the raw numbers for the heatmap
data = cm.values

# Set up clean labels for the axes
# If the index is just 0 and 1, we'll map them to readable text
y_labels = cm.index.tolist()
if y_labels == [0, 1] or y_labels == ['0', '1']:
    y_labels = ["Actual Legitimate", "Actual Phishing"]

x_labels = ["Predicted Legitimate", "Predicted Phishing"]

plt.figure(figsize=(5, 4), dpi=300)
sns.heatmap(data, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=x_labels, yticklabels=y_labels,
            annot_kws={"size": 14, "weight": "bold"})

plt.xlabel("Predicted Label", fontsize=12)
plt.ylabel("Actual Label", fontsize=12)
plt.title("Confusion Matrix (Threshold = 0.65)", fontsize=14)

plt.tight_layout()
plt.savefig("results/figure_10_confusion_matrix.png", dpi=300)
print("\nSaved: results/figure_10_confusion_matrix.png")
