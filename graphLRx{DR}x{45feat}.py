import matplotlib.pyplot as plt
import numpy as np

# === STEP 1: Plug in your MCC data ===
# Rows = models, Columns = DR methods
mcc_data = {
    "TF-IDF":        [0.1701,    0.4158,    0.4662,    0.1892,    0.5966],
    "GloVe":         [0.8660,    0.8500,    0.7992,    0.8710,    0.8384],
    "BIGRAM":        [0.8720,    0.8522,    0.8100,    0.8670,    0.8417],
    "TRIGRAM":       [0.8260,    0.8391,    0.7337,    0.8345,    0.8435],
    "BoW":           [0.8426,    0.8175,    0.7983,    0.8301,    0.8223]
}

dr_methods = ["PCA", "ICA", "NMF", "SVD", "Chi2"]
colors = ['#d9d9d9', '#a6a6a6', '#737373', '#404040', '#000000']  # Gray scale

# === STEP 2: Plotting ===
models = list(mcc_data.keys())
num_models = len(models)
num_dr = len(dr_methods)

x = np.arange(num_models)
bar_width = 0.15

fig, ax = plt.subplots(figsize=(10, 6))

# Plot each DR method as a separate group of bars
for i in range(num_dr):
    offsets = [mcc_data[model][i] for model in models]
    ax.bar(x + i * bar_width, offsets, bar_width, label=dr_methods[i], color=colors[i])

# === STEP 3: Labels & Aesthetics ===
ax.set_ylabel("MCC", fontsize=12)
ax.set_xlabel("Classifiers", fontsize=12)
ax.set_title("MCC of Logistic Regression with Different Dimensionality Reduction Methods \n using 45 features of different types", fontsize=14)
ax.set_xticks(x + bar_width * (num_dr - 1) / 2)
ax.set_xticklabels(models, rotation=25, ha='right', fontsize=10)
ax.set_ylim([0.0, 1.0])
ax.legend(title="Dimensionality Reduction", fontsize=9, title_fontsize=10)

plt.tight_layout()
plt.show()
