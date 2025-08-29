# main.py
# Run: uv run python main.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
import skfuzzy  # scikit-fuzzy

# ============ 0) LOAD ============
df = pd.read_csv('data/credit_card_clients.csv')
print("✅ Loaded 'data/credit_card_clients.csv'")
print("   Shape:", df.shape)

# ============ 1) Feature prep ============
# Create BILL TOTAL 
df['BILL TOTAL'] = (
    df['BILL_AMT1'] + df['BILL_AMT2'] + df['BILL_AMT3'] +
    df['BILL_AMT4'] + df['BILL_AMT5'] + df['BILL_AMT6']
)

# Two features: LIMIT_BAL (index 1) and BILL TOTAL (index 25)
# (Name-based selection is safer; equivalent to iloc[:, [1,25]])
X = df[['LIMIT_BAL', 'BILL TOTAL']].values
print("\n🔹 Raw feature stats:")
print("   LIMIT_BAL  min/max:", X[:,0].min(), X[:,0].max())
print("   BILL TOTAL min/max:", X[:,1].min(), X[:,1].max())

# ============ 2) Scale to [0,1] ============
scaler = MinMaxScaler()
X = scaler.fit_transform(X)   # shape: (n_samples, 2)
print("\n🔧 Scaled shape:", X.shape)

# skfuzzy expects (features, samples), so transpose for cmeans calls
XT = X.T
print("   Transposed for skfuzzy:", XT.shape, "(features x samples)")

# ============ 3) Single run example (c=5) with prints ============
print("\n▶ Running c-means with c=5")
clustering = skfuzzy.cmeans(
    data=XT, c=5, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
)
centers_5, u_5 = clustering[0], clustering[1]
fpc_5 = clustering[6]
print(f"   FPC(c=5): {fpc_5:.4f}")
print("   Membership matrix shape:", u_5.shape, "(c x n)")
print("   First sample membership (sums to 1):", u_5[:, 0], " sum=", u_5[:, 0].sum())

pred_5 = u_5.argmax(axis=0)
uniq_5, cnts_5 = np.unique(pred_5, return_counts=True)
print("   Cluster sizes (c=5):", dict(zip(uniq_5, cnts_5)))

# ============ 4) Sweep c=2..10 for FPC + small cluster plots grid ============
os.makedirs("docs", exist_ok=True)

colors = ['blue', 'orange', 'green', 'red', 'yellow', 'black', 'brown', 'cyan', 'magenta', 'forestgreen']

fig, axes = plt.subplots(3, 3, figsize=(15, 15))
fpcs = []

print("\n🔎 FPC sweep (c=2..10):")
for n_clusters, ax in zip(range(2, 11), axes.reshape(-1)):
    centers, memberships, _, _, _, _, fpc = skfuzzy.cmeans(
        data=XT, c=n_clusters, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
    )
    fpcs.append(fpc)
    preds = np.argmax(memberships, axis=0)
    uniq, cnts = np.unique(preds, return_counts=True)
    print(f"   c={n_clusters}: FPC={fpc:.4f} | sizes={dict(zip(uniq, cnts))}")

    for i in range(n_clusters):
        ax.plot(X[preds == i, 0], X[preds == i, 1], '.', color=colors[i % len(colors)])
    for c in centers:
        ax.plot(c[0], c[1], 'rs')
    ax.set_title(f'c={n_clusters} | FPC={fpc:.3f}')
    ax.set_xlabel('Limit (scaled)')
    ax.set_ylabel('Bill total (scaled)')
    ax.grid(True)

fig.tight_layout()
grid_path = "docs/clusters_grid_c2_to_c10.png"
fig.savefig(grid_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"🖼  Saved grid plot -> {grid_path}")

# ============ 5) FPC curve and chosen c ============
fig, ax = plt.subplots()
ax.plot(range(2, 11), fpcs, marker='o')
ax.set_xlabel('Number of clusters (c)')
ax.set_ylabel('FPC')
ax.set_title('FPC vs c')
ax.grid(True)
fpc_plot_path = "docs/fpc_vs_c.png"
fig.savefig(fpc_plot_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"🖼  Saved FPC curve -> {fpc_plot_path}")

best_idx = int(np.argmax(fpcs))
best_c = 2 + best_idx
print(f"\n🏆 Best c by FPC: c={best_c} (FPC={fpcs[best_idx]:.4f})")

# ============ 6) Final run with best c + final scatter ============
centers, memberships, _, _, _, _, fpc = skfuzzy.cmeans(
    data=XT, c=best_c, m=2.0, error=0.005, maxiter=1000, init=None, seed=42
)
preds = np.argmax(memberships, axis=0)
uniq, cnts = np.unique(preds, return_counts=True)
print(f"🎯 Final run c={best_c}: FPC={fpc:.4f} | sizes={dict(zip(uniq, cnts))}")

fig, ax = plt.subplots()
for i in range(best_c):
    ax.scatter(X[preds == i, 0], X[preds == i, 1], s=8, label=f'Cluster {i}')
ax.scatter(centers[:, 0], centers[:, 1], c='k', s=120, marker='X', label='centers')
ax.set_xlabel('Limit (scaled)')
ax.set_ylabel('Bill total (scaled)')
ax.set_title(f'Fuzzy C-Means (c={best_c})')
ax.legend(loc='best', fontsize=8)
ax.grid(True)
final_scatter_path = f"docs/fcm_scatter_c{best_c}.png"
fig.savefig(final_scatter_path, dpi=160, bbox_inches="tight")
plt.close(fig)
print(f"🖼  Saved final scatter -> {final_scatter_path}")

print("\n✅ Done.")
