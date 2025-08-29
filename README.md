# Fuzzy C-Means (C-Means) — README

This project applies **Fuzzy C-Means (FCM)** clustering (via `scikit-fuzzy`) to the *Default of Credit Card Clients* dataset using two features:

- `LIMIT_BAL` — credit limit  
- `BILL TOTAL` — `BILL_AMT1+…+BILL_AMT6`

The pipeline:

1. Load `data/credit_card_clients.csv` (your exact path).  
2. Create `BILL TOTAL`.  
3. Scale `['LIMIT_BAL', 'BILL TOTAL']` to `[0,1]`.  
4. Run FCM for a sweep of cluster counts (`c = 2..10`).  
5. Plot FPC vs `c` and a grid of mini-scatter plots.  
6. Pick the `c` with the **highest FPC** and plot the final clustering.

---

## What is Fuzzy C-Means?

Like K-Means, FCM finds `c` cluster **centers**, but it assigns **soft memberships** `u_{ik} ∈ [0,1]` for each point `k` in each cluster `i` (columns of `U` sum to 1). This is useful when clusters **overlap** or you want a measure of **how strongly** a point belongs to each cluster.

**Objective (Euclidean)**  
Minimize  
`J_m(U,V) = Σ_i Σ_k (u_{ik}^m) * || x_k − v_i ||^2`  
with `m > 1` (the *fuzzifier*, typically `1.6–2.0`). Higher `m` → softer memberships.

**Alternating updates**  
- Centers: `v_i = (Σ_k u_{ik}^m * x_k) / (Σ_k u_{ik}^m)`  
- Memberships (normalized inverse-distance rule)

Stop when memberships change less than a small `error` or when `maxiter` is reached.

---

## How we choose the number of clusters

We report the **Fuzzy Partition Coefficient (FPC)** for each `c`.  
`FPC = (1/n) * Σ_k Σ_i (u_{ik}^2)` (higher is better; 1.0 is perfectly crisp, `≈1/c` is very fuzzy).

You’ll usually see the **best FPC at a small c**; as `c` grows, FPC tends to drop because partitions get fuzzier and centers creep together.

---

## Results (your plots)

> These image paths assume you ran `main.py` and the script wrote plots to `docs/`.

### 1) FPC vs number of clusters
![FPC vs c](docs/fpc_vs_c.png)

**How to read:**  
- The curve shows FPC for `c = 2..10`.  
- **Pick the peak** (here it’s at `c = 2`).  
- The downward slope after `c=2` means adding more clusters makes the partition fuzzier (less crisp separation) for these two features.

---

### 2) Cluster grids (c = 2…10)
![Cluster grids](docs/clusters_grid_c2_to_c10.png)

**What you’re seeing:**  
- Each panel is an FCM run for a specific `c`.  
- Colors = hard labels from the fuzzy memberships (`argmax` across clusters).  
- Black/red squares = cluster centers (in **scaled** space).  
- As `c` increases, the algorithm keeps subdividing the dense region at low limit / low bill totals.  
- FPC shown in each title steadily **declines** with `c`, indicating the split becomes less crisp.

**Takeaway:** For these two features, **few clusters** (especially `c=2`) summarize the structure best. Large `c` just slices the same mass in arbitrary ways.

---

### 3) Final clustering (chosen c by FPC)
![Final scatter](docs/fcm_scatter_c2.png)

**Interpretation:**  
- Two broad groups appear in scaled space:  
  1) smaller limits & small bills;  
  2) higher limits & larger bills.  
- The **“X” markers** are the fuzzy centers.  
- Remember: points near the boundary have **non-trivial memberships in both clusters**; colors show the *hard* label for visualization only.

---

## Practical tips

- **Always scale** features before FCM (Euclidean metric).  
- If you switch to more features, avoid heavy **collinearity** (or use a compact subset).  
- If you ever get FPC ≈ `1/c` across all `c`, that’s a **degenerate** run (uniform memberships). Rerun with different initializations or adjust features/scale.  
- Soft memberships are great for: thresholding borderline points, ranking “how typical” a point is for a cluster, and flagging outliers (low max-membership).

---
