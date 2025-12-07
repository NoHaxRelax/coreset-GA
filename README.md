# 🧬 Core-Set Optimization via Multi-Objective Genetic Search
Maximizing informativeness, diversity, and class balance in dataset subset selection

This project investigates how to automatically select high-value training subsets for supervised learning using Search-Based Software Engineering (SBSE) techniques. Instead of training a full model for every candidate subset (computationally infeasible), we estimate the learning utility of each image using a committee of pretrained classifiers, then optimize subset composition with a multi-objective genetic algorithm (NSGA-II).

The goal is to build a Pareto frontier of model accuracy vs. dataset size; showing how much data is needed for a performant classifier when the subset is chosen intelligently rather than randomly.

---

## Problem Overview
Given a labeled dataset \(D\), we seek subsets \(S_k \subset D\) of size \(k\) that maximize three properties:
- **Informativeness (difficulty):** images the committee finds confusing (high entropy).
- **Representation (class balance):** proportional coverage of all classes.
- **Non-redundancy (diversity):** avoid near-duplicates; cover distinct regions of the data manifold.

We treat this as a multi-objective search problem and use NSGA-II to find Pareto-optimal subsets.

---

## Objectives
Each subset \(S\) has three objective scores to maximize:

1) **Difficulty**  
Use an ensemble of pretrained models. For each sample \(x\):
- \(p_i(x)\): softmax from committee model \(i\)  
- \(\bar{p}(x) = \frac{1}{M}\sum_i p_i(x)\)  
- \(H(\cdot)\): entropy  

Difficulty of a subset:
\[
\text{difficulty}(S) = \frac{1}{|S|} \sum_{x \in S} H(\bar{p}(x))
\]
High entropy means the committee is unsure, implying informative samples.

2) **Diversity**  
Use cosine distance between embeddings to avoid high-dimensional L2 collapse. With \(f(x)\) as a pretrained 512-dim embedding:
\[
\text{diversity}(S) = 1 - \frac{1}{|S|(|S|-1)} \sum_{i \neq j} \cos\big(f(x_i), f(x_j)\big)
\]
High diversity means samples cover different regions of the data manifold.

3) **Class Balance**  
Let \(S_c\) be the count of samples in class \(c\) and \(C\) the number of classes. Define:
\[
\text{balance}(S) = 1 - \sum_{c=1}^{C} \left|\frac{S_c}{|S|} - \frac{1}{C}\right|
\]
High balance means proportional representation of all classes.

---

## GA Formulation (NSGA-II)
Custom operators and constraints tuned for this problem.

### Chromosome Representation
- Fixed-length vector of \(k\) unique dataset indices. Example: `[14, 88, 233, 419, ...]`
- Order does not matter; uniqueness enforced after crossover/mutation.

### Population Size and Generations
- **Population:** 30  
- **Generations:** 20  
- Rationale: fits an 8 A100-hour (or ~35 T4-hour) budget and allows multiple \(k\) runs (e.g., 50, 100, 200, 500, 1000) instead of one massive GA run.

### Mutation Operator (one per child, probabilistic)
- **Index replacement (70%):** replace 1–5 random indices with new valid ones.
- **Segment shuffle (20%):** shuffle a random 10–20% segment.
- **Swap mutation (10%):** swap two indices.
- **Post-step:** enforce uniqueness; if duplicates appear, replace them with random new indices.

### Crossover Operator
- **Set-aware uniform crossover:** choose each gene from either parent; if duplicates arise, fill remaining slots with random unseen indices. Preserves diversity while respecting fixed subset size.

### Selection and Ranking
- NSGA-II handles non-dominated sorting, crowding distance, and Pareto-front preservation. No manual weighting of objectives.

---

## Evaluation Plan
For each \(k \in \{50, 100, 200, 500, 750, 1000\}\):
1. Run NSGA-II and extract the final Pareto-optimal subsets.  
2. Pick one representative subset (closest to ideal: max difficulty, max diversity, max balance).  
3. Train a small CNN on that subset.  
4. Compare test accuracy against:
   - random subsets
   - full dataset
   - (maybe) hardest-only subsets
   - (maybe) balanced-only subsets

**Deliverable:** Accuracy vs. dataset size Pareto curve showing how curated subsets outperform random sampling.

---

## Model Training and Evaluation

After NSGA-II converges for each subset size \(k\), we select the best subset and train a classifier to validate that our proxy objectives (difficulty, diversity, balance) translate to actual model performance.

### Subset Selection from Pareto Frontier

The GA produces a set of Pareto-optimal solutions—subsets that cannot be improved in one objective without degrading another. To pick a single representative subset for training, we use a **weighted distance to ideal point**:

\[
\text{score}(S) = w_1 \cdot \text{difficulty}(S) + w_2 \cdot \text{diversity}(S) + w_3 \cdot \text{balance}(S)
\]

Where \(w_1 = w_2 = w_3 = 1/3\) (equal weighting after per-objective standardization). Alternatively, we select the subset closest to the utopian point (max difficulty, max diversity, max balance) in normalized objective space.

**Selection criteria:**
- If multiple subsets tie, prefer the one with highest minimum objective (maximin strategy).
- Ensure the selected subset has at least one sample per class (enforced during GA).

### Model Architecture

We use a **lightweight CNN** designed for fast training and fair comparison across subset sizes:

**Architecture:**
- **Input:** 32×32×3 (or dataset-native resolution)
- **Conv Block 1:** 32 filters, 3×3, ReLU, BatchNorm, MaxPool(2×2)
- **Conv Block 2:** 64 filters, 3×3, ReLU, BatchNorm, MaxPool(2×2)
- **Conv Block 3:** 128 filters, 3×3, ReLU, BatchNorm, MaxPool(2×2)
- **Flatten**
- **Dense:** 128 units, ReLU, Dropout(0.5)
- **Output:** \(C\) units (softmax), where \(C\) is the number of classes

**Rationale:** Small enough to train quickly on tiny subsets (k=50), but expressive enough to show performance differences between subset selection strategies.

### Training Procedure

For each selected subset \(S_k\):

1. **Data splits:**
   - Training: 100% of \(S_k\) (we're testing subset quality, not generalization from small splits)
   - Validation: held-out test set (same for all experiments)
   - No data augmentation (to isolate subset selection effects)

2. **Training hyperparameters:**
   - **Optimizer:** Adam (lr=0.001, decay=1e-6)
   - **Loss:** Categorical cross-entropy
   - **Batch size:** min(32, \(|S_k|/2\)) to ensure multiple batches even for k=50
   - **Epochs:** 50 (with early stopping: patience=10, monitor='val_accuracy')
   - **Initialization:** He normal

3. **Training protocol:**
   - Train from scratch (no pretraining) to measure raw subset utility
   - Use validation accuracy as the primary metric
   - Save best model based on validation performance
   - Report final test accuracy on held-out test set

### Baseline Comparisons

For each \(k\), we compare the GA-selected subset against:

1. **Random baseline:** 5 random subsets of size \(k\) (mean ± std accuracy)
2. **Full dataset:** Train on entire \(D\) (upper bound; shows data efficiency)
3. **(maybe) Hardest-only:** Top-\(k\) samples by difficulty score (no diversity/balance)
4. **(maybe) Balanced-only:** Top-\(k\) samples maximizing balance (no difficulty/diversity)

**Why these baselines?**
- Random: establishes that GA selection is better than chance
- Full dataset: shows how much data is actually needed
- Hardest-only: tests if difficulty alone is sufficient
- Balanced-only: tests if class balance alone is sufficient

### Evaluation Metrics

**Primary metrics:**
- **Test accuracy:** Classification accuracy on held-out test set
- **Training efficiency:** Accuracy achieved per training sample (accuracy / k)

**Secondary metrics (for analysis):**
- **Convergence speed:** Epochs to reach 90% of final accuracy
- **Class-wise F1:** Per-class performance (especially for imbalanced datasets)
- **Calibration:** Expected calibration error (ECE) to check if models are overconfident

### Results Visualization

The final deliverable is a **Pareto efficiency curve** plotting:
- **X-axis:** Subset size \(k\)
- **Y-axis:** Test accuracy
- **Series:**
  - GA-selected subsets (our method)
  - Random baseline (mean ± std)
  - Hardest-only baseline
  - Balanced-only baseline
  - Full dataset (horizontal line)

This visualization shows:
- How much data is needed for a given accuracy target
- The gap between smart selection and random sampling
- Whether multi-objective optimization outperforms single-objective strategies

### Computational Considerations

**Training budget allocation:**
- GA runs: ~10–30 minutes per \(k\) (CPU-bound, uses precomputed embeddings)
- Model training: ~1–3 hours per \(k\) (GPU-bound)
- Total per \(k\): ~2–4 hours
- All \(k\) values: ~12–24 hours (well within 8 A100-hour budget)

**Reproducibility:**
- Fixed random seeds for GA initialization, model training, and data splits
- Save selected subset indices for each \(k\) to enable exact reproduction
- Log all hyperparameters and training curves

---

## Repository Structure
```
.
├── data/
├── embeddings/
├── pretrained_committee_models/
├── ga/
│   ├── nsga2.py
│   ├── population.py
│   ├── mutation.py
│   ├── crossover.py
│   ├── evaluation.py
├── experiments/
│   ├── run_k50.py
│   ├── run_k100.py
│   └── ...
├── analysis/
│   ├── plot_pareto.ipynb
│   ├── visualize_subsets.ipynb
├── results/
├── training/
├── final_models/
├── README.md
└── requirements.txt
```

---

## Project Goals
- Show that smart data selection yields better accuracy with fewer samples.
- Demonstrate NSGA-II for dataset optimization in an SBSE framing.
- Produce interpretable subsets ranked by informativeness, diversity, and class balance.
- Deliver a Pareto efficiency curve of data size vs. model accuracy.

---

## Compute Budget Notes
- Models are not trained inside the GA loop.  
- Difficulty, diversity, and balance scores use:
  - precomputed embeddings
  - one pass of committee softmaxes
- Total GA runtime per \(k\): minutes to tens of minutes. Compute hours mainly go to the final evaluation classifiers.

---

## License
MIT