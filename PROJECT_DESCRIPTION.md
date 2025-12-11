# Core-Set Optimization via Multi-Objective Genetic Search
## Project Description for PowerPoint Presentation

---

## 1. PROBLEM STATEMENT

### The Core Challenge
Given a large labeled dataset **D** (e.g., 20,000 MNIST images), we want to find the **optimal subset S_k of size k** that maximizes model performance when used for training, while minimizing the amount of data required.

### Why This Matters
- **Data efficiency**: Can we achieve good accuracy with fewer training samples?
- **Resource constraints**: Limited computational budget, storage, or labeling costs
- **Quality over quantity**: Not all data points are equally valuable for learning

### The Multi-Objective Nature
We seek subsets that simultaneously maximize:
1. **Informativeness (Difficulty)**: Samples that are hard to classify → more learning value
2. **Diversity**: Samples spread across different regions of the data manifold → avoid redundancy
3. **Class Balance**: Proportional representation of all classes → fair and robust learning

---

## 2. WHY THIS IS A SEARCH-BASED SOFTWARE ENGINEERING (SBSE) PROBLEM

### SBSE Characteristics
**Search-Based Software Engineering** applies metaheuristic search techniques (like genetic algorithms) to solve software engineering problems that are:
- **Combinatorially complex**: The search space is enormous (e.g., choosing k=100 from 20,000 samples = C(20000,100) ≈ 10^240 possible subsets)
- **Multi-objective**: Multiple conflicting goals that cannot be optimized independently
- **No closed-form solution**: Cannot be solved analytically; requires exploration of solution space

### Our Problem as SBSE
1. **Search Space**: All possible k-sized subsets from dataset D
   - Size: C(|D|, k) - exponentially large
   - Example: C(20000, 100) ≈ 10^240 combinations

2. **Fitness Function**: Multi-objective evaluation (difficulty, diversity, balance)
   - No single "best" solution
   - Trade-offs between objectives
   - Requires Pareto-optimal solutions

3. **Search Strategy**: NSGA-II (Non-dominated Sorting Genetic Algorithm II)
   - Population-based metaheuristic
   - Evolves solutions over generations
   - Maintains diversity in solution space

4. **Software Engineering Context**: 
   - Optimizing a software artifact (training dataset)
   - Automated selection process
   - Quality attributes (accuracy, efficiency) as objectives

---

## 3. SOLUTION APPROACH: PROXY-BASED FITNESS FUNCTION

### The Computational Challenge
**Why we can't train a CNN for every candidate subset:**

If we used actual model accuracy as the fitness function:
- **Population size**: 30 candidates per generation
- **Generations**: 20 generations
- **Total evaluations**: 30 × 20 = **600 CNN training runs per k value**
- **Training time**: ~1-3 hours per CNN (even for small models)
- **Total time**: 600 × 2 hours = **1,200 hours** (50 days!) per k value
- **For 6 k values**: 300 days of continuous training

**This is computationally infeasible!**

### Our Proxy Solution
Instead of training CNNs, we use **fast-to-compute proxy metrics** that correlate with actual model performance:

#### **Fitness Function Components:**

**1. Difficulty (Informativeness)**
- Uses a **committee of 3 pretrained models** (ResNet18, VGG11, MobileNetV2)
- For each sample x:
  - Get softmax predictions from all committee models: p₁(x), p₂(x), p₃(x)
  - Average predictions: p̄(x) = (p₁(x) + p₂(x) + p₃(x)) / 3
  - Compute entropy: H(p̄(x)) = -Σ p̄(x) log(p̄(x))
- **Subset difficulty** = mean entropy across all samples in subset
- **Rationale**: High entropy = committee is uncertain = sample is informative/hard
- **Computation**: O(1) per sample (just lookup precomputed scores)

**2. Diversity (Non-redundancy)**
- Extract **512-dimensional embeddings** using pretrained ResNet50 feature extractor
- For subset S, compute pairwise cosine similarities between all embeddings
- **Diversity** = 1 - mean cosine similarity
- **Rationale**: Low similarity = samples are spread out = covers more of data manifold
- **Computation**: O(k²) per subset (fast for k < 1000)

**3. Balance (Class Representation)**
- Count samples per class in subset
- Compute deviation from uniform distribution
- **Balance** = 1 - normalized deviation
- **Rationale**: Balanced classes = better generalization, avoids bias
- **Computation**: O(k) per subset (very fast)

### Why Proxies Work
- **Committee entropy** correlates with learning difficulty (uncertain samples are more informative)
- **Embedding diversity** correlates with coverage of data distribution
- **Class balance** is directly related to model robustness
- **Validation**: We train actual CNNs only on final selected subsets to verify proxy quality

### Computational Savings
- **Proxy evaluation**: ~0.1 seconds per candidate
- **Total GA time**: 600 × 0.1s = **60 seconds** per k value
- **vs. CNN training**: 1,200 hours per k value
- **Speedup**: ~72,000× faster!

---

## 4. PROJECT PIPELINE

### Phase 1: Data Preparation
1. **Dataset Split**
   - Selection pool: 20,000 samples (candidates for GA)
   - Validation set: 2,000 samples (for early stopping)
   - Test set: 2,000 samples (final evaluation)

### Phase 2: Precomputation (One-time setup)
1. **Committee Model Preparation**
   - Load 3 pretrained ImageNet models
   - Adapt for MNIST (1-channel input, 10-class output)
   - Save adapted models

2. **Difficulty Score Computation**
   - Run committee inference on all 20,000 selection pool samples
   - Compute entropy for each sample
   - Save difficulty scores to disk

3. **Embedding Extraction**
   - Extract 512-dim embeddings for all selection pool samples
   - Save embeddings to disk

### Phase 3: Genetic Algorithm (NSGA-II)
For each subset size k ∈ {50, 100, 200, 500, 750, 1000}:

1. **Initialization**
   - Create population of 30 random k-sized subsets
   - Evaluate fitness (difficulty, diversity, balance) for each

2. **Evolution Loop** (20 generations)
   - **Selection**: Tournament selection based on Pareto rank + crowding distance
   - **Crossover**: Set-aware uniform crossover (80% probability)
   - **Mutation**: Index replacement (70%), segment shuffle (20%), swap (10%)
   - **Evaluation**: Compute proxy fitness for all offspring
   - **Survival**: Non-dominated sorting + crowding distance to select next generation

3. **Pareto Front Extraction**
   - Extract all non-dominated solutions
   - Save Pareto-optimal subsets

### Phase 4: Subset Selection
- From Pareto front, select one representative subset
- Method: Weighted score (equal weights) or ideal point distance
- Save selected subset indices

### Phase 5: Model Training & Validation
1. **Train CNN on GA-selected subset**
   - Architecture: 3 conv blocks (32, 64, 128 filters) + dense layer
   - Training: 50 epochs, early stopping, Adam optimizer
   - Save trained model

2. **Train Random Baselines** (5 runs)
   - Train same CNN on 5 random k-sized subsets
   - Compute mean ± std accuracy

3. **Train Full Dataset Model** (upper bound)
   - Train on entire 20,000 sample dataset
   - Shows maximum achievable performance

### Phase 6: Evaluation & Visualization
- Compare test accuracies: GA-selected vs. random vs. full dataset
- Generate Pareto efficiency curve: Accuracy vs. Dataset Size
- Analyze training efficiency (accuracy per sample)

---

## 5. GENETIC ALGORITHM SPECIFICATIONS

### Representation
- **Chromosome**: Fixed-length vector of k unique dataset indices
- Example: `[14, 88, 233, 419, 1024, ...]` (k indices)

### Population & Evolution
- **Population size**: 30
- **Generations**: 20
- **Total evaluations**: 600 per k value

### Genetic Operators

**Crossover** (80% probability):
- Set-aware uniform crossover
- Each gene (index) chosen from either parent
- Handle duplicates by filling with random unseen indices

**Mutation** (applied to all offspring):
- **Index replacement** (70%): Replace 1-5 random indices
- **Segment shuffle** (20%): Shuffle random 10-20% segment
- **Swap mutation** (10%): Swap two indices
- Always enforce uniqueness

**Selection**:
- Tournament selection (size 2)
- Based on Pareto rank (lower = better) + crowding distance (higher = better)

### Multi-Objective Handling
- **NSGA-II algorithm**: Non-dominated sorting + crowding distance
- Maintains diverse Pareto front
- No manual objective weighting needed

---

## 6. KEY INNOVATIONS & CONTRIBUTIONS

1. **Proxy-based fitness function**: Enables fast GA evaluation without training models
2. **Multi-objective optimization**: Balances informativeness, diversity, and class balance
3. **SBSE framing**: Treats dataset selection as a search problem
4. **Pareto efficiency analysis**: Shows data efficiency gains over random sampling
5. **Scalable approach**: Can handle large datasets (20k+ samples) efficiently

---

## 7. EXPECTED OUTCOMES

### Deliverables
1. **Pareto efficiency curve**: Accuracy vs. Dataset Size
   - Shows GA-selected subsets outperform random sampling
   - Demonstrates data efficiency (e.g., 500 GA-selected samples ≈ 2000 random samples)

2. **Pareto fronts**: 3D visualization of (difficulty, diversity, balance) trade-offs

3. **Selected subsets**: Optimal k-sized subsets for each k value

4. **Trained models**: CNNs trained on GA-selected subsets, validated on test set

### Success Metrics
- GA-selected subsets achieve **higher accuracy** than random subsets of same size
- GA-selected subsets achieve **similar accuracy** to larger random subsets (data efficiency)
- Pareto fronts show **meaningful trade-offs** between objectives
- Computational cost remains **feasible** (hours, not days)

---

## SUMMARY FOR PRESENTATION

**Problem**: Find optimal k-sized training subsets that maximize model performance

**Challenge**: Combinatorially complex search space (10^240+ combinations) with multiple conflicting objectives

**SBSE Approach**: Use genetic algorithm (NSGA-II) to search solution space

**Key Innovation**: Proxy fitness function (committee entropy + embedding diversity + class balance) instead of expensive CNN training

**Result**: Efficiently find Pareto-optimal subsets that outperform random sampling, demonstrating data efficiency gains

