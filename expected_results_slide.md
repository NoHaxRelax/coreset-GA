# Expected Results (PPTX slide)
- Headline: Multi-objective GA (difficulty + diversity + balance) should deliver better accuracy-per-sample than random; current figures are placeholders until runs complete.
- What to show on the slide (with synthetic placeholders until real results land):
  - Main plot: `results/accuracy_vs_size.png` — test accuracy vs subset size k ∈ {50, 100, 200, 500, 750, 1000}; series for GA-selected, random mean ± std, hardest-only, balanced-only, full-dataset line.
  - Supporting plots: `results/pareto_fronts_3d.png` (difficulty/diversity/balance), `results/convergence_curves.png` (GA generations), `results/training_efficiency.png` (accuracy ÷ k).
  - Summary table: `results/summary_table.csv` and `results/evaluation.json` (per-k metrics).
- Placeholder quantitative expectations (replace with actuals):
  - Full MNIST baseline (upper bound): ~99% test accuracy.
  - k=50: GA ~93–95% vs random ~90–92%.
  - k=100: GA ~95–97% vs random ~93–95%.
  - k=200: GA ~97–98% vs random ~95–97%.
  - k=500: GA ~98–99% vs random ~97–98%.
  - Efficiency: GA curve should dominate random in accuracy-per-sample across k.
- How to generate (and regenerate) synthetic placeholder visuals:
  - Activate env and run the helper script (already run once):  
    `source .venv/bin/activate && python generate_synthetic_results.py`
  - Outputs land in `results/`:
    - `accuracy_vs_size.png`
    - `pareto_fronts_3d.png`
    - `convergence_curves.png`
    - `training_efficiency.png`
    - `summary_table.csv`
    - `evaluation.json` (marked `"synthetic": true`)
- Notes for the slide:
  - Clearly label visuals as synthetic placeholders.
  - Replace with real outputs after running the pipeline (`quick_start.py --k ...` or `experiments/run_k*.py` + `training/*` + `analysis/*` per README).
  - If time-constrained, keep the placeholder figures but call out that results are illustrative only.

