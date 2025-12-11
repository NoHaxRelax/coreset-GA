"""
Generate synthetic placeholder results (plots + tables) so the Expected Results
slide can be populated before real experiments complete.
Outputs are written to the `results/` directory.
"""

import json
import os
import numpy as np
import pandas as pd
import matplotlib

# Use a non-interactive backend for environments without display
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # needed for 3D projection


def main() -> None:
    os.makedirs("results", exist_ok=True)

    ks = np.array([50, 100, 200, 500, 750, 1000])
    rng = np.random.default_rng(0)

    def noise(scale: float = 0.3) -> np.ndarray:
        return rng.normal(0, scale, size=ks.size)

    ga = np.clip(np.array([94, 96, 97.5, 98.6, 98.8, 99.0]) + noise(0.1), 0, 100)
    rnd = np.clip(ga - (2 + noise(0.25)), 0, 100)
    hardest = np.clip(ga - (3 + noise(0.35)), 0, 100)
    balanced = np.clip(ga - (2.5 + noise(0.3)), 0, 100)
    full = 99.2

    # Accuracy vs subset size
    plt.figure(figsize=(6, 4))
    for series, label in [
        (ga, "GA"),
        (rnd, "Random"),
        (hardest, "Hardest-only"),
        (balanced, "Balanced-only"),
    ]:
        plt.plot(ks, series, marker="o", label=label)
    plt.axhline(full, color="k", ls="--", label="Full dataset")
    plt.xlabel("Subset size k")
    plt.ylabel("Test accuracy (synthetic %)")
    plt.title("Accuracy vs subset size (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/accuracy_vs_size.png", dpi=200)

    # Pareto front (synthetic 3D scatter)
    fig = plt.figure(figsize=(5, 4))
    ax = fig.add_subplot(111, projection="3d")
    diff = rng.uniform(0.7, 1.0, size=30)
    div = rng.uniform(0.7, 1.0, size=30)
    bal = rng.uniform(0.7, 1.0, size=30)
    sc = ax.scatter(diff, div, bal, c=diff + div + bal, cmap="viridis")
    ax.set_xlabel("Difficulty")
    ax.set_ylabel("Diversity")
    ax.set_zlabel("Balance")
    ax.set_title("Pareto front (synthetic)")
    fig.colorbar(sc, ax=ax, shrink=0.6, pad=0.1)
    plt.tight_layout()
    plt.savefig("results/pareto_fronts_3d.png", dpi=200)

    # Convergence curves (hypervolume proxy)
    plt.figure(figsize=(6, 4))
    gens = np.arange(1, 21)
    hv = np.linspace(0.5, 0.95, gens.size) + rng.normal(0, 0.01, gens.size)
    plt.plot(gens, hv, marker="o")
    plt.xlabel("Generation")
    plt.ylabel("Hypervolume (synthetic)")
    plt.title("GA convergence (synthetic)")
    plt.tight_layout()
    plt.savefig("results/convergence_curves.png", dpi=200)

    # Training efficiency
    plt.figure(figsize=(6, 4))
    eff_ga = ga / ks
    eff_rnd = rnd / ks
    plt.plot(ks, eff_ga, marker="o", label="GA")
    plt.plot(ks, eff_rnd, marker="o", label="Random")
    plt.xlabel("Subset size k")
    plt.ylabel("Accuracy per sample (synthetic %)")
    plt.title("Training efficiency (synthetic)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/training_efficiency.png", dpi=200)

    # Summary table and JSON
    df = pd.DataFrame(
        {
            "k": ks,
            "acc_ga": ga,
            "acc_random": rnd,
            "acc_hardest": hardest,
            "acc_balanced": balanced,
        }
    )
    df.to_csv("results/summary_table.csv", index=False)

    json.dump(
        {
            "synthetic": True,
            "ks": ks.tolist(),
            "acc_ga": ga.tolist(),
            "acc_random": rnd.tolist(),
            "acc_hardest": hardest.tolist(),
            "acc_balanced": balanced.tolist(),
            "acc_full": full,
        },
        open("results/evaluation.json", "w"),
        indent=2,
    )

    print("Synthetic outputs written to results/:")
    for fname in [
        "accuracy_vs_size.png",
        "pareto_fronts_3d.png",
        "convergence_curves.png",
        "training_efficiency.png",
        "summary_table.csv",
        "evaluation.json",
    ]:
        print(f" - results/{fname}")


if __name__ == "__main__":
    main()

