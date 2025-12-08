"""
Mutation operators for genetic algorithm.

Implements three mutation strategies:
1. Index replacement (70%): replace 1-5 random indices
2. Segment shuffle (20%): shuffle a random 10-20% segment
3. Swap mutation (10%): swap two indices
"""

import numpy as np
from typing import Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config
from ga.population import enforce_uniqueness


def mutate_index_replacement(
    chromosome: np.ndarray,
    pool_size: int,
    num_replacements: int = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: replace random indices with new ones.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        num_replacements: Number of indices to replace. If None, random between min and max.
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if num_replacements is None:
        num_replacements = rng.integers(
            config.MUTATION_MIN_REPLACEMENTS,
            config.MUTATION_MAX_REPLACEMENTS + 1
        )
    
    num_replacements = min(num_replacements, len(chromosome))
    
    # Create a copy
    mutated = chromosome.copy()
    
    # Select indices to replace
    indices_to_replace = rng.choice(len(mutated), size=num_replacements, replace=False)
    
    # Get currently used indices
    used_indices = set(mutated)
    
    # Find available indices
    available_indices = np.array([i for i in range(pool_size) if i not in used_indices])
    
    if len(available_indices) < num_replacements:
        # Not enough available indices, just replace what we can
        num_replacements = len(available_indices)
        indices_to_replace = indices_to_replace[:num_replacements]
    
    # Replace with random new indices
    new_indices = rng.choice(available_indices, size=num_replacements, replace=False)
    mutated[indices_to_replace] = new_indices
    
    # Enforce uniqueness and sort
    mutated = enforce_uniqueness(mutated, pool_size, rng=rng)
    
    return mutated


def mutate_segment_shuffle(
    chromosome: np.ndarray,
    pool_size: int,
    segment_size: int = None,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: shuffle a random segment of the chromosome.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        segment_size: Size of segment to shuffle. If None, random between min and max %.
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(chromosome) < 2:
        return chromosome.copy()
    
    if segment_size is None:
        # Random segment size between min and max percentage
        min_size = max(1, int(len(chromosome) * config.MUTATION_SEGMENT_MIN_PCT))
        max_size = max(1, int(len(chromosome) * config.MUTATION_SEGMENT_MAX_PCT))
        segment_size = rng.integers(min_size, max_size + 1)
    
    segment_size = min(segment_size, len(chromosome))
    
    # Create a copy
    mutated = chromosome.copy()
    
    # Select random segment
    start_idx = rng.integers(0, len(mutated) - segment_size + 1)
    end_idx = start_idx + segment_size
    
    # Shuffle the segment
    segment = mutated[start_idx:end_idx]
    rng.shuffle(segment)
    mutated[start_idx:end_idx] = segment
    
    # No need to enforce uniqueness (shuffling doesn't change values)
    return np.sort(mutated)


def mutate_swap(
    chromosome: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Mutation: swap two random indices.
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    if len(chromosome) < 2:
        return chromosome.copy()
    
    # Create a copy
    mutated = chromosome.copy()
    
    # Select two random indices to swap
    idx1, idx2 = rng.choice(len(mutated), size=2, replace=False)
    mutated[idx1], mutated[idx2] = mutated[idx2], mutated[idx1]
    
    # No need to enforce uniqueness (swapping doesn't change values)
    return np.sort(mutated)


def mutate(
    chromosome: np.ndarray,
    pool_size: int,
    rng: np.random.Generator = None
) -> np.ndarray:
    """
    Apply mutation to a chromosome using probabilistic selection.
    
    Selects mutation operator based on configured probabilities:
    - Index replacement: 70%
    - Segment shuffle: 20%
    - Swap: 10%
    
    Args:
        chromosome: Original chromosome
        pool_size: Size of the selection pool
        rng: Random number generator. If None, creates a new one.
        
    Returns:
        Mutated chromosome
    """
    if rng is None:
        rng = np.random.default_rng()
    
    # Select mutation operator based on probabilities
    rand = rng.random()
    
    if rand < config.MUTATION_INDEX_REPLACEMENT_PROB:
        return mutate_index_replacement(chromosome, pool_size, rng=rng)
    elif rand < config.MUTATION_INDEX_REPLACEMENT_PROB + config.MUTATION_SEGMENT_SHUFFLE_PROB:
        return mutate_segment_shuffle(chromosome, pool_size, rng=rng)
    else:
        return mutate_swap(chromosome, pool_size, rng=rng)


if __name__ == "__main__":
    # Test mutation operators
    print("Testing mutation operators...")
    
    k = 10
    pool_size = 100
    rng = np.random.default_rng(42)
    
    # Create test chromosome
    chromosome = np.sort(rng.choice(pool_size, size=k, replace=False))
    print(f"Original chromosome: {chromosome}")
    
    # Test index replacement
    print("\n1. Index replacement mutation:")
    mutated = mutate_index_replacement(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test segment shuffle
    print("\n2. Segment shuffle mutation:")
    mutated = mutate_segment_shuffle(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test swap
    print("\n3. Swap mutation:")
    mutated = mutate_swap(chromosome, pool_size, rng=rng)
    print(f"   Mutated: {mutated}")
    print(f"   Changed: {np.sum(chromosome != mutated)} indices")
    
    # Test probabilistic mutation
    print("\n4. Probabilistic mutation (10 runs):")
    for i in range(10):
        mutated = mutate(chromosome, pool_size, rng=rng)
        mutation_type = "replacement" if np.sum(chromosome != mutated) > 2 else \
                        "shuffle" if np.any(np.sort(chromosome) != np.sort(mutated)) == False else "swap"
        print(f"   Run {i+1}: {mutation_type} - {mutated[:5]}...")
