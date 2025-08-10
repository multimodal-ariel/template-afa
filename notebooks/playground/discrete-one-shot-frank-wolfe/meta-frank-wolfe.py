# %%
from __future__ import annotations

import random
from typing import List, Set

import matplotlib.pyplot as plt
import numpy as np


# %%
class DiscreteSubmodularFunction:
    """Base class for discrete submodular functions"""

    def __init__(self, ground_set_size: int):
        self.n = ground_set_size

    def value(self, S: Set[int]) -> float:
        """Compute f(S) for subset S"""
        raise NotImplementedError

    def marginal_gain(self, S: Set[int], i: int) -> float:
        """Compute f(S ∪ {i}) - f(S)"""
        if i in S:
            return 0.0
        return self.value(S | {i}) - self.value(S)


class CoverageFunction(DiscreteSubmodularFunction):
    """
    Coverage function: Each element covers some items,
    we want to maximize total coverage
    """

    def __init__(self, coverage_sets: List[Set[int]]):
        super().__init__(len(coverage_sets))
        self.coverage_sets = coverage_sets

    def value(self, S: Set[int]) -> float:
        """Returns number of items covered by elements in S"""
        covered = set()
        for i in S:
            covered = covered | self.coverage_sets[i]
        return len(covered)


class MatroidConstraint:
    """Cardinality constraint: |S| <= k"""

    def __init__(self, n: int, k: int):
        self.n = n
        self.k = k

    def is_feasible(self, S: Set[int]) -> bool:
        return len(S) <= self.k

    def maximize_linear(self, gradient: np.ndarray) -> np.ndarray:
        """
        Maximize <gradient, x> over matroid polytope
        For cardinality: pick top-k elements
        """
        x = np.zeros(self.n)
        top_k_indices = np.argsort(gradient)[-self.k :]  # Largest k values
        x[top_k_indices] = 1.0
        return x


def estimate_multilinear_gradient_one_sample(
    f: DiscreteSubmodularFunction, x: np.ndarray
) -> np.ndarray:
    """
    One-sample gradient estimation for multilinear extension
    From Eq. (1) in the paper: ∂f̄/∂x_i = E[f(R ∪ {i}) - f(R)]
    where R is random set with Pr[j ∈ R] = x_j for j ≠ i
    """
    n = len(x)
    gradient = np.zeros(n)

    for i in range(n):
        # Sample random set R from distribution x (excluding i)
        R = set()
        for j in range(n):
            if j != i and random.random() < x[j]:
                R.add(j)

        # Estimate gradient coordinate i
        gradient[i] = f.marginal_gain(R, i)

    return gradient


def round_solution(x: np.ndarray, constraint: MatroidConstraint) -> Set[int]:
    """
    Simple randomized rounding: treat x as probabilities
    Then fix to ensure feasibility
    """
    S = set()

    # First, randomly include elements based on probabilities
    for i in range(len(x)):
        if random.random() < x[i]:
            S.add(i)

    # Ensure feasibility (for cardinality, keep top-k by x values)
    if len(S) > constraint.k:
        # Keep elements with highest x values
        sorted_elements = sorted(S, key=lambda i: x[i], reverse=True)
        S = set(sorted_elements[: constraint.k])

    return S


def one_shot_frank_wolfe_discrete(
    functions: List[DiscreteSubmodularFunction],
    constraint: MatroidConstraint,
    T: int,
    verbose: bool = True,
) -> tuple:
    """
    Algorithm 2 adapted for discrete submodular maximization

    Returns:
        - solutions_continuous: List of fractional solutions x_t
        - solutions_discrete: List of discrete sets X_t
        - values: List of function values f_t(X_t)
        - gradients_history: History of gradient estimates
    """
    n = constraint.n

    # Initialize (Line 1: d_0 ← 0)
    d = np.zeros(n)
    x = np.zeros(n)  # Start from empty set

    # Storage for visualization
    solutions_continuous = []
    solutions_discrete = []
    values = []
    gradients_history = []

    # Main loop (Line 2: for t = 1, ..., T)
    for t in range(1, T + 1):
        if verbose and t % 10 == 0:
            print(f"Round {t}/{T}")

        # Line 3: Play x_t
        solutions_continuous.append(x.copy())

        # Round to discrete solution
        X_t = round_solution(x, constraint)
        solutions_discrete.append(X_t)

        # Get function value
        f_t = functions[t - 1]
        value = f_t.value(X_t)
        values.append(value)

        # Line 4: Update variance-reduced gradient
        # d_t ← (1 - ρ_t)d_{t-1} + ρ_t∇̃f_t(x_t)
        rho_t = 2.0 / ((t + 3) ** (2.0 / 3.0))

        # One-sample gradient estimate
        grad_estimate = estimate_multilinear_gradient_one_sample(f_t, x)
        gradients_history.append(grad_estimate.copy())

        # Variance reduction
        d = (1 - rho_t) * d + rho_t * grad_estimate

        # Line 5: Linear optimization
        # v_t ← arg max_{v ∈ K} <d_t, v>
        v = constraint.maximize_linear(d)

        # Line 6: Frank-Wolfe update
        # For submodular: x_{t+1} ← x_t + η_t(v_t - x_t)
        eta_t = 1.0 / (t + 3) ** 0.5  # Can tune this
        x = x + eta_t * (v - x)

        # Ensure x stays in [0,1]^n
        x = np.clip(x, 0, 1)

    return solutions_continuous, solutions_discrete, values, gradients_history


def create_example_coverage_problem(n_elements: int = 20, n_items: int = 50):
    """
    Create a synthetic coverage problem
    Each element can cover some random subset of items
    """
    coverage_sets = []
    for i in range(n_elements):
        # Each element covers 10-30% of items randomly
        coverage_size = random.randint(int(0.1 * n_items), int(0.3 * n_items))
        covered_items = set(random.sample(range(n_items), coverage_size))
        coverage_sets.append(covered_items)

    return CoverageFunction(coverage_sets)


def visualize_discrete_optimization(
    solutions_continuous: List[np.ndarray],
    solutions_discrete: List[Set[int]],
    values: List[float],
    gradients_history: List[np.ndarray],
    functions: List[DiscreteSubmodularFunction],
    constraint: MatroidConstraint,
):
    """Create comprehensive visualization of the discrete optimization process"""

    T = len(values)
    n = constraint.n

    fig = plt.figure(figsize=(16, 10))

    # 1. Function values over time
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(range(1, T + 1), values, "b-", linewidth=2, label="Algorithm")

    # Compute offline optimum for each function (greedy)
    greedy_values = []
    for f in functions:
        S = set()
        for _ in range(constraint.k):
            best_elem = None
            best_gain = -1
            for i in range(n):
                if i not in S:
                    gain = f.marginal_gain(S, i)
                    if gain > best_gain:
                        best_gain = gain
                        best_elem = i
            if best_elem is not None:
                S.add(best_elem)
        greedy_values.append(f.value(S))

    ax1.plot(range(1, T + 1), greedy_values, "r--", alpha=0.7, label="Greedy (offline)")
    ax1.set_xlabel("Round t")
    ax1.set_ylabel("Function Value f_t(X_t)")
    ax1.set_title("Coverage Value Over Time")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 2. Continuous solution evolution (heatmap)
    ax2 = plt.subplot(2, 3, 2)
    continuous_matrix = np.array(solutions_continuous).T
    im = ax2.imshow(continuous_matrix, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
    ax2.set_xlabel("Round t")
    ax2.set_ylabel("Element")
    ax2.set_title("Fractional Solution x_t (Heatmap)")
    plt.colorbar(im, ax=ax2, fraction=0.046)

    # 3. Set size over time
    ax3 = plt.subplot(2, 3, 3)
    set_sizes = [len(S) for S in solutions_discrete]
    ax3.plot(range(1, T + 1), set_sizes, "g-", linewidth=2)
    ax3.axhline(
        y=constraint.k,
        color="r",
        linestyle="--",
        label=f"Constraint (k={constraint.k})",
    )
    ax3.set_xlabel("Round t")
    ax3.set_ylabel("|X_t|")
    ax3.set_title("Selected Set Size")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # 4. Element selection frequency
    ax4 = plt.subplot(2, 3, 4)
    selection_counts = np.zeros(n)
    for S in solutions_discrete:
        for i in S:
            selection_counts[i] += 1

    bars = ax4.bar(range(n), selection_counts / T, color="skyblue")
    ax4.set_xlabel("Element")
    ax4.set_ylabel("Selection Frequency")
    ax4.set_title("How Often Each Element is Selected")
    ax4.set_xticks(range(0, n, max(1, n // 10)))

    # Color top-k bars differently
    top_k_elements = np.argsort(selection_counts)[-constraint.k :]
    for i in top_k_elements:
        bars[i].set_color("darkblue")

    # 5. Gradient magnitude over time
    ax5 = plt.subplot(2, 3, 5)
    gradient_norms = [np.linalg.norm(g) for g in gradients_history]
    ax5.plot(range(1, T + 1), gradient_norms, "purple", linewidth=2)
    ax5.set_xlabel("Round t")
    ax5.set_ylabel("||∇f_t||")
    ax5.set_title("Gradient Magnitude (Shows Convergence)")
    ax5.grid(True, alpha=0.3)

    # 6. Regret analysis
    ax6 = plt.subplot(2, 3, 6)
    cumulative_algo = np.cumsum(values)
    cumulative_greedy = np.cumsum(greedy_values)

    # Also compare with best fixed set
    print("Computing best fixed set (this may take a moment)...")
    best_fixed_value = 0
    best_fixed_set = set()

    # Simple approximation: try several random sets
    for _ in range(100):
        S = set(random.sample(range(n), min(constraint.k, n)))
        total_value = sum(f.value(S) for f in functions)
        if total_value > best_fixed_value:
            best_fixed_value = total_value
            best_fixed_set = S

    best_fixed_values = [functions[t].value(best_fixed_set) for t in range(T)]
    cumulative_fixed = np.cumsum(best_fixed_values)

    ax6.plot(range(1, T + 1), cumulative_algo, "b-", linewidth=2, label="Algorithm")
    ax6.plot(range(1, T + 1), cumulative_greedy, "r--", label="Greedy (offline)")
    ax6.plot(range(1, T + 1), cumulative_fixed, "g:", label="Best Fixed Set")

    regret = cumulative_fixed[-1] - cumulative_algo[-1]
    ax6.set_xlabel("Round t")
    ax6.set_ylabel("Cumulative Value")
    ax6.set_title(f"Cumulative Performance (Regret = {regret:.1f})")
    ax6.legend()
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("discrete_os_fw_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print summary statistics
    print("\n📊 Summary Statistics:")
    print(f"Average value: {np.mean(values):.2f}")
    print(f"Final value: {values[-1]:.2f}")
    print(f"Best single-round value: {max(values):.2f}")
    print(f"Average set size: {np.mean(set_sizes):.1f}")
    print(f"Regret vs best fixed set: {regret:.2f}")
    print(f"Most selected elements: {np.argsort(selection_counts)[-5:]}")


# %%
# Example usage
# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Problem parameters
n_elements = 20  # Number of elements to choose from
n_items = 50  # Number of items to cover
k = 5  # Can select at most k elements
T = 100  # Number of rounds

print(f"🎯 Discrete Submodular Maximization")
print(f"Elements: {n_elements}, Items to cover: {n_items}")
print(f"Constraint: Select at most {k} elements")
print(f"Rounds: {T}\n")

# Create sequence of coverage functions
# In adversarial setting, coverage sets can change each round
functions = []
for t in range(T):
    # Create slightly different coverage function each round
    # (simulating changing item importance or availability)
    functions.append(create_example_coverage_problem(n_elements, n_items))

# Create constraint
constraint = MatroidConstraint(n_elements, k)

# Run algorithm
print("Running One-Shot Frank-Wolfe for discrete optimization...")
solutions_cont, solutions_disc, values, gradients = one_shot_frank_wolfe_discrete(
    functions, constraint, T
)

print(f"✅ Completed {T} rounds")
print(f"Average coverage: {np.mean(values):.2f} items")

# Visualize results
print("\nGenerating visualizations...")
visualize_discrete_optimization(
    solutions_cont, solutions_disc, values, gradients, functions, constraint
)

print("\n💡 What to look for:")
print("1. Function values should be reasonably high")
print("2. Fractional solution should converge to good elements")
print("3. Frequently selected elements are likely high-value")
print("4. Gradient magnitude decreases (algorithm converges)")
print("5. Cumulative value grows steadily")

# %%
