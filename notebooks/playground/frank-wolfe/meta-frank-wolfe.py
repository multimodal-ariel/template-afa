# %%
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Literal

import matplotlib.pyplot as plt
import numpy as np


# %%
class SimpleLinearOracle:
    """
    A simple online linear optimization oracle.
    This is a basic implementation - you can replace with more sophisticated ones.
    """

    def __init__(self, constraint_set_optimizer, learning_rate=0.01):
        """
        Args:
            constraint_set_optimizer: Function that solves max/min <d, v> over constraint set K
                                     Takes gradient d, returns optimal point v
            learning_rate: Learning rate for the online algorithm
        """
        self.optimizer = constraint_set_optimizer
        self.history = []  # Store past gradients
        self.learning_rate = learning_rate
        self.last_output = None

    def get_output_from_previous_round(self):
        """Returns the solution from the previous round (Line 4 of Algorithm 1)"""
        return self.last_output

    def receive_feedback_and_compute_next(self, v_t, d_t):
        """
        Receive feedback and compute solution for next round (Line 8 of Algorithm 1)

        Args:
            v_t: The point that was used
            d_t: The gradient at that point
        """
        # Store feedback
        self.history.append((v_t, d_t))

        # Simple strategy: Use exponential moving average of past gradients
        if len(self.history) == 1:
            avg_gradient = d_t
        else:
            # Weighted average with more weight on recent gradients
            avg_gradient = d_t
            for i, (_, past_d) in enumerate(self.history[:-1]):
                weight = np.exp(-self.learning_rate * (len(self.history) - i - 1))
                avg_gradient = avg_gradient + weight * past_d
            avg_gradient = avg_gradient / (
                1
                + sum(
                    np.exp(-self.learning_rate * j) for j in range(1, len(self.history))
                )
            )

        # Compute optimal solution for this gradient
        self.last_output = self.optimizer(avg_gradient)
        return self.last_output


# %%
# Example: Simple constraint set (box constraints [0,1]^n)
class ConstraintSet(ABC):
    def __init__(self, dimension):
        self.dim = dimension

    @abstractmethod
    def minimize_linear(self, gradient) -> np.ndarray:
        """Minimize <gradient, v> over [0,1]^n"""

    @abstractmethod
    def maximize_linear(self, gradient) -> np.ndarray:
        """Maximize <gradient, v> over [0,1]^n"""

    @abstractmethod
    def get_any_point(self) -> np.ndarray:
        """Return any feasible point"""


# Example: Simple constraint set (box constraints [0,1]^n)
class BoxConstraintSet(ConstraintSet):
    def minimize_linear(self, gradient):
        """Minimize <gradient, v> over [0,1]^n"""
        # Minimum is achieved at 0 where gradient is positive, 1 where negative
        return (gradient < 0).astype(float)

    def maximize_linear(self, gradient):
        """Maximize <gradient, v> over [0,1]^n"""
        # Maximum is achieved at 1 where gradient is positive, 0 where negative
        return (gradient > 0).astype(float)

    def get_any_point(self):
        """Return any feasible point"""
        return 0.5 * np.ones(self.dim)


# %%
def meta_frank_wolfe(
    constraint_set_K: ConstraintSet,
    time_horizon_T: int,
    num_fw_steps_K: int,
    gradient_oracle: Callable,
    objective_functions: list[Callable],  # Added to track actual objective values
    problem_type: Literal["convex", "submodular"] = "convex",
    initial_point=None,
):
    """
    Algorithm 1: Meta-Frank-Wolfe

    Args:
        constraint_set_K: Object with methods for linear optimization
        time_horizon_T: Number of rounds
        num_fw_steps_K: Number of Frank-Wolfe steps per round
        gradient_oracle: Function that returns stochastic gradient at a point
        problem_type: Either 'convex' (minimization) or 'submodular' (maximization)
        initial_point: Starting point x_1
        diameter_D: Diameter of constraint set
        smoothness_L: Smoothness parameter
        variance_sigma: Variance bound for stochastic gradients

    Returns:
        List of points x_t for t = 1, ..., T
    """

    # Line 1: Initialize online linear optimization oracles E^(1), ..., E^(K)
    oracles: list[SimpleLinearOracle] = list()
    for k in range(num_fw_steps_K):
        if problem_type == "convex":
            # For convex, we minimize
            optimizer = lambda d: constraint_set_K.minimize_linear(d)
        else:
            # For submodular, we maximize
            optimizer = lambda d: constraint_set_K.maximize_linear(d)

        oracles.append(SimpleLinearOracle(optimizer))

    # Line 2: Initialize d^(0)_t = 0 and x^(1)_t = x_1

    # Initialize x_1
    if initial_point is None:
        if problem_type == "submodular":
            x_current = np.zeros_like(constraint_set_K.get_any_point())
        else:
            x_current = constraint_set_K.get_any_point()
    else:
        x_current = initial_point.copy()

    # d[k][t] represents d^(k)_t in the paper
    d = np.zeros((num_fw_steps_K + 1, x_current.shape[0]))  # d^(0) to d^(K)

    # Store all iterates
    all_x_t = []
    all_losses = list()

    # Line 3: Main loop for t = 1, 2, ..., T
    for t in range(1, time_horizon_T + 1):
        # Store x^(k) for k = 1, ..., K+1 for this round
        x_k = np.zeros((num_fw_steps_K + 1, x_current.shape[0]))
        x_k[0] = x_current.copy()  # x^(1)_t = x_t

        # Line 4: Get v^(k)_t from oracle E^(k) output in round t-1
        v_k = []
        for k in range(num_fw_steps_K):
            v_from_oracle = oracles[k].get_output_from_previous_round()
            if v_from_oracle is None:
                # First round - oracle has no previous output
                if problem_type == "convex":
                    v_from_oracle = constraint_set_K.minimize_linear(
                        np.zeros_like(x_current)
                    )
                else:
                    v_from_oracle = constraint_set_K.maximize_linear(
                        np.zeros_like(x_current)
                    )
            v_k.append(v_from_oracle)

        # Line 5: x^(k+1)_t ← update(x^(k)_t, v^(k)_t, η_k) for k = 1...K
        for k in range(num_fw_steps_K):
            # Step size from paper
            if problem_type == "convex":
                eta_k = 1.0 / (k + 4)  # 1/(k+3) but 0-indexed
                # Convex update: x^(k+1) = (1 - η_k) * x^(k) + η_k * v^(k)
                x_k[k + 1] = (1 - eta_k) * x_k[k] + eta_k * v_k[k]
            else:
                eta_k = 1.0 / num_fw_steps_K
                # Submodular update: x^(k+1) = x^(k) + η_k * (v^(k) - x^(k))
                x_k[k + 1] = x_k[k] + eta_k * (v_k[k] - x_k[k])

        # Line 6: Play x_t = x^(K+1)_t, then obtain value f_t(x_t) and oracle access
        x_current = x_k[num_fw_steps_K].copy()
        all_x_t.append(x_current.copy())

        # Track the actual loss
        current_loss = objective_functions[t - 1](x_current)
        all_losses.append(current_loss)

        # In real usage, you would get f_t(x_t) here
        # For now, we just get gradient estimates at each x^(k)_t

        # Line 7: d^(k)_t ← (1 - ρ_k)d^(k-1)_t + ρ_k * ∇f_t(x^(k-1)_t) for k = 1...K
        for k in range(1, num_fw_steps_K + 1):
            # Averaging parameter from paper
            rho_k = 2.0 / ((k + 3) ** (2.0 / 3.0))

            # Get stochastic gradient at x^(k-1)_t
            grad_estimate = gradient_oracle(x_k[k - 1], t)

            # Variance reduction update
            d[k] = (1 - rho_k) * d[k - 1] + rho_k * grad_estimate

        # Line 8: Feedback <v^(k)_t, d^(k)_t> to E^(k) for k = 1...K
        for k in range(num_fw_steps_K):
            oracles[k].receive_feedback_and_compute_next(v_k[k], d[k + 1])

    return all_x_t, all_losses


# %%
# # Example usage
# # Problem setup
# dimension = 10
# T = 100  # Time horizon
# K = 5  # Number of Frank-Wolfe steps

# # Create constraint set
# constraint_set = BoxConstraintSet(dimension)

# # Simple quadratic objective: f_t(x) = ||x - target_t||^2
# # where target_t changes each round (adversarial)
# targets = [np.random.rand(dimension) for _ in range(T)]


# def stochastic_gradient_oracle(x, round_t):
#     """Returns stochastic gradient of f_t at x"""
#     # True gradient: 2(x - target)
#     true_grad = 2 * (x - targets[round_t - 1])
#     # Add noise
#     noise = np.random.randn(dimension) * 0.1
#     return true_grad + noisej


# # Run algorithm
# print("Running Meta-Frank-Wolfe...")
# solutions = meta_frank_wolfe(
#     constraint_set_K=constraint_set,
#     time_horizon_T=T,
#     num_fw_steps_K=K,
#     gradient_oracle=stochastic_gradient_oracle,
#     problem_type="convex",
#     initial_point=np.ones(dimension) * 0.5,
# )

# print(f"Completed {T} rounds")
# print(f"Final solution: {solutions[-1]}")

# # Compute regret (simplified)
# total_loss = sum(np.linalg.norm(solutions[t] - targets[t]) ** 2 for t in range(T))
# print(f"Total loss: {total_loss:.2f}")


# %%
def visualize_minimization():
    """Create a 2D example that's easy to visualize"""

    # Setup: 2D problem so we can visualize
    dimension = 2
    T = 50  # Time horizon
    K = 3  # Frank-Wolfe steps

    # Create moving targets in a pattern (so we can see if algorithm learns)
    # Targets move in a predictable pattern: circular motion
    angles = np.linspace(0, 2 * np.pi, T)
    radius = 0.3
    center = np.array([0.5, 0.5])
    targets = [center + radius * np.array([np.cos(a), np.sin(a)]) for a in angles]

    # Define objective functions: f_t(x) = ||x - target_t||^2
    objective_functions = [
        lambda x, t=t: np.linalg.norm(x - targets[t]) ** 2 for t in range(T)
    ]

    # Stochastic gradient oracle
    def stochastic_gradient_oracle(x, round_t):
        true_grad = 2 * (x - targets[round_t - 1])
        noise = np.random.randn(dimension) * 0.05  # Small noise
        return true_grad + noise

    # Run our algorithm
    constraint_set = BoxConstraintSet(dimension)
    initial = np.array([0.5, 0.5])

    print("Running Meta-Frank-Wolfe...")
    solutions, losses = meta_frank_wolfe(
        constraint_set_K=constraint_set,
        time_horizon_T=T,
        num_fw_steps_K=K,
        gradient_oracle=stochastic_gradient_oracle,
        objective_functions=objective_functions,
        problem_type="convex",
        initial_point=initial,
    )

    # Compute the best fixed point in hindsight
    print("Computing best fixed point in hindsight...")
    best_fixed_point = None
    best_fixed_loss = float("inf")

    # Grid search for best fixed point
    for x1 in np.linspace(0, 1, 20):
        for x2 in np.linspace(0, 1, 20):
            point = np.array([x1, x2])
            total_loss = sum(f(point) for f in objective_functions)
            if total_loss < best_fixed_loss:
                best_fixed_loss = total_loss
                best_fixed_point = point

    # Compute regret
    algorithm_total_loss = sum(losses)
    regret = algorithm_total_loss - best_fixed_loss

    print(f"\nResults:")
    print(f"Algorithm's total loss: {algorithm_total_loss:.3f}")
    print(f"Best fixed point's total loss: {best_fixed_loss:.3f}")
    print(f"Regret: {regret:.3f}")
    print(f"Average regret per round: {regret/T:.3f}")

    # Create visualizations
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # 1. Trajectory plot
    ax = axes[0, 0]
    solutions = np.array(solutions)
    targets_array = np.array(targets)

    # Plot constraint set boundary
    rectangle = plt.Rectangle((0, 0), 1, 1, fill=False, edgecolor="black", linewidth=2)
    ax.add_patch(rectangle)

    # Plot targets trajectory
    ax.plot(targets_array[:, 0], targets_array[:, 1], "r--", alpha=0.5, label="Targets")
    ax.scatter(targets_array[:, 0], targets_array[:, 1], c="red", s=20, alpha=0.3)

    # Plot algorithm's trajectory
    ax.plot(
        solutions[:, 0],
        solutions[:, 1],
        "b-",
        alpha=0.7,
        label="Algorithm",
        linewidth=2,
    )
    ax.scatter(solutions[:, 0], solutions[:, 1], c="blue", s=30, alpha=0.5)

    # Mark start and end
    ax.scatter(
        solutions[0, 0],
        solutions[0, 1],
        c="green",
        s=100,
        marker="o",
        label="Start",
        zorder=5,
    )
    ax.scatter(
        solutions[-1, 0],
        solutions[-1, 1],
        c="orange",
        s=100,
        marker="s",
        label="End",
        zorder=5,
    )

    # Mark best fixed point
    ax.scatter(
        best_fixed_point[0],
        best_fixed_point[1],
        c="purple",
        s=150,
        marker="*",
        label=f"Best fixed point",
        zorder=6,
    )

    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_title("Algorithm Trajectory vs Moving Targets")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)

    # 2. Loss over time
    ax = axes[0, 1]
    ax.plot(range(1, T + 1), losses, "b-", label="Algorithm loss", linewidth=2)
    best_fixed_losses = [f(best_fixed_point) for f in objective_functions]
    ax.plot(
        range(1, T + 1),
        best_fixed_losses,
        "r--",
        label="Best fixed point loss",
        alpha=0.7,
    )
    ax.fill_between(range(1, T + 1), losses, best_fixed_losses, alpha=0.3, color="gray")
    ax.set_xlabel("Round t")
    ax.set_ylabel("Loss f_t(x_t)")
    ax.set_title("Loss at Each Round")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Cumulative loss
    ax = axes[0, 2]
    cumulative_algo = np.cumsum(losses)
    cumulative_best = np.cumsum(best_fixed_losses)
    ax.plot(range(1, T + 1), cumulative_algo, "b-", label="Algorithm", linewidth=2)
    ax.plot(
        range(1, T + 1), cumulative_best, "r--", label="Best fixed point", linewidth=2
    )
    ax.fill_between(
        range(1, T + 1),
        cumulative_algo,
        cumulative_best,
        alpha=0.3,
        color="yellow",
        label="Regret",
    )
    ax.set_xlabel("Round t")
    ax.set_ylabel("Cumulative Loss")
    ax.set_title(f"Cumulative Loss (Final Regret = {regret:.2f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Distance to target over time
    ax = axes[1, 0]
    distances = [np.linalg.norm(solutions[t] - targets[t]) for t in range(T)]
    ax.plot(range(1, T + 1), distances, "g-", linewidth=2)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Distance ||x_t - target_t||")
    ax.set_title("Distance to Current Target")
    ax.grid(True, alpha=0.3)

    # 5. Regret growth
    ax = axes[1, 1]
    regret_over_time = cumulative_algo - cumulative_best
    ax.plot(range(1, T + 1), regret_over_time, "purple", linewidth=2)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Regret")
    ax.set_title("Regret Growth Over Time")
    ax.grid(True, alpha=0.3)

    # 6. Average regret
    ax = axes[1, 2]
    avg_regret = regret_over_time / np.arange(1, T + 1)
    ax.plot(range(1, T + 1), avg_regret, "orange", linewidth=2)
    ax.set_xlabel("Round t")
    ax.set_ylabel("Average Regret (Regret/t)")
    ax.set_title("Average Regret (Should Decrease)")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("meta_fw_visualization.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Print analysis
    print("\n📊 What to Look For:")
    print("1. Blue trajectory should adapt to follow the red targets")
    print("2. Loss should be reasonably low (algorithm is minimizing)")
    print("3. Cumulative loss grows sublinearly (regret is sublinear)")
    print("4. Average regret decreases over time (algorithm is learning)")

    if avg_regret[-1] < avg_regret[5]:  # Compare late vs early performance
        print(
            "\n✅ SUCCESS: Algorithm is learning! Average regret decreased from {:.3f} to {:.3f}".format(
                avg_regret[5], avg_regret[-1]
            )
        )
    else:
        print("\n⚠️  Algorithm may need tuning")

    return solutions, targets, losses


visualize_minimization()

# %%
