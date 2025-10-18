"""
Target function datasets for toy experiments.

This module provides datasets for studying interpolation capabilities
of regression vs flow models in low-data regimes.
"""

import logging
from typing import Dict, Literal, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def get_primes(n: int) -> list[int]:
    """Get first n prime numbers for frequency/phase generation."""
    primes = []
    candidate = 2
    while len(primes) < n:
        is_prime = True
        for p in primes:
            if p * p > candidate:
                break
            if candidate % p == 0:
                is_prime = False
                break
        if is_prime:
            primes.append(candidate)
        candidate += 1
    return primes


class TargetFunctionDataset(Dataset):
    """
    Dataset for target function f(c) = Σ wᵢ * trig_i(freq_i * c + phase_i)

    Supports both scalar and vector-valued functions with configurable:
    - Number of sine/cosine components
    - Uniform or inverse-frequency weighting
    - Prime-based frequencies/phases (no overlaps)
    - Low training samples, high eval samples

    Args:
        num_samples: Number of data points to generate
        target_dim: Dimension of target function (1 for scalar, >1 for vector)
        condition_dim: Dimension of conditioning variable (typically 1)
        num_components: Number of sine/cosine terms per output dimension
        c_min: Minimum value of conditioning variable
        c_max: Maximum value of conditioning variable
        weight_strategy: 'uniform' (all weights=1) or 'inverse_freq' (weight ∝ 1/freq)
        sampling_strategy: 'grid' (uniform spacing) or 'random' (uniform random)
        freq_seed: Random seed for frequency generation
        phase_seed: Random seed for phase generation
        weight_seed: Random seed for weight generation
        sample_seed: Random seed for sampling c values
        use_cosines: If True, alternate between sine and cosine terms
    """

    def __init__(
        self,
        num_samples: int = 64,
        target_dim: int = 1,
        condition_dim: int = 1,
        num_components: int = 5,
        c_min: float = 0.0,
        c_max: float = 1.0,
        weight_strategy: Literal["uniform", "inverse_freq"] = "uniform",
        sampling_strategy: Literal["grid", "random"] = "grid",
        freq_seed: int = 42,
        phase_seed: int = 43,
        weight_seed: int = 44,
        sample_seed: int = 45,
        use_cosines: bool = True,
    ):
        super().__init__()

        # Store configuration
        self.num_samples = num_samples
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.num_components = num_components
        self.c_min = c_min
        self.c_max = c_max
        self.weight_strategy = weight_strategy
        self.sampling_strategy = sampling_strategy
        self.freq_seed = freq_seed
        self.phase_seed = phase_seed
        self.weight_seed = weight_seed
        self.sample_seed = sample_seed
        self.use_cosines = use_cosines

        # Generate function parameters (frequencies, phases, weights)
        self._generate_function_params()

        # Generate data points
        self.c_values = self._generate_c_values()
        self.x_values = self._compute_target(self.c_values)

    def _generate_function_params(self):
        """
        Generate frequencies, phases, and weights using prime numbers.
        Each output dimension gets unique parameters to ensure independence.
        """
        # Need enough primes for all components across all dimensions
        total_components = self.target_dim * self.num_components
        primes = get_primes(total_components * 2)  # *2 for freq and phase

        self.frequencies = []
        self.phases = []
        self.weights = []

        prime_idx = 0
        for dim in range(self.target_dim):
            dim_freqs = []
            dim_phases = []
            dim_weights = []

            for comp in range(self.num_components):
                # Assign unique prime-based frequency and phase
                freq = float(primes[prime_idx])
                prime_idx += 1
                phase = (
                    float(primes[prime_idx]) / 10.0
                )  # Scale phase to reasonable range
                prime_idx += 1

                dim_freqs.append(freq)
                dim_phases.append(phase)

                # Compute weight based on strategy
                if self.weight_strategy == "uniform":
                    weight = 1.0
                elif self.weight_strategy == "inverse_freq":
                    weight = 1.0 / freq
                else:
                    raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")

                dim_weights.append(weight)

            self.frequencies.append(dim_freqs)
            self.phases.append(dim_phases)
            self.weights.append(dim_weights)

    def _generate_c_values(self) -> torch.Tensor:
        """Generate conditioning variable values."""
        np.random.seed(self.sample_seed)

        if self.sampling_strategy == "grid":
            # Uniform grid in [c_min, c_max]
            c = np.linspace(self.c_min, self.c_max, self.num_samples)
        elif self.sampling_strategy == "random":
            # Uniform random in [c_min, c_max]
            c = np.random.uniform(self.c_min, self.c_max, self.num_samples)
            c = np.sort(c)  # Sort for easier visualization
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

        return torch.tensor(c, dtype=torch.float32).reshape(-1, self.condition_dim)

    def _compute_target(self, c: torch.Tensor) -> torch.Tensor:
        """
        Compute target function f(c).

        Args:
            c: [n_samples, condition_dim] conditioning values

        Returns:
            x: [n_samples, target_dim] target values
        """
        c_np = c.numpy().flatten()
        x_values = np.zeros((len(c_np), self.target_dim))

        for dim in range(self.target_dim):
            for comp in range(self.num_components):
                freq = self.frequencies[dim][comp]
                phase = self.phases[dim][comp]
                weight = self.weights[dim][comp]

                # Alternate between sine and cosine if enabled
                if self.use_cosines and comp % 2 == 1:
                    x_values[:, dim] += weight * np.cos(freq * c_np + phase)
                else:
                    x_values[:, dim] += weight * np.sin(freq * c_np + phase)

        return torch.tensor(x_values, dtype=torch.float32)

    def generate_eval_data(
        self,
        num_samples: int,
        eval_seed: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate evaluation data with different c values.
        Does not cache results - useful for large evaluation sets.

        Args:
            num_samples: Number of evaluation samples
            eval_seed: Random seed for sampling

        Returns:
            Dict with 'c' and 'x' tensors
        """
        np.random.seed(eval_seed)

        if self.sampling_strategy == "grid":
            c = np.linspace(self.c_min, self.c_max, num_samples)
        else:
            c = np.random.uniform(self.c_min, self.c_max, num_samples)
            c = np.sort(c)

        c_tensor = torch.tensor(c, dtype=torch.float32).reshape(-1, self.condition_dim)
        x_tensor = self._compute_target(c_tensor)

        return {"c": c_tensor, "x": x_tensor}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                'c': [condition_dim] conditioning value
                'x': [target_dim] target value
        """
        return {
            "c": self.c_values[idx],
            "x": self.x_values[idx],
        }

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data at once (useful for full-batch training).

        Returns:
            Dict with:
                'c': [num_samples, condition_dim] conditioning values
                'x': [num_samples, target_dim] target values
        """
        return {
            "c": self.c_values,
            "x": self.x_values,
        }

    def get_function_description(self) -> str:
        """Get human-readable description of the target function."""
        desc_lines = [f"Target function f: ℝ → ℝ^{self.target_dim}"]
        desc_lines.append(f"Domain: c ∈ [{self.c_min:.2f}, {self.c_max:.2f}]")
        desc_lines.append(f"Weighting: {self.weight_strategy}")
        desc_lines.append(f"Components per dimension: {self.num_components}")
        desc_lines.append("")

        for dim in range(min(self.target_dim, 3)):  # Show first 3 dims max
            terms = []
            for comp in range(self.num_components):
                freq = self.frequencies[dim][comp]
                phase = self.phases[dim][comp]
                weight = self.weights[dim][comp]
                trig = "cos" if self.use_cosines and comp % 2 == 1 else "sin"
                terms.append(f"{weight:.3f}·{trig}({freq:.1f}c + {phase:.2f})")

            desc_lines.append(f"f_{dim}(c) = {' + '.join(terms)}")

        if self.target_dim > 3:
            desc_lines.append(f"... ({self.target_dim - 3} more dimensions)")

        return "\n".join(desc_lines)


class ProjectedTargetFunctionDataset(Dataset):
    """
    Dataset for projected target function: g(c) = P @ f(c)

    Creates a high-dimensional target function and projects it onto a
    lower-dimensional subspace using a rank-deficient projection matrix.

    The projection matrix P ∈ ℝ^(high_dim × high_dim) has rank = low_dim,
    constructed as P = A(A^T A)^(-1) A^T where A ∈ ℝ^(high_dim × low_dim).

    This setup allows studying:
    - Reconstruction in projected subspaces
    - Rank-deficient regression problems
    - Subspace learning capabilities

    Args:
        num_samples: Number of data points
        target_dim: Dimension of base target function f(c) and output g(c)
        condition_dim: Dimension of conditioning variable
        low_dim: Rank of projection matrix (low_dim < target_dim)
        num_components: Number of sine/cosine terms per dimension of f
        c_min: Minimum value of conditioning variable
        c_max: Maximum value of conditioning variable
        weight_strategy: 'uniform' or 'inverse_freq'
        sampling_strategy: 'grid' or 'random'
        freq_seed: Random seed for frequency generation
        phase_seed: Random seed for phase generation
        weight_seed: Random seed for weight generation
        proj_seed: Random seed for projection matrix
        sample_seed: Random seed for sampling c values
    """

    def __init__(
        self,
        num_samples: int = 64,
        target_dim: int = 8,
        condition_dim: int = 1,
        low_dim: int = 3,
        num_components: int = 5,
        c_min: float = 0.0,
        c_max: float = 1.0,
        weight_strategy: Literal["uniform", "inverse_freq"] = "uniform",
        sampling_strategy: Literal["grid", "random"] = "grid",
        freq_seed: int = 42,
        phase_seed: int = 43,
        weight_seed: int = 44,
        proj_seed: int = 46,
        sample_seed: int = 45,
    ):
        super().__init__()

        assert low_dim < target_dim, "low_dim must be < target_dim"

        self.num_samples = num_samples
        self.target_dim = target_dim
        self.condition_dim = condition_dim
        self.low_dim = low_dim
        self.proj_seed = proj_seed

        # Create base high-dimensional target function
        self.base_dataset = TargetFunctionDataset(
            num_samples=num_samples,
            target_dim=target_dim,
            condition_dim=condition_dim,
            num_components=num_components,
            c_min=c_min,
            c_max=c_max,
            weight_strategy=weight_strategy,
            sampling_strategy=sampling_strategy,
            freq_seed=freq_seed,
            phase_seed=phase_seed,
            weight_seed=weight_seed,
            sample_seed=sample_seed,
            use_cosines=True,
        )

        # Generate projection matrix (target_dim × target_dim, rank = low_dim)
        self.P = self._generate_projection_matrix()

        # Compute projected values
        self.c_values = self.base_dataset.c_values
        self.x_high = self.base_dataset.x_values
        self.x_projected = self._project(self.x_high)

    def _generate_projection_matrix(self) -> torch.Tensor:
        """
        Generate projection matrix P ∈ ℝ^(target_dim × target_dim) with rank = low_dim.

        The projection matrix projects onto a low_dim subspace of ℝ^target_dim.
        Formula: P = A(A^T A)^(-1) A^T where A ∈ ℝ^(target_dim × low_dim)
        """
        np.random.seed(self.proj_seed)

        # Generate random target_dim × low_dim matrix
        A = np.random.randn(self.target_dim, self.low_dim)

        # Compute A^T A
        ATA = A.T @ A

        # Add small regularization for numerical stability
        ATA_inv = np.linalg.inv(ATA + 1e-10 * np.eye(self.low_dim))

        # Compute projection matrix P = A(A^T A)^(-1) A^T
        # This projects onto the column space of A (rank = low_dim)
        P = A @ ATA_inv @ A.T

        return torch.tensor(P, dtype=torch.float32)

    def _project(self, x_high: torch.Tensor) -> torch.Tensor:
        """
        Project high-dimensional values to low-dimensional subspace.

        Args:
            x_high: [n_samples, target_dim]

        Returns:
            x_projected: [n_samples, target_dim] projected onto low_dim subspace
        """
        # P is target_dim × target_dim, so P @ x^T gives target_dim output
        # This projects x onto the low_dim subspace
        return x_high @ self.P.T

    def generate_eval_data(
        self,
        num_samples: int,
        eval_seed: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate evaluation data with projection.

        Args:
            num_samples: Number of evaluation samples
            eval_seed: Random seed for sampling

        Returns:
            Dict with 'c' and 'x' tensors (x is projected)
        """
        # Generate base data
        base_data = self.base_dataset.generate_eval_data(num_samples, eval_seed)

        # Project the target values
        x_projected = self._project(base_data["x"])

        return {"c": base_data["c"], "x": x_projected}

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                'c': [condition_dim] conditioning value
                'x': [target_dim] projected target value (lives in low_dim subspace)
        """
        return {
            "c": self.c_values[idx],
            "x": self.x_projected[idx],
        }

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data at once.

        Returns:
            Dict with:
                'c': [num_samples, condition_dim]
                'x': [num_samples, target_dim] projected onto low_dim subspace
        """
        return {
            "c": self.c_values,
            "x": self.x_projected,
        }

    def get_high_dim_data(self) -> Dict[str, torch.Tensor]:
        """
        Get unprojected high-dimensional data.

        Returns:
            Dict with:
                'c': [num_samples, condition_dim]
                'x': [num_samples, target_dim] unprojected
        """
        return {
            "c": self.c_values,
            "x": self.x_high,
        }

    def get_function_description(self) -> str:
        """Get human-readable description."""
        desc_lines = [
            "Projected target function g(c) = P @ f(c)",
            f"f: ℝ → ℝ^{self.target_dim}",
            f"P: ℝ^{self.target_dim}×{self.target_dim} projection matrix (rank {self.low_dim})",
            f"g: ℝ → ℝ^{self.target_dim} (living in {self.low_dim}-dim subspace)",
            "",
            "Base function f(c):",
        ]

        base_desc = self.base_dataset.get_function_description()
        desc_lines.append(base_desc)

        return "\n".join(desc_lines)


if __name__ == "__main__":
    # Test code
    logging.basicConfig(level=logging.INFO)
    logger.info("Testing datasets...")

    # Test TargetFunctionDataset
    ds1 = TargetFunctionDataset(num_samples=10, target_dim=1)
    logger.info(f"TargetFunctionDataset: {len(ds1)} samples")
    sample = ds1[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"c shape: {sample['c'].shape}, x shape: {sample['x'].shape}")

    # Test generate_eval_data
    eval_data = ds1.generate_eval_data(num_samples=20, eval_seed=999)
    logger.info(f"Eval data keys: {eval_data.keys()}")
    logger.info(
        f"Eval c shape: {eval_data['c'].shape}, x shape: {eval_data['x'].shape}"
    )

    # Test ProjectedTargetFunctionDataset
    ds2 = ProjectedTargetFunctionDataset(num_samples=10, target_dim=8, low_dim=3)
    logger.info(f"\nProjectedTargetFunctionDataset: {len(ds2)} samples")
    sample = ds2[0]
    logger.info(f"Sample keys: {sample.keys()}")
    logger.info(f"c shape: {sample['c'].shape}, x shape: {sample['x'].shape}")

    # Test generate_eval_data
    eval_data = ds2.generate_eval_data(num_samples=20, eval_seed=999)
    logger.info(f"Eval data keys: {eval_data.keys()}")
    logger.info(
        f"Eval c shape: {eval_data['c'].shape}, x shape: {eval_data['x'].shape}"
    )

    logger.info("\n✓ All dataset tests passed!")
