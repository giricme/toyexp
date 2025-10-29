"""
Target function datasets for toy experiments.

This module provides datasets for studying interpolation capabilities
of regression vs flow models in low-data regimes.
"""

import logging
from typing import Dict, List, Literal, Optional, Tuple

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
    Dataset for target function f(c) = Î£ wáµ¢ * trig_i(freq_i * c + phase_i)

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
        weight_strategy: 'uniform' (all weights=1) or 'inverse_freq' (weight âˆ 1/freq)
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
        desc_lines = [f"Target function f: â„ â†’ â„^{self.target_dim}"]
        desc_lines.append(f"Domain: c âˆˆ [{self.c_min:.2f}, {self.c_max:.2f}]")
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
                terms.append(f"{weight:.3f}Â·{trig}({freq:.1f}c + {phase:.2f})")

            desc_lines.append(f"f_{dim}(c) = {' + '.join(terms)}")

        if self.target_dim > 3:
            desc_lines.append(f"... ({self.target_dim - 3} more dimensions)")

        return "\n".join(desc_lines)


class ProjectedTargetFunctionDataset(Dataset):
    """
    Dataset for projected target function: g(c) = P @ f(c)

    Creates a high-dimensional target function and projects it onto a
    lower-dimensional subspace using a rank-deficient projection matrix.

    The projection matrix P âˆˆ â„^(high_dim Ã— high_dim) has rank = low_dim,
    constructed as P = A(A^T A)^(-1) A^T where A âˆˆ â„^(high_dim Ã— low_dim).

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

        # Generate projection matrix (target_dim Ã— target_dim, rank = low_dim)
        self.P = self._generate_projection_matrix()

        # Compute projected values
        self.c_values = self.base_dataset.c_values
        self.x_high = self.base_dataset.x_values
        self.x_projected = self._project(self.x_high)

    def _generate_projection_matrix(self) -> torch.Tensor:
        """
        Generate projection matrix P âˆˆ â„^(target_dim Ã— target_dim) with rank = low_dim.

        The projection matrix projects onto a low_dim subspace of â„^target_dim.
        Formula: P = A(A^T A)^(-1) A^T where A âˆˆ â„^(target_dim Ã— low_dim)
        """
        np.random.seed(self.proj_seed)

        # Generate random target_dim Ã— low_dim matrix
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
        # P is target_dim Ã— target_dim, so P @ x^T gives target_dim output
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
            f"f: â„ â†’ â„^{self.target_dim}",
            f"P: â„^{self.target_dim}Ã—{self.target_dim} projection matrix (rank {self.low_dim})",
            f"g: â„ â†’ â„^{self.target_dim} (living in {self.low_dim}-dim subspace)",
            "",
            "Base function f(c):",
        ]

        base_desc = self.base_dataset.get_function_description()
        desc_lines.append(base_desc)

        return "\n".join(desc_lines)


"""
New LieAlgebraRotationDataset implementation for redesigned target function.

Target function: fᵢ(α, c) = wᵢ(c) · exp(αᵢ·c · A) · e₁
Output: concat(f₁, f₂, ..., fₖ) ∈ ℝ^(K·rotation_dim)
"""


class LieAlgebraRotationDataset(Dataset):
    """
    Dataset for Lie algebra rotation functions (REDESIGNED).

    Target function: fᵢ(α, c) = wᵢ(c) · exp(αᵢ·c · A) · e₁
    Output: concat(f₁, f₂, ..., fₖ) ∈ ℝ^(K·rotation_dim)

    Key differences from old design:
    - Single rotation matrix A (same for all components)
    - Single initial vector e₁ (same for all components)
    - Each component i has velocity αᵢ
    - Rotation angle is αᵢ·c (product of velocity and conditioning)
    - Output is concatenation of all K rotated vectors

    Where:
    - K: number of rotation components (num_rotations)
    - A ∈ so(n): single skew-symmetric matrix (rotation generator)
    - e₁ ∈ ℝⁿ: single initial unit vector
    - wᵢ(c) ∈ ℝ: weight functions (constant or high-frequency)
    - αᵢ: rotation velocities (one per component)
    - c ∈ [c_min, c_max]: conditioning variable

    For SO(2): A = [[0, -1], [1, 0]] (standard 2D rotation)

    Args:
        num_c_samples: Number of c samples (α is implicit via αᵢ·c)
        rotation_dim: Dimension of rotation (2 for SO(2), 3 for SO(3))
        num_rotations: Number of rotation components K
        c_min: Minimum c value
        c_max: Maximum c value
        alpha_values: List of K velocity values [α₁, α₂, ..., αₖ]
                     If None, generates linearly spaced in [alpha_min, alpha_max]
        alpha_min: Minimum α value (if alpha_values not provided)
        alpha_max: Maximum α value (if alpha_values not provided)
        weight_mode: 'constant' or 'high_frequency'
        num_weight_components: Number of sine/cosine terms per weight (if high_frequency)
        weight_strategy: 'uniform' or 'inverse_freq' (if high_frequency)
        sampling_strategy: 'grid' or 'random'
        rotation_seed: Random seed for A matrix and e₁ vector
        weight_seed: Random seed for weight functions
        sample_seed: Random seed for sampling c values
    """

    def __init__(
        self,
        num_c_samples: int = 50,
        rotation_dim: int = 2,
        num_rotations: int = 5,
        c_min: float = 0.0,
        c_max: float = 1.0,
        alpha_values: Optional[List[float]] = None,
        alpha_min: float = 0.0,
        alpha_max: float = 2 * np.pi,
        weight_mode: Literal["constant", "high_frequency"] = "constant",
        num_weight_components: int = 3,
        weight_strategy: Literal["uniform", "inverse_freq"] = "uniform",
        sampling_strategy: Literal["grid", "random"] = "grid",
        rotation_seed: int = 42,
        weight_seed: int = 43,
        sample_seed: int = 44,
    ):
        super().__init__()

        # Store configuration
        self.num_c_samples = num_c_samples
        self.rotation_dim = rotation_dim
        self.num_rotations = num_rotations
        self.c_min = c_min
        self.c_max = c_max
        self.weight_mode = weight_mode
        self.num_weight_components = num_weight_components
        self.weight_strategy = weight_strategy
        self.sampling_strategy = sampling_strategy
        self.rotation_seed = rotation_seed
        self.weight_seed = weight_seed
        self.sample_seed = sample_seed

        # Validate
        assert rotation_dim in [2, 3], "Only SO(2) and SO(3) supported"
        assert weight_mode in ["constant", "high_frequency"]

        # Generate or use provided alpha velocities
        if alpha_values is not None:
            assert (
                len(alpha_values) == num_rotations
            ), f"alpha_values must have length {num_rotations}"
            self.alpha_velocities = np.array(alpha_values, dtype=np.float32)
        else:
            # Linearly spaced velocities
            self.alpha_velocities = np.linspace(alpha_min, alpha_max, num_rotations)

        # Generate single rotation matrix A and initial vector e₁
        self._generate_rotation_params()

        # Generate weight functions
        if weight_mode == "high_frequency":
            self._generate_weight_functions()
        else:
            # Constant weights (random in [-1, 1])
            np.random.seed(weight_seed)
            self.constant_weights = np.random.uniform(-1, 1, num_rotations)

        # Generate c samples
        self.c_samples = self._generate_c_samples()

        # Compute target values for all samples
        # Shape: [num_c_samples, num_rotations * rotation_dim]
        self.targets = self._compute_all_targets()

    def _generate_rotation_params(self):
        """Generate single rotation matrix A and initial vector e₁."""
        np.random.seed(self.rotation_seed)

        if self.rotation_dim == 2:
            # Standard 2D rotation matrix generator
            self.A = np.array([[0, -1], [1, 0]], dtype=np.float32)
        else:  # rotation_dim == 3
            # Random skew-symmetric matrix for SO(3)
            # Generate random 3x3 matrix
            rand_mat = np.random.randn(3, 3)
            # Make it skew-symmetric: A = (M - M^T) / 2
            self.A = (rand_mat - rand_mat.T) / 2
            self.A = self.A.astype(np.float32)

        # Single initial vector e₁ (random unit vector)
        self.initial_vector = np.random.randn(self.rotation_dim).astype(np.float32)
        self.initial_vector = self.initial_vector / np.linalg.norm(self.initial_vector)

    def _generate_weight_functions(self):
        """Generate high-frequency weight functions wᵢ(c)."""
        np.random.seed(self.weight_seed)

        # Get primes for frequencies and phases
        total_components = self.num_rotations * self.num_weight_components
        primes = get_primes(total_components * 2)

        self.weight_frequencies = []
        self.weight_phases = []
        self.weight_coefficients = []

        prime_idx = 0
        for rot in range(self.num_rotations):
            rot_freqs = []
            rot_phases = []
            rot_weights = []

            for comp in range(self.num_weight_components):
                freq = float(primes[prime_idx])
                prime_idx += 1
                phase = float(primes[prime_idx]) / 10.0
                prime_idx += 1

                # Weight coefficient
                if self.weight_strategy == "uniform":
                    weight = 1.0
                elif self.weight_strategy == "inverse_freq":
                    weight = 1.0 / freq
                else:
                    raise ValueError(f"Unknown weight_strategy: {self.weight_strategy}")

                rot_freqs.append(freq)
                rot_phases.append(phase)
                rot_weights.append(weight)

            self.weight_frequencies.append(rot_freqs)
            self.weight_phases.append(rot_phases)
            self.weight_coefficients.append(rot_weights)

    def _compute_weight(self, c: float, rotation_idx: int) -> float:
        """Compute weight wᵢ(c) for rotation i."""
        if self.weight_mode == "constant":
            return self.constant_weights[rotation_idx]
        else:  # high_frequency
            weight = 0.0
            for comp in range(self.num_weight_components):
                freq = self.weight_frequencies[rotation_idx][comp]
                phase = self.weight_phases[rotation_idx][comp]
                coeff = self.weight_coefficients[rotation_idx][comp]

                # Alternate sine/cosine
                if comp % 2 == 0:
                    weight += coeff * np.sin(freq * c + phase)
                else:
                    weight += coeff * np.cos(freq * c + phase)

            return weight

    def _rotation_matrix(self, angle: float) -> np.ndarray:
        """
        Compute rotation matrix exp(angle·A).

        Args:
            angle: Rotation angle (αᵢ·c)

        Returns:
            Rotation matrix R = exp(angle·A)
        """
        if self.rotation_dim == 2:
            # For 2D rotation with A = [[0, -1], [1, 0]]:
            # exp(θ·A) = [[cos(θ), -sin(θ)], [sin(θ), cos(θ)]]
            cos_angle = np.cos(angle)
            sin_angle = np.sin(angle)
            return np.array(
                [[cos_angle, -sin_angle], [sin_angle, cos_angle]], dtype=np.float32
            )
        else:  # rotation_dim == 3
            # Use matrix exponential for general skew-symmetric matrix
            from scipy.linalg import expm

            return expm(angle * self.A).astype(np.float32)

    def _compute_component(self, c: float, component_idx: int) -> np.ndarray:
        """
        Compute single component fᵢ(α, c) = wᵢ(c) · exp(αᵢ·c · A) · e₁

        Args:
            c: Conditioning value
            component_idx: Which component i to compute

        Returns:
            Rotated vector of shape [rotation_dim]
        """
        # Get weight
        weight = self._compute_weight(c, component_idx)

        # Get velocity for this component
        alpha_i = self.alpha_velocities[component_idx]

        # Compute rotation angle
        angle = alpha_i * c

        # Get rotation matrix
        R = self._rotation_matrix(angle)

        # Apply rotation to initial vector and scale by weight
        rotated = R @ self.initial_vector
        return weight * rotated

    def _compute_target(self, c: float) -> np.ndarray:
        """
        Compute full target: concat(f₁, f₂, ..., fₖ)

        Args:
            c: Conditioning value

        Returns:
            Concatenated output of shape [num_rotations * rotation_dim]
        """
        components = []
        for i in range(self.num_rotations):
            comp = self._compute_component(c, i)
            components.append(comp)

        return np.concatenate(components)

    def _generate_c_samples(self) -> np.ndarray:
        """Generate conditioning variable samples."""
        np.random.seed(self.sample_seed)

        if self.sampling_strategy == "grid":
            # Uniform grid
            c_vals = np.linspace(self.c_min, self.c_max, self.num_c_samples)
        elif self.sampling_strategy == "random":
            # Random uniform sampling
            c_vals = np.random.uniform(self.c_min, self.c_max, self.num_c_samples)
            c_vals = np.sort(c_vals)  # Sort for visualization
        else:
            raise ValueError(f"Unknown sampling_strategy: {self.sampling_strategy}")

        return c_vals.astype(np.float32)

    def _compute_all_targets(self) -> torch.Tensor:
        """Compute target values for all samples."""
        targets = []
        for c in self.c_samples:
            target = self._compute_target(c)
            targets.append(target)

        return torch.tensor(np.array(targets), dtype=torch.float32)

    def generate_eval_data(
        self,
        num_c_samples: int,
        eval_seed: int,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate evaluation data with different c samples.

        Args:
            num_c_samples: Number of c samples for evaluation
            eval_seed: Random seed for sampling

        Returns:
            Dict with 'c' and 'x' tensors
        """
        np.random.seed(eval_seed)

        if self.sampling_strategy == "grid":
            c_vals = np.linspace(self.c_min, self.c_max, num_c_samples)
        else:
            c_vals = np.random.uniform(self.c_min, self.c_max, num_c_samples)
            c_vals = np.sort(c_vals)

        # Compute targets
        targets = []
        for c in c_vals:
            target = self._compute_target(c)
            targets.append(target)

        c_tensor = torch.tensor(c_vals, dtype=torch.float32).reshape(-1, 1)
        targets_tensor = torch.tensor(np.array(targets), dtype=torch.float32)

        return {"c": c_tensor, "x": targets_tensor}

    def get_manifold_vectors(self, c_values: np.ndarray) -> np.ndarray:
        """
        Get the manifold vectors exp(αᵢ·c · A) · e₁ for each component.

        This is used for computing cosine similarity metrics.

        Args:
            c_values: Array of c values, shape [num_samples]

        Returns:
            Manifold vectors, shape [num_samples, num_rotations, rotation_dim]
        """
        manifold_vecs = []

        for c in c_values:
            c_vecs = []
            for i in range(self.num_rotations):
                alpha_i = self.alpha_velocities[i]
                angle = alpha_i * c
                R = self._rotation_matrix(angle)
                vec = R @ self.initial_vector
                c_vecs.append(vec)
            manifold_vecs.append(c_vecs)

        return np.array(manifold_vecs)  # [num_samples, num_rotations, rotation_dim]

    def __len__(self) -> int:
        return len(self.c_samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            Dict with:
                'c': [1] tensor containing c value
                'x': [num_rotations * rotation_dim] tensor containing concat output
        """
        return {
            "c": torch.tensor([self.c_samples[idx]], dtype=torch.float32),
            "x": self.targets[idx],
        }

    def get_all_data(self) -> Dict[str, torch.Tensor]:
        """
        Get all data at once.

        Returns:
            Dict with:
                'c': [num_samples, 1] conditioning values
                'x': [num_samples, num_rotations * rotation_dim] target values
        """
        return {
            "c": torch.tensor(self.c_samples, dtype=torch.float32).reshape(-1, 1),
            "x": self.targets,
        }

    def get_function_description(self) -> str:
        """Get human-readable description of the target function."""
        desc_lines = [
            f"Lie Algebra Rotation Function (REDESIGNED): fᵢ(α, c) = wᵢ(c) · exp(αᵢ·c · A) · e₁",
            f"Output: concat(f₁, ..., f₍{self.num_rotations}₎) ∈ ℝ^{self.num_rotations * self.rotation_dim}",
            f"Rotation dimension: SO({self.rotation_dim})",
            f"Number of components K: {self.num_rotations}",
            f"c ∈ [{self.c_min:.2f}, {self.c_max:.2f}]",
            f"Weight mode: {self.weight_mode}",
            "",
            "Rotation velocities α:",
        ]

        for i in range(min(self.num_rotations, 10)):
            desc_lines.append(f"  α_{i} = {self.alpha_velocities[i]:.4f}")

        if self.num_rotations > 10:
            desc_lines.append(f"  ... ({self.num_rotations - 10} more velocities)")

        if self.weight_mode == "constant":
            desc_lines.append("")
            desc_lines.append("Constant weights:")
            for i in range(min(self.num_rotations, 5)):
                desc_lines.append(f"  w_{i} = {self.constant_weights[i]:.3f}")
        else:
            desc_lines.append("")
            desc_lines.append(
                f"High-frequency weights with {self.num_weight_components} components each"
            )

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

    logger.info("\nâœ“ All dataset tests passed!")
