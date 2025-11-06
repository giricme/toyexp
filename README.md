# Toy Experiments: Regression vs Flow Matching vs MIP

Comparing regression, flow matching, and manifold interpolation (MIP) models for function approximation in low-data regimes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Executive Summary

This project explores **implicit biases** of different training paradigms when learning target functions from limited data:

- **Objective**: Compare three training paradigms (regression, flow matching, MIP) across three tasks (reconstruction, projection, Lie algebra) using identical network architectures trained with L1 and L2 losses.

- **Key Result**: Clear trade-off between reconstruction fidelity and geometric structure preservation:
  - **Regression-L2** achieves best reconstruction (26× better than flow on recon task)
  - **MIP-L2** provides best balance (within 1.7× of regression, significantly better geometry)
  - **Flow matching** struggles with reconstruction but shows geometry potential

- **Recommendation**: Start with regression-L2 for baseline performance. Switch to MIP-L2 when geometric constraints (manifold adherence, subspace structure) are critical.

- **Experimental Setup**: Results averaged over 3 random seeds with 50 training samples and 100,000 evaluation samples per experiment.

---

## 🎯 Overview

### Three Training Paradigms

1. **Regression**: Direct function approximation `f(c)` - optimizes reconstruction error
2. **Flow Matching**: Learning velocity fields `dx/dt = v(x_t, c, t)` via ODE integration
3. **MIP (Manifold Interpolation)**: Flow matching + denoising term at fixed time t* for manifold adherence

### Three Experiment Types

1. **Reconstruction**: Learn scalar target functions `f: ℝ → ℝ`
2. **Projection**: Learn high-dimensional functions constrained to low-dimensional subspaces
3. **Lie Algebra**: Learn rotation components evolving on SO(2) manifolds

### Key Features

- **Modular, clean codebase** with configuration-driven experiments
- **Comprehensive evaluation**: Reconstruction metrics (L1/L2) + geometric metrics (subspace angles, manifold adherence)
- **Multiple seeds support** for robust statistical analysis
- **Detailed logging and visualization**

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/toyexp.git
cd toyexp

# Install dependencies
pip install torch numpy matplotlib pyyaml scipy
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- PyYAML 6.0+
- SciPy 1.10+ (for Lie algebra experiments)

---

## 🚀 Quick Start

### 1. Run Your First Experiment (2 minutes)

```bash
# Regression mode - reconstruction task
python train_recon.py --config config_recon.yaml --mode regression

# This will:
# - Train a regression model for 50,000 epochs
# - Save results to ./outputs/recon/
# - Create plots and logs automatically
```

### 2. Check Results

```bash
# View training log
cat ./outputs/recon/train.log

# Results structure:
./outputs/recon/
├── config.yaml              # Configuration used
├── train.log                # Training logs
├── evaluation.csv           # Evaluation metrics
├── checkpoints/
│   └── final_model.pt       # Final model
└── plots/
    ├── training_loss.png    # Loss over time
    └── predictions.png      # Model predictions vs ground truth
```

### 3. Compare Training Paradigms

```bash
# Run all three modes
python train_recon.py --config config_recon.yaml --mode regression
python train_recon.py --config config_recon.yaml --mode flow
python train_recon.py --config config_recon.yaml --mode mip

# Compare results
python run_mode_comparison.py
```

### 4. Try Other Experiments

```bash
# Projection experiment (8D → 3D subspace)
python train_proj.py --config config_proj.yaml --mode mip

# Lie algebra experiment (SO(2) rotations)
python train_lie.py --config config_lie.yaml --mode mip
```

---

## 📊 Experiments

### Reconstruction Experiment

**Goal**: Learn scalar target functions `f(c) = Σ wᵢ·sin(freqᵢ·c + phaseᵢ)`

**Setup**:
- Input: `c ∈ [0, 1]`
- Output: Scalar `f(c) ∈ ℝ`
- Training: 50 samples
- Evaluation: 100,000 samples

**Mathematical Formulation**:
```
f(c) = Σᵢ₌₁ᴷ wᵢ · trigᵢ(ωᵢc + φᵢ)
```
where K=3 components, frequencies ωᵢ are prime-based, weights wᵢ=1 (uniform).

**Key Finding**: Regression-L2 achieves 0.0023 L1 error, 26× better than flow matching.

### Projection Experiment

**Goal**: Learn 8D functions living in 3D subspaces with interval-dependent projections

**Setup**:
- Input: `c ∈ [0, 1]`
- Output: `g(c) = Pᵢ(c) f(c) ∈ ℝ⁸` constrained to rank-3 subspaces
- Domain split into 10 intervals, each with unique projection matrix

**Mathematical Formulation**:
```
g(c) = Pᵢ(c) f(c)
where Pᵢ = Aᵢ(Aᵢᵀ Aᵢ)⁻¹ Aᵢᵀ,  Aᵢ ∈ ℝ⁸ˣ³
```

**Metrics**: L1/L2 reconstruction + subspace angles/distances

**Key Finding**: MIP-L2 only 4.7% worse than regression while preserving geometric structure.

### Lie Algebra Experiment

**Goal**: Learn rotation components evolving on SO(2) manifolds

**Setup**:
- Input: `c ∈ [0, 1]`
- Output: 8 rotation components, each 2D vector
- High-frequency weight functions modulate rotations

**Mathematical Formulation**:
```
fᵢ(α, c) = wᵢ(c) · exp(αᵢc · A) · e₁
where A = [[0, -1], [1, 0]] (SO(2) generator)
Output: concat(f₁, ..., f₈) ∈ ℝ¹⁶
```

**Metrics**: L1/L2 reconstruction + cosine similarity + perpendicular error

**Key Finding**: MIP-L2 achieves best manifold adherence (0.081 avg perpendicular error vs 0.092 for regression).

---

## ⚙️ Configuration

### Basic Config Structure

```yaml
experiment:
  name: "recon_experiment"
  mode: "mip"                    # 'regression', 'flow', or 'mip'
  seed: 42
  device: "cuda"
  output_dir: "./outputs/recon"

dataset:
  num_train: 50                  # Training samples
  num_eval: 100000               # Evaluation samples
  target_dim: 1                  # Output dimension
  num_components: 3              # Frequency components
  sampling_strategy: "grid"      # 'grid' or 'random'

network:
  architecture: "concat"         # 'concat' or 'film'
  hidden_dim: 256
  num_layers: 3
  activation: "relu"

training:
  loss_type: "l2"               # 'l1' or 'l2'
  batch_size: 32
  num_epochs: 50000
  learning_rate: 0.001
  log_interval: 1000
  eval_interval: 50000
  
  # MIP-specific
  mip_t_star: 0.9               # Fixed time for denoising term

evaluation:
  num_eval_steps: [1, 9]        # NFE for flow models
  integration_method: "euler"    # 'euler' or 'rk4'
```

### Override Config Parameters

```bash
# Change mode and loss
python train_recon.py --mode flow --loss l1

# Change network architecture
python train_recon.py --hidden_dim 512 --num_layers 4

# Change output directory
python train_recon.py --output_dir ./my_results
```

---

## 🎮 Usage Examples

### Compare All Methods on One Task

```bash
# Reconstruction with L2 loss
for mode in regression flow mip; do
    python train_recon.py --config config_recon.yaml \
        --mode $mode \
        --output_dir ./outputs/recon_${mode}_l2
done

# Analyze results
python run_mode_comparison.py
```

### Multiple Seeds for Statistical Analysis

```bash
# Run 3 seeds for each method
for seed in 0 1 2; do
    for mode in regression flow mip; do
        python train_recon.py \
            --mode $mode \
            --seed $seed \
            --output_dir ./outputs/recon_${mode}_seed${seed}
    done
done

# Generate averaged results
python run_mode_comparison.py
```

### L1 vs L2 Loss Comparison

```bash
# Train with L1 loss
python train_recon.py --config config_recon.yaml --loss l1

# Train with L2 loss
python train_recon.py --config config_recon.yaml --loss l2
```

### Quick Test Run

```bash
# Reduce epochs for fast testing
python train_recon.py --num_epochs 1000 --eval_interval 500
```

---

## 📈 Results and Analysis

### Viewing Results

After training, check:

```bash
# Training metrics
cat ./outputs/recon/train.log

# Evaluation metrics (CSV format)
cat ./outputs/recon/evaluation.csv

# Visualizations
ls ./outputs/recon/plots/
```

### Mode Comparison

Generate comparison tables and plots:

```bash
python run_mode_comparison.py

# Creates:
# - results_table_averaged.tex (mean ± std across seeds)
# - results_table_seedwise.tex (individual seed results)
# - Comparison plots
```

### Analyzing Results

The `run_mode_comparison.py` script:
- Aggregates results across multiple seeds
- Computes mean ± standard deviation
- Generates LaTeX tables for papers
- Creates comparison visualizations

**Output Format**:
```
Mode    | Training Loss | L1 Error        | L2 Error        | Geometry Metrics
--------|---------------|-----------------|-----------------|------------------
regress | l2           | 0.002 ± 0.000   | 0.003 ± 0.000   | ...
flow    | l2           | 0.060 ± 0.009   | 0.072 ± 0.007   | ...
mip     | l2           | 0.004 ± 0.000   | 0.005 ± 0.000   | ...
```

---

## 🏗️ Project Structure

```
.
├── config_recon.yaml          # Reconstruction config
├── config_proj.yaml           # Projection config
├── config_lie.yaml            # Lie algebra config
├── train_recon.py             # Reconstruction training
├── train_proj.py              # Projection training
├── train_lie.py               # Lie algebra training
├── run_mode_comparison.py     # Multi-seed analysis
├── datasets.py                # Dataset implementations
├── networks.py                # Neural network architectures
├── losses.py                  # Loss functions
├── integrate.py               # ODE integration
├── logging_utils.py           # Logging utilities
├── config.py                  # Configuration management
├── utils.py                   # Utility functions
└── outputs/                   # Experiment results
```

---

## 🔬 Network Architectures

### ConcatMLP
Concatenation-based conditioning:
```
[x, c, t] → Linear → ReLU → ... → Linear → output
```

**When to use**: Default choice, simple and effective

### FiLMMLP
Feature-wise Linear Modulation:
```
x → Linear → FiLM(c, t) → ReLU → ... → output
```

**When to use**: When conditioning should modulate features rather than concatenate

---

## 📝 Evaluation Metrics

### Reconstruction Metrics
- **L1 Error**: `||f_θ(c) - f*(c)||₁`
- **L2 Error**: `||f_θ(c) - f*(c)||₂`

Measures how accurately the learned function approximates the true target.

### Geometric Metrics (Projection)
- **Average Angle**: Mean principal angle between learned and true subspaces
- **Max Angle**: Worst-case subspace alignment
- **Subspace Distance**: Distance from predictions to true subspace

Measures whether outputs lie in correct low-dimensional subspaces.

### Geometric Metrics (Lie Algebra)
- **Cosine Similarity**: Alignment of learned vectors with manifold directions
- **Perpendicular Error**: Distance from predictions to manifold
- **Average/Min/Max** variants for each

Measures whether outputs lie on the rotation manifold.

---

## 🎯 Method Selection Guide

| Priority | Task Type | Recommended Method |
|----------|-----------|-------------------|
| Best reconstruction | Any | Regression-L2 |
| Geometric constraints | Manifolds | MIP-L2 |
| Balanced performance | Subspaces | MIP-L2 |
| Exploratory | Novel problems | Flow-L2 (NFE=9) |

### Loss Function Selection

- **L2 training loss**: Better for matching L2 test metrics (recommended default)
- **L1 training loss**: Occasionally provides better robustness

---

## 🔧 Troubleshooting

### CUDA Out of Memory

```yaml
# Reduce batch size in config
training:
  batch_size: 16  # or 8

# Or use CPU
experiment:
  device: "cpu"
```

### Poor Convergence

```yaml
# Try different learning rates
training:
  learning_rate: 0.0001  # Smaller
  # or
  learning_rate: 0.01    # Larger
```

### Missing Metrics

Check that evaluation interval divides num_epochs:
```yaml
training:
  num_epochs: 50000
  eval_interval: 50000  # Must divide evenly!
```

### Flow Models Performing Poorly

- Ensure sufficient ODE integration steps (NFE=9 recommended)
- Check that `initial_dist: "zeros"` for evaluation
- Try adjusting `mip_t_star` if using MIP mode

---

## 📚 Documentation

### Module Docstrings

All modules have comprehensive inline documentation:

```python
# View dataset documentation
python -c "import datasets; help(datasets.TargetFunctionDataset)"

# View network documentation
python -c "import networks; help(networks.ConcatMLP)"
```

### Testing Modules

Run built-in tests:

```bash
# Test datasets
python datasets.py

# Test networks
python networks.py

# Test integration
python integrate.py
```

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional architectures (Transformers, attention mechanisms)
- [ ] More sophisticated target functions
- [ ] Advanced ODE solvers (adaptive step size)
- [ ] Experiment tracking (Weights & Biases, MLflow)
- [ ] Unit tests with pytest
- [ ] More geometric constraints and metrics

---

## 📄 License

This project is licensed under the MIT License.

---

## 🙏 Acknowledgments

Built using:
- [PyTorch](https://pytorch.org/) for deep learning
- [NumPy](https://numpy.org/) for numerical computing
- [SciPy](https://scipy.org/) for scientific computing
- [Matplotlib](https://matplotlib.org/) for visualization
- [PyYAML](https://pyyaml.org/) for configuration

---

## 📧 Contact

For questions or issues, please open an issue on GitHub.

---

---

**Happy Experimenting! 🚀**