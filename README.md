# Toy Experiments: Regression vs Flow Models

Comparing regression and flow matching models for function approximation in low-data regimes.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🎯 Overview

This project explores the **implicit biases** of regression versus flow-based models when learning target functions from limited data. The key research question: *What interpolation behaviors emerge from different training paradigms?*

### Key Features

- **Two Training Paradigms**:
  - **Regression**: Direct function approximation `f(c)`
  - **Flow Matching**: Learning velocity fields `dx/dt = v(x_t, c, t)`

- **Two Experiment Types**:
  - **Reconstruction**: Learn scalar target functions `f: ℝ → ℝ`
  - **Projection**: Learn high-dimensional functions constrained to low-dimensional subspaces

- **Clean, Modular Codebase**:
  - Well-tested components
  - Configuration-driven experiments
  - Comprehensive logging and visualization

---

## 📦 Installation

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/toyexp.git
cd toyexp

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package
pip install -e .
```

### With Development Tools

```bash
pip install -e ".[dev]"
```

### Requirements

- Python 3.9+
- PyTorch 2.0+
- NumPy 1.24+
- Matplotlib 3.7+
- PyYAML 6.0+

---

## 🚀 Quick Start

### 1. Run a Quick Test (100 epochs, ~2 minutes)

```bash
python toyexp/train_recon.py --config toyexp/configs/config_test.yaml
```

**Expected output:**
- Training logs in `./outputs/test/train.log`
- Checkpoints in `./outputs/test/checkpoints/`
- Plots in `./outputs/test/plots/`

### 2. Run Full Reconstruction Experiment

```bash
# Regression mode
python toyexp/train_recon.py \
    --config toyexp/configs/config_recon.yaml \
    --mode regression

# Flow mode
python toyexp/train_recon.py \
    --config toyexp/configs/config_recon.yaml \
    --mode flow
```

### 3. Run Projection Experiment

```bash
python toyexp/train_proj.py \
    --config toyexp/configs/config_proj.yaml \
    --mode regression
```

### 4. Analyze Results

```bash
python toyexp/analyze_results.py \
    --results_dir ./outputs \
    --output_dir ./analysis
```

---

## 📁 Project Structure

```
toyexp/
├── toyexp/                      # Main package
|   ├── common/                    
│       ├── datasets.py             # Dataset implementations
│       ├── networks.py             # Neural network architectures
│       ├── losses.py               # Loss functions
│       ├── integrate.py            # ODE integration methods
│       ├── logging_utils.py        # Logging utilities
│       ├── config.py               # Configuration management
│       ├── utils.py                # Utility functions
│       |── analyze_results.py      # Results analysis
|   ├── scripts/ 
│       ├── train_recon.py          # Reconstruction training script
│       ├── train_proj.py           # Projection training script
│   └── configs/                # Configuration files
│       ├── config_recon.yaml   # Reconstruction config
│       ├── config_proj.yaml    # Projection config
├── pyproject.toml              # Project metadata
├── README.md                   # This file
```

---

## 🎮 Usage

### Basic Training

```bash
python toyexp/train_recon.py --config toyexp/configs/config_recon.yaml
```

### Command-Line Overrides

Override any config parameter using dot notation:

```bash
python toyexp/train_recon.py \
    --config toyexp/configs/config_recon.yaml \
    experiment.mode=flow \
    training.learning_rate=0.001 \
    training.num_epochs=10000 \
    dataset.num_train=100 \
    network.hidden_dim=512
```

### Multiple Seeds

```bash
for seed in 42 43 44 45 46; do
    python toyexp/train_recon.py \
        --config toyexp/configs/config_recon.yaml \
        --mode regression \
        --seed $seed
done
```

### Hyperparameter Search

```bash
# Try different learning rates
for lr in 0.001 0.0001 0.00001; do
    python toyexp/train_recon.py \
        --config toyexp/configs/config_recon.yaml \
        training.learning_rate=$lr \
        experiment.output_dir=./outputs/lr_$lr
done
```

---

## 🔬 Experiments

### Reconstruction Experiment

**Goal**: Learn scalar target functions `f(c) = Σ wᵢ·sin(freqᵢ·c + phaseᵢ)`

**Setup**:
- **Input**: Conditioning variable `c ∈ ℝ`
- **Output**: Scalar value `f(c) ∈ ℝ`
- **Training data**: 50 samples (low-data regime)
- **Evaluation**: 1000 samples (dense sampling)

**Methods**:
1. **Regression**: Directly predict `f(c)` from `c`
2. **Flow**: Learn velocity field `v(x_t, c, t)` and integrate

### Projection Experiment

**Goal**: Learn high-dimensional functions living in low-dimensional subspaces

**Setup**:
- **Input**: Conditioning variable `c ∈ ℝ`
- **Output**: `g(c) = P @ f(c)` where `P` is rank-deficient projection
- **Dimensions**: Output in `ℝ^8` constrained to `ℝ^2` subspace
- **Challenge**: Can models discover the subspace structure?

**Metrics**:
- L1/L2 reconstruction error
- Subspace alignment error
- Explained variance ratio

---

## 📊 Datasets

### TargetFunctionDataset

Generates target functions as combinations of sine/cosine terms:

```python
f(c) = Σ wᵢ · trig_i(freq_i · c + phase_i)
```

**Features**:
- Prime-based frequencies (no harmonics)
- Configurable weighting (uniform or inverse-frequency)
- Reproducible with multiple seed controls
- Grid or random sampling

### ProjectedTargetFunctionDataset

Projects high-dimensional functions onto low-dimensional subspaces:

```python
g(c) = P @ f(c)
```

Where `P` is a rank-deficient projection matrix.

**Features**:
- Configurable output/subspace dimensions
- True projection matrix for analysis
- Subspace alignment metrics

---

## 🏗️ Architecture

### Network Architectures

**ConcatMLP**: Concatenation-based conditioning
```
[x, c, t] → Linear → ReLU → ... → Linear → output
```

**FiLMMLP**: Feature-wise Linear Modulation
```
x → Linear → FiLM(c, t) → ReLU → ... → output
```

### Training Modes

**Regression Mode**:
- Direct prediction: `pred = model(zeros, c)`
- Loss: `||pred - target||²`
- Single forward pass

**Flow Mode**:
- Learns velocity field: `v = model(x_t, c, t)`
- Loss: `||v - (x_1 - x_0)||²` (flow matching)
- ODE integration at inference

---

## 📈 Results

After training, each experiment produces:

```
outputs/experiment_name/
├── config.yaml              # Saved configuration
├── train.log                # Training logs
├── checkpoints/             # Model checkpoints
│   ├── best_model.pt       # Best validation model
│   ├── final_model.pt      # Final epoch model
│   └── checkpoint_*.pt     # Periodic checkpoints
└── plots/                   # Visualizations
    ├── training_curves.png # Loss curves
    ├── predictions.png     # Predictions vs truth
    └── errors.png          # Error distribution
```

### Analysis

Compare methods across multiple seeds:

```bash
python toyexp/analyze_results.py --results_dir ./outputs
```

Produces:
- Comparison plots with error bars
- Summary statistics (mean ± std)
- Subspace analysis (for projection experiments)

---

## ⚙️ Configuration

### Config File Structure

```yaml
experiment:
  name: "my_experiment"
  mode: "regression"  # or "flow"
  seed: 42
  device: "cuda"
  output_dir: "./outputs/my_exp"

dataset:
  num_train: 50
  num_eval: 1000
  target_dim: 1
  condition_dim: 1
  num_components: 3
  weight_strategy: "uniform"  # or "inverse_freq"
  sampling_strategy: "grid"   # or "random"

network:
  architecture: "concat"  # or "film"
  hidden_dim: 256
  num_layers: 3
  activation: "relu"

training:
  batch_size: 32
  num_epochs: 5000
  learning_rate: 0.001
  log_interval: 100
  eval_interval: 500
  save_interval: 1000
  initial_dist: "zeros"  # or "gaussian"

evaluation:
  num_eval_steps: 1
  integration_method: "euler"  # or "rk4"
```

### Override Any Parameter

```bash
python toyexp/train_recon.py --config config.yaml \
    training.num_epochs=10000 \
    network.hidden_dim=512 \
    dataset.num_train=200
```

---

## 🧪 Testing

### Run Module Tests

```bash
# Test individual modules
python toyexp/datasets.py
python toyexp/networks.py
python toyexp/losses.py
python toyexp/integrate.py
python toyexp/config.py
python toyexp/utils.py
```

### Quick Integration Test

```bash
python toyexp/train_recon.py --config toyexp/configs/config_test.yaml
```

---

## 📚 Documentation

- **[PROJECT_STRUCTURE.md](PROJECT_STRUCTURE.md)**: Detailed project structure
- **[RESULTS.md](RESULTS.md)**: Experimental results and analysis (TODO)
- **Module docstrings**: Comprehensive inline documentation

---

## 🤝 Contributing

Contributions welcome! Areas for improvement:

- [ ] Additional architectures (Transformers, etc.)
- [ ] More sophisticated target functions
- [ ] Advanced ODE solvers
- [ ] Experiment tracking integration (Weights & Biases, MLflow)
- [ ] Unit tests for all modules
- [ ] More analysis tools

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

Built using:
- [PyTorch](https://pytorch.org/) for deep learning
- [NumPy](https://numpy.org/) for numerical computing
- [Matplotlib](https://matplotlib.org/) for visualization
- [PyYAML](https://pyyaml.org/) for configuration

---

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com](mailto:your.email@example.com).

---

## 🔗 Citation

If you use this code in your research, please cite:

```bibtex
@software{toyexp2025,
  title = {Toy Experiments: Regression vs Flow Models},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/toyexp}
}
```

---

## 📝 Changelog

### v0.1.0 (2025-01-XX)
- Initial release
- Reconstruction and projection experiments
- Regression and flow matching modes
- Comprehensive configuration system
- Command-line override support
- Analysis tools

---

**Happy Experimenting! 🚀**