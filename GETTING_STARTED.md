# Getting Started with Toy Experiments

Quick start guide to run your first experiments comparing regression vs flow models.

## Prerequisites

```bash
# Install dependencies
pip install torch numpy matplotlib pyyaml
```

## 5-Minute Quickstart

### 1. Run Your First Experiment

```bash
# Regression mode
python train_recon.py --config config_recon.yaml --mode regression

# This will:
# - Train a regression model for 5000 epochs
# - Save results to ./outputs/recon_experiment/
# - Create plots and logs automatically
```

### 2. Check Results

```bash
# View training log
cat ./outputs/recon_experiment/train.log

# Results structure:
./outputs/recon_experiment/
├── config.yaml              # Configuration used
├── train.log                # Training logs
├── checkpoints/
│   ├── best_model.pt        # Best model
│   └── final_model.pt       # Final model
└── plots/
    ├── training_curves.png  # Loss over time
    ├── predictions.png      # Model predictions
    └── errors.png           # Prediction errors
```

### 3. Compare with Flow Model

```bash
# Flow mode
python train_recon.py --config config_recon.yaml --mode flow \
    --output_dir ./outputs/recon_flow
```

### 4. Run Batch Experiments

```bash
# Run all experiments with multiple seeds
chmod +x run_experiments.sh
./run_experiments.sh
```

### 5. Analyze Results

```bash
# Compare all methods
python analyze_results.py --results_dir ./outputs --output_dir ./analysis
```

## Understanding the Experiments

### Reconstruction Experiment

**What it does**: Learns a scalar function f(c) where c ∈ [0, 1]

**Why it's interesting**: Tests if flow models add value for simple 1D outputs

**Key config**:
```yaml
dataset:
  num_train: 50      # Low-data regime
  target_dim: 1      # Scalar output
```

### Projection Experiment

**What it does**: Learns an 8D function that lives on a 2D subspace

**Why it's interesting**: Tests if flows capture low-dimensional structure better

**Key config**:
```yaml
dataset:
  num_train: 50      # Low-data regime
  target_dim: 8      # High-dimensional output
  low_dim: 2         # But only 2D of freedom
```

## Configuration Quick Reference

### Change Training Settings

Edit `config_recon.yaml`:

```yaml
training:
  batch_size: 32           # Batch size
  num_epochs: 5000         # Training epochs
  learning_rate: 0.001     # Learning rate
```

### Change Network Architecture

```yaml
network:
  architecture: "concat"   # or "film" for FiLM conditioning
  hidden_dim: 256          # Hidden layer size
  num_layers: 3            # Number of hidden layers
  activation: "relu"       # or "gelu", "silu", "tanh"
```

### Enable EMA (Exponential Moving Average)

```yaml
network:
  ema:
    enabled: true          # Enable EMA
    decay: 0.999           # EMA decay rate
```

## Command-Line Options

### Override Config Values

```bash
# Change experiment mode
python train_recon.py --mode flow

# Change random seed
python train_recon.py --seed 123

# Change output directory
python train_recon.py --output_dir ./my_results

# Combine multiple overrides
python train_recon.py --mode flow --seed 456 --output_dir ./flow_456
```

## Common Workflows

### 1. Quick Test Run

Reduce epochs for testing:

```yaml
training:
  num_epochs: 100    # Quick test
```

```bash
python train_recon.py --config config_recon.yaml
```

### 2. Compare Architectures

```bash
# Standard concatenation
python train_recon.py --mode regression --output_dir ./concat

# Edit config to use FiLM, then:
python train_recon.py --mode regression --output_dir ./film
```

### 3. Multiple Seeds for Robustness

```bash
for seed in 42 123 456; do
    python train_recon.py --mode regression --seed $seed \
        --output_dir ./outputs/regression_seed_${seed}
done
```

### 4. Compare Regression vs Flow

```bash
# Regression
python train_recon.py --mode regression --output_dir ./reg_results

# Flow
python train_recon.py --mode flow --output_dir ./flow_results

# Analyze both
python analyze_results.py --results_dir ./
```

## Troubleshooting

### CUDA Out of Memory

```yaml
# In config file, reduce batch size:
training:
  batch_size: 16  # or 8

# Or use CPU:
experiment:
  device: "cpu"
```

### Poor Convergence

Try different learning rates:

```yaml
training:
  learning_rate: 0.0001  # Smaller LR
  # or
  learning_rate: 0.01    # Larger LR
```

### Logs Not Appearing

Check that output directory is writable:

```bash
ls -la ./outputs/recon_experiment/
cat ./outputs/recon_experiment/train.log
```

## Next Steps

### 1. Read Full Documentation

See `README.md` for complete API reference and advanced usage.

### 2. Customize Datasets

Edit dataset parameters in config files:

```yaml
dataset:
  num_components: 3      # Number of frequency components
  weight_strategy: "uniform"  # or "inverse_freq"
  sampling_strategy: "grid"   # or "random"
```

### 3. Implement Custom Functions

Create custom target functions by extending `TargetFunctionDataset`:

```python
from toyexp.common.datasets import TargetFunctionDataset

class MyDataset(TargetFunctionDataset):
    def _generate_target_function(self, c):
        # Your custom function
        return my_function(c)
```

### 4. Add New Architectures

Implement the standard interface:

```python
class MyNetwork(nn.Module):
    def forward(self, x, c, t):
        # x: current state
        # c: conditioning variable
        # t: time (for flow models)
        return prediction
```

## Quick Command Reference

```bash
# Basic training
python train_recon.py

# With options
python train_recon.py --mode flow --seed 42

# Projection experiment
python train_proj.py --mode regression

# Batch experiments
./run_experiments.sh

# Analysis
python analyze_results.py

# Find print statements (for cleanup)
grep -rn "print(" toyexp/

# Run tests
python -m pytest tests/  # (if tests exist)
```

## Expected Output

After training completes, you should see:

```
================================================================================
Starting recon_experiment
================================================================================
2024-01-01 12:00:00 - INFO - Loading configuration from config_recon.yaml
2024-01-01 12:00:00 - INFO - Using device: cuda
2024-01-01 12:00:01 - INFO - Training dataset: 50 samples
2024-01-01 12:00:01 - INFO - Model: ConcatMLP with 197,377 parameters
2024-01-01 12:00:02 - INFO - Starting training...
2024-01-01 12:00:05 - INFO - Epoch 100/5000, Loss: 0.0234
...
2024-01-01 12:05:00 - INFO - Training complete!
2024-01-01 12:05:00 - INFO - Results saved to ./outputs/recon_experiment
```

## Getting Help

1. **Check logs**: `cat ./outputs/*/train.log`
2. **Read README**: `README.md` has full documentation
3. **Check config**: Verify your YAML files are valid
4. **Run with fewer epochs**: Test quickly with `num_epochs: 100`

## What's Next?

Once you're comfortable with basic experiments:

1. Try different network architectures (concat vs FiLM)
2. Experiment with different activation functions
3. Compare different integration methods (euler vs rk4)
4. Run multiple seeds and analyze variance
5. Implement custom target functions
6. Create your own experiment configs

Happy experimenting! 🚀