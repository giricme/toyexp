"""
Training script for projection experiment.

Compare regression vs flow models with high-dimensional output in low-dimensional subspace.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from toyexp.common.config import (
    apply_overrides,
    load_config,
    parse_override_args,
    save_config,
    validate_config,
)
from toyexp.common.datasets import ProjectedTargetFunctionDataset
from toyexp.common.integrate import integrate
from toyexp.common.logging_utils import (
    create_metrics_logger,
    get_logger,
    log_config,
    log_evaluation,
    log_model_info,
    log_training_step,
    setup_logging,
)
from toyexp.common.losses import LossManager
from toyexp.common.networks import create_model
from toyexp.common.utils import (
    plot_errors,
    plot_predictions,
    plot_training_curves,
    save_checkpoint,
    set_seed,
)

logger = get_logger(__name__)


def create_datasets(config):
    """Create training and evaluation datasets from config."""
    logger.info("Creating datasets...")

    # Training dataset
    train_dataset = ProjectedTargetFunctionDataset(
        num_samples=config.dataset.num_train,
        target_dim=config.dataset.target_dim,
        condition_dim=config.dataset.condition_dim,
        low_dim=config.dataset.low_dim,
        num_components=config.dataset.num_components,
        c_min=config.dataset.c_min,
        c_max=config.dataset.c_max,
        weight_strategy=config.dataset.weight_strategy,
        sampling_strategy=config.dataset.sampling_strategy,
        freq_seed=config.dataset.freq_seed,
        phase_seed=config.dataset.phase_seed,
        weight_seed=config.dataset.weight_seed,
        proj_seed=config.dataset.proj_seed,
        sample_seed=config.dataset.sample_seed,
    )

    logger.info(f"Training dataset: {len(train_dataset)} samples")
    logger.info(
        f"Projection: {config.dataset.target_dim}D ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Â ÃƒÂ¢Ã¢â€šÂ¬Ã¢â€žÂ¢ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™ÃƒÂ¢Ã¢â€šÂ¬Ã…Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â ÃƒÆ’Ã†â€™Ãƒâ€ Ã¢â‚¬â„¢ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¡ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¬ÃƒÆ’Ã†â€™Ãƒâ€šÃ‚Â¢ÃƒÆ’Ã‚Â¢ÃƒÂ¢Ã¢â‚¬Å¡Ã‚Â¬Ãƒâ€¦Ã‚Â¾ÃƒÆ’Ã¢â‚¬Å¡Ãƒâ€šÃ‚Â¢ {config.dataset.low_dim}D subspace"
    )

    return train_dataset


def train_epoch(model, dataloader, loss_manager, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0

    for batch in dataloader:
        c = batch["c"].to(device)
        x_1 = batch["x"].to(device)

        # Compute loss
        if config.experiment.mode == "regression":
            loss = loss_manager.compute_loss(model, c, x_1)
        elif config.experiment.mode == "mip":
            # MIP mode: needs x_0, x_1, c (no t sampled)
            # Initial distribution
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn_like(x_1)
            else:
                x_0 = torch.zeros_like(x_1)

            loss = loss_manager.compute_loss(model, x_0, x_1, c)
        else:  # flow
            # Sample time uniformly
            batch_size = c.shape[0]
            t = torch.rand(batch_size, 1, device=device)

            # Initial distribution
            if config.training.initial_dist == "gaussian":
                x_0 = torch.randn_like(x_1)
            else:
                x_0 = torch.zeros_like(x_1)

            loss = loss_manager.compute_loss(model, x_0, x_1, c, t)

        # Optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        num_batches += 1

    return epoch_loss / num_batches


def analyze_subspace(model, dataset, device, config):
    """
    Analyze learned vs true subspace structure.

    Returns dictionary with subspace analysis metrics.
    """
    if not config.evaluation.analyze_subspace:
        return {}

    model.eval()

    # Sample many points to estimate learned subspace
    eval_data = dataset.generate_eval_data(
        num_samples=1000,
        eval_seed=config.experiment.seed + 2000,
    )

    c_eval = eval_data["c"].to(device)
    x_true = eval_data["x"]

    with torch.no_grad():
        # Initial distribution
        if config.training.initial_dist == "gaussian":
            x_0 = torch.randn_like(x_true).to(device)
        else:
            x_0 = torch.zeros_like(x_true).to(device)

        # Get predictions
        x_pred = integrate(
            model=model,
            x_0=x_0,
            c=c_eval,
            n_steps=config.evaluation.num_eval_steps,
            method=config.evaluation.integration_method,
            mode=config.experiment.mode,
        )

        x_pred_np = x_pred.cpu().numpy()

    # Compute PCA on predictions
    from numpy.linalg import svd

    # Center the data
    x_mean = np.mean(x_pred_np, axis=0)
    x_centered = x_pred_np - x_mean

    # SVD
    U, S, Vt = svd(x_centered, full_matrices=False)

    # Get explained variance
    total_variance = np.sum(S**2)
    explained_variance = S**2 / total_variance

    # Get true projection matrix
    P_true = dataset.P.cpu().numpy()

    # Compute alignment between learned and true subspaces
    # Use top low_dim principal components
    V_learned = Vt[: config.dataset.low_dim].T  # (target_dim, low_dim)

    # True subspace basis (columns of projection matrix span the range)
    U_true, _, _ = svd(P_true, full_matrices=False)
    V_true = U_true[:, : config.dataset.low_dim]  # (target_dim, low_dim)

    # Compute subspace alignment using Frobenius norm of difference of projection matrices
    P_learned = V_learned @ V_learned.T
    P_true_normalized = V_true @ V_true.T

    alignment_error = np.linalg.norm(P_learned - P_true_normalized, "fro")

    metrics = {
        "subspace_alignment_error": alignment_error,
        "explained_variance_ratio": explained_variance[: config.dataset.low_dim].sum(),
        "top_singular_values": S[: config.dataset.low_dim].tolist(),
    }

    logger.info("Subspace analysis:")
    logger.info(f"  Alignment error: {alignment_error:.4f}")
    logger.info(
        f"  Explained variance ratio: {metrics['explained_variance_ratio']:.4f}"
    )

    return metrics


def evaluate(model, dataset, device, config):
    """Evaluate model on dataset."""
    model.eval()

    # Generate evaluation data
    eval_data = dataset.generate_eval_data(
        num_samples=config.dataset.num_eval,
        eval_seed=config.experiment.seed + 1000,
    )

    c_eval = eval_data["c"].to(device)
    x_true = eval_data["x"].cpu().numpy()

    with torch.no_grad():
        # Initial distribution
        if config.evaluation.initial_dist == "gaussian":
            x_0 = torch.randn_like(eval_data["x"]).to(device)
        else:
            x_0 = torch.zeros_like(eval_data["x"]).to(device)

        # Get predictions
        x_pred = integrate(
            model=model,
            x_0=x_0,
            c=c_eval,
            n_steps=config.evaluation.num_eval_steps,
            method=config.evaluation.integration_method,
            mode=config.experiment.mode,
        )

        x_pred = x_pred.cpu().numpy()

    # Compute metrics
    l1_error = np.mean(np.abs(x_pred - x_true))
    l2_error = np.sqrt(np.mean((x_pred - x_true) ** 2))

    # Per-dimension errors
    l1_per_dim = np.mean(np.abs(x_pred - x_true), axis=0)
    l2_per_dim = np.sqrt(np.mean((x_pred - x_true) ** 2, axis=0))

    metrics = {
        "l1_error": l1_error,
        "l2_error": l2_error,
        "l1_per_dim_mean": np.mean(l1_per_dim),
        "l1_per_dim_std": np.std(l1_per_dim),
        "l2_per_dim_mean": np.mean(l2_per_dim),
        "l2_per_dim_std": np.std(l2_per_dim),
    }

    # Subspace analysis
    subspace_metrics = analyze_subspace(model, dataset, device, config)
    metrics.update(subspace_metrics)

    # Prepare data for plotting (all dimensions)
    plot_data = {
        "c_values": c_eval.cpu().numpy().flatten(),
        "true_values": x_true,  # All dimensions [num_samples, target_dim]
        "pred_values": x_pred,  # All dimensions [num_samples, target_dim]
        "l1_per_dim": l1_per_dim,  # Per-dimension L1 errors
        "l2_per_dim": l2_per_dim,  # Per-dimension L2 errors
    }

    return metrics, plot_data


def main(config_path: str, overrides: dict = None):
    """Main training function."""
    # Load and validate configuration
    config = load_config(config_path)

    if overrides:
        config = apply_overrides(config, overrides)

    validate_config(config)

    # Build output directory with subdirectories for mode and loss_type
    base_output_dir = Path(config.experiment.output_dir)
    output_dir = base_output_dir / config.experiment.mode / config.training.loss_type
    output_dir.mkdir(parents=True, exist_ok=True)

    # Setup logging
    setup_logging(
        name="train_proj",
        level=logging.INFO,
        log_file=output_dir / "train.log",
    )

    logger.info("=" * 80)
    logger.info(f"Starting {config.experiment.name}")
    logger.info("=" * 80)

    # Save config
    save_config(config, output_dir / "config.yaml")
    log_config(config.to_dict())

    # Set seed
    set_seed(config.experiment.seed)

    # Device
    device = torch.device(
        config.experiment.device
        if torch.cuda.is_available() and config.experiment.device == "cuda"
        else "cpu"
    )
    logger.info(f"Using device: {device}")

    # Create datasets
    train_dataset = create_datasets(config)

    # Create dataloader
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.training.batch_size,
        shuffle=True,
        num_workers=0,
    )

    # Create model
    model = create_model(
        architecture=config.network.architecture,
        x_dim=config.dataset.target_dim,
        c_dim=config.dataset.condition_dim,
        output_dim=config.dataset.target_dim,
        hidden_dim=config.network.hidden_dim,
        n_layers=config.network.num_layers,
        activation=config.network.activation,
        use_time=(config.experiment.mode in ["flow", "mip"]),
    ).to(device)

    log_model_info(model)

    # Create loss manager
    loss_manager = LossManager(
        mode=config.experiment.mode,
        loss_type=config.training.loss_type,
        x_dim=config.dataset.target_dim,
        mip_t_star=config.training.get("mip_t_star", 0.9),
    )

    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)

    # Create CSV metrics logger
    metrics_logger = create_metrics_logger(output_dir, experiment_type="proj")

    # Training loop
    logger.info("Starting training...")
    train_losses = []
    best_l2_error = float("inf")

    for epoch in range(config.training.num_epochs):
        # Train
        epoch_loss = train_epoch(
            model, train_loader, loss_manager, optimizer, device, config
        )
        train_losses.append(epoch_loss)

        # Log to CSV
        metrics_logger.log("training", {"epoch": epoch + 1, "loss": epoch_loss})

        # Log training
        if (epoch + 1) % config.training.log_interval == 0:
            log_training_step(
                epoch=epoch + 1,
                step=epoch + 1,
                loss=epoch_loss,
            )

        # Evaluate
        if (epoch + 1) % config.training.eval_interval == 0:
            metrics, plot_data = evaluate(model, train_dataset, device, config)
            log_evaluation(metrics, prefix=f"Epoch {epoch + 1}")

            # Log to CSV
            metrics_logger.log(
                "evaluation",
                {
                    "epoch": epoch + 1,
                    "l1_error": metrics["l1_error"],
                    "l2_error": metrics["l2_error"],
                },
            )

            # Save best model
            if metrics["l2_error"] < best_l2_error:
                best_l2_error = metrics["l2_error"]
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_dir=output_dir / "checkpoints",
                    filename="best_model.pt",
                    additional_info={"metrics": metrics},
                )

        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
                save_dir=output_dir / "checkpoints",
                filename=f"checkpoint_epoch_{epoch+1}.pt",
            )

    # Final evaluation
    logger.info("=" * 80)
    logger.info("Final evaluation...")
    logger.info("=" * 80)

    metrics, plot_data = evaluate(model, train_dataset, device, config)
    log_evaluation(metrics, prefix="Final")

    # Create plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Training curves
    plot_training_curves(
        {"train_loss": train_losses},
        save_path=plots_dir / "training_curves.png",
        title=f"{config.experiment.name} - Training Curves",
    )

    # Grid visualization for all dimensions
    target_dim = config.dataset.target_dim
    c_values = plot_data["c_values"]
    true_values = plot_data["true_values"]
    pred_values = plot_data["pred_values"]
    l1_per_dim = plot_data["l1_per_dim"]
    l2_per_dim = plot_data["l2_per_dim"]

    # Determine grid layout
    n_cols = min(4, target_dim)
    n_rows = (target_dim + n_cols - 1) // n_cols

    # 1. Predictions grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if target_dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for dim in range(target_dim):
        ax = axes[dim]
        ax.scatter(c_values, true_values[:, dim], alpha=0.5, s=20, label="True")
        ax.scatter(c_values, pred_values[:, dim], alpha=0.5, s=20, label="Predicted")
        ax.set_xlabel("c")
        ax.set_ylabel(f"x[{dim}]")
        ax.set_title(f"Dimension {dim}")
        ax.legend()
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(target_dim, len(axes)):
        axes[dim].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "predictions_all_dims_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 2. L1 Errors grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if target_dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for dim in range(target_dim):
        ax = axes[dim]
        l1_errors = np.abs(pred_values[:, dim] - true_values[:, dim])
        ax.scatter(c_values, l1_errors, alpha=0.5, s=20, c="red")
        ax.set_xlabel("c")
        ax.set_ylabel("L1 Error")
        ax.set_title(f"Dimension {dim} (mean: {l1_per_dim[dim]:.4f})")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(target_dim, len(axes)):
        axes[dim].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "l1_errors_all_dims_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 3. L2 Errors grid
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if target_dim == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for dim in range(target_dim):
        ax = axes[dim]
        l2_errors = (pred_values[:, dim] - true_values[:, dim]) ** 2
        ax.scatter(c_values, l2_errors, alpha=0.5, s=20, c="orange")
        ax.set_xlabel("c")
        ax.set_ylabel("L2 Error (squared)")
        ax.set_title(f"Dimension {dim} (RMSE: {l2_per_dim[dim]:.4f})")
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for dim in range(target_dim, len(axes)):
        axes[dim].axis("off")

    plt.tight_layout()
    plt.savefig(plots_dir / "l2_errors_all_dims_grid.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # 4. Summary plots: Error vs Dimension
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # L1 error vs dimension
    axes[0].bar(range(target_dim), l1_per_dim, color="red", alpha=0.6)
    axes[0].set_xlabel("Dimension")
    axes[0].set_ylabel("Mean L1 Error")
    axes[0].set_title("L1 Error vs Dimension")
    axes[0].grid(True, alpha=0.3, axis="y")
    axes[0].set_xticks(range(target_dim))

    # L2 error vs dimension
    axes[1].bar(range(target_dim), l2_per_dim, color="orange", alpha=0.6)
    axes[1].set_xlabel("Dimension")
    axes[1].set_ylabel("RMSE")
    axes[1].set_title("L2 Error (RMSE) vs Dimension")
    axes[1].grid(True, alpha=0.3, axis="y")
    axes[1].set_xticks(range(target_dim))

    plt.tight_layout()
    plt.savefig(plots_dir / "error_vs_dimension.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs,
        loss=train_losses[-1],
        save_dir=output_dir / "checkpoints",
        filename="final_model.pt",
        additional_info={"metrics": metrics},
    )

    logger.info("Training complete!")
    logger.info(f"Results saved to {output_dir}")

    # Close CSV loggers
    metrics_logger.close_all()

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train projection experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_proj.py --config config_proj.yaml
  
  # Override config values using full key paths
  python train_proj.py --config config_proj.yaml \\
      experiment.mode=flow \\
      experiment.seed=123 \\
      training.learning_rate=0.001 \\
      dataset.low_dim=3
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_proj.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "overrides",
        nargs="*",
        help="Config overrides in key=value format using full paths (e.g., experiment.mode=flow training.learning_rate=0.01)",
    )

    args = parser.parse_args()

    # Parse overrides from positional arguments
    overrides = None
    if args.overrides:
        overrides = parse_override_args(args.overrides)

    main(args.config, overrides)