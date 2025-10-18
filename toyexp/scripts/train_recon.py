"""
Training script for reconstruction experiment.

Compare regression vs flow models on scalar target functions f(c).
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from toyexp.common.datasets import TargetFunctionDataset
from toyexp.common.integrate import integrate
from toyexp.common.logging_utils import (
    setup_logging,
    get_logger,
    log_config,
    log_evaluation,
    log_model_info,
    log_training_step,
)
from toyexp.common.losses import LossManager
from toyexp.common.networks import create_model
from toyexp.common.config import load_config, save_config, validate_config
from toyexp.common.utils import (
    plot_errors,
    plot_predictions,
    plot_training_curves,
    save_checkpoint,
    set_seed,
    parse_override_args,
    build_experiment_name,
)
from toyexp.common.config import merge_configs, Config

logger = get_logger(__name__)


def create_datasets(config):
    """Create training and evaluation datasets from config."""
    logger.info("Creating datasets...")
    
    # Training dataset
    train_dataset = TargetFunctionDataset(
        num_samples=config.dataset.num_train,
        target_dim=config.dataset.target_dim,
        condition_dim=config.dataset.condition_dim,
        num_components=config.dataset.num_components,
        c_min=config.dataset.c_min,
        c_max=config.dataset.c_max,
        weight_strategy=config.dataset.weight_strategy,
        sampling_strategy=config.dataset.sampling_strategy,
        freq_seed=config.dataset.freq_seed,
        phase_seed=config.dataset.phase_seed,
        weight_seed=config.dataset.weight_seed,
        sample_seed=config.dataset.sample_seed,
    )
    
    logger.info(f"Training dataset: {len(train_dataset)} samples")
    
    return train_dataset


def train_epoch(model, dataloader, loss_manager, optimizer, device, config):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    num_batches = 0
    
    for batch in dataloader:
        c = batch['c'].to(device)
        x_1 = batch['x'].to(device)
        
        # Compute loss
        if config.experiment.mode == 'regression':
            loss = loss_manager.compute_loss(model, c, x_1)
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


def evaluate(model, dataset, device, config):
    """Evaluate model on dataset."""
    model.eval()
    
    # Generate evaluation data
    eval_data = dataset.generate_eval_data(
        num_samples=config.dataset.num_eval,
        eval_seed=config.experiment.seed + 1000,
    )
    
    c_eval = eval_data['c'].to(device)
    x_true = eval_data['x'].cpu().numpy()
    
    with torch.no_grad():
        # Initial distribution
        if config.training.initial_dist == "gaussian":
            x_0 = torch.randn_like(eval_data['x']).to(device)
        else:
            x_0 = torch.zeros_like(eval_data['x']).to(device)
        
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
    
    metrics = {
        'l1_error': l1_error,
        'l2_error': l2_error,
    }
    
    # Prepare data for plotting
    plot_data = {
        'c_values': c_eval.cpu().numpy().flatten(),
        'true_values': x_true.flatten(),
        'pred_values': x_pred.flatten(),
    }
    
    return metrics, plot_data


def main(config_path: str, overrides: dict = None):
    """Main training function."""
    # Load and validate configuration
    config = load_config(config_path)
    
    if overrides:
        config = merge_configs(config, overrides)
    
    validate_config(config)
    
    # Setup
    output_dir = Path(config.experiment.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    setup_logging(
        name='train_recon',
        level=logging.INFO,
        log_file=output_dir / 'train.log',
    )
    
    logger.info("="*80)
    logger.info(f"Starting {config.experiment.name}")
    logger.info("="*80)
    
    # Save config
    save_config(config, output_dir / 'config.yaml')
    log_config(config.to_dict())
    
    # Set seed
    set_seed(config.experiment.seed)
    
    # Device
    device = torch.device(config.experiment.device 
                         if torch.cuda.is_available() and config.experiment.device == "cuda" 
                         else "cpu")
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
        use_time=(config.experiment.mode == 'flow'),
    ).to(device)
    
    log_model_info(model)
    
    # Create loss manager
    loss_manager = LossManager(
        mode=config.experiment.mode,
        loss_type=config.training.loss_type,
        x_dim=config.dataset.target_dim,
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=config.training.learning_rate)
    
    # Training loop
    logger.info("Starting training...")
    train_losses = []
    best_l2_error = float('inf')
    
    for epoch in range(config.training.num_epochs):
        # Train
        epoch_loss = train_epoch(model, train_loader, loss_manager, optimizer, device, config)
        train_losses.append(epoch_loss)
        
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
            
            # Save best model
            if metrics['l2_error'] < best_l2_error:
                best_l2_error = metrics['l2_error']
                save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    save_dir=output_dir / 'checkpoints',
                    filename='best_model.pt',
                    additional_info={'metrics': metrics},
                )
        
        # Save periodic checkpoint
        if (epoch + 1) % config.training.save_interval == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
                save_dir=output_dir / 'checkpoints',
                filename=f'checkpoint_epoch_{epoch+1}.pt',
            )
    
    # Final evaluation
    logger.info("="*80)
    logger.info("Final evaluation...")
    logger.info("="*80)
    
    metrics, plot_data = evaluate(model, train_dataset, device, config)
    log_evaluation(metrics, prefix="Final")
    
    # Create plots
    plots_dir = output_dir / 'plots'
    plots_dir.mkdir(exist_ok=True)
    
    # Training curves
    plot_training_curves(
        {'train_loss': train_losses},
        save_path=plots_dir / 'training_curves.png',
        title=f"{config.experiment.name} - Training Curves",
    )
    
    # Predictions
    plot_predictions(
        plot_data['c_values'],
        plot_data['true_values'],
        plot_data['pred_values'],
        save_path=plots_dir / 'predictions.png',
        title=f"{config.experiment.name} - Predictions",
    )
    
    # Errors
    errors = np.abs(plot_data['pred_values'] - plot_data['true_values'])
    plot_errors(
        plot_data['c_values'],
        errors,
        save_path=plots_dir / 'errors.png',
        title=f"{config.experiment.name} - Prediction Errors",
    )
    
    # Save final checkpoint
    save_checkpoint(
        model=model,
        optimizer=optimizer,
        epoch=config.training.num_epochs,
        loss=train_losses[-1],
        save_dir=output_dir / 'checkpoints',
        filename='final_model.pt',
        additional_info={'metrics': metrics},
    )
    
    logger.info("Training complete!")
    logger.info(f"Results saved to {output_dir}")
    
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train reconstruction experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python train_recon.py --config config_recon.yaml
  
  # Override mode and seed
  python train_recon.py --config config_recon.yaml --mode flow --seed 123
  
  # Use config overrides (dot notation)
  python train_recon.py --config config_recon.yaml \\
      experiment.mode=flow \\
      training.learning_rate=0.001 \\
      dataset.num_train=100
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_recon.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=['regression', 'flow'],
        help="Override experiment mode",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override random seed",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Override output directory",
    )
    parser.add_argument(
        'overrides',
        nargs='*',
        help='Config overrides in key=value format (e.g., training.learning_rate=0.01)',
    )
    
    args = parser.parse_args()
    
    # Build overrides dict from named arguments
    overrides = {}
    if args.mode:
        overrides['experiment'] = {'mode': args.mode}
    if args.seed:
        if 'experiment' not in overrides:
            overrides['experiment'] = {}
        overrides['experiment']['seed'] = args.seed
    if args.output_dir:
        if 'experiment' not in overrides:
            overrides['experiment'] = {}
        overrides['experiment']['output_dir'] = args.output_dir
    
    # Parse additional overrides from positional arguments
    if args.overrides:
        additional_overrides = parse_override_args(args.overrides)
        # Merge with named argument overrides
        overrides = merge_configs(Config(overrides), additional_overrides).to_dict() if overrides else additional_overrides
    
    main(args.config, overrides if overrides else None)