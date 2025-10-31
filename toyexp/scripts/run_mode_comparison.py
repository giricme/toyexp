"""
Batch wrapper script for running mode comparison experiments.

This script automates running training across different modes (regression/flow/mip)
and loss types (l1/l2), then generates a LaTeX table comparing the results.

Usage:
    python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml
    python -m toyexp.scripts.run_mode_comparison --experiment proj --config toyexp/configs/config_proj.yaml
    python -m toyexp.scripts.run_mode_comparison --experiment lie --config toyexp/configs/config_lie.yaml
"""

import argparse
import importlib
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import yaml


# Experiment configurations
EXPERIMENT_CONFIGS = {
    "recon": {
        "module": "toyexp.scripts.train_recon",
        "modes": ["regression", "flow", "mip"],
        "loss_types": ["l1", "l2"],
        "metrics": ["L1", "L2"],
    },
    "proj": {
        "module": "toyexp.scripts.train_proj",
        "modes": ["regression", "flow", "mip"],
        "loss_types": ["l1", "l2"],
        "metrics": ["L1", "L2"],
    },
    "lie": {
        "module": "toyexp.scripts.train_lie",
        "modes": ["regression", "flow", "mip"],
        "loss_types": ["l1", "l2"],
        "metrics": [
            "L1",
            "L2",
            "Avg Cos Sim",
            "Min Cos Sim",
            "Avg Perp Error",
            "Max Perp Error",
        ],
    },
}


def setup_logging():
    """Setup logging for the batch wrapper."""
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def run_training(module_name: str, config_path: str, mode: str, loss_type: str, logger):
    """
    Run a single training job with specified mode and loss_type.
    
    Args:
        module_name: Full module name (e.g., 'toyexp.train_recon')
        config_path: Path to config file
        mode: Training mode (regression/flow/mip)
        loss_type: Loss type (l1/l2)
        logger: Logger instance
        
    Returns:
        bool: True if training completed successfully
    """
    logger.info("=" * 80)
    logger.info(f"Starting training: mode={mode}, loss_type={loss_type}")
    logger.info("=" * 80)
    
    try:
        # Import the training module
        train_module = importlib.import_module(module_name)
        
        # Build overrides dict
        overrides = {
            "experiment": {"mode": mode},
            "training": {"loss_type": loss_type},
        }
        
        logger.info(f"Module: {module_name}")
        logger.info(f"Config: {config_path}")
        logger.info(f"Overrides: {overrides}")
        
        # Call the main function with overrides
        train_module.main(config_path, overrides)
        
        logger.info(f"✓ Training completed successfully: mode={mode}, loss_type={loss_type}")
        return True
        
    except Exception as e:
        logger.error(f"✗ Training failed: mode={mode}, loss_type={loss_type}")
        logger.error(f"Error: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return False


def load_results(output_dir: Path, mode: str, loss_type: str, logger):
    """
    Load results from a completed training run.
    
    Args:
        output_dir: Base output directory
        mode: Training mode
        loss_type: Loss type
        logger: Logger instance
        
    Returns:
        dict: Metrics from best checkpoint, or None if not found
    """
    # Results are saved in: output_dir / mode / loss_type / checkpoints / best_model.pt
    results_path = output_dir / mode / loss_type / "checkpoints" / "final_model.pt"
    
    if not results_path.exists():
        logger.warning(f"Results not found: {results_path}")
        return None
    
    try:
        checkpoint = torch.load(results_path, map_location="cpu", weights_only=False)
        metrics = checkpoint.get("metrics", {})
        
        logger.info(f"Loaded results from {results_path}")
        logger.info(f"  L1: {metrics.get('l1_error', np.nan):.6f}")
        logger.info(f"  L2: {metrics.get('l2_error', np.nan):.6f}")
        
        return metrics
        
    except Exception as e:
        logger.error(f"Failed to load results from {results_path}: {e}")
        return None


def generate_latex_table(
    results: dict,
    experiment_type: str,
    save_path: Path,
    logger,
):
    """
    Generate LaTeX table comparing results across modes and loss types.
    
    Args:
        results: Dict mapping (mode, loss_type) -> metrics
        experiment_type: Type of experiment (recon/proj/lie)
        save_path: Where to save the LaTeX table
        logger: Logger instance
    """
    logger.info("=" * 80)
    logger.info("Generating LaTeX table")
    logger.info("=" * 80)
    
    exp_config = EXPERIMENT_CONFIGS[experiment_type]
    modes = exp_config["modes"]
    loss_types = exp_config["loss_types"]
    metric_names = exp_config["metrics"]
    
    # Start building LaTeX table
    lines = []
    lines.append("\\begin{table}[h]")
    lines.append("\\centering")
    lines.append("\\begin{tabular}{|l|l|" + "c|" * len(metric_names) + "}")
    lines.append("\\hline")
    
    # Header row
    header = "Mode & Loss & " + " & ".join(metric_names) + " \\\\"
    lines.append(header)
    lines.append("\\hline")
    
    # Data rows
    for mode in modes:
        for loss_type in loss_types:
            key = (mode, loss_type)
            
            if key not in results or results[key] is None:
                # Missing results - fill with dashes
                values = ["---"] * len(metric_names)
            else:
                metrics = results[key]
                values = []
                
                # Extract metrics based on experiment type
                if experiment_type in ["recon", "proj"]:
                    # Simple: L1, L2
                    values.append(f"{metrics.get('l1_error', np.nan):.6f}")
                    values.append(f"{metrics.get('l2_error', np.nan):.6f}")
                    
                elif experiment_type == "lie":
                    # Complex: L1, L2, Avg Cos Sim, Min Cos Sim, Avg Perp Error, Max Perp Error
                    values.append(f"{metrics.get('l1_error', np.nan):.6f}")
                    values.append(f"{metrics.get('l2_error', np.nan):.6f}")
                    values.append(f"{metrics.get('avg_cos_similarity', np.nan):.6f}")
                    values.append(f"{metrics.get('min_cos_similarity', np.nan):.6f}")
                    values.append(f"{metrics.get('avg_perp_error', np.nan):.6f}")
                    values.append(f"{metrics.get('max_perp_error', np.nan):.6f}")
            
            # Format row
            row = f"{mode} & {loss_type} & " + " & ".join(values) + " \\\\"
            lines.append(row)
            lines.append("\\hline")
    
    # Close table
    lines.append("\\end{tabular}")
    lines.append(f"\\caption{{Comparison of {experiment_type} experiment results across modes and loss types}}")
    lines.append(f"\\label{{tab:{experiment_type}_comparison}}")
    lines.append("\\end{table}")
    
    # Save to file
    latex_content = "\n".join(lines)
    save_path.write_text(latex_content)
    
    logger.info(f"Saved LaTeX table to {save_path}")
    logger.info("\nTable preview:")
    logger.info(latex_content)


def main():
    """Main function for batch mode comparison."""
    parser = argparse.ArgumentParser(
        description="Run mode comparison experiments and generate LaTeX tables",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run reconstruction experiment
  python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml
  
  # Run projection experiment
  python -m toyexp.scripts.run_mode_comparison --experiment proj --config toyexp/configs/config_proj.yaml
  
  # Run Lie experiment
  python -m toyexp.scripts.run_mode_comparison --experiment lie --config toyexp/configs/config_lie.yaml
  
  # Skip training and only generate table from existing results
  python -m toyexp.scripts.run_mode_comparison --experiment recon --config toyexp/configs/config_recon.yaml --skip-training
        """,
    )
    
    parser.add_argument(
        "--experiment",
        type=str,
        required=True,
        choices=["recon", "proj", "lie"],
        help="Type of experiment to run",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to base config file",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training and only generate table from existing results",
    )
    
    args = parser.parse_args()
    
    # Setup
    logger = setup_logging()
    config_path = Path(args.config)
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        sys.exit(1)
    
    # Get experiment configuration
    exp_config = EXPERIMENT_CONFIGS[args.experiment]
    module_name = exp_config["module"]
    
    logger.info("=" * 80)
    logger.info(f"Batch Mode Comparison: {args.experiment.upper()}")
    logger.info("=" * 80)
    logger.info(f"Config: {config_path}")
    logger.info(f"Module: {module_name}")
    logger.info(f"Modes: {exp_config['modes']}")
    logger.info(f"Loss types: {exp_config['loss_types']}")
    logger.info("")
    
    # Load base config to get output directory
    with open(config_path) as f:
        base_config = yaml.safe_load(f)
    base_output_dir = Path(base_config["experiment"]["output_dir"])
    
    logger.info(f"Output directory: {base_output_dir}")
    logger.info("")
    
    # Run training for each mode/loss_type combination
    if not args.skip_training:
        success_count = 0
        total_count = len(exp_config["modes"]) * len(exp_config["loss_types"])
        
        for mode in exp_config["modes"]:
            for loss_type in exp_config["loss_types"]:
                success = run_training(
                    module_name,
                    str(config_path),
                    mode,
                    loss_type,
                    logger,
                )
                if success:
                    success_count += 1
                logger.info("")
        
        logger.info("=" * 80)
        logger.info(f"Training Summary: {success_count}/{total_count} completed successfully")
        logger.info("=" * 80)
        logger.info("")
    else:
        logger.info("Skipping training (--skip-training flag set)")
        logger.info("")
    
    # Load all results
    logger.info("=" * 80)
    logger.info("Loading results")
    logger.info("=" * 80)
    
    results = {}
    for mode in exp_config["modes"]:
        for loss_type in exp_config["loss_types"]:
            metrics = load_results(base_output_dir, mode, loss_type, logger)
            results[(mode, loss_type)] = metrics
            logger.info("")
    
    # Generate LaTeX table
    table_path = base_output_dir / "results_table.tex"
    generate_latex_table(results, args.experiment, table_path, logger)
    
    logger.info("")
    logger.info("=" * 80)
    logger.info("Batch comparison complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()