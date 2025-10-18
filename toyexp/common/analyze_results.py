"""
Analyze and compare experimental results.

This script loads results from multiple experiment runs and creates
comparison plots and summary statistics.
"""

import argparse
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from toyexp.common.config import load_config
from toyexp.common.logging_utils import setup_logging, get_logger

logger = get_logger(__name__)


def load_experiment_results(output_dir: Path):
    """
    Load results from a single experiment directory.
    
    Returns:
        dict with config, metrics, and checkpoint info
    """
    output_dir = Path(output_dir)
    
    if not output_dir.exists():
        logger.warning(f"Directory not found: {output_dir}")
        return None
    
    results = {
        'dir': str(output_dir),
        'config': None,
        'best_checkpoint': None,
        'final_checkpoint': None,
    }
    
    # Load config
    config_path = output_dir / 'config.yaml'
    if config_path.exists():
        results['config'] = load_config(config_path)
    
    # Load best checkpoint
    best_ckpt_path = output_dir / 'checkpoints' / 'best_model.pt'
    if best_ckpt_path.exists():
        ckpt = torch.load(best_ckpt_path, map_location='cpu')
        results['best_checkpoint'] = ckpt
    
    # Load final checkpoint
    final_ckpt_path = output_dir / 'checkpoints' / 'final_model.pt'
    if final_ckpt_path.exists():
        ckpt = torch.load(final_ckpt_path, map_location='cpu')
        results['final_checkpoint'] = ckpt
    
    return results


def extract_metrics(results):
    """Extract key metrics from results dict."""
    if results is None or results['best_checkpoint'] is None:
        return None
    
    ckpt = results['best_checkpoint']
    metrics = ckpt.get('metrics', {})
    
    return {
        'l1_error': metrics.get('l1_error', np.nan),
        'l2_error': metrics.get('l2_error', np.nan),
        'epoch': ckpt.get('epoch', np.nan),
        'loss': ckpt.get('loss', np.nan),
    }


def compare_methods(results_dict, save_dir=None):
    """
    Compare different methods and create summary plots.
    
    Args:
        results_dict: Dict mapping method names to lists of results
                     (from multiple seeds)
        save_dir: Optional directory to save plots
    """
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Collect metrics for each method
    summary = {}
    for method_name, results_list in results_dict.items():
        metrics_list = [extract_metrics(r) for r in results_list]
        metrics_list = [m for m in metrics_list if m is not None]
        
        if not metrics_list:
            logger.warning(f"No valid metrics for {method_name}")
            continue
        
        # Compute statistics across seeds
        l1_errors = [m['l1_error'] for m in metrics_list]
        l2_errors = [m['l2_error'] for m in metrics_list]
        
        summary[method_name] = {
            'l1_mean': np.mean(l1_errors),
            'l1_std': np.std(l1_errors),
            'l2_mean': np.mean(l2_errors),
            'l2_std': np.std(l2_errors),
            'num_seeds': len(metrics_list),
        }
    
    # Print summary table
    logger.info("\n" + "="*80)
    logger.info("RESULTS SUMMARY")
    logger.info("="*80)
    logger.info(f"{'Method':<30} {'L1 Error':<20} {'L2 Error':<20} {'Seeds':<10}")
    logger.info("-"*80)
    
    for method_name, stats in summary.items():
        l1_str = f"{stats['l1_mean']:.4f} ± {stats['l1_std']:.4f}"
        l2_str = f"{stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}"
        logger.info(f"{method_name:<30} {l1_str:<20} {l2_str:<20} {stats['num_seeds']:<10}")
    
    logger.info("="*80)
    
    # Create comparison bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    methods = list(summary.keys())
    x = np.arange(len(methods))
    
    # L1 errors
    l1_means = [summary[m]['l1_mean'] for m in methods]
    l1_stds = [summary[m]['l1_std'] for m in methods]
    ax1.bar(x, l1_means, yerr=l1_stds, capsize=5, alpha=0.7)
    ax1.set_xlabel('Method', fontsize=12)
    ax1.set_ylabel('L1 Error', fontsize=12)
    ax1.set_title('L1 Error Comparison', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3)
    
    # L2 errors
    l2_means = [summary[m]['l2_mean'] for m in methods]
    l2_stds = [summary[m]['l2_std'] for m in methods]
    ax2.bar(x, l2_means, yerr=l2_stds, capsize=5, alpha=0.7, color='orange')
    ax2.set_xlabel('Method', fontsize=12)
    ax2.set_ylabel('L2 Error', fontsize=12)
    ax2.set_title('L2 Error Comparison', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_dir:
        save_path = save_dir / 'method_comparison.png'
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved comparison plot to {save_path}")
    
    plt.close(fig)
    
    return summary


def main():
    """Main analysis function."""
    parser = argparse.ArgumentParser(
        description="Analyze experiment results"
    )
    parser.add_argument(
        '--results_dir',
        type=str,
        default='./outputs',
        help='Base directory containing experiment results'
    )
    parser.add_argument(
        '--experiment',
        type=str,
        choices=['recon', 'proj', 'all'],
        default='all',
        help='Which experiments to analyze'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./analysis',
        help='Directory to save analysis results'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(
        name='analyze_results',
        level=logging.INFO,
    )
    
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("="*80)
    logger.info("Experiment Results Analysis")
    logger.info("="*80)
    logger.info(f"Results directory: {results_dir}")
    logger.info(f"Output directory: {output_dir}")
    logger.info("")
    
    # Define experiment patterns
    experiments_to_analyze = []
    
    if args.experiment in ['recon', 'all']:
        experiments_to_analyze.extend([
            ('Recon - Regression', 'recon_regression_seed*'),
            ('Recon - Flow', 'recon_flow_seed*'),
        ])
    
    if args.experiment in ['proj', 'all']:
        experiments_to_analyze.extend([
            ('Proj - Regression', 'proj_regression_seed*'),
            ('Proj - Flow', 'proj_flow_seed*'),
        ])
    
    # Load all results
    all_results = {}
    
    for method_name, pattern in experiments_to_analyze:
        logger.info(f"Loading results for: {method_name}")
        
        # Find matching directories
        matching_dirs = sorted(results_dir.glob(pattern))
        
        if not matching_dirs:
            logger.warning(f"No results found for pattern: {pattern}")
            continue
        
        logger.info(f"Found {len(matching_dirs)} runs")
        
        # Load results from each seed
        results_list = []
        for exp_dir in matching_dirs:
            results = load_experiment_results(exp_dir)
            if results is not None:
                results_list.append(results)
        
        all_results[method_name] = results_list
        logger.info(f"Loaded {len(results_list)} valid results")
        logger.info("")
    
    # Compare methods
    if all_results:
        logger.info("Comparing methods...")
        summary = compare_methods(all_results, save_dir=output_dir)
        
        # Save summary to file
        summary_file = output_dir / 'summary.txt'
        with open(summary_file, 'w') as f:
            f.write("EXPERIMENT RESULTS SUMMARY\n")
            f.write("="*80 + "\n\n")
            
            for method_name, stats in summary.items():
                f.write(f"{method_name}:\n")
                f.write(f"  L1 Error: {stats['l1_mean']:.4f} ± {stats['l1_std']:.4f}\n")
                f.write(f"  L2 Error: {stats['l2_mean']:.4f} ± {stats['l2_std']:.4f}\n")
                f.write(f"  Number of seeds: {stats['num_seeds']}\n\n")
        
        logger.info(f"Saved summary to {summary_file}")
    else:
        logger.warning("No results to compare!")
    
    logger.info("")
    logger.info("="*80)
    logger.info("Analysis complete!")
    logger.info("="*80)


if __name__ == "__main__":
    main()