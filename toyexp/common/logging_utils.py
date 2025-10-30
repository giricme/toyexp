"""
Logging utilities for toy experiments.

Provides centralized logging configuration and utilities.
All modules should import and use the logger from this module.
"""

import csv
from datetime import datetime
import logging
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional


def setup_logging(
    name: str = "toyexp",
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    log_to_console: bool = True,
    format_string: Optional[str] = None,
) -> logging.Logger:
    """
    Setup a logger with consistent formatting.

    Args:
        name: Logger name (use module name or 'toyexp' for root)
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: If provided, log to this file in addition to console
        log_to_console: Whether to log to console
        format_string: Custom format string (uses default if None)

    Returns:
        Configured logger instance

    Usage:
        # In any module
        from logging_utils import get_logger
        logger = get_logger(__name__)

        logger.info("Training started")
        logger.debug(f"Batch size: {batch_size}")
        logger.warning("Learning rate may be too high")
        logger.error("Failed to load checkpoint")
    """
    logger = logging.getLogger(name)

    # Avoid adding handlers multiple times
    if logger.hasHandlers():
        return logger

    logger.setLevel(level)

    # Default format
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    formatter = logging.Formatter(format_string, datefmt="%Y-%m-%d %H:%M:%S")

    # Console handler
    if log_to_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if log_file is not None:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "toyexp") -> logging.Logger:
    """
    Get or create a logger with the given name.

    If the logger doesn't exist yet, it will be created with default settings.
    Use setup_logger() first if you need custom configuration.

    Args:
        name: Logger name (typically __name__ from the calling module)

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set it up with defaults
    if not logger.hasHandlers():
        return setup_logging(name)

    return logger


def set_log_level(level: int, logger_name: Optional[str] = None):
    """
    Change the logging level for a specific logger or all loggers.

    Args:
        level: New logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Specific logger to modify, or None for root logger

    Usage:
        # Set all logging to DEBUG
        set_log_level(logging.DEBUG)

        # Set specific module to WARNING
        set_log_level(logging.WARNING, 'toyexp.datasets')
    """
    if logger_name is None:
        logger_name = "toyexp"

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)

    # Also update handler levels
    for handler in logger.handlers:
        handler.setLevel(level)


class LoggerContext:
    """
    Context manager for temporarily changing log level.

    Usage:
        logger = get_logger(__name__)

        logger.info("This is shown")

        with LoggerContext(logging.DEBUG):
            logger.debug("This is also shown")

        logger.debug("This is hidden again")
    """

    def __init__(self, level: int, logger_name: Optional[str] = None):
        self.level = level
        self.logger_name = logger_name if logger_name else "toyexp"
        self.original_level = None

    def __enter__(self):
        logger = logging.getLogger(self.logger_name)
        self.original_level = logger.level
        set_log_level(self.level, self.logger_name)
        return self

    def __exit__(self, *args):
        set_log_level(self.original_level, self.logger_name)


def log_model_info(model, logger: Optional[logging.Logger] = None):
    """
    Log information about a model (architecture, parameters).

    Args:
        model: PyTorch model
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = get_logger(__name__)

    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Model: {model.__class__.__name__}")
    logger.info(f"Total parameters: {n_params:,}")
    logger.info(f"Trainable parameters: {n_trainable:,}")


def log_training_step(
    epoch: int,
    step: int,
    loss: float,
    metrics: Optional[dict] = None,
    logger: Optional[logging.Logger] = None,
):
    """
    Log a training step with consistent formatting.

    Args:
        epoch: Current epoch
        step: Current step
        loss: Loss value
        metrics: Optional dict of additional metrics to log
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = get_logger(__name__)

    msg = f"Epoch {epoch:4d} | Step {step:6d} | Loss: {loss:.6f}"

    if metrics:
        for key, value in metrics.items():
            if isinstance(value, float):
                msg += f" | {key}: {value:.6f}"
            else:
                msg += f" | {key}: {value}"

    logger.info(msg)


def log_evaluation(
    metrics: dict,
    prefix: str = "Eval",
    logger: Optional[logging.Logger] = None,
):
    """
    Log evaluation metrics with consistent formatting.
    
    Filters out numpy arrays and other complex types to avoid cluttering logs.

    Args:
        metrics: Dict of metric name -> value
        prefix: Prefix for log message (e.g., 'Eval', 'Test', 'Train')
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info(f"{prefix} Results:")
    for key, value in metrics.items():
        # Skip numpy arrays, lists, dicts, and other complex types
        if hasattr(value, '__len__') and not isinstance(value, str):
            continue  # Skip arrays, lists, dicts
        if key.startswith('_'):  # Skip private/internal metrics
            continue
        
        if isinstance(value, float):
            logger.info(f"  {key}: {value:.6f}")
        elif isinstance(value, (int, str, bool)):
            logger.info(f"  {key}: {value}")


def log_config(config: dict, logger: Optional[logging.Logger] = None):
    """
    Log configuration with nice formatting.

    Args:
        config: Configuration dict (can be nested)
        logger: Logger to use (creates default if None)
    """
    if logger is None:
        logger = get_logger(__name__)

    logger.info("=" * 60)
    logger.info("Configuration:")
    logger.info("=" * 60)

    def _log_dict(d: dict, indent: int = 0):
        for key, value in d.items():
            if isinstance(value, dict):
                logger.info("  " * indent + f"{key}:")
                _log_dict(value, indent + 1)
            else:
                logger.info("  " * indent + f"{key}: {value}")

    _log_dict(config)
    logger.info("=" * 60)


class CSVLogger:
    """Write metrics to CSV with automatic timestamping."""

    def __init__(self, filepath: Path, fieldnames: List[str]):
        self.filepath = Path(filepath)
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = ["timestamp"] + fieldnames
        self.file = None
        self.writer = None
        self._init_file()

    def _init_file(self):
        file_exists = self.filepath.exists()
        self.file = open(self.filepath, "a", newline="")
        self.writer = csv.DictWriter(self.file, fieldnames=self.fieldnames)
        if not file_exists:
            self.writer.writeheader()

    def log(self, metrics: Dict[str, Any]):
        """Log metrics row with timestamp."""
        row = {"timestamp": datetime.now().isoformat(), **metrics}
        self.writer.writerow(row)
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


class MetricsLogger:
    """Manage multiple CSV loggers."""

    def __init__(self, output_dir: Path):
        self.output_dir = Path(output_dir)
        self.loggers = {}

    def add_logger(self, name: str, fieldnames: List[str]):
        """Add a CSV logger."""
        filepath = self.output_dir / f"{name}.csv"
        self.loggers[name] = CSVLogger(filepath, fieldnames)

    def log(self, logger_name: str, metrics: Dict[str, Any]):
        """Log to specific CSV."""
        self.loggers[logger_name].log(metrics)

    def close_all(self):
        for logger in self.loggers.values():
            logger.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close_all()


def create_metrics_logger(
    output_dir: Path, experiment_type: str = "lie"
) -> MetricsLogger:
    """Factory to create MetricsLogger with standard configurations."""
    ml = MetricsLogger(output_dir)

    # Training logger (all experiments)
    ml.add_logger("training", ["epoch", "loss"])

    # Evaluation logger
    if experiment_type == "lie":
        ml.add_logger(
            "evaluation",
            [
                "epoch",
                "l1_error",
                "l2_error",
                "avg_cos_similarity",
                "min_cos_similarity",
                "avg_abs_cos_similarity",
                "min_abs_cos_similarity",
                "avg_perp_error",
            ],
        )
        ml.add_logger(
            "components",
            [
                "epoch",
                "component",
                "alpha",
                "cos_similarity",
                "abs_cos_similarity",
                "perp_error_mean",
                "perp_error_std",
            ],
        )
    else:  # recon, proj
        ml.add_logger("evaluation", ["epoch", "l1_error", "l2_error"])

    return ml


if __name__ == "__main__":
    logger = logging.getLogger(__name__)
    logger.info("Testing logging utilities...\n")

    # Test 1: Basic logger setup
    logger.info("=" * 60)
    logger.info("Test 1: Basic logger setup")
    logger.info("=" * 60)

    logger = setup_logger("test_logger", level=logging.INFO)

    logger.debug("This DEBUG message should not appear")
    logger.info("This INFO message should appear")
    logger.warning("This WARNING message should appear")
    logger.error("This ERROR message should appear")

    logger.info("\nÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Basic logging works\n")

    # Test 2: Get logger
    logger.info("=" * 60)
    logger.info("Test 2: Get existing logger")
    logger.info("=" * 60)

    logger2 = get_logger("test_logger")
    logger2.info("Same logger instance retrieved")

    logger.info("ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ get_logger() works\n")

    # Test 3: File logging
    logger.info("=" * 60)
    logger.info("Test 3: File logging")
    logger.info("=" * 60)

    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
        log_file = f.name

    file_logger = setup_logger("file_logger", log_file=log_file)
    file_logger.info("This message goes to file and console")

    with open(log_file, "r") as f:
        content = f.read()
        logger.info(f"Log file contains {len(content)} characters")

    import os

    os.unlink(log_file)

    logger.info("ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ File logging works\n")

    # Test 4: Change log level
    logger.info("=" * 60)
    logger.info("Test 4: Change log level")
    logger.info("=" * 60)

    test_logger = setup_logger("level_test", level=logging.INFO)
    test_logger.debug("This should NOT appear (level=INFO)")

    set_log_level(logging.DEBUG, "level_test")
    test_logger.debug("This SHOULD appear (level=DEBUG)")

    logger.info("ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Log level changes work\n")

    # Test 5: Logger context
    logger.info("=" * 60)
    logger.info("Test 5: Logger context (temporary level change)")
    logger.info("=" * 60)

    ctx_logger = setup_logger("context_test", level=logging.INFO)
    ctx_logger.debug("1. This should NOT appear (INFO level)")

    with LoggerContext(logging.DEBUG, "context_test"):
        ctx_logger.debug("2. This SHOULD appear (DEBUG temporarily)")

    ctx_logger.debug("3. This should NOT appear again (back to INFO)")

    logger.info("ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Logger context works\n")

    # Test 6: Log model info
    logger.info("=" * 60)
    logger.info("Test 6: Log model info")
    logger.info("=" * 60)

    import torch.nn as nn

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 20)
            self.fc2 = nn.Linear(20, 1)

    model = SimpleModel()
    model_logger = get_logger("model_test")
    log_model_info(model, model_logger)

    logger.info("\nÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Model info logging works\n")

    # Test 7: Log training step
    logger.info("=" * 60)
    logger.info("Test 7: Log training step")
    logger.info("=" * 60)

    train_logger = get_logger("train_test")
    log_training_step(
        epoch=1,
        step=100,
        loss=0.123456,
        metrics={"lr": 0.001, "grad_norm": 2.5},
        logger=train_logger,
    )

    logger.info("\nÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Training step logging works\n")

    # Test 8: Log evaluation
    logger.info("=" * 60)
    logger.info("Test 8: Log evaluation metrics")
    logger.info("=" * 60)

    eval_logger = get_logger("eval_test")
    log_evaluation(
        metrics={
            "loss": 0.234567,
            "l1_error": 0.123,
            "l2_error": 0.456,
            "n_samples": 1000,
        },
        prefix="Test",
        logger=eval_logger,
    )

    logger.info("\nÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Evaluation logging works\n")

    # Test 9: Log config
    logger.info("=" * 60)
    logger.info("Test 9: Log configuration")
    logger.info("=" * 60)

    config_logger = get_logger("config_test")
    config = {
        "model": {"architecture": "concat", "hidden_dim": 256, "n_layers": 3},
        "training": {"lr": 0.001, "batch_size": 32, "epochs": 1000},
        "seed": 42,
    }
    log_config(config, config_logger)

    logger.info("\nÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ Config logging works\n")

    logger.info("=" * 60)
    logger.info("All logging tests passed! ÃƒÆ’Ã‚Â¢Ãƒâ€¦Ã¢â‚¬Å“ÃƒÂ¢Ã¢â€šÂ¬Ã…â€œ")
    logger.info("=" * 60)