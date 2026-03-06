"""
TensorBoard Logger — drop-in replacement for WandBLogger.

Provides the same `log(data, step)`` interface so training scripts can switch
between WandB and TensorBoard by changing a single import / factory call.

Uses TensorFlow's tf.summary as backend (already installed in this environment).
TF is configured to NOT use GPU (GPU is reserved for JAX).
"""

import datetime
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = os.environ.get("TF_CPP_MIN_LOG_LEVEL", "3")

import tensorflow as tf  # noqa: E402

# Prevent TF from allocating GPU memory — GPU belongs to JAX.
try:
    tf.config.set_visible_devices([], "GPU")
except (RuntimeError, ValueError):
    pass


def _to_python_scalar(v):
    """Convert JAX/numpy arrays or Python numbers to a plain float.
    For multi-element arrays, return the mean."""
    if isinstance(v, (int, float)):
        return float(v)
    if hasattr(v, "shape"):
        import numpy as np
        arr = np.asarray(v)
        if arr.size == 0:
            return None
        return float(arr.mean())
    if hasattr(v, "item"):
        return float(v.item())
    return None


def _recursive_flatten_dict(d: dict, prefix: str = ""):
    """Flatten nested dicts into 'key/subkey' format (same convention as WandBLogger)."""
    items = {}
    for key, value in d.items():
        full_key = f"{prefix}/{key}" if prefix else key
        if isinstance(value, dict):
            items.update(_recursive_flatten_dict(value, full_key))
        else:
            scalar = _to_python_scalar(value)
            if scalar is not None:
                items[full_key] = scalar
    return items


class TensorBoardLogger:
    """
    Minimal TensorBoard logger with the same ``log(data, step)`` interface
    as WandBLogger.

    Usage::

        logger = TensorBoardLogger(log_dir="runs/my_experiment")
        logger.log({"critic_loss": 0.5, "actor_loss": 0.3}, step=100)
        logger.log({"environment": {"episode": {"r": 1.0}}}, step=200)
    """

    def __init__(self, log_dir: str, description: str = ""):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"{description}_{timestamp}" if description else timestamp
        self.log_dir = os.path.join(log_dir, run_name)
        self.writer = tf.summary.create_file_writer(self.log_dir)
        print(f"[TensorBoard] Logging to: {self.log_dir}")
        print(f"[TensorBoard] View with:  tensorboard --logdir {os.path.abspath(log_dir)}")

    def log(self, data: dict, step: int = None):
        """
        Log a (possibly nested) dict of scalars.
        Interface-compatible with WandBLogger.log().
        """
        flat = _recursive_flatten_dict(data)
        with self.writer.as_default(step=step or 0):
            for tag, value in flat.items():
                tf.summary.scalar(tag, value, step=step or 0)
        self.writer.flush()

    def close(self):
        self.writer.close()
