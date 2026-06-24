"""
TensorFlow compatibility helpers for this project.
"""

from __future__ import annotations

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None


def require_tensorflow():
    """
    Return the TensorFlow module or raise a project-specific install error.
    """
    if tf is None:
        raise ModuleNotFoundError(
            "TensorFlow is required for `leaf-disease-classifier` but is not "
            "installed in the current Python environment.\n"
            "Install project dependencies with:\n"
            "  pip install -r requirements.txt"
        )
    return tf
