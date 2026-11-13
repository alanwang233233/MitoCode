"""Configuration file location for MitoCode.

A single entry point answers "which configuration file should this module
use?": the project copy under ``<workspace>/.mitocode`` wins, the user-level
copy is the fallback, and a missing pair is an explicit error rather than an
empty configuration.
"""

from .manager import ConfigManager, ConfigScope, default_global_dir

__all__ = [
    "ConfigManager",
    "ConfigScope",
    "default_global_dir",
]
