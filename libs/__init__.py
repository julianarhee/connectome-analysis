"""
Connectome analysis shared libraries package.
"""

# Import all modules to make them available when importing the package
from . import neuprint_funcs
from . import plotting  
from . import utils

# Make modules available at package level
__all__ = ['neuprint_funcs', 'plotting', 'utils']

# Version info
__version__ = "0.1.0"
