#!/usr/bin/env python3
"""
Example app showing how to import from the shared libs module
"""

# Import from the shared libs package
import sys
import os


import neuprint_funcs as npf
import plotting as putil
import utils as util

# Example usage
def main():
    print("Example app importing from shared libs:")
    print(f"Available functions from neuprint_funcs: {dir(npf)}")
    print(f"Available functions from plotting: {dir(putil)}")
    print(f"Available functions from utils: {dir(util)}")
    
    # Test that imports work
    print("\nImport test successful!")

if __name__ == "__main__":
    main()