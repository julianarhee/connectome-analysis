#!/usr/bin/env python3
"""
TuTu-LC1a0-AOTU analysis app
"""

# Import from the shared libs package
import neuprint_funcs as npf
import plotting as putil
import utils as util

def main():
    print("TuTu-LC1a0-AOTU analysis app:")
    print(f"Available functions from neuprint_funcs: {dir(npf)}")
    print(f"Available functions from plotting: {dir(putil)}")
    print(f"Available functions from utils: {dir(util)}")
    
    # TODO: Add TuTu-LC1a0-AOTU specific analysis code here
    print("\nApp ready for TuTu-LC1a0-AOTU analysis!")

if __name__ == "__main__":
    main()
