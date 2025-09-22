#!/usr/bin/env python3
import sys
import os

# Test the path calculation
libs_path = os.path.join(os.getcwd(), 'libs', 'src')
print('Current directory:', os.getcwd())
print('Libs path:', libs_path)
print('Path exists:', os.path.exists(libs_path))
if os.path.exists(libs_path):
    print('Files in libs/src:', os.listdir(libs_path))

# Add to path and test import
sys.path.insert(0, libs_path)
try:
    import neuprint_funcs as npf
    print('Import successful!')
    print('Available functions:', [f for f in dir(npf) if not f.startswith('_')])
except ImportError as e:
    print('Import failed:', e)


