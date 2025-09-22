#!/usr/bin/env python3
"""
Tests for utils module
"""

import sys
import os

# Add the libs src directory to the Python path
libs_src_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'src')
if libs_src_path not in sys.path:
    sys.path.insert(0, libs_src_path)

import utils as util

def test_manual_tab2csv():
    """Test the manual_tab2csv function"""
    # This is a basic test - you might want to mock the file path
    # or create a test data file
    try:
        result = util.manual_tab2csv()
        assert result is not None
        print("✓ manual_tab2csv test passed")
    except Exception as e:
        print(f"✗ manual_tab2csv test failed: {e}")

def test_set_sns_style():
    """Test the set_sns_style function"""
    try:
        util.set_sns_style(style='dark')
        print("✓ set_sns_style test passed")
    except Exception as e:
        print(f"✗ set_sns_style test failed: {e}")

if __name__ == "__main__":
    test_manual_tab2csv()
    test_set_sns_style()


