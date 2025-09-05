#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test imports for debugging the diagnostic server
"""

import sys
import os

# Add current directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

print(f"Testing imports from: {current_dir}")
print(f"Python path: {sys.path[:3]}")

try:
    print("Testing basic imports...")
    import cv2
    print("✅ OpenCV imported successfully")
except ImportError as e:
    print(f"❌ OpenCV import error: {e}")

try:
    import easyocr
    print("✅ EasyOCR imported successfully")
except ImportError as e:
    print(f"❌ EasyOCR import error: {e}")

try:
    from TestRun.opencv_adaptive_detector import OpenCVAdaptiveDetector
    print("✅ OpenCVAdaptiveDetector imported successfully")
except ImportError as e:
    print(f"❌ OpenCVAdaptiveDetector import error: {e}")

try:
    from TestRun.username_extractor import UsernameExtractor
    print("✅ UsernameExtractor imported successfully")
except ImportError as e:
    print(f"❌ UsernameExtractor import error: {e}")

# Test basic functionality
try:
    extractor = UsernameExtractor()
    print("✅ UsernameExtractor instantiated successfully")
except Exception as e:
    print(f"❌ UsernameExtractor instantiation error: {e}")

print("\nImport test complete!")