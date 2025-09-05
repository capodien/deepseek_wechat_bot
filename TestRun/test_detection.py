#!/usr/bin/env python3
"""
Test script to analyze red dot detection in WeChat screenshots
"""
import cv2
import numpy as np
import os
from glob import glob

def analyze_red_dot(image_path):
    """Analyze and find red dots in WeChat screenshot"""
    print(f"\n🔍 Analyzing: {os.path.basename(image_path)}")
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print("❌ Failed to load image")
        return
    
    height, width = image.shape[:2]
    print(f"📐 Image size: {width}x{height}")
    
    # Current detection parameters (macOS)
    TARGET_COLOR_MAC = np.array([88, 94, 231])  # BGR
    X_RANGE_MAC = (60, 320)
    
    # Windows detection parameters  
    TARGET_COLOR_WIN = np.array([81, 81, 255])  # BGR
    COLOR_TOLERANCE = 10
    X_RANGE_WIN = (66, 380)
    
    # Test both detection methods
    print("\n🍎 Testing macOS detection parameters:")
    test_detection(image, TARGET_COLOR_MAC, X_RANGE_MAC, tolerance=0)
    
    print("\n🪟 Testing Windows detection parameters:")  
    test_detection(image, TARGET_COLOR_WIN, X_RANGE_WIN, tolerance=COLOR_TOLERANCE)
    
    # Scan for red-ish colors in the expected region
    print("\n🔍 Scanning for red-ish colors in chat list area:")
    scan_red_colors(image, X_RANGE_MAC)

def test_detection(image, target_color, x_range, tolerance=0):
    """Test detection with given parameters"""
    height, width = image.shape[:2]
    
    # Create coordinate grids
    x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
    
    if tolerance > 0:
        # Windows method with tolerance
        lower_bound = target_color - tolerance
        upper_bound = target_color + tolerance
        color_mask = np.all((lower_bound <= image) & (image <= upper_bound), axis=-1)
    else:
        # macOS method - exact match
        color_mask = np.all(image == target_color, axis=-1)
    
    # Apply region filter
    region_mask = (x_coords >= x_range[0]) & (x_coords <= x_range[1])
    
    # Find matches
    matched_points = np.column_stack((
        x_coords[color_mask & region_mask],
        y_coords[color_mask & region_mask]
    ))
    
    print(f"   Target color: {target_color} (BGR)")
    print(f"   Search region: x={x_range[0]}-{x_range[1]}")
    print(f"   Tolerance: {tolerance}")
    print(f"   Found {len(matched_points)} matching pixels")
    
    if len(matched_points) > 0:
        # Sort by y-coordinate (bottom to top)
        sorted_points = matched_points[np.argsort(-matched_points[:, 1])]
        best_point = sorted_points[0]
        print(f"   ✅ Best match at: ({best_point[0]}, {best_point[1]})")
        return tuple(best_point.astype(int))
    else:
        print(f"   ❌ No matches found")
        return None

def scan_red_colors(image, x_range):
    """Scan for red-ish colors in the chat list region"""
    height, width = image.shape[:2]
    
    # Define red color ranges (in BGR)
    red_ranges = [
        ([0, 0, 200], [100, 100, 255], "Bright Red"),
        ([0, 0, 150], [150, 150, 255], "Medium Red"), 
        ([50, 50, 200], [150, 150, 255], "WeChat Red Range")
    ]
    
    for lower, upper, name in red_ranges:
        lower = np.array(lower)
        upper = np.array(upper)
        
        # Create mask for this red range
        mask = cv2.inRange(image, lower, upper)
        
        # Apply region filter
        y_indices, x_indices = np.where(mask > 0)
        region_matches = [(x, y) for x, y in zip(x_indices, y_indices) 
                         if x_range[0] <= x <= x_range[1]]
        
        if region_matches:
            print(f"   🔴 {name}: Found {len(region_matches)} pixels")
            # Show a few sample coordinates
            for i, (x, y) in enumerate(region_matches[:3]):
                color = image[y, x]  # BGR
                print(f"      Sample {i+1}: ({x}, {y}) color={color}")
        else:
            print(f"   ⚪ {name}: No pixels found")

def main():
    # Find the most recent screenshot
    screenshots_dir = "pic/screenshots"
    if not os.path.exists(screenshots_dir):
        print(f"❌ Screenshots directory not found: {screenshots_dir}")
        return
    
    pattern = os.path.join(screenshots_dir, "wechat_*.png")
    files = glob(pattern)
    
    if not files:
        print(f"❌ No screenshots found in {screenshots_dir}")
        return
    
    # Sort by modification time, get most recent
    latest_file = max(files, key=os.path.getmtime)
    analyze_red_dot(latest_file)

if __name__ == "__main__":
    main()