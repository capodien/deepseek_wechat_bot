#!/usr/bin/env python3
"""
OpenCV Timestamp Left Boundary Detection
Uses OpenCV and numpy for timestamp boundary detection.
"""

import cv2
import numpy as np
from datetime import datetime

def detect_timestamp_boundary(image_path):
    """Detect timestamp left boundary using OpenCV methods."""
    
    # Load image
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    print(f"📸 Processing {width}x{height} image")
    
    # Method 1: Gradient-based detection using Sobel
    print("\n🔍 Method 1: Sobel Gradient Detection")
    
    # Focus on timestamp area (right portion, top area)
    timestamp_region = gray[:min(80, height), int(width*0.6):]
    
    # Apply Sobel X (vertical edges)
    sobel_x = cv2.Sobel(timestamp_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_abs = np.absolute(sobel_x)
    
    # Find significant edges
    threshold = np.mean(sobel_abs) + 2 * np.std(sobel_abs)
    print(f"   Gradient threshold: {threshold:.2f}")
    
    left_boundaries_method1 = []
    for row in range(sobel_abs.shape[0]):
        edge_positions = np.where(sobel_abs[row] > threshold)[0]
        if len(edge_positions) > 0:
            # Convert back to full image coordinates
            boundary = edge_positions[0] + int(width*0.6)
            left_boundaries_method1.append(boundary)
    
    method1_result = int(np.median(left_boundaries_method1)) if left_boundaries_method1 else None
    print(f"   Found {len(left_boundaries_method1)} boundary candidates")
    print(f"   Result: x = {method1_result}")
    
    # Method 2: Color-based detection for grey timestamp
    print("\n🎯 Method 2: HSV Color Detection")
    
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Define grey color range for timestamp (adjust based on actual image)
    lower_grey = np.array([0, 0, 100])    # Low saturation, medium-high value
    upper_grey = np.array([180, 50, 200])  # Any hue, low saturation, high value
    
    # Create mask for grey colors
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    method2_result = None
    if contours:
        # Find the rightmost contour (likely the timestamp)
        rightmost_contour = max(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, y, w, h = cv2.boundingRect(rightmost_contour)
        method2_result = x
        print(f"   Found {len(contours)} grey regions")
        print(f"   Timestamp region: x={x}, y={y}, w={w}, h={h}")
        print(f"   Result: x = {method2_result}")
    else:
        print("   No grey regions found")
    
    # Method 3: Canny edge detection
    print("\n⚡ Method 3: Canny Edge Detection")
    
    # Apply Gaussian blur and Canny edge detection
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    # Focus on right portion where timestamp is located
    right_region = edges[:min(80, height), int(width*0.6):]
    
    method3_result = None
    # Find leftmost edge in timestamp area
    for col in range(right_region.shape[1]):
        if np.any(right_region[:, col]):
            method3_result = col + int(width*0.6)
            break
    
    print(f"   Result: x = {method3_result}")
    
    # Create visualization using OpenCV
    vis_img = img_rgb.copy()
    
    # Draw detection regions
    cv2.rectangle(vis_img, (int(width*0.6), 0), (width, min(80, height)), (255, 255, 0), 2)
    
    results = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]  # Red, Green, Blue
    method_names = ['Gradient', 'Color', 'Edge']
    symbols = ['G', 'C', 'E']
    
    # Draw boundary lines
    for i, (method_result, color, name, symbol) in enumerate(zip(
        [method1_result, method2_result, method3_result], 
        colors, method_names, symbols)):
        
        if method_result is not None:
            # Draw vertical line
            cv2.line(vis_img, (method_result, 0), (method_result, height), color, 3)
            
            # Add label
            cv2.putText(vis_img, f'{symbol}: {method_result}', 
                       (method_result - 30, 30 + i*25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            results.append((name, method_result))
    
    # Save visualization
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/erli/coding/deepseek_wechat_bot/WorkingOn/{timestamp}_opencv_timestamp_analysis.png"
    
    # Convert back to BGR for saving
    vis_bgr = cv2.cvtColor(vis_img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(output_path, vis_bgr)
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS - OpenCV Timestamp Detection")
    print("="*60)
    
    for name, boundary in results:
        print(f"📍 {name:12} Method: x = {boundary:3d} pixels")
    
    if results:
        boundaries = [r[1] for r in results]
        avg_boundary = int(np.mean(boundaries))
        std_dev = np.std(boundaries)
        
        print(f"\n📊 Statistical Analysis:")
        print(f"   Average Boundary: x = {avg_boundary} pixels")
        print(f"   Standard Deviation: {std_dev:.1f}")
        print(f"   Range: {min(boundaries)} - {max(boundaries)}")
        
        # Confidence assessment
        if std_dev < 5:
            confidence = "🎯 High - Methods agree closely"
        elif std_dev < 15:
            confidence = "⚠️ Medium - Some variation between methods"
        else:
            confidence = "❌ Low - Significant disagreement between methods"
            
        print(f"   Confidence Level: {confidence}")
        
        # Recommendation
        print(f"\n💡 Recommendation:")
        if std_dev < 10:
            print(f"   Use x = {avg_boundary} as the timestamp left boundary")
            print(f"   All methods show good agreement")
        else:
            # Find most consistent method
            best_method = min(results, key=lambda x: abs(x[1] - avg_boundary))
            print(f"   Consider using {best_method[0]} method result: x = {best_method[1]}")
            print(f"   Methods show some disagreement - validate manually")
    
    else:
        print("❌ No timestamp boundary detected by any method")
        print("   Consider adjusting detection parameters or image preprocessing")
    
    print(f"\n🎨 Visualization saved to: {output_path}")
    
    # Also save individual processing images for debugging
    debug_path_base = f"/Users/erli/coding/deepseek_wechat_bot/WorkingOn/{timestamp}"
    
    # Save Sobel result
    sobel_normalized = cv2.normalize(sobel_abs, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
    cv2.imwrite(f"{debug_path_base}_sobel_gradient.png", sobel_normalized)
    
    # Save color mask
    cv2.imwrite(f"{debug_path_base}_color_mask.png", mask)
    
    # Save Canny edges
    cv2.imwrite(f"{debug_path_base}_canny_edges.png", edges)
    
    print(f"🔍 Debug images saved with prefix: {debug_path_base}_")
    
    return results, output_path

if __name__ == "__main__":
    image_path = "/Users/erli/coding/deepseek_wechat_bot/WorkingOn/Attempt_Card_NameBoundryDetection_Photo.png"
    results, output_path = detect_timestamp_boundary(image_path)