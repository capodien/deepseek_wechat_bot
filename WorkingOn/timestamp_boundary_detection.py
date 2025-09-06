#!/usr/bin/env python3
"""
Timestamp Left Boundary Detection
Demonstrates multiple methods to detect the left boundary of grey timestamp text.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def load_and_prepare_image(image_path):
    """Load image and convert to different color spaces."""
    img = cv2.imread(image_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img, img_rgb, gray

def method1_gradient_detection(gray_img, roi_top=0, roi_bottom=None):
    """Method 1: Gradient-based detection using Sobel filter."""
    if roi_bottom is None:
        roi_bottom = gray_img.shape[0]
    
    # Focus on timestamp area (right portion of image)
    height, width = gray_img.shape
    timestamp_region = gray_img[roi_top:roi_bottom, width//2:]
    
    # Apply horizontal Sobel filter to detect vertical edges
    sobel_x = cv2.Sobel(timestamp_region, cv2.CV_64F, 1, 0, ksize=3)
    sobel_abs = np.absolute(sobel_x)
    
    # Find the leftmost significant edge in each row
    threshold = np.mean(sobel_abs) + 2 * np.std(sobel_abs)
    
    left_boundaries = []
    for row in range(sobel_abs.shape[0]):
        edge_positions = np.where(sobel_abs[row] > threshold)[0]
        if len(edge_positions) > 0:
            # Add back the offset since we cropped to right half
            left_boundaries.append(edge_positions[0] + width//2)
    
    if left_boundaries:
        return int(np.median(left_boundaries))
    return None

def method2_color_based_detection(img_rgb):
    """Method 2: Color-based detection for grey timestamp."""
    # Define grey color range for timestamp
    # Convert to HSV for better color detection
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    
    # Grey timestamp color range (adjust based on actual colors)
    lower_grey = np.array([0, 0, 100])    # Low saturation, medium-high value
    upper_grey = np.array([180, 50, 200])  # Any hue, low saturation, high value
    
    # Create mask for grey colors
    mask = cv2.inRange(hsv, lower_grey, upper_grey)
    
    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Find the rightmost contour (likely the timestamp)
        rightmost_contour = max(contours, key=lambda c: cv2.boundingRect(c)[0])
        x, y, w, h = cv2.boundingRect(rightmost_contour)
        return x, mask
    
    return None, mask

def method3_edge_detection(gray_img):
    """Method 3: Canny edge detection with morphological operations."""
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray_img, (3, 3), 0)
    
    # Canny edge detection
    edges = cv2.Canny(blurred, 50, 150)
    
    # Focus on right portion where timestamp is located
    height, width = edges.shape
    right_region = edges[:, width//2:]
    
    # Find leftmost edge in timestamp area
    for col in range(right_region.shape[1]):
        if np.any(right_region[:, col]):
            return col + width//2, edges
    
    return None, edges

def detect_timestamp_boundary(image_path):
    """Main function to detect timestamp left boundary using multiple methods."""
    img, img_rgb, gray = load_and_prepare_image(image_path)
    
    # Method 1: Gradient-based detection
    gradient_boundary = method1_gradient_detection(gray, roi_top=0, roi_bottom=60)
    
    # Method 2: Color-based detection  
    color_boundary, color_mask = method2_color_based_detection(img_rgb)
    
    # Method 3: Edge detection
    edge_boundary, edges = method3_edge_detection(gray)
    
    # Create visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Original image
    axes[0, 0].imshow(img_rgb)
    axes[0, 0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Method 1: Gradient detection
    axes[0, 1].imshow(gray, cmap='gray')
    if gradient_boundary:
        axes[0, 1].axvline(x=gradient_boundary, color='red', linewidth=3, label=f'Gradient: x={gradient_boundary}')
        axes[0, 1].legend()
    axes[0, 1].set_title('Method 1: Gradient Detection', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Method 2: Color-based detection
    axes[0, 2].imshow(img_rgb)
    if color_boundary:
        x = color_boundary[0] if isinstance(color_boundary, tuple) else color_boundary
        axes[0, 2].axvline(x=x, color='green', linewidth=3, label=f'Color: x={x}')
        axes[0, 2].legend()
    axes[0, 2].set_title('Method 2: Color Detection', fontsize=14, fontweight='bold')
    axes[0, 2].axis('off')
    
    # Color mask
    axes[1, 0].imshow(color_mask, cmap='gray')
    axes[1, 0].set_title('Color Mask (Grey Detection)', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Method 3: Edge detection
    axes[1, 1].imshow(edges, cmap='gray')
    if edge_boundary:
        axes[1, 1].axvline(x=edge_boundary, color='blue', linewidth=3, label=f'Edge: x={edge_boundary}')
        axes[1, 1].legend()
    axes[1, 1].set_title('Method 3: Edge Detection', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    # Combined results
    axes[1, 2].imshow(img_rgb)
    
    results = []
    if gradient_boundary:
        axes[1, 2].axvline(x=gradient_boundary, color='red', linewidth=2, alpha=0.8, label=f'Gradient: {gradient_boundary}')
        results.append(('Gradient', gradient_boundary))
    
    if color_boundary:
        x = color_boundary[0] if isinstance(color_boundary, tuple) else color_boundary
        axes[1, 2].axvline(x=x, color='green', linewidth=2, alpha=0.8, label=f'Color: {x}')
        results.append(('Color', x))
    
    if edge_boundary:
        axes[1, 2].axvline(x=edge_boundary, color='blue', linewidth=2, alpha=0.8, label=f'Edge: {edge_boundary}')
        results.append(('Edge', edge_boundary))
    
    axes[1, 2].legend()
    axes[1, 2].set_title('Combined Results', fontsize=14, fontweight='bold')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/erli/coding/deepseek_wechat_bot/WorkingOn/{timestamp}_timestamp_boundary_detection.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print results
    print("🔍 Timestamp Left Boundary Detection Results:")
    print("=" * 50)
    
    for method, boundary in results:
        print(f"📍 {method:12} Method: x = {boundary}")
    
    if results:
        boundaries = [r[1] for r in results]
        avg_boundary = int(np.mean(boundaries))
        print(f"📊 Average Boundary: x = {avg_boundary}")
        
        # Confidence assessment
        std_dev = np.std(boundaries)
        if std_dev < 5:
            confidence = "High"
        elif std_dev < 15:
            confidence = "Medium"  
        else:
            confidence = "Low"
            
        print(f"🎯 Confidence Level: {confidence} (std: {std_dev:.1f})")
    
    print(f"🎨 Visualization saved: {output_path}")
    
    return results, output_path

if __name__ == "__main__":
    image_path = "/Users/erli/coding/deepseek_wechat_bot/WorkingOn/Attempt_Card_NameBoundryDetection_Photo.png"
    results, output_path = detect_timestamp_boundary(image_path)