#!/usr/bin/env python3
"""
Simple Timestamp Left Boundary Detection
Uses PIL and numpy for timestamp boundary detection without OpenCV dependency.
"""

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
from datetime import datetime

def load_image(image_path):
    """Load image and convert to numpy array."""
    img = Image.open(image_path)
    img_array = np.array(img)
    return img, img_array

def rgb_to_gray(img_array):
    """Convert RGB to grayscale using weighted average."""
    if len(img_array.shape) == 3:
        return np.dot(img_array[...,:3], [0.299, 0.587, 0.114])
    return img_array

def method1_gradient_detection(gray_img):
    """Detect left boundary using horizontal gradient."""
    height, width = gray_img.shape
    
    # Focus on timestamp area (right 40% of image, top 80 pixels)
    timestamp_region = gray_img[:min(80, height), int(width*0.6):]
    
    # Calculate horizontal gradients (simple difference)
    grad_x = np.diff(timestamp_region.astype(float), axis=1)
    
    # Find significant edges (absolute gradient > threshold)
    threshold = np.std(grad_x) * 2
    
    left_boundaries = []
    for row in range(grad_x.shape[0]):
        edges = np.where(np.abs(grad_x[row]) > threshold)[0]
        if len(edges) > 0:
            # Convert back to full image coordinates
            left_boundaries.append(edges[0] + int(width*0.6))
    
    if left_boundaries:
        return int(np.median(left_boundaries))
    return None

def method2_color_threshold(img_array):
    """Detect timestamp using color thresholding for grey text."""
    height, width = img_array.shape[:2]
    
    # Convert to grayscale
    if len(img_array.shape) == 3:
        gray = rgb_to_gray(img_array)
    else:
        gray = img_array
    
    # Focus on timestamp area
    timestamp_region = gray[:min(80, height), int(width*0.5):]
    
    # Threshold for grey text (adjust based on image)
    # Grey text typically has values between 100-180
    text_mask = (timestamp_region > 100) & (timestamp_region < 200)
    
    # Find leftmost text pixel in each row
    left_boundaries = []
    for row in range(text_mask.shape[0]):
        text_pixels = np.where(text_mask[row])[0]
        if len(text_pixels) > 0:
            left_boundaries.append(text_pixels[0] + int(width*0.5))
    
    if left_boundaries:
        return int(np.median(left_boundaries))
    return None

def method3_intensity_transition(gray_img):
    """Detect boundary based on intensity transitions."""
    height, width = gray_img.shape
    
    # Focus on timestamp area
    timestamp_region = gray_img[:min(80, height), int(width*0.6):]
    
    # For each row, find the first significant intensity change
    left_boundaries = []
    
    for row in range(timestamp_region.shape[0]):
        row_pixels = timestamp_region[row]
        
        # Look for transitions from dark to light (background to text)
        for col in range(1, len(row_pixels)-1):
            # Check if there's a significant increase in brightness
            if (row_pixels[col] - row_pixels[col-1]) > 30:
                left_boundaries.append(col + int(width*0.6))
                break
    
    if left_boundaries:
        return int(np.median(left_boundaries))
    return None

def detect_timestamp_boundary(image_path):
    """Main detection function with visualization."""
    img, img_array = load_image(image_path)
    gray = rgb_to_gray(img_array)
    
    # Apply detection methods
    method1_result = method1_gradient_detection(gray)
    method2_result = method2_color_threshold(img_array)  
    method3_result = method3_intensity_transition(gray)
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Original image
    axes[0, 0].imshow(img)
    axes[0, 0].set_title('Original Image', fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grayscale with method 1
    axes[0, 1].imshow(gray, cmap='gray')
    if method1_result:
        axes[0, 1].axvline(x=method1_result, color='red', linewidth=2, 
                          label=f'Gradient: x={method1_result}')
        axes[0, 1].legend()
    axes[0, 1].set_title('Method 1: Gradient Detection', fontweight='bold')
    axes[0, 1].axis('off')
    
    # Method 2: Color threshold
    axes[1, 0].imshow(img)
    if method2_result:
        axes[1, 0].axvline(x=method2_result, color='green', linewidth=2,
                          label=f'Color: x={method2_result}')
        axes[1, 0].legend()
    axes[1, 0].set_title('Method 2: Color Threshold', fontweight='bold')
    axes[1, 0].axis('off')
    
    # Combined results
    axes[1, 1].imshow(img)
    
    results = []
    colors = ['red', 'green', 'blue']
    method_names = ['Gradient', 'Color', 'Intensity']
    
    for i, (method_result, color, name) in enumerate(zip(
        [method1_result, method2_result, method3_result], 
        colors, method_names)):
        
        if method_result:
            axes[1, 1].axvline(x=method_result, color=color, linewidth=2, 
                              alpha=0.7, label=f'{name}: {method_result}')
            results.append((name, method_result))
    
    if results:
        axes[1, 1].legend()
    axes[1, 1].set_title('Combined Results', fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/erli/coding/deepseek_wechat_bot/WorkingOn/{timestamp}_simple_timestamp_detection.png"
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
        std_dev = np.std(boundaries)
        
        print(f"📊 Average Boundary: x = {avg_boundary}")
        print(f"📈 Standard Deviation: {std_dev:.1f}")
        
        # Confidence assessment
        if std_dev < 5:
            confidence = "🎯 High"
        elif std_dev < 15:
            confidence = "⚠️ Medium"
        else:
            confidence = "❌ Low"
            
        print(f"🔥 Confidence Level: {confidence}")
    
    print(f"🎨 Visualization saved: {output_path}")
    
    return results, output_path

if __name__ == "__main__":
    image_path = "/Users/erli/coding/deepseek_wechat_bot/WorkingOn/Attempt_Card_NameBoundryDetection_Photo.png"
    results, output_path = detect_timestamp_boundary(image_path)