#!/usr/bin/env python3
"""
Basic Timestamp Left Boundary Detection
Uses only matplotlib and numpy for timestamp boundary detection.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from datetime import datetime

def detect_timestamp_boundary(image_path):
    """Detect timestamp left boundary using basic image processing."""
    
    # Load image
    img = mpimg.imread(image_path)
    print(f"📸 Image shape: {img.shape}")
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        # RGB to grayscale conversion
        gray = np.dot(img[...,:3], [0.299, 0.587, 0.114])
    else:
        gray = img
    
    height, width = gray.shape
    print(f"🖼️  Processing {width}x{height} image")
    
    # Method 1: Gradient-based detection
    print("\n🔍 Method 1: Gradient Detection")
    
    # Focus on timestamp area (right portion, top area)
    timestamp_region = gray[:min(60, height), int(width*0.6):]
    
    # Simple horizontal gradient using numpy diff
    grad_x = np.diff(timestamp_region, axis=1)
    
    # Find significant gradients
    threshold = np.std(np.abs(grad_x)) * 1.5
    print(f"   Gradient threshold: {threshold:.2f}")
    
    left_boundaries_method1 = []
    for row in range(grad_x.shape[0]):
        significant_edges = np.where(np.abs(grad_x[row]) > threshold)[0]
        if len(significant_edges) > 0:
            # Convert back to full image coordinates
            boundary = significant_edges[0] + int(width*0.6)
            left_boundaries_method1.append(boundary)
    
    method1_result = int(np.median(left_boundaries_method1)) if left_boundaries_method1 else None
    print(f"   Found {len(left_boundaries_method1)} boundary candidates")
    print(f"   Result: x = {method1_result}")
    
    # Method 2: Intensity-based detection  
    print("\n🎯 Method 2: Intensity Analysis")
    
    # Look for grey text pixels in timestamp area
    timestamp_region_full = gray[:min(60, height), int(width*0.5):]
    
    # Grey text typically has intermediate intensity values
    text_candidates = (timestamp_region_full > 0.4) & (timestamp_region_full < 0.8)
    
    left_boundaries_method2 = []
    for row in range(text_candidates.shape[0]):
        text_pixels = np.where(text_candidates[row])[0]
        if len(text_pixels) > 5:  # Need sufficient pixels to be text
            boundary = text_pixels[0] + int(width*0.5)
            left_boundaries_method2.append(boundary)
    
    method2_result = int(np.median(left_boundaries_method2)) if left_boundaries_method2 else None
    print(f"   Found {len(left_boundaries_method2)} text candidates")  
    print(f"   Result: x = {method2_result}")
    
    # Method 3: Edge transition detection
    print("\n⚡ Method 3: Edge Transition")
    
    timestamp_region_edge = gray[:min(60, height), int(width*0.65):]
    
    left_boundaries_method3 = []
    for row in range(timestamp_region_edge.shape[0]):
        row_pixels = timestamp_region_edge[row]
        
        # Look for significant brightness increases (dark to light transition)
        for col in range(1, len(row_pixels)-2):
            brightness_change = row_pixels[col] - row_pixels[col-1]
            if brightness_change > 0.1:  # Significant increase
                boundary = col + int(width*0.65)
                left_boundaries_method3.append(boundary)
                break
    
    method3_result = int(np.median(left_boundaries_method3)) if left_boundaries_method3 else None
    print(f"   Found {len(left_boundaries_method3)} transitions")
    print(f"   Result: x = {method3_result}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Original image
    axes[0, 0].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    axes[0, 0].set_title('🖼️ Original Image', fontsize=14, fontweight='bold')
    axes[0, 0].axis('off')
    
    # Grayscale with gradient detection
    axes[0, 1].imshow(gray, cmap='gray')
    if method1_result:
        axes[0, 1].axvline(x=method1_result, color='red', linewidth=3, 
                          label=f'Gradient: x={method1_result}')
        # Add detection region rectangle
        axes[0, 1].add_patch(plt.Rectangle((int(width*0.6), 0), 
                                         width - int(width*0.6), min(60, height),
                                         fill=False, edgecolor='red', alpha=0.3, linewidth=2))
    axes[0, 1].legend(loc='upper left')
    axes[0, 1].set_title('🔍 Method 1: Gradient Detection', fontsize=14, fontweight='bold')
    axes[0, 1].axis('off')
    
    # Intensity analysis
    axes[1, 0].imshow(gray, cmap='gray')
    if method2_result:
        axes[1, 0].axvline(x=method2_result, color='green', linewidth=3,
                          label=f'Intensity: x={method2_result}')
        # Add detection region rectangle
        axes[1, 0].add_patch(plt.Rectangle((int(width*0.5), 0), 
                                         width - int(width*0.5), min(60, height),
                                         fill=False, edgecolor='green', alpha=0.3, linewidth=2))
    axes[1, 0].legend(loc='upper left')
    axes[1, 0].set_title('🎯 Method 2: Intensity Analysis', fontsize=14, fontweight='bold')
    axes[1, 0].axis('off')
    
    # Combined results
    axes[1, 1].imshow(img, cmap='gray' if len(img.shape) == 2 else None)
    
    results = []
    colors = ['red', 'green', 'blue']
    method_names = ['Gradient', 'Intensity', 'Transition']
    symbols = ['🔍', '🎯', '⚡']
    
    for i, (method_result, color, name, symbol) in enumerate(zip(
        [method1_result, method2_result, method3_result], 
        colors, method_names, symbols)):
        
        if method_result is not None:
            axes[1, 1].axvline(x=method_result, color=color, linewidth=2, 
                              alpha=0.8, label=f'{symbol} {name}: {method_result}')
            results.append((name, method_result))
    
    if results:
        axes[1, 1].legend(loc='upper left')
        
        # Add timestamp region highlight
        axes[1, 1].add_patch(plt.Rectangle((int(width*0.5), 0), 
                                         width - int(width*0.5), min(60, height),
                                         fill=False, edgecolor='yellow', alpha=0.5, linewidth=2,
                                         linestyle='--', label='Search Region'))
    
    axes[1, 1].set_title('🔥 Combined Results', fontsize=14, fontweight='bold')
    axes[1, 1].axis('off')
    
    plt.suptitle('Timestamp Left Boundary Detection Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"/Users/erli/coding/deepseek_wechat_bot/WorkingOn/{timestamp}_timestamp_analysis.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    # Print comprehensive results
    print("\n" + "="*60)
    print("🏆 FINAL RESULTS - Timestamp Left Boundary Detection")
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
    
    print(f"\n🎨 Detailed visualization saved to: {output_path}")
    
    return results, output_path

if __name__ == "__main__":
    image_path = "/Users/erli/coding/deepseek_wechat_bot/WorkingOn/Attempt_Card_NameBoundryDetection_Photo.png"
    results, output_path = detect_timestamp_boundary(image_path)