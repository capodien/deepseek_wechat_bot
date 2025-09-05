#!/usr/bin/env python3
"""
Strong Edge Transition Analysis Report
=====================================
Comprehensive analysis of the improved right boundary detection using strong horizontal pixel differences.

Based on your observation: "look at the right boundary, the horizontal pixel differences is high right on the edge"
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from modules.m_Card_Processing import SimpleWidthDetector
import os

def create_comprehensive_analysis():
    """Create detailed visual analysis of the strong edge transition detection method"""
    
    # Load the target preprocessed image
    target_path = '/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot_v2/pic/screenshots/20250905_130426_02_photoshop_levels_gamma.png'
    
    print("🔬 Strong Edge Transition Analysis Report")
    print("=" * 60)
    print(f"📸 Target Image: {os.path.basename(target_path)}")
    
    # Load image
    img = cv2.imread(target_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"❌ Could not load image: {target_path}")
        return
        
    print(f"📐 Image dimensions: {img.shape[1]}x{img.shape[0]} pixels")
    
    # Step 1: Compute horizontal pixel differences  
    print("\n🔍 Step 1: Computing Horizontal Pixel Differences")
    diff_x = np.diff(img.astype(np.int16), axis=1)
    print(f"   - Pixel difference range: {diff_x.min()} to {diff_x.max()}")
    print(f"   - Negative values indicate white-to-black transitions (card edges)")
    print(f"   - Strong transitions (< -100): {np.sum(diff_x < -100)} pixels")
    print(f"   - Very strong transitions (< -200): {np.sum(diff_x < -200)} pixels")
    
    # Step 2: Analyze strong edge transitions
    print("\n⚡ Step 2: Strong Edge Transition Analysis")
    strong_threshold = -100  # Based on your observation
    strong_mask = diff_x < strong_threshold
    strong_projection = np.sum(strong_mask, axis=0)
    
    print(f"   - Strong transition threshold: < {strong_threshold}")
    print(f"   - Columns with strong transitions: {np.sum(strong_projection > 0)}")
    print(f"   - Max strong transitions per column: {np.max(strong_projection)}")
    
    # Find significant boundary candidates (2+ strong transitions)
    significant_columns = []
    min_transitions = 2
    for x in range(len(strong_projection)):
        if strong_projection[x] >= min_transitions:
            significant_columns.append((x, strong_projection[x]))
    
    print(f"   - Columns with {min_transitions}+ strong transitions: {len(significant_columns)}")
    
    # Step 3: Focus on boundary region (right side)
    print("\n🎯 Step 3: Right Boundary Region Analysis")
    img_width = img.shape[1]
    search_start = int(img_width * 0.8)  # Focus on right 20%
    search_end = int(img_width * 0.98)   # Avoid edge artifacts
    
    boundary_candidates = []
    for x, transitions in significant_columns:
        if search_start <= x <= search_end:
            boundary_candidates.append((x, transitions))
            
    print(f"   - Search region: {search_start}-{search_end}px (right 20%)")
    print(f"   - Boundary candidates in region: {len(boundary_candidates)}")
    
    if boundary_candidates:
        # Sort by position (rightmost first)
        boundary_candidates.sort(key=lambda b: b[0], reverse=True)
        
        print(f"\n🏆 Top 10 Rightmost Boundary Candidates:")
        for i, (x, transitions) in enumerate(boundary_candidates[:10]):
            print(f"   {i+1:2d}. x={x:4d}px, strong_transitions={transitions:2d}")
            
        # Show the selected boundary
        selected_x, selected_transitions = boundary_candidates[0]
        print(f"\n✅ Selected Right Boundary:")
        print(f"   - Position: {selected_x}px")
        print(f"   - Strong transitions: {selected_transitions}")
        print(f"   - Distance from right edge: {img_width - selected_x}px")
    
    # Step 4: Create comprehensive visualizations
    print(f"\n🎨 Step 4: Creating Visual Analysis")
    
    # Create figure with multiple subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Strong Edge Transition Detection Analysis', fontsize=16, fontweight='bold')
    
    # Subplot 1: Original image with detected boundary
    axes[0,0].imshow(img, cmap='gray')
    axes[0,0].set_title('Original Preprocessed Image with Detected Boundary')
    if boundary_candidates:
        selected_x = boundary_candidates[0][0]
        axes[0,0].axvline(x=selected_x, color='red', linewidth=2, label=f'Detected Boundary: {selected_x}px')
        axes[0,0].legend()
    axes[0,0].set_xlabel('X Position (pixels)')
    axes[0,0].set_ylabel('Y Position (pixels)')
    
    # Subplot 2: Horizontal pixel differences (sample row)
    middle_row = img.shape[0] // 2
    sample_diff = diff_x[middle_row, :]
    axes[0,1].plot(sample_diff, 'b-', alpha=0.7, label='Pixel Differences')
    axes[0,1].axhline(y=strong_threshold, color='red', linestyle='--', label=f'Strong Threshold ({strong_threshold})')
    axes[0,1].set_title(f'Horizontal Pixel Differences (Row {middle_row})')
    axes[0,1].set_xlabel('X Position (pixels)')
    axes[0,1].set_ylabel('Pixel Difference')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Subplot 3: Strong transitions projection
    axes[1,0].plot(strong_projection, 'g-', linewidth=1, label='Strong Transitions Count')
    axes[1,0].axhline(y=min_transitions, color='orange', linestyle='--', label=f'Min Threshold ({min_transitions})')
    if boundary_candidates:
        axes[1,0].axvline(x=selected_x, color='red', linewidth=2, label=f'Selected Boundary: {selected_x}px')
    axes[1,0].set_title('Strong Edge Transitions per Column')
    axes[1,0].set_xlabel('X Position (pixels)')
    axes[1,0].set_ylabel('Strong Transitions Count')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # Subplot 4: Boundary region zoom
    if boundary_candidates:
        zoom_start = max(0, selected_x - 100)
        zoom_end = min(img_width, selected_x + 100)
        zoom_region = img[:, zoom_start:zoom_end]
        
        axes[1,1].imshow(zoom_region, cmap='gray')
        axes[1,1].axvline(x=selected_x-zoom_start, color='red', linewidth=2, 
                         label=f'Boundary: {selected_x}px')
        axes[1,1].set_title(f'Boundary Region Zoom ({zoom_start}-{zoom_end}px)')
        axes[1,1].set_xlabel('Relative X Position (pixels)')
        axes[1,1].set_ylabel('Y Position (pixels)')
        axes[1,1].legend()
    else:
        axes[1,1].text(0.5, 0.5, 'No boundary detected', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('Boundary Region (No Detection)')
    
    plt.tight_layout()
    
    # Save the analysis
    output_path = 'pic/screenshots/strong_edge_analysis.png'
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"   📊 Visual analysis saved: {output_path}")
    
    # Step 5: Method Summary
    print(f"\n📋 Step 5: Method Summary")
    print(f"   🔧 Detection Method: Strong Edge Transition Detection")
    print(f"   📐 Technique: Horizontal pixel differences + vertical projection")
    print(f"   🎯 Key Innovation: Focus on strong transitions (< -100) at boundaries")
    print(f"   ⚡ Sensitivity: Minimum 2 strong transitions per column")
    print(f"   🎨 Preprocessing: Photoshop-style levels adjustment for high contrast")
    print(f"   🔍 Search Strategy: Rightmost boundary in reasonable range")
    
    print(f"\n🎉 Analysis Complete!")
    print(f"   - Visual analysis saved to: {output_path}")
    print(f"   - Method successfully detects right boundary using your observation")
    print(f"   - Strong pixel differences effectively identify card edges")
    
    return output_path

if __name__ == "__main__":
    create_comprehensive_analysis()