#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual demonstration of Phase 1: Upper Card Region Extraction Areas

Shows exactly what areas are extracted for grey timestamp detection.
"""

import cv2
import numpy as np
from datetime import datetime

def create_visual_demonstration():
    """Create a visual demonstration of the extraction areas"""
    print("🎯 Phase 1: Upper Card Region Extraction Areas")
    print("=" * 55)
    
    # Create a mock WeChat card layout for demonstration
    img_width, img_height = 800, 600
    demo_img = np.ones((img_height, img_width, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Define mock card data (simulating detected cards)
    mock_cards = [
        {"bbox": [50, 100, 300, 80], "name": "Card 1"},   # [x, y, w, h]
        {"bbox": [50, 200, 300, 70], "name": "Card 2"}, 
        {"bbox": [50, 290, 300, 85], "name": "Card 3"},
        {"bbox": [50, 395, 300, 75], "name": "Card 4"}
    ]
    
    print("📊 Upper Region Extraction Logic:")
    print("   • Extract upper 40% of each card OR maximum 35 pixels")
    print("   • Add 5px margins on all sides")
    print("   • Focus on area where grey timestamps appear")
    print()
    
    for i, card in enumerate(mock_cards):
        rx, ry, rw, rh = card["bbox"]
        card_name = card["name"]
        
        # Apply Phase 1 extraction logic
        upper_region_height = min(rh * 0.4, 35)  # Upper 40% or max 35px
        
        analysis_left = rx + 5      # Card left + margin
        analysis_right = rx + rw - 5  # Card right - margin  
        analysis_top = ry + 5       # Card top + margin
        analysis_bottom = ry + int(upper_region_height)  # Upper portion only
        
        # Calculate dimensions
        extracted_width = analysis_right - analysis_left
        extracted_height = analysis_bottom - analysis_top
        percentage = (upper_region_height / rh) * 100
        
        print(f"📄 {card_name}:")
        print(f"   • Full card: {rw}×{rh}px")
        print(f"   • Upper region limit: {upper_region_height:.1f}px ({percentage:.1f}% of card)")
        print(f"   • Extracted area: {extracted_width}×{extracted_height}px")
        print(f"   • Coordinates: [{analysis_left}, {analysis_top}, {extracted_width}×{extracted_height}]")
        
        # Draw full card boundary (blue)
        cv2.rectangle(demo_img, (rx, ry), (rx + rw, ry + rh), (255, 100, 100), 2)
        cv2.putText(demo_img, card_name, (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 100, 100), 2)
        
        # Draw upper extraction region (green)
        cv2.rectangle(demo_img, (analysis_left, analysis_top), (analysis_right, analysis_bottom), (0, 200, 0), 3)
        cv2.putText(demo_img, "UPPER REGION", (analysis_left + 5, analysis_top + 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
        cv2.putText(demo_img, f"{extracted_width}×{extracted_height}", 
                   (analysis_left + 5, analysis_bottom - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 150, 0), 1)
        
        # Simulate avatar area (gray circle)
        avatar_x, avatar_y = rx + 15, ry + rh//2
        cv2.circle(demo_img, (avatar_x, avatar_y), 20, (150, 150, 150), -1)
        cv2.putText(demo_img, "Avatar", (avatar_x - 15, avatar_y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 100), 1)
        
        # Simulate timestamp area (light blue rectangle in upper region)
        if extracted_height > 10:
            timestamp_x = analysis_right - 60
            timestamp_y = analysis_top + 5
            cv2.rectangle(demo_img, (timestamp_x, timestamp_y), (timestamp_x + 50, timestamp_y + 12), (200, 200, 100), -1)
            cv2.putText(demo_img, "Timestamp", (timestamp_x, timestamp_y + 9), cv2.FONT_HERSHEY_SIMPLEX, 0.25, (100, 100, 50), 1)
        
        print()
    
    # Add legend
    legend_y = 50
    cv2.rectangle(demo_img, (450, legend_y), (750, legend_y + 120), (255, 255, 255), -1)
    cv2.rectangle(demo_img, (450, legend_y), (750, legend_y + 120), (0, 0, 0), 1)
    cv2.putText(demo_img, "Legend:", (460, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    cv2.rectangle(demo_img, (460, legend_y + 30), (480, legend_y + 45), (255, 100, 100), 2)
    cv2.putText(demo_img, "Full card boundaries", (490, legend_y + 42), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.rectangle(demo_img, (460, legend_y + 50), (480, legend_y + 65), (0, 200, 0), 3)
    cv2.putText(demo_img, "Upper region extraction", (490, legend_y + 62), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.rectangle(demo_img, (460, legend_y + 70), (480, legend_y + 85), (200, 200, 100), -1)
    cv2.putText(demo_img, "Grey timestamp area", (490, legend_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    cv2.circle(demo_img, (470, legend_y + 100), 8, (150, 150, 150), -1)
    cv2.putText(demo_img, "Avatar position", (490, legend_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
    
    # Save the demonstration
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = f"pic/screenshots/{timestamp}_Phase1_Upper_Region_Extraction.png"
    cv2.imwrite(output_path, demo_img)
    
    print("🎨 Visual demonstration saved:", output_path)
    print()
    print("📋 Summary of Phase 1 Extraction:")
    print("   ✅ Takes precise card coordinates from CardBoundaryDetector")
    print("   ✅ Extracts upper 40% of each card (or max 35px)")
    print("   ✅ Applies 5px margins on all sides for safety")
    print("   ✅ Creates ROI containing the timestamp area")
    print("   ✅ Passes this ROI to Phase 2 for grey text isolation")
    print()
    print("🎯 Purpose: Focus processing on the specific area where")
    print("   WeChat places grey timestamps (upper portion of cards)")

if __name__ == "__main__":
    create_visual_demonstration()
    print("\\n🏁 Phase 1 area demonstration complete!")