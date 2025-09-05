#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Demonstrate Phase 1: Upper Card Region Extraction

This script shows exactly what area is extracted by the grey timestamp detection
in Phase 1, highlighting the upper portion of each card where timestamps are located.
"""

import os
import sys
import cv2
import numpy as np
from datetime import datetime

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.m_Card_Processing import CardBoundaryDetector
    print("✅ Successfully imported CardBoundaryDetector")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def demonstrate_upper_region_extraction():
    """Demonstrate the upper card region extraction process"""
    print("🔍 Demonstrating Phase 1: Upper Card Region Extraction")
    print("=" * 60)
    
    # Find a suitable screenshot for demonstration
    screenshot_dir = "pic/screenshots"
    if not os.path.exists(screenshot_dir):
        print(f"❌ Screenshot directory not found: {screenshot_dir}")
        return
    
    # Find any suitable screenshot
    screenshots = []
    for file in os.listdir(screenshot_dir):
        if file.endswith('.png') and not ('Debug_' in file or 'horizontal_differences' in file):
            file_path = os.path.join(screenshot_dir, file)
            screenshots.append((file_path, os.path.getmtime(file_path)))
    
    if not screenshots:
        print(f"❌ No suitable screenshots found in {screenshot_dir}")
        return
    
    # Sort by modification time and get the most recent
    screenshots.sort(key=lambda x: x[1], reverse=True) 
    diagnostic_screenshot = screenshots[0][0]
    
    print(f"📸 Using screenshot: {os.path.basename(diagnostic_screenshot)}")
    
    try:
        # Step 1: Get card boundaries using CardBoundaryDetector
        print("🔧 Detecting card boundaries...")
        card_detector = CardBoundaryDetector(debug_mode=False)
        cards, detection_info = card_detector.detect_cards(diagnostic_screenshot)
        
        if not cards:
            print("❌ No cards detected")
            return
            
        print(f"📄 Found {len(cards)} cards")
        
        # Load the original image
        img = cv2.imread(diagnostic_screenshot)
        if img is None:
            print(f"❌ Failed to load image: {diagnostic_screenshot}")
            return
        
        # Create visualization showing upper regions
        vis_img = img.copy()
        
        print("\\n🎯 Phase 1: Upper Region Extraction Details:")
        print("-" * 50)
        
        for i, card in enumerate(cards[:5]):  # Show first 5 cards
            card_id = i + 1
            card_bbox = card["bbox"]  # [x, y, w, h]
            rx, ry, rw, rh = map(int, card_bbox)
            
            # Apply the same logic as in _detect_grey_timestamp_left_edge
            upper_region_height = min(rh * 0.4, 35)  # Upper 40% or max 35px
            
            analysis_left = rx + 5      # Card left + margin
            analysis_right = rx + rw - 5  # Card right - margin
            analysis_top = ry + 5       # Card top + margin  
            analysis_bottom = ry + int(upper_region_height)  # Upper portion only
            
            # Calculate actual dimensions
            actual_width = analysis_right - analysis_left
            actual_height = analysis_bottom - analysis_top
            percentage_of_card = (upper_region_height / rh) * 100
            
            print(f"📄 Card {card_id}:")
            print(f"   • Full card: [{rx}, {ry}, {rw}×{rh}]")
            print(f"   • Upper region: [{analysis_left}, {analysis_top}, {actual_width}×{actual_height}]")
            print(f"   • Height limit: {upper_region_height:.1f}px ({percentage_of_card:.1f}% of card)")
            print(f"   • Extraction area: {actual_width}×{actual_height}px")
            
            # Draw the full card boundary in blue
            cv2.rectangle(vis_img, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            cv2.putText(vis_img, f"Card {card_id}", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw the upper region extraction area in green
            if analysis_left < analysis_right and analysis_top < analysis_bottom:
                cv2.rectangle(vis_img, (analysis_left, analysis_top), (analysis_right, analysis_bottom), (0, 255, 0), 3)
                cv2.putText(vis_img, "Upper Region", (analysis_left, analysis_top - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Add dimension labels
                cv2.putText(vis_img, f"{actual_width}×{actual_height}", 
                          (analysis_left, analysis_bottom + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            print()
        
        # Save the visualization
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"pic/screenshots/{timestamp}_Upper_Region_Extraction_Demo.png"
        cv2.imwrite(output_path, vis_img)
        
        print(f"🎨 Visualization saved: {os.path.basename(output_path)}")
        print("\\n📊 Summary:")
        print(f"   • ✅ Blue rectangles: Full card boundaries")
        print(f"   • ✅ Green rectangles: Upper region extraction areas (Phase 1)")
        print(f"   • ✅ Logic: Upper 40% of card height OR maximum 35 pixels")
        print(f"   • ✅ Margins: 5px on all sides within the upper region")
        print(f"   • ✅ Purpose: Extract area where grey timestamps are typically located")
        
        # Show ROI extraction example for first card
        if cards:
            first_card = cards[0]
            rx, ry, rw, rh = map(int, first_card["bbox"])
            upper_region_height = min(rh * 0.4, 35)
            
            analysis_left = rx + 5
            analysis_right = rx + rw - 5
            analysis_top = ry + 5
            analysis_bottom = ry + int(upper_region_height)
            
            if analysis_left < analysis_right and analysis_top < analysis_bottom:
                # Extract the actual ROI
                roi = img[analysis_top:analysis_bottom, analysis_left:analysis_right]
                roi_output_path = f"pic/screenshots/{timestamp}_Upper_ROI_Example_Card1.png"
                cv2.imwrite(roi_output_path, roi)
                
                print(f"\\n📸 Example ROI extracted: {os.path.basename(roi_output_path)}")
                print(f"   • This is the actual image data passed to Phase 2 (Grey Text Isolation)")
                print(f"   • Dimensions: {roi.shape[1]}×{roi.shape[0]} pixels")
        
    except Exception as e:
        print(f"❌ Demo failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    demonstrate_upper_region_extraction()
    print("\\n🏁 Upper region extraction demonstration complete!")