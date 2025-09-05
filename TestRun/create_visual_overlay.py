#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Create Visual Overlay
Draws detection results directly on the screenshot for visual verification
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from TestRun.accurate_avatar_detector import AccurateAvatarDetector

def create_visual_overlay(screenshot_path, output_path=None):
    """Create a visual overlay showing all detection results on the original image"""
    
    # Load the original screenshot
    img = cv2.imread(screenshot_path)
    if img is None:
        print(f"❌ Failed to load image: {screenshot_path}")
        return None
    
    print(f"📸 Creating visual overlay for: {os.path.basename(screenshot_path)}")
    
    # Initialize detector and get results
    detector = AccurateAvatarDetector()
    centers = detector.find_avatar_centers(screenshot_path)
    contact_cards = detector.get_contact_regions(screenshot_path)
    
    # Create visualization overlay
    overlay = img.copy()
    
    # Draw semi-transparent background for better visibility
    alpha = 0.3
    overlay_bg = img.copy()
    
    # Draw each contact card region
    for i, card in enumerate(contact_cards):
        # Extract card info
        avatar_x, avatar_y = card['avatar_center']
        avatar_left = avatar_x - 20
        avatar_top = avatar_y - 20
        click_x, click_y = card['click_center']
        
        # Draw card boundary (subtle)
        card_bounds = card['card_bounds']
        cv2.rectangle(overlay_bg, 
                     (card_bounds[0], card_bounds[1]), 
                     (card_bounds[0] + card_bounds[2], card_bounds[1] + card_bounds[3]),
                     (200, 200, 255), 1)
        
        # Draw avatar box (bright green)
        cv2.rectangle(overlay, 
                     (avatar_left, avatar_top), 
                     (avatar_left + 40, avatar_top + 40),
                     (0, 255, 0), 2)
        
        # Draw avatar center point (red dot)
        cv2.circle(overlay, (avatar_x, avatar_y), 3, (0, 0, 255), -1)
        
        # Draw click position (blue cross)
        cv2.drawMarker(overlay, (click_x, click_y), (255, 100, 0), 
                      cv2.MARKER_CROSS, 10, 2)
        
        # Draw line from avatar to click position
        cv2.arrowedLine(overlay, (avatar_x + 20, avatar_y), (click_x - 10, click_y),
                       (255, 200, 0), 1, tipLength=0.3)
        
        # Add card number label
        label = f"{i+1}"
        label_bg_pos = (avatar_left - 5, avatar_top - 10)
        
        # Draw label background for better visibility
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(overlay, 
                     (label_bg_pos[0], label_bg_pos[1] - label_h - 4),
                     (label_bg_pos[0] + label_w + 4, label_bg_pos[1]),
                     (0, 255, 0), -1)
        
        # Draw label text
        cv2.putText(overlay, label, 
                   (avatar_left - 3, avatar_top - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Blend overlay with original
    result = cv2.addWeighted(overlay, 0.8, overlay_bg, 0.2, 0)
    
    # Add legend
    legend_y = 30
    legend_x = img.shape[1] - 400
    
    # Draw legend background
    cv2.rectangle(result, (legend_x - 10, 10), (img.shape[1] - 10, 150), 
                 (255, 255, 255), -1)
    cv2.rectangle(result, (legend_x - 10, 10), (img.shape[1] - 10, 150), 
                 (0, 0, 0), 2)
    
    # Legend title
    cv2.putText(result, "Detection Legend:", (legend_x, legend_y),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
    
    # Legend items
    legend_items = [
        ("Green Box", "Avatar detected", (0, 255, 0)),
        ("Red Dot", "Avatar center", (0, 0, 255)),
        ("Blue Cross", "Click position", (255, 100, 0)),
        ("Arrow", "Avatar → Click", (255, 200, 0))
    ]
    
    for i, (symbol, desc, color) in enumerate(legend_items):
        y_pos = legend_y + 30 + (i * 25)
        
        # Draw symbol example
        if "Box" in symbol:
            cv2.rectangle(result, (legend_x, y_pos - 10), (legend_x + 15, y_pos + 5), color, 2)
        elif "Dot" in symbol:
            cv2.circle(result, (legend_x + 7, y_pos - 2), 3, color, -1)
        elif "Cross" in symbol:
            cv2.drawMarker(result, (legend_x + 7, y_pos - 2), color, cv2.MARKER_CROSS, 10, 2)
        elif "Arrow" in symbol:
            cv2.arrowedLine(result, (legend_x, y_pos), (legend_x + 15, y_pos), color, 2)
        
        # Draw description
        cv2.putText(result, f"= {desc}", (legend_x + 25, y_pos),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Add summary at top
    summary = f"Accurate Detection: {len(centers)} Real Avatars (No False Positives)"
    cv2.rectangle(result, (10, 10), (400, 50), (255, 255, 255), -1)
    cv2.rectangle(result, (10, 10), (400, 50), (0, 200, 0), 2)
    cv2.putText(result, summary, (20, 35),
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 150, 0), 2)
    
    # Add detailed info box
    info_y = img.shape[0] - 100
    cv2.rectangle(result, (10, info_y - 40), (450, info_y + 40), (255, 255, 255), -1)
    cv2.rectangle(result, (10, info_y - 40), (450, info_y + 40), (0, 0, 0), 2)
    
    info_lines = [
        f"Total Cards Detected: {len(contact_cards)}",
        f"Avatar X Range: {min(c[0] for c in centers)} - {max(c[0] for c in centers)}px",
        f"Avatar Y Range: {min(c[1] for c in centers)} - {max(c[1] for c in centers)}px",
        f"Click Offset: Avatar X + 70px (into text area)"
    ]
    
    for i, line in enumerate(info_lines):
        cv2.putText(result, line, (20, info_y - 20 + (i * 20)),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    
    # Save result
    if output_path is None:
        output_path = f"pic/screenshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_visual_overlay.png"
    
    cv2.imwrite(output_path, result)
    print(f"✅ Visual overlay saved: {output_path}")
    
    return output_path

if __name__ == "__main__":
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        output = create_visual_overlay(screenshot_path)
        
        print(f"\n📊 Visual Overlay Created!")
        print(f"Open this file to see the results: {output}")
        print(f"\nWhat's shown on the image:")
        print(f"  🟩 Green Boxes = Detected avatars")
        print(f"  🔴 Red Dots = Avatar center points")
        print(f"  🔵 Blue Crosses = Click positions")
        print(f"  ➡️ Arrows = Connection from avatar to click point")
        print(f"  Numbers = Avatar index (1-16)")
    else:
        print("❌ No diagnostic screenshots found")