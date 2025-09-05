#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Debug Avatar Detection
Visualize where avatars are being detected vs actual positions
"""

import cv2
import numpy as np
import os
from datetime import datetime

def analyze_contact_list_structure(image_path):
    """Analyze the actual structure of WeChat contact list"""
    
    img = cv2.imread(image_path)
    if img is None:
        print(f"Failed to load image: {image_path}")
        return
    
    height, width = img.shape[:2]
    print(f"Image dimensions: {width}x{height}")
    
    # Create visualization
    vis = img.copy()
    
    # Look for avatars by detecting circular regions with content
    # WeChat avatars are typically circular profile pictures
    
    # Check different x positions to find where avatars actually are
    x_positions_to_check = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    
    print("\n🔍 Scanning for avatar positions...")
    
    # Check multiple y positions (typical contact card positions)
    y_positions = range(100, min(height-50, 1400), 60)  # Check every 60 pixels
    
    found_avatars = []
    
    for y in y_positions:
        for x in x_positions_to_check:
            # Define a region where avatar might be
            avatar_size = 40  # Typical avatar size
            
            if x + avatar_size < width and y + avatar_size < height:
                roi = img[y:y+avatar_size, x:x+avatar_size]
                
                # Check if this region has enough variation (not empty)
                gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                variance = np.var(gray_roi)
                
                # Also check for circular content (avatars are circular)
                # Create a circular mask
                mask = np.zeros((avatar_size, avatar_size), dtype=np.uint8)
                cv2.circle(mask, (avatar_size//2, avatar_size//2), avatar_size//2 - 2, 255, -1)
                
                # Apply mask and check content
                masked_roi = cv2.bitwise_and(gray_roi, gray_roi, mask=mask)
                masked_variance = np.var(masked_roi[mask > 0])
                
                if masked_variance > 200:  # Has significant content
                    # Check if we already found an avatar at this y position
                    if not any(abs(y - ay) < 20 for ax, ay in found_avatars):
                        found_avatars.append((x, y))
                        print(f"  Found potential avatar at X={x}, Y={y} (variance={masked_variance:.0f})")
                        
                        # Draw on visualization
                        cv2.rectangle(vis, (x, y), (x+avatar_size, y+avatar_size), (0, 255, 0), 2)
                        cv2.putText(vis, f"{len(found_avatars)}", (x+5, y+25), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                        break  # Found avatar at this y, move to next y
    
    print(f"\n📊 Found {len(found_avatars)} potential avatar positions")
    
    if found_avatars:
        # Analyze x positions
        x_positions = [x for x, y in found_avatars]
        unique_x = list(set(x_positions))
        print(f"  X positions: {unique_x}")
        
        # Most common x position is likely the correct one
        from collections import Counter
        x_counts = Counter(x_positions)
        most_common_x = x_counts.most_common(1)[0][0]
        print(f"  Most common X position: {most_common_x}")
        
        # Calculate average spacing
        y_positions = sorted([y for x, y in found_avatars if x == most_common_x])
        if len(y_positions) > 1:
            spacings = [y_positions[i+1] - y_positions[i] for i in range(len(y_positions)-1)]
            avg_spacing = sum(spacings) / len(spacings)
            print(f"  Average Y spacing: {avg_spacing:.1f} pixels")
    
    # Also check the current analyzer's detection
    print("\n🔍 Checking current analyzer detection at X=60...")
    test_x = 60
    for y in range(100, min(height-50, 500), 60):
        if test_x + 40 < width and y + 40 < height:
            roi = img[y:y+40, test_x:test_x+40]
            gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            variance = np.var(gray_roi)
            
            # Draw what current analyzer might be seeing
            cv2.rectangle(vis, (test_x, y), (test_x+40, y+40), (0, 0, 255), 1)
            cv2.putText(vis, f"v={variance:.0f}", (test_x+45, y+20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
    
    # Save visualization
    output_path = os.path.join("pic/screenshots", f"avatar_debug_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    cv2.imwrite(output_path, vis)
    print(f"\n💾 Saved visualization to: {output_path}")
    
    return found_avatars


if __name__ == "__main__":
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        print(f"🎯 Analyzing: {latest}")
        avatars = analyze_contact_list_structure(screenshot_path)
        
        if avatars and len(avatars) > 0:
            # Suggest fix
            x_positions = [x for x, y in avatars]
            from collections import Counter
            x_counts = Counter(x_positions)
            correct_x = x_counts.most_common(1)[0][0]
            
            print(f"\n✅ RECOMMENDATION:")
            print(f"   Change AVATAR_LEFT_MARGIN from 62 to {correct_x}")
            print(f"   This will align avatar detection with actual WeChat UI")
    else:
        print("No screenshots found")