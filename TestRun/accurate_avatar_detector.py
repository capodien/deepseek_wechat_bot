#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Accurate Avatar Detector
Precisely finds only real avatar centers, avoiding false positives
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from datetime import datetime

class AccurateAvatarDetector:
    """Precisely detect only real avatar centers with strict validation"""
    
    def __init__(self):
        # Avatar characteristics based on WeChat UI analysis
        self.AVATAR_SIZE = 40
        self.AVATAR_LEFT_MARGIN = 60  # More precise based on actual UI
        self.CARD_HEIGHT = 70  # Standard WeChat contact card height
        
        # Stricter detection thresholds to avoid false positives
        self.MIN_STD_THRESHOLD = 40  # Higher threshold for real content
        self.MAX_STD_THRESHOLD = 120 # Avoid overly noisy regions
        self.MIN_MEAN_THRESHOLD = 50  # Avoid too-dark empty regions
        self.MAX_MEAN_THRESHOLD = 200 # Avoid too-bright empty regions
        
        # Positional constraints
        self.TOP_MARGIN = 100  # Don't look too high (header area)
        self.BOTTOM_MARGIN = 50  # Don't look too low
        
        # Validation parameters
        self.MIN_VERTICAL_SPACING = 55  # Real cards are ~70px apart
        self.MAX_CARDS_EXPECTED = 20   # Reasonable limit
        
    def find_avatar_centers(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Find only real avatar centers with strict validation
        Returns sorted list of (center_x, center_y) tuples
        """
        print(f"\n🎯 Accurate avatar detection in: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return []
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"📐 Image: {width}x{height}")
        
        avatars = []
        scan_bottom = min(height - self.BOTTOM_MARGIN, 1400)  # Don't scan too far down
        
        # Scan with consistent spacing based on card height
        y = self.TOP_MARGIN
        while y < scan_bottom:
            avatar_found = self._find_avatar_at_y_level(gray, y, width)
            if avatar_found:
                center_x, center_y = avatar_found
                
                # Validate this isn't too close to existing avatars
                if self._is_valid_position(avatars, center_x, center_y):
                    avatars.append((center_x, center_y))
                    print(f"  ✅ Avatar {len(avatars)}: ({center_x}, {center_y})")
                    
                    # Move to next expected card position
                    y += self.CARD_HEIGHT
                else:
                    # Small step if position was invalid
                    y += 20
            else:
                # Small step to keep scanning
                y += 15
            
            # Safety limit
            if len(avatars) >= self.MAX_CARDS_EXPECTED:
                break
        
        print(f"✅ Found {len(avatars)} real avatars")
        return avatars
    
    def _find_avatar_at_y_level(self, gray, y, width) -> Tuple[int, int] or None:
        """Find avatar at specific Y level with strict validation"""
        
        # Scan around expected avatar X position
        best_x = None
        best_score = 0
        
        for x in range(self.AVATAR_LEFT_MARGIN - 10, self.AVATAR_LEFT_MARGIN + 30):
            if x + self.AVATAR_SIZE >= width or y + self.AVATAR_SIZE >= gray.shape[0]:
                continue
                
            # Extract potential avatar region
            roi = gray[y:y+self.AVATAR_SIZE, x:x+self.AVATAR_SIZE]
            
            if roi.size == 0:
                continue
            
            # Calculate statistics
            mean_val = np.mean(roi)
            std_val = np.std(roi)
            
            # Strict validation for real avatar content
            if (self.MIN_STD_THRESHOLD <= std_val <= self.MAX_STD_THRESHOLD and
                self.MIN_MEAN_THRESHOLD <= mean_val <= self.MAX_MEAN_THRESHOLD):
                
                # Additional validation: check for avatar-like patterns
                score = self._validate_avatar_pattern(roi)
                
                if score > best_score and score > 0.5:
                    best_score = score
                    best_x = x
        
        if best_x is not None:
            center_x = best_x + self.AVATAR_SIZE // 2
            center_y = y + self.AVATAR_SIZE // 2
            return (center_x, center_y)
        
        return None
    
    def _validate_avatar_pattern(self, roi) -> float:
        """Validate that ROI looks like a real avatar (0.0 to 1.0 score)"""
        
        # Check for circular content (avatars are usually circular)
        center = roi.shape[0] // 2
        
        # Create circular mask
        y, x = np.ogrid[:roi.shape[0], :roi.shape[1]]
        mask = (x - center)**2 + (y - center)**2 <= (center - 3)**2
        
        # Compare circular vs corner content
        if np.sum(mask) == 0:
            return 0.0
            
        circular_std = np.std(roi[mask])
        corner_std = np.std(roi[~mask])
        
        # Real avatars have more content in center than corners
        if circular_std > corner_std * 0.8:
            score = min(1.0, circular_std / 60.0)  # Normalize to 0-1
            return score
        
        return 0.0
    
    def _is_valid_position(self, existing_avatars, new_x, new_y) -> bool:
        """Check if new position is valid (not too close to existing)"""
        
        for ex_x, ex_y in existing_avatars:
            distance = abs(new_y - ex_y)
            if distance < self.MIN_VERTICAL_SPACING:
                return False
        
        return True
    
    def visualize_detection(self, image_path: str, output_path: str = None) -> str:
        """Create visualization showing only real detected avatars"""
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Detect avatars
        centers = self.find_avatar_centers(image_path)
        
        # Create visualization
        vis = img.copy()
        
        # Draw scan region
        height = img.shape[0]
        scan_bottom = min(height - self.BOTTOM_MARGIN, 1400)
        cv2.rectangle(vis, (self.AVATAR_LEFT_MARGIN - 20, self.TOP_MARGIN), 
                     (self.AVATAR_LEFT_MARGIN + 60, scan_bottom), (255, 200, 0), 2)
        
        # Draw detected avatars
        for i, (cx, cy) in enumerate(centers):
            # Draw center point (green dot)
            cv2.circle(vis, (cx, cy), 4, (0, 255, 0), -1)
            
            # Draw estimated bounding box
            half_size = self.AVATAR_SIZE // 2
            x1, y1 = cx - half_size, cy - half_size
            x2, y2 = cx + half_size, cy + half_size
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add index label
            cv2.putText(vis, str(i+1), (cx - 8, cy - half_size - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        # Add detection summary
        summary = f"Accurate Detection: {len(centers)} real avatars"
        cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Add scan info
        info = f"Scanned Y: {self.TOP_MARGIN} to {scan_bottom}"
        cv2.putText(vis, info, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Save visualization
        if output_path is None:
            output_path = f"pic/screenshots/accurate_avatar_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        cv2.imwrite(output_path, vis)
        print(f"💾 Visualization saved: {output_path}")
        
        return output_path
    
    def get_contact_regions(self, image_path: str) -> List[dict]:
        """Get complete contact card information based on accurate avatar centers"""
        centers = self.find_avatar_centers(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        contact_cards = []
        
        for i, (avatar_x, avatar_y) in enumerate(centers):
            # Define contact card boundaries based on avatar position
            card_left = 0
            card_right = min(width, 480)
            card_top = max(0, avatar_y - 35)  # Card starts above avatar
            card_bottom = min(height, avatar_y + 35)  # Card extends below avatar
            
            # Avatar bounds
            avatar_left = avatar_x - self.AVATAR_SIZE // 2
            avatar_top = avatar_y - self.AVATAR_SIZE // 2
            
            # Click position (in text area to the right of avatar)
            click_x = avatar_left + self.AVATAR_SIZE + 50  # Into text area
            click_y = avatar_y  # Same as avatar center
            
            card_info = {
                'index': i,
                'avatar_center': (avatar_x, avatar_y),
                'avatar_bounds': (avatar_left, avatar_top, self.AVATAR_SIZE, self.AVATAR_SIZE),
                'card_bounds': (card_left, card_top, card_right - card_left, card_bottom - card_top),
                'text_region': (avatar_left + self.AVATAR_SIZE + 10, card_top, 
                              card_right - avatar_left - self.AVATAR_SIZE - 20, card_bottom - card_top),
                'click_center': (click_x, click_y),
                'has_red_dot': False,  # Would need separate detection
                'has_message': True,   # Assume true if avatar exists
                'contact_name': 'Unknown'
            }
            
            contact_cards.append(card_info)
        
        return contact_cards


if __name__ == "__main__":
    # Test the accurate detector
    detector = AccurateAvatarDetector()
    
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        print(f"🎯 Testing accurate detection with: {latest}")
        
        # Find real avatars only
        centers = detector.find_avatar_centers(screenshot_path)
        
        print(f"\n📊 Accurate Detection Summary:")
        print(f"  Real avatars found: {len(centers)}")
        print(f"  Expected for visible area: ~16-17")
        
        # Create visualization
        output = detector.visualize_detection(screenshot_path)
        print(f"\n✅ Visualization: {output}")
        
        # Show coordinates
        if centers:
            print(f"\n📍 Avatar Centers:")
            for i, (x, y) in enumerate(centers):
                click_x = x + 70  # Approximate click position
                print(f"  {i+1}: Avatar({x}, {y}) → Click({click_x}, {y})")
        
    else:
        print("❌ No diagnostic screenshots found")