#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Robust Avatar Detector
Combines multiple detection methods to reliably find all avatar centers in WeChat message list
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from datetime import datetime

class RobustAvatarDetector:
    """Find avatar centers using multiple robust detection methods"""
    
    def __init__(self):
        # Avatar characteristics (refined based on analysis)
        self.AVATAR_SIZE = 40
        self.MIN_STD_THRESHOLD = 30  # Based on debug data showing 40-80 std
        self.LEFT_SCAN_START = 30
        self.LEFT_SCAN_END = 100
        self.VERTICAL_STEP = 15  # Fine-grained vertical scanning
        self.MIN_VERTICAL_SPACING = 50  # Minimum distance between avatars
        
    def find_avatar_centers(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Find all avatar center coordinates using robust detection
        Returns sorted list of (center_x, center_y) tuples
        """
        print(f"\n🎯 Finding avatar centers in: {os.path.basename(image_path)}")
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return []
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"📐 Image: {width}x{height}")
        
        # Method 1: Standard deviation scanning (most reliable based on debug data)
        avatars = self._scan_by_std(gray, height)
        
        # Method 2: Edge density scanning (backup method)
        if len(avatars) < 5:  # If std method didn't find enough
            print("🔄 Using edge density as backup...")
            avatars.extend(self._scan_by_edges(gray, height))
        
        # Remove duplicates and sort
        avatars = self._remove_duplicates(avatars)
        avatars.sort(key=lambda pos: pos[1])  # Sort by Y position
        
        print(f"✅ Found {len(avatars)} avatar centers")
        
        # Debug: Show first few positions
        for i, (x, y) in enumerate(avatars[:5]):
            print(f"  Avatar {i}: Center at ({x}, {y})")
        
        return avatars
    
    def _scan_by_std(self, gray, height) -> List[Tuple[int, int]]:
        """Scan using standard deviation threshold (primary method)"""
        
        avatars = []
        
        # Scan with fine-grained steps
        for y in range(80, height - self.AVATAR_SIZE, self.VERTICAL_STEP):
            best_x = None
            best_std = 0
            
            # Find the best X position at this Y level
            for x in range(self.LEFT_SCAN_START, self.LEFT_SCAN_END):
                if x + self.AVATAR_SIZE >= gray.shape[1]:
                    break
                
                # Extract avatar-sized region
                roi = gray[y:y+self.AVATAR_SIZE, x:x+self.AVATAR_SIZE]
                
                if roi.size == 0:
                    continue
                
                std = np.std(roi)
                
                # Check if this looks like an avatar
                if std > self.MIN_STD_THRESHOLD and std > best_std:
                    best_std = std
                    best_x = x
            
            # If we found a good candidate at this Y level
            if best_x is not None:
                center_x = best_x + self.AVATAR_SIZE // 2
                center_y = y + self.AVATAR_SIZE // 2
                
                # Check if it's far enough from existing avatars
                is_unique = True
                for ex, ey in avatars:
                    if abs(center_y - ey) < self.MIN_VERTICAL_SPACING:
                        is_unique = False
                        break
                
                if is_unique:
                    avatars.append((center_x, center_y))
        
        print(f"📊 STD method found: {len(avatars)} avatars")
        return avatars
    
    def _scan_by_edges(self, gray, height) -> List[Tuple[int, int]]:
        """Scan using edge detection (backup method)"""
        
        avatars = []
        edges = cv2.Canny(gray, 50, 150)
        
        for y in range(80, height - self.AVATAR_SIZE, 20):
            for x in range(self.LEFT_SCAN_START, self.LEFT_SCAN_END):
                if x + self.AVATAR_SIZE >= gray.shape[1]:
                    break
                
                # Extract region
                roi_edges = edges[y:y+self.AVATAR_SIZE, x:x+self.AVATAR_SIZE]
                roi_gray = gray[y:y+self.AVATAR_SIZE, x:x+self.AVATAR_SIZE]
                
                if roi_edges.size == 0 or roi_gray.size == 0:
                    continue
                
                # Check edge density and standard deviation
                edge_density = np.sum(roi_edges > 0) / roi_edges.size
                std = np.std(roi_gray)
                
                if edge_density > 0.05 and std > 25:
                    center_x = x + self.AVATAR_SIZE // 2
                    center_y = y + self.AVATAR_SIZE // 2
                    
                    # Check uniqueness
                    is_unique = True
                    for ex, ey in avatars:
                        if abs(center_y - ey) < self.MIN_VERTICAL_SPACING:
                            is_unique = False
                            break
                    
                    if is_unique:
                        avatars.append((center_x, center_y))
        
        print(f"🔍 Edge method found: {len(avatars)} additional avatars")
        return avatars
    
    def _remove_duplicates(self, avatars) -> List[Tuple[int, int]]:
        """Remove duplicate detections"""
        
        unique = []
        
        for x, y in avatars:
            is_duplicate = False
            
            for ux, uy in unique:
                # If too close, it's a duplicate
                if abs(y - uy) < 30 and abs(x - ux) < 30:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append((x, y))
        
        return unique
    
    def visualize_detection(self, image_path: str, output_path: str = None) -> str:
        """Create visualization showing all detected avatar centers"""
        
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        # Detect avatars
        centers = self.find_avatar_centers(image_path)
        
        # Create visualization
        vis = img.copy()
        
        # Draw avatar centers and estimated boundaries
        for i, (cx, cy) in enumerate(centers):
            # Draw center point (green dot)
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
            
            # Draw estimated bounding box
            half_size = self.AVATAR_SIZE // 2
            x1, y1 = cx - half_size, cy - half_size
            x2, y2 = cx + half_size, cy + half_size
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add index label
            cv2.putText(vis, str(i), (cx - 8, cy - half_size - 8),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Draw scan region indicator
        cv2.line(vis, (self.LEFT_SCAN_START, 0), (self.LEFT_SCAN_START, img.shape[0]), (255, 0, 0), 1)
        cv2.line(vis, (self.LEFT_SCAN_END, 0), (self.LEFT_SCAN_END, img.shape[0]), (255, 0, 0), 1)
        
        # Add detection summary
        summary = f"Detected {len(centers)} avatars"
        cv2.putText(vis, summary, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save visualization
        if output_path is None:
            output_path = f"pic/screenshots/{datetime.now().strftime('%Y%m%d_%H%M%S')}_robust_avatar_detection.png"
        
        cv2.imwrite(output_path, vis)
        print(f"💾 Visualization saved: {output_path}")
        
        return output_path
    
    def get_contact_regions(self, image_path: str) -> List[dict]:
        """
        Get complete contact card information based on avatar centers
        Returns list of contact card dictionaries with all regions
        """
        centers = self.find_avatar_centers(image_path)
        
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        contact_cards = []
        
        for i, (avatar_x, avatar_y) in enumerate(centers):
            # Define contact card boundaries based on avatar position
            card_left = 0
            card_right = min(width, 480)  # Typical WeChat contact list width
            card_top = max(0, avatar_y - 20)  # Card starts above avatar
            card_bottom = min(height, avatar_y + 50)  # Card extends below avatar
            
            # Avatar bounds (already known)
            avatar_left = avatar_x - self.AVATAR_SIZE // 2
            avatar_top = avatar_y - self.AVATAR_SIZE // 2
            
            # Text region (to the right of avatar)
            text_left = avatar_left + self.AVATAR_SIZE + 10
            text_right = card_right - 10
            
            # Click position (center of text area)
            click_x = (text_left + text_right) // 2
            click_y = avatar_y
            
            card_info = {
                'index': i,
                'avatar_center': (avatar_x, avatar_y),
                'avatar_bounds': (avatar_left, avatar_top, self.AVATAR_SIZE, self.AVATAR_SIZE),
                'card_bounds': (card_left, card_top, card_right - card_left, card_bottom - card_top),
                'text_region': (text_left, card_top, text_right - text_left, card_bottom - card_top),
                'click_center': (click_x, click_y),
                'has_red_dot': False,  # Would need color detection
                'has_message': True,   # Assume true if avatar exists
                'contact_name': 'Unknown'
            }
            
            contact_cards.append(card_info)
        
        return contact_cards


if __name__ == "__main__":
    # Test the robust detector
    detector = RobustAvatarDetector()
    
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        print(f"🎯 Testing robust detection with: {latest}")
        
        # Find all avatars
        centers = detector.find_avatar_centers(screenshot_path)
        
        print(f"\n📊 Summary:")
        print(f"  Total avatars found: {len(centers)}")
        
        # Create visualization
        output = detector.visualize_detection(screenshot_path)
        print(f"\n✅ Visualization: {output}")
        
        # Test contact card generation
        cards = detector.get_contact_regions(screenshot_path)
        print(f"\n📋 Generated {len(cards)} contact card definitions")
        for i, card in enumerate(cards[:3]):  # Show first 3
            print(f"  Card {i}: Avatar({card['avatar_center'][0]}, {card['avatar_center'][1]}) → Click({card['click_center'][0]}, {card['click_center'][1]})")
    else:
        print("❌ No diagnostic screenshots found")