#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Avatar Contour Detector
Uses edge detection and contour analysis to find all avatars in WeChat message list
"""

import cv2
import numpy as np
import os
from typing import List, Tuple
from datetime import datetime

class AvatarContourDetector:
    """Detect avatars using contour and blob detection methods"""
    
    def __init__(self):
        # Avatar characteristics based on WeChat UI
        self.AVATAR_MIN_SIZE = 30
        self.AVATAR_MAX_SIZE = 50
        self.LEFT_REGION_WIDTH_RATIO = 0.25  # Focus on left 25% of image
        self.ASPECT_RATIO_TOLERANCE = 0.2  # Allow 20% deviation from perfect square
        
    def detect_avatars(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Detect avatar center positions using contour detection
        Returns list of (center_x, center_y) coordinates
        """
        print(f"\n🎯 Detecting avatars in: {os.path.basename(image_path)}")
        
        # Step 1: Load and preprocess image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return []
        
        height, width = img.shape[:2]
        print(f"📐 Image dimensions: {width}x{height}")
        
        # Step 2: Focus on left column where avatars are
        left_region_width = int(width * self.LEFT_REGION_WIDTH_RATIO)
        left_region = img[:, :left_region_width]
        print(f"🔍 Focusing on left {left_region_width}px of image")
        
        # Convert to grayscale
        gray = cv2.cvtColor(left_region, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Step 3: Edge detection to find avatar boundaries
        # Use adaptive thresholding for better results with varying avatar styles
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
        
        # Also try Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Combine both methods
        combined = cv2.bitwise_or(thresh, edges)
        
        # Step 4: Find contours
        contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"📦 Found {len(contours)} initial contours")
        
        # Step 5: Filter contours to find avatars
        avatars = []
        for contour in contours:
            # Get bounding rectangle
            x, y, w, h = cv2.boundingRect(contour)
            
            # Filter by size
            if not (self.AVATAR_MIN_SIZE <= w <= self.AVATAR_MAX_SIZE and
                   self.AVATAR_MIN_SIZE <= h <= self.AVATAR_MAX_SIZE):
                continue
            
            # Filter by aspect ratio (should be roughly square)
            aspect_ratio = w / h if h > 0 else 0
            if not (1 - self.ASPECT_RATIO_TOLERANCE <= aspect_ratio <= 1 + self.ASPECT_RATIO_TOLERANCE):
                continue
            
            # Filter by position (avatars typically start around x=40-60)
            if x < 20 or x > 100:
                continue
            
            # Calculate center coordinates
            center_x = x + w // 2
            center_y = y + h // 2
            
            # Avoid duplicates (avatars too close together)
            is_duplicate = False
            for existing_x, existing_y in avatars:
                if abs(center_y - existing_y) < 30:  # Too close vertically
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                avatars.append((center_x, center_y))
        
        # Step 6: Sort by vertical position (top to bottom)
        avatars.sort(key=lambda pos: pos[1])
        
        print(f"✅ Detected {len(avatars)} avatars")
        return avatars
    
    def visualize_detection(self, image_path: str, output_path: str = None) -> str:
        """Create visualization showing detected avatar centers"""
        
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image")
            return None
        
        # Detect avatars
        avatar_centers = self.detect_avatars(image_path)
        
        # Create visualization
        vis = img.copy()
        
        # Draw avatar centers and bounding boxes
        for i, (cx, cy) in enumerate(avatar_centers):
            # Draw center point
            cv2.circle(vis, (cx, cy), 3, (0, 255, 0), -1)
            
            # Draw estimated bounding box
            half_size = self.AVATAR_MIN_SIZE // 2 + 5
            x1, y1 = cx - half_size, cy - half_size
            x2, y2 = cx + half_size, cy + half_size
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Add index label
            cv2.putText(vis, str(i), (cx - 10, cy - half_size - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add detection info
        info_text = f"Detected {len(avatar_centers)} avatars"
        cv2.putText(vis, info_text, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # Save visualization
        if output_path is None:
            output_path = f"pic/screenshots/avatar_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        
        cv2.imwrite(output_path, vis)
        print(f"💾 Visualization saved: {output_path}")
        
        return output_path
    
    def detect_with_template_matching(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Alternative method using template matching to find square regions
        """
        img = cv2.imread(image_path)
        if img is None:
            return []
        
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        avatars = []
        
        # Create a template of expected avatar size (filled square)
        template_size = 40
        template = np.ones((template_size, template_size), dtype=np.uint8) * 255
        
        # Scan the left region with sliding window
        left_region = int(width * 0.25)
        
        for y in range(0, height - template_size, 20):  # Step by 20 pixels
            for x in range(30, min(100, left_region)):  # Avatar x range
                roi = gray[y:y+template_size, x:x+template_size]
                
                # Check if ROI has enough variation (not empty space)
                if np.std(roi) > 20:  # Has content
                    # Check if roughly square shaped
                    edges = cv2.Canny(roi, 50, 150)
                    edge_pixels = np.sum(edges > 0)
                    
                    if edge_pixels > 100:  # Has enough edges
                        center_x = x + template_size // 2
                        center_y = y + template_size // 2
                        
                        # Avoid duplicates
                        is_duplicate = False
                        for ex, ey in avatars:
                            if abs(center_y - ey) < 30:
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            avatars.append((center_x, center_y))
        
        avatars.sort(key=lambda pos: pos[1])
        return avatars


if __name__ == "__main__":
    # Test the detector
    detector = AvatarContourDetector()
    
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        print(f"🖼️ Testing with: {latest}")
        
        # Method 1: Contour detection
        print("\n📊 Method 1: Contour Detection")
        centers = detector.detect_avatars(screenshot_path)
        for i, (x, y) in enumerate(centers[:10]):  # Show first 10
            print(f"  Avatar {i}: Center at ({x}, {y})")
        
        # Method 2: Template matching
        print("\n📊 Method 2: Template Matching")
        centers2 = detector.detect_with_template_matching(screenshot_path)
        print(f"  Found {len(centers2)} avatars")
        
        # Create visualization
        output = detector.visualize_detection(screenshot_path)
        if output:
            print(f"\n✅ Check visualization at: {output}")