#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Adaptive Width Calculator for Message Card OCR Regions
Dynamically calculates optimal OCR window width based on card content
"""

import cv2
import numpy as np
from typing import Dict, Tuple, List

class AdaptiveWidthCalculator:
    """Calculate adaptive OCR window dimensions for message cards"""
    
    def __init__(self):
        # Default fallback values
        self.DEFAULT_USERNAME_WIDTH = 200
        self.DEFAULT_MESSAGE_WIDTH = 250
        self.DEFAULT_TIMESTAMP_WIDTH = 100
        
        # Minimum and maximum widths
        self.MIN_USERNAME_WIDTH = 120
        self.MAX_USERNAME_WIDTH = 300
        self.MIN_MESSAGE_WIDTH = 150
        self.MAX_MESSAGE_WIDTH = 400
        self.MIN_TIMESTAMP_WIDTH = 80
        self.MAX_TIMESTAMP_WIDTH = 150
        
        # Edge detection parameters
        self.EDGE_THRESHOLD = 50
        self.PADDING = 10  # Extra padding for OCR accuracy
        
    def calculate_message_card_width(self, img: np.ndarray, avatar_bounds: Tuple, 
                                   screen_width: int) -> Dict:
        """
        Calculate adaptive width for message card OCR regions
        
        Args:
            img: Screenshot image
            avatar_bounds: (x, y, w, h) of detected avatar
            screen_width: Total screen width
            
        Returns:
            Dict with calculated dimensions for username, message, timestamp
        """
        x, y, w, h = avatar_bounds
        
        # Method 1: Detect card boundary by looking for vertical edges
        card_width = self._detect_card_boundary(img, avatar_bounds, screen_width)
        
        # Method 2: Use proportion-based calculation as fallback
        if card_width is None:
            card_width = self._calculate_proportional_width(avatar_bounds, screen_width)
        
        # Calculate individual region widths
        regions = self._calculate_region_widths(card_width, avatar_bounds)
        
        return {
            'card_width': card_width,
            'username_width': regions['username_width'],
            'message_width': regions['message_width'], 
            'timestamp_width': regions['timestamp_width'],
            'method': regions['method']
        }
    
    def _detect_card_boundary(self, img: np.ndarray, avatar_bounds: Tuple, 
                            screen_width: int) -> int:
        """
        Detect message card boundary using edge detection
        """
        try:
            x, y, w, h = avatar_bounds
            
            # Create search region to the right of avatar
            search_start_x = x + w + 5
            search_end_x = min(screen_width - 20, search_start_x + 500)
            search_y = y + h // 4  # Search in middle area of card
            search_height = h // 2
            
            if search_start_x >= search_end_x or search_y >= img.shape[0]:
                return None
            
            # Extract horizontal strip for edge detection
            strip = img[search_y:search_y+search_height, search_start_x:search_end_x]
            
            if strip.size == 0:
                return None
            
            # Convert to grayscale
            if len(strip.shape) == 3:
                gray_strip = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY)
            else:
                gray_strip = strip
            
            # Calculate horizontal gradient to find card edges
            grad_x = cv2.Sobel(gray_strip, cv2.CV_64F, 1, 0, ksize=3)
            grad_x = np.abs(grad_x)
            
            # Sum gradients vertically to get horizontal edge profile  
            edge_profile = np.sum(grad_x, axis=0)
            
            # Smooth the profile
            edge_profile = cv2.GaussianBlur(edge_profile.reshape(1, -1), (1, 5), 0).flatten()
            
            # Find significant edges (card boundaries)
            edges = self._find_significant_edges(edge_profile)
            
            if edges:
                # Use the rightmost significant edge as card boundary
                card_boundary_offset = max(edges)
                card_width = search_start_x + card_boundary_offset - x
                
                # Validate reasonable card width
                if 200 <= card_width <= 600:
                    return card_width
            
            return None
            
        except Exception as e:
            print(f"⚠️ Edge detection failed: {e}")
            return None
    
    def _find_significant_edges(self, edge_profile: np.ndarray) -> List[int]:
        """Find significant edges in the gradient profile"""
        if len(edge_profile) < 10:
            return []
        
        # Calculate threshold based on profile statistics
        mean_val = np.mean(edge_profile)
        std_val = np.std(edge_profile)
        threshold = mean_val + std_val * 1.5
        
        # Find peaks above threshold
        edges = []
        for i in range(1, len(edge_profile) - 1):
            if (edge_profile[i] > threshold and 
                edge_profile[i] > edge_profile[i-1] and 
                edge_profile[i] > edge_profile[i+1]):
                edges.append(i)
        
        return edges
    
    def _calculate_proportional_width(self, avatar_bounds: Tuple, 
                                    screen_width: int) -> int:
        """
        Calculate card width based on screen proportions
        """
        x, y, w, h = avatar_bounds
        
        # Method: Use proportion of remaining screen width
        remaining_width = screen_width - (x + w) - 20  # 20px margin from right edge
        
        # Use 85% of remaining width for card content
        card_width = int(remaining_width * 0.85)
        
        # Clamp to reasonable bounds
        card_width = max(250, min(card_width, 500))
        
        return card_width
    
    def _calculate_region_widths(self, card_width: int, avatar_bounds: Tuple) -> Dict:
        """
        Calculate individual OCR region widths based on subtraction method:
        username_width = card_width - avatar_width - timestamp_width
        """
        x, y, w, h = avatar_bounds
        
        # Fixed timestamp width (timestamps like "08:46", "Yesterday 21:55" are fairly consistent)
        timestamp_width = 100  # Fixed width for timestamps
        
        # Calculate username width using subtraction method
        # Total usable width = card_width - spacing
        usable_width = card_width - 20  # 20px for spacing/padding
        username_width = usable_width - timestamp_width
        
        # Apply bounds checking
        username_width = max(self.MIN_USERNAME_WIDTH, 
                           min(username_width, self.MAX_USERNAME_WIDTH))
        
        # Message width uses full card width for message preview area
        message_width = max(self.MIN_MESSAGE_WIDTH,
                          min(int(card_width * 0.80), self.MAX_MESSAGE_WIDTH))
        
        return {
            'username_width': username_width,
            'message_width': message_width,
            'timestamp_width': timestamp_width,
            'method': 'subtraction_method'
        }
    
    def calculate_adaptive_regions(self, img: np.ndarray, avatar_bounds: Tuple) -> Dict:
        """
        Calculate all adaptive OCR regions for a message card
        
        Returns:
            Dict with all region coordinates and dimensions
        """
        x, y, w, h = avatar_bounds
        screen_width = img.shape[1]
        
        # Get adaptive widths
        adaptive_dims = self.calculate_message_card_width(img, avatar_bounds, screen_width)
        
        # Calculate region coordinates
        regions = {
            'username_region': {
                'x': x + w + 10,
                'y': y,
                'width': adaptive_dims['username_width'],
                'height': int(h * 0.6)  # Top 60% for username
            },
            'message_region': {
                'x': x + w + 10, 
                'y': y + int(h * 0.6),  # Bottom 40% for message
                'width': adaptive_dims['message_width'],
                'height': int(h * 0.4)
            },
            'timestamp_region': {
                'x': x + adaptive_dims['card_width'] - adaptive_dims['timestamp_width'] - 10,  # 10px margin from right edge
                'y': y,
                'width': adaptive_dims['timestamp_width'], 
                'height': int(h * 0.6)  # Same height as username region (top portion)
            },
            'adaptive_info': adaptive_dims
        }
        
        # Validate all regions are within screen bounds
        for region_name, region in regions.items():
            if region_name == 'adaptive_info':
                continue
                
            # Clamp to screen boundaries
            region['x'] = max(0, min(region['x'], screen_width - region['width']))
            region['y'] = max(0, min(region['y'], img.shape[0] - region['height']))
            region['width'] = min(region['width'], screen_width - region['x'])
            region['height'] = min(region['height'], img.shape[0] - region['y'])
        
        return regions


def test_adaptive_width():
    """Test adaptive width calculation with latest screenshot"""
    import os
    import sys
    
    # Add current directory to path for imports
    current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, current_dir)
    
    from TestRun.opencv_adaptive_detector import OpenCVAdaptiveDetector
    
    # Find latest screenshot
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) 
                  if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if not screenshots:
        print("❌ No diagnostic screenshots found")
        return
    
    latest_screenshot = sorted(screenshots)[-1]
    screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
    
    print(f"🔍 Testing adaptive width with: {latest_screenshot}")
    
    # Detect avatars
    detector = OpenCVAdaptiveDetector()
    avatars = detector.detect_avatars(screenshot_path)
    
    if not avatars:
        print("❌ No avatars detected")
        return
    
    # Load image
    img = cv2.imread(screenshot_path)
    if img is None:
        print("❌ Failed to load image")
        return
    
    # Test adaptive width calculation
    calculator = AdaptiveWidthCalculator()
    
    print(f"\n📏 Adaptive Width Analysis:")
    for i, avatar in enumerate(avatars, 1):
        avatar_bounds = avatar['card_bounds']
        regions = calculator.calculate_adaptive_regions(img, avatar_bounds)
        
        print(f"\n🎯 Avatar #{i} - {avatar_bounds}")
        print(f"   Card Width: {regions['adaptive_info']['card_width']}px ({regions['adaptive_info']['method']})")
        print(f"   Username: {regions['username_region']['width']}px")
        print(f"   Message: {regions['message_region']['width']}px") 
        print(f"   Timestamp: {regions['timestamp_region']['width']}px")


if __name__ == "__main__":
    test_adaptive_width()