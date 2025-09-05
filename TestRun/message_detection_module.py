#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
New Message Detection Module
Combines text change detection with red dot detection for reliable message monitoring

Requirements:
1. Detect new messages using multiple methods
2. Provide accurate click coordinates for detected messages
3. Integrate with improved screenshot capture system
4. Support both text change and red dot detection strategies
"""

import cv2
import numpy as np
import easyocr
import os
import time
import platform
import hashlib
import pyautogui
from datetime import datetime
from typing import Optional, Tuple, List, Dict
import json

class MessageDetector:
    """Comprehensive message detection system with multiple detection strategies"""
    
    def __init__(self, output_dir: str = "TestRun/message_detection"):
        self.system = platform.system()
        self.output_dir = output_dir
        self.ocr_reader = None  # Initialize lazily
        
        # Detection state
        self.previous_text_hash = None
        self.last_screenshot_path = None
        self.detection_history = []
        
        # Region definitions for different detection methods
        self.contact_region = (60, 100, 320, 800)  # Contact list area
        self.red_dot_region = (60, 100, 380, 800)  # Red dot search area
        
        # Red dot detection parameters
        self.red_dot_colors = {
            'primary': np.array([84, 98, 227]),      # Main red dot color
            'windows': np.array([81, 81, 255]),       # Windows variant
            'fallback': np.array([80, 100, 230])     # Fallback color
        }
        self.color_tolerance = 15
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🔍 Message Detector initialized")
        print(f"📁 Output: {self.output_dir}")
        print(f"🖥️  System: {self.system}")
    
    def _get_ocr_reader(self):
        """Lazily initialize OCR reader to save startup time"""
        if self.ocr_reader is None:
            print("📖 Initializing OCR reader...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True, verbose=False)
        return self.ocr_reader
    
    def detect_new_messages(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]], str]:
        """
        Primary message detection method combining multiple strategies
        
        Args:
            screenshot_path: Path to current WeChat screenshot
            
        Returns:
            (detected, coordinates, method) where:
            - detected: True if new message found
            - coordinates: (x, y) click coordinates if found
            - method: Detection method used ('text_change', 'red_dot', 'combined')
        """
        try:
            print(f"\n🔍 Analyzing screenshot: {screenshot_path}")
            
            # Strategy 1: Text change detection (primary)
            text_detected, text_coords = self._detect_text_changes(screenshot_path)
            
            # Strategy 2: Red dot detection (fallback)
            dot_detected, dot_coords = self._detect_red_dots(screenshot_path)
            
            # Combine results with priority logic
            if text_detected and text_coords:
                print(f"✅ Text change detection: Message found at {text_coords}")
                self._log_detection('text_change', text_coords, screenshot_path)
                return True, text_coords, 'text_change'
            
            elif dot_detected and dot_coords:
                print(f"✅ Red dot detection: Message found at {dot_coords}")
                self._log_detection('red_dot', dot_coords, screenshot_path)
                return True, dot_coords, 'red_dot'
            
            else:
                print("🔍 No new messages detected")
                return False, None, 'none'
                
        except Exception as e:
            print(f"❌ Message detection error: {e}")
            return False, None, 'error'
    
    def _detect_text_changes(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Detect messages by monitoring text changes in contact list"""
        try:
            # Load screenshot
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"❌ Failed to load screenshot for text detection")
                return False, None
            
            # Extract text from contact area
            text_content = self._extract_contact_text(image)
            if not text_content:
                return False, None
            
            # Create hash of current text
            text_string = '|'.join(text_content)
            current_hash = hashlib.md5(text_string.encode()).hexdigest()
            
            # Check for changes
            has_changed = (self.previous_text_hash is not None and 
                          current_hash != self.previous_text_hash)
            
            # Update stored hash
            self.previous_text_hash = current_hash
            
            if has_changed:
                print(f"📝 Text change detected in contact list")
                # Find click position for the changed contact
                coords = self._find_target_contact_position(image, text_content)
                if coords:
                    return True, coords
            
            return False, None
            
        except Exception as e:
            print(f"❌ Text change detection error: {e}")
            return False, None
    
    def _extract_contact_text(self, image) -> List[str]:
        """Extract text from contact list region using OCR"""
        try:
            # Crop to contact region
            x, y, w, h = self.contact_region
            cropped = image[y:y+h, x:x+w]
            
            # Use OCR to extract text
            reader = self._get_ocr_reader()
            results = reader.readtext(cropped)
            
            # Extract text content
            text_content = []
            for result in results:
                text = result[1].strip()
                if text and len(text) > 1:  # Filter out single characters
                    text_content.append(text)
            
            return text_content
            
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return []
    
    def _find_target_contact_position(self, image, text_content: List[str]) -> Optional[Tuple[int, int]]:
        """Find click position for target contact (Rio_Old or others with new messages)"""
        try:
            # Look for specific contacts or message indicators
            target_contacts = ['Rio_Old', 'Rio', 'KK', 'File Transfer']
            
            x, y, w, h = self.contact_region
            cropped = image[y:y+h, x:x+w]
            
            reader = self._get_ocr_reader()
            results = reader.readtext(cropped)
            
            # Find target contact with precise coordinates
            for result in results:
                text = result[1].strip()
                
                for target in target_contacts:
                    if target in text:
                        # Get bounding box and calculate click position
                        bbox = result[0]
                        center_x = int((bbox[0][0] + bbox[2][0]) / 2) + x
                        center_y = int((bbox[0][1] + bbox[2][1]) / 2) + y
                        
                        print(f"🎯 Found target contact '{target}' at ({center_x}, {center_y})")
                        return (center_x, center_y)
            
            # Fallback: estimate position of most likely candidate
            for i, text in enumerate(text_content):
                for target in target_contacts:
                    if target in text:
                        estimated_x = x + w//2
                        estimated_y = y + 150 + (i * 75)
                        print(f"📍 Estimated position for '{target}': ({estimated_x}, {estimated_y})")
                        return (estimated_x, estimated_y)
            
            return None
            
        except Exception as e:
            print(f"❌ Position finding error: {e}")
            return None
    
    def _detect_red_dots(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Detect red notification dots using color-based detection"""
        try:
            # Load screenshot
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"❌ Failed to load screenshot for red dot detection")
                return False, None
            
            # Try different red dot color variants
            for color_name, target_color in self.red_dot_colors.items():
                coords = self._find_red_dots_with_color(image, target_color)
                if coords:
                    print(f"🔴 Red dot found using {color_name} color at {coords}")
                    return True, coords
            
            return False, None
            
        except Exception as e:
            print(f"❌ Red dot detection error: {e}")
            return False, None
    
    def _find_red_dots_with_color(self, image, target_color: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find red dots using specific color parameters"""
        try:
            # Define search region
            x_start, y_start, x_width, y_height = self.red_dot_region
            
            # Create coordinate grids for efficient processing
            x_coords, y_coords = np.meshgrid(
                np.arange(image.shape[1]),
                np.arange(image.shape[0])
            )
            
            # Color matching with tolerance
            lower_bound = target_color - self.color_tolerance
            upper_bound = target_color + self.color_tolerance
            color_mask = np.all((lower_bound <= image) & (image <= upper_bound), axis=-1)
            
            # Region filtering
            region_mask = ((x_coords >= x_start) & (x_coords <= x_start + x_width) &
                          (y_coords >= y_start) & (y_coords <= y_start + y_height))
            
            # Find matching points
            matched_points = np.column_stack((
                x_coords[color_mask & region_mask],
                y_coords[color_mask & region_mask]
            ))
            
            if matched_points.size > 0:
                # Return the bottommost red dot (most recent message)
                sorted_points = matched_points[np.argsort(-matched_points[:, 1])]
                coords = sorted_points[0].astype(int)
                return (int(coords[0]), int(coords[1]))
            
            return None
            
        except Exception as e:
            print(f"❌ Color matching error: {e}")
            return None
    
    def click_message_coordinates(self, coordinates: Tuple[int, int], delay_range: Tuple[float, float] = (0.2, 0.5)) -> bool:
        """Click on detected message coordinates with natural timing"""
        try:
            x, y = coordinates
            
            # Generate random delay for natural behavior
            import random
            delay = random.uniform(delay_range[0], delay_range[1])
            
            print(f"🖱️  Clicking message at ({x}, {y}) with {delay:.2f}s delay")
            
            # Move and click
            pyautogui.moveTo(x, y, duration=delay)
            time.sleep(0.1)  # Small pause before click
            pyautogui.click()
            
            print(f"✅ Successfully clicked message coordinates")
            return True
            
        except Exception as e:
            print(f"❌ Click error: {e}")
            return False
    
    def _log_detection(self, method: str, coordinates: Tuple[int, int], screenshot_path: str):
        """Log detection events for analysis and debugging"""
        try:
            # Convert numpy types to Python native types for JSON serialization
            coords = (int(coordinates[0]), int(coordinates[1]))
            
            detection_event = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'coordinates': coords,
                'screenshot': os.path.basename(screenshot_path),
                'system': self.system
            }
            
            self.detection_history.append(detection_event)
            
            # Save to file
            log_path = os.path.join(self.output_dir, "detection_log.json")
            with open(log_path, 'w') as f:
                json.dump(self.detection_history, f, indent=2)
                
        except Exception as e:
            print(f"⚠️ Logging error: {e}")
    
    def get_detection_stats(self) -> Dict:
        """Get statistics about detection performance"""
        if not self.detection_history:
            return {'total_detections': 0}
        
        methods = {}
        for event in self.detection_history:
            method = event['method']
            methods[method] = methods.get(method, 0) + 1
        
        return {
            'total_detections': len(self.detection_history),
            'methods_used': methods,
            'latest_detection': self.detection_history[-1] if self.detection_history else None
        }
    
    def test_detection_sequence(self, screenshot_path: str) -> bool:
        """Run complete detection test sequence"""
        print("\n" + "="*50)
        print("🧪 MESSAGE DETECTION TEST")
        print("="*50)
        
        if not os.path.exists(screenshot_path):
            print(f"❌ Screenshot not found: {screenshot_path}")
            return False
        
        # Test detection methods
        print(f"\n1. Testing detection on: {os.path.basename(screenshot_path)}")
        detected, coords, method = self.detect_new_messages(screenshot_path)
        
        if detected and coords:
            print(f"\n✅ MESSAGE DETECTED")
            print(f"Method: {method}")
            print(f"Coordinates: {coords}")
            
            # Test clicking (with safety check)
            print(f"\n2. Testing click functionality...")
            # Note: In test mode, we might want to skip actual clicking
            print(f"🖱️  Would click at coordinates: {coords}")
            
            # Show stats
            print(f"\n3. Detection Statistics:")
            stats = self.get_detection_stats()
            for key, value in stats.items():
                print(f"  {key}: {value}")
            
            return True
        else:
            print(f"\n❌ No messages detected")
            return False


def detect_new_messages_integrated(screenshot_path: str) -> Tuple[Optional[Tuple[int, int]], str]:
    """
    Integration function for main bot - maintains compatibility
    
    Returns:
        (coordinates, method) where coordinates is (x, y) if found, method is detection type
    """
    global _global_detector
    if '_global_detector' not in globals():
        _global_detector = MessageDetector()
    
    detected, coords, method = _global_detector.detect_new_messages(screenshot_path)
    
    if detected and coords:
        return coords, method
    else:
        return (None, None), 'none'


def main():
    """Test the message detection module"""
    print("🚀 Message Detection Module Test")
    
    # Create detector instance
    detector = MessageDetector()
    
    # Look for recent screenshots to test
    screenshot_dirs = [
        "pic/screenshots",
        "../pic/screenshots", 
        "TestRun/screenshots"
    ]
    
    test_screenshot = None
    for dir_path in screenshot_dirs:
        if os.path.exists(dir_path):
            screenshots = [f for f in os.listdir(dir_path) if f.endswith('.png')]
            if screenshots:
                latest = max(screenshots, key=lambda f: os.path.getmtime(os.path.join(dir_path, f)))
                test_screenshot = os.path.join(dir_path, latest)
                break
    
    if test_screenshot:
        success = detector.test_detection_sequence(test_screenshot)
        if success:
            print(f"\n🎉 Detection module ready for integration!")
        else:
            print(f"\n💡 Tips for better detection:")
            print(f"   • Ensure WeChat has visible new message indicators")
            print(f"   • Check that contact list area is visible")
            print(f"   • Verify red notification dots are present")
    else:
        print(f"\n⚠️ No screenshots found for testing")
        print(f"   Searched in: {screenshot_dirs}")


if __name__ == "__main__":
    main()