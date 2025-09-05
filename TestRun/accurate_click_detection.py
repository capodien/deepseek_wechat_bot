#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Click Coordinate Detection Module
Improves accuracy by aligning click coordinates to contact row centers
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

class AccurateClickDetector:
    """Enhanced message detection with precise click coordinate calculation"""
    
    def __init__(self, output_dir: str = "TestRun/message_detection"):
        self.system = platform.system()
        self.output_dir = output_dir
        self.ocr_reader = None
        
        # Detection state
        self.previous_text_hash = None
        self.detection_history = []
        
        # Enhanced region definitions - will be auto-calibrated
        self.contact_region = (60, 100, 320, 800)  # Contact list area
        self.red_dot_region = (60, 100, 380, 800)  # Red dot search area
        
        # Contact list structure parameters
        self.contact_row_height = 75  # Approximate row height in WeChat
        self.contact_text_width = 280  # Width for contact text area
        self.click_offset_x = 200  # X offset from left edge for optimal clicking
        
        # Red dot detection parameters
        self.red_dot_colors = {
            'primary': np.array([84, 98, 227]),      
            'windows': np.array([81, 81, 255]),       
            'fallback': np.array([80, 100, 230])     
        }
        self.color_tolerance = 15
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        print(f"🎯 Accurate Click Detector initialized")
        print(f"📁 Output: {self.output_dir}")
        print(f"🖥️  System: {self.system}")
    
    def detect_new_messages(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]], str]:
        """Enhanced detection with accurate click coordinates"""
        try:
            print(f"\n🔍 Starting accurate message detection on: {screenshot_path}")
            
            if not os.path.exists(screenshot_path):
                print(f"❌ Screenshot not found: {screenshot_path}")
                return False, None, 'file_not_found'
            
            # Try red dot detection first (more reliable for new messages)
            print("🔴 Attempting red dot detection...")
            has_red_dot, red_coords = self._detect_red_dots_accurate(screenshot_path)
            
            if has_red_dot and red_coords:
                # Convert red dot pixel to accurate contact row center
                accurate_coords = self._align_to_contact_row(red_coords, screenshot_path)
                if accurate_coords:
                    print(f"✅ Red dot detected and aligned: {red_coords} → {accurate_coords}")
                    self._log_detection('red_dot_aligned', accurate_coords, screenshot_path)
                    return True, accurate_coords, 'red_dot_aligned'
            
            # Fallback to text change detection
            print("📝 Attempting text change detection...")
            has_text_change, text_coords = self._detect_text_changes_accurate(screenshot_path)
            
            if has_text_change and text_coords:
                print(f"✅ Text change detected: {text_coords}")
                self._log_detection('text_change', text_coords, screenshot_path)
                return True, text_coords, 'text_change'
            
            print("ℹ️  No new messages detected")
            return False, None, 'none'
                
        except Exception as e:
            print(f"❌ Message detection error: {e}")
            return False, None, 'error'
    
    def _align_to_contact_row(self, red_dot_coords: Tuple[int, int], screenshot_path: str) -> Optional[Tuple[int, int]]:
        """Convert red dot coordinates to accurate contact row center"""
        try:
            x_red, y_red = red_dot_coords
            
            # Load screenshot for contact structure analysis
            image = cv2.imread(screenshot_path)
            if image is None:
                return None
            
            # Define contact list boundaries
            contact_x, contact_y, contact_w, contact_h = self.contact_region
            
            # Calculate which contact row this red dot belongs to
            relative_y = y_red - contact_y  # Y position relative to contact list start
            row_index = max(0, int(relative_y / self.contact_row_height))
            
            # Calculate accurate click coordinates for this contact row
            click_x = contact_x + self.click_offset_x  # Centered in contact text area
            click_y = contact_y + (row_index * self.contact_row_height) + (self.contact_row_height // 2)
            
            # Validate coordinates are within reasonable bounds
            max_y = contact_y + contact_h
            if click_y > max_y:
                click_y = max_y - (self.contact_row_height // 2)
            
            # Ensure minimum Y coordinate
            min_y = contact_y + (self.contact_row_height // 2)
            if click_y < min_y:
                click_y = min_y
            
            print(f"🎯 Row alignment: Red dot ({x_red}, {y_red}) → Row {row_index} → Click ({click_x}, {click_y})")
            
            return (click_x, click_y)
            
        except Exception as e:
            print(f"❌ Row alignment error: {e}")
            return red_dot_coords  # Return original coordinates as fallback
    
    def _detect_red_dots_accurate(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Enhanced red dot detection with improved accuracy"""
        try:
            image = cv2.imread(screenshot_path)
            if image is None:
                return False, None
            
            # Try each red dot color variant
            for color_name, target_color in self.red_dot_colors.items():
                coords = self._find_red_dots_with_clustering(image, target_color)
                if coords:
                    print(f"🔴 Red dot found using {color_name} at {coords}")
                    return True, coords
            
            return False, None
            
        except Exception as e:
            print(f"❌ Red dot detection error: {e}")
            return False, None
    
    def _find_red_dots_with_clustering(self, image, target_color: np.ndarray) -> Optional[Tuple[int, int]]:
        """Find red dots using clustering to improve accuracy"""
        try:
            x_start, y_start, x_width, y_height = self.red_dot_region
            
            # Create coordinate grids
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
            
            if matched_points.size == 0:
                return None
            
            # Cluster nearby points to find distinct red dots
            clusters = self._cluster_red_dots(matched_points)
            
            if not clusters:
                return None
            
            # Return the bottommost cluster (most recent message)
            bottommost_cluster = max(clusters, key=lambda c: c[1])
            return (int(bottommost_cluster[0]), int(bottommost_cluster[1]))
            
        except Exception as e:
            print(f"❌ Red dot clustering error: {e}")
            return None
    
    def _cluster_red_dots(self, points: np.ndarray, cluster_distance: int = 20) -> List[Tuple[int, int]]:
        """Cluster nearby red dot pixels to find distinct notification dots"""
        if len(points) == 0:
            return []
        
        clusters = []
        processed = set()
        
        for i, point in enumerate(points):
            if i in processed:
                continue
            
            # Find all points within cluster distance
            distances = np.sqrt(np.sum((points - point) ** 2, axis=1))
            cluster_indices = np.where(distances <= cluster_distance)[0]
            
            # Mark as processed
            for idx in cluster_indices:
                processed.add(idx)
            
            # Calculate cluster center
            cluster_points = points[cluster_indices]
            center_x = int(np.mean(cluster_points[:, 0]))
            center_y = int(np.mean(cluster_points[:, 1]))
            
            clusters.append((center_x, center_y))
        
        return clusters
    
    def _detect_text_changes_accurate(self, screenshot_path: str) -> Tuple[bool, Optional[Tuple[int, int]]]:
        """Enhanced text change detection with accurate positioning"""
        try:
            image = cv2.imread(screenshot_path)
            if image is None:
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
                # Find accurate click position for the changed contact
                coords = self._find_target_contact_position_accurate(image, text_content)
                if coords:
                    return True, coords
            
            return False, None
            
        except Exception as e:
            print(f"❌ Text change detection error: {e}")
            return False, None
    
    def _find_target_contact_position_accurate(self, image, text_content: List[str]) -> Optional[Tuple[int, int]]:
        """Find accurate click position using OCR bounding boxes"""
        try:
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
                        # Get bounding box and calculate accurate click position
                        bbox = result[0]
                        
                        # Use contact row center instead of text center
                        text_center_y = int((bbox[0][1] + bbox[2][1]) / 2) + y
                        
                        # Align to contact row
                        relative_y = text_center_y - y
                        row_index = max(0, int(relative_y / self.contact_row_height))
                        
                        click_x = x + self.click_offset_x
                        click_y = y + (row_index * self.contact_row_height) + (self.contact_row_height // 2)
                        
                        print(f"🎯 Target contact '{target}' aligned to row {row_index}: ({click_x}, {click_y})")
                        return (click_x, click_y)
            
            return None
            
        except Exception as e:
            print(f"❌ Position finding error: {e}")
            return None
    
    def _extract_contact_text(self, image) -> List[str]:
        """Extract text from contact list region using OCR"""
        try:
            x, y, w, h = self.contact_region
            cropped = image[y:y+h, x:x+w]
            
            reader = self._get_ocr_reader()
            results = reader.readtext(cropped)
            
            text_content = []
            for result in results:
                text = result[1].strip()
                if text and len(text) > 1:
                    text_content.append(text)
            
            return text_content
            
        except Exception as e:
            print(f"❌ Text extraction error: {e}")
            return []
    
    def _get_ocr_reader(self):
        """Get OCR reader with lazy initialization"""
        if self.ocr_reader is None:
            print("🔍 Initializing OCR reader...")
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        return self.ocr_reader
    
    def click_message_coordinates(self, coordinates: Tuple[int, int], delay_range: Tuple[float, float] = (0.2, 0.5)) -> bool:
        """Click on detected message coordinates with natural timing"""
        try:
            x, y = coordinates
            
            import random
            delay = random.uniform(delay_range[0], delay_range[1])
            
            print(f"🖱️  Clicking message at ({x}, {y}) with {delay:.2f}s delay")
            
            # Move and click with smooth motion
            pyautogui.moveTo(x, y, duration=delay)
            time.sleep(0.1)
            pyautogui.click()
            
            print(f"✅ Successfully clicked message coordinates")
            return True
            
        except Exception as e:
            print(f"❌ Click error: {e}")
            return False
    
    def calibrate_regions(self, screenshot_path: str) -> Dict[str, Tuple[int, int, int, int]]:
        """Auto-calibrate detection regions based on actual WeChat window"""
        try:
            print(f"🔧 Calibrating detection regions from: {screenshot_path}")
            
            # This would analyze the screenshot to detect contact list bounds
            # For now, return current regions
            return {
                'contact_region': self.contact_region,
                'red_dot_region': self.red_dot_region
            }
            
        except Exception as e:
            print(f"❌ Calibration error: {e}")
            return {}
    
    def _log_detection(self, method: str, coordinates: Tuple[int, int], screenshot_path: str):
        """Log detection events for analysis"""
        try:
            coords = (int(coordinates[0]), int(coordinates[1]))
            
            detection_event = {
                'timestamp': datetime.now().isoformat(),
                'method': method,
                'coordinates': coords,
                'screenshot': os.path.basename(screenshot_path),
                'system': self.system
            }
            
            # Load existing log
            log_path = os.path.join(self.output_dir, 'accurate_detection_log.json')
            log_data = []
            
            if os.path.exists(log_path):
                with open(log_path, 'r', encoding='utf-8') as f:
                    log_data = json.load(f)
            
            # Add new event
            log_data.append(detection_event)
            
            # Keep only last 50 events
            log_data = log_data[-50:]
            
            # Save updated log
            with open(log_path, 'w', encoding='utf-8') as f:
                json.dump(log_data, f, indent=2, ensure_ascii=False)
            
            print(f"📝 Detection logged: {method} at {coords}")
            
        except Exception as e:
            print(f"❌ Logging error: {e}")


if __name__ == '__main__':
    """Test the accurate click detector"""
    detector = AccurateClickDetector()
    
    # Find latest screenshot
    screenshot_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest_screenshot = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"🧪 Testing accurate detection on: {latest_screenshot}")
        
        has_message, coords, method = detector.detect_new_messages(screenshot_path)
        
        if has_message and coords:
            print(f"✅ Detection successful: {method} at {coords}")
            
            # Optionally test click (uncomment to actually click)
            # detector.click_message_coordinates(coords)
        else:
            print(f"ℹ️  No messages detected using method: {method}")
    else:
        print("❌ No test screenshots found")