#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simple Dynamic Width Detector
Focused ONLY on determining the width of WeChat message cards
No avatars, no OCR zones, no complex features - just width detection
"""

import cv2
import numpy as np
import os
from typing import Optional, Tuple
from datetime import datetime

def find_vertical_edge_x(img, x0=0, x1=None, y0=0, y1=None, rightmost=True):
    """
    Return the x (in original image coords) of the dominant vertical edge inside ROI.
    Works on narrow strips like your screenshot.
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    H, W = gray.shape
    x1 = W if x1 is None else x1
    y1 = H if y1 is None else y1
    roi = gray[y0:y1, x0:x1]

    # Optional denoise/normalize for dark UIs
    roi = cv2.bilateralFilter(roi, d=5, sigmaColor=25, sigmaSpace=25)

    # Compute horizontal gradient (vertical edges) → 1D profile
    # (Sobel is a bit smoother than simple diff)
    sobelx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
    prof = np.mean(np.abs(sobelx), axis=0)  # average over rows → shape (roiW,)

    # Smooth & pick peak
    prof = cv2.GaussianBlur(prof.reshape(1,-1), (1,7), 0).ravel()
    if rightmost:
        idx = int(np.argmax(prof[::-1]))          # strongest from right
        xr  = (x1 - 1) - idx
    else:
        xr  = int(np.argmax(prof)) + x0

    # Confidence (0–1): peak vs neighborhood
    peak = prof[(xr - x0)]
    med  = float(np.median(prof))
    mad  = float(np.median(np.abs(prof - med)) + 1e-6)
    conf = max(0.0, min(1.0, (peak - med) / (6*mad)))  # rough score

    return xr, conf, prof

class SimpleWidthDetector:
    """Simple width detector focused ONLY on finding message card width"""
    
    def __init__(self):
        # Simple parameters for width detection only
        self.CONVERSATION_WIDTH_RATIO = 0.65  # Focus on left 65% of screen where cards are
        self.EDGE_THRESHOLD_LOW = 30
        self.EDGE_THRESHOLD_HIGH = 100
        
    def detect_width(self, image_path: str) -> Optional[Tuple[int, int, int]]:
        """
        Detect the width of WeChat message cards
        Returns: (left_boundary, right_boundary, width) or None if failed
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        print(f"🎯 Simple Width Detection: {os.path.basename(image_path)}")
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Focus on conversation area (left side where message cards are)
        img_height, img_width = img.shape[:2]
        conversation_width = int(img_width * self.CONVERSATION_WIDTH_RATIO)
        conversation_area = img[:, :conversation_width]
        
        print(f"  💬 Conversation area: {conversation_area.shape[1]}x{conversation_area.shape[0]}")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(conversation_area, cv2.COLOR_BGR2GRAY)
        
        # Apply gentle blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Edge detection to find card boundaries
        edges = cv2.Canny(blurred, self.EDGE_THRESHOLD_LOW, self.EDGE_THRESHOLD_HIGH)
        
        # Find vertical edges by summing horizontally
        # This gives us the strength of vertical edges at each x position
        vertical_edge_strength = np.sum(edges, axis=0)
        
        # Find the left boundary using sophisticated edge detection
        left_boundary = self._find_left_boundary_edge_based(conversation_area)
        
        # Find the right boundary (last significant edge before conversation ends)
        right_boundary = self._find_right_boundary(vertical_edge_strength, conversation_width)
        
        if left_boundary is None or right_boundary is None:
            print("❌ Could not detect card boundaries")
            return None
        
        width = right_boundary - left_boundary
        
        print(f"  ✅ Width detected: {width}px (left: {left_boundary}, right: {right_boundary})")
        
        return left_boundary, right_boundary, width
    
    def _find_left_boundary_edge_based(self, conversation_area: np.ndarray) -> Optional[int]:
        """Find the left boundary using sophisticated vertical edge detection"""
        print(f"  🕸️ Edge-based left boundary detection - looking for sidebar boundary")
        
        try:
            # Get the complete edge profile to analyze all peaks
            left_x, confidence, profile = find_vertical_edge_x(
                img=conversation_area,
                x0=0,              # Start from left edge
                x1=100,            # Only search first 100px (sidebar region)
                y0=0,              # Full height
                y1=None,           # Full height
                rightmost=False    # Find leftmost edge (sidebar boundary)
            )
            
            print(f"  📊 Strongest edge: x={left_x}px, confidence={confidence:.3f}")
            
            # Find ALL significant peaks in the profile, not just the strongest
            profile_peaks = []
            mean_profile = np.mean(profile)
            threshold = mean_profile * 1.2  # Lower threshold to catch weaker edges
            
            for i in range(1, len(profile) - 1):
                if (profile[i] > profile[i-1] and profile[i] > profile[i+1] and 
                    profile[i] > threshold):
                    # Calculate local confidence for this peak
                    local_region = profile[max(0, i-5):min(len(profile), i+6)]
                    local_med = np.median(local_region)
                    local_mad = np.median(np.abs(local_region - local_med)) + 1e-6
                    local_conf = (profile[i] - local_med) / (6 * local_mad)
                    
                    profile_peaks.append((i, profile[i], local_conf))
            
            # Sort peaks by position (left to right)
            profile_peaks.sort(key=lambda x: x[0])
            
            print(f"  📈 Found {len(profile_peaks)} significant edges:")
            for i, (pos, strength, conf) in enumerate(profile_peaks):
                print(f"    Edge {i+1}: x={pos}px, strength={strength:.0f}, conf={conf:.3f}")
            
            # Strategy: Use strongest edge and subtract 8px for actual sidebar boundary
            # Strongest edge is likely in message content, so subtract 8px to get true boundary
            if left_x is not None:
                # Apply 8px offset to get actual sidebar boundary
                actual_boundary = left_x - 8
                print(f"  ✅ Strongest edge at {left_x}px, applying -8px offset")
                print(f"  ✅ Actual sidebar boundary: {actual_boundary}px")
                return actual_boundary
            
            # Fallback: Use the original strongest edge if no reasonable alternatives
            if left_x >= 20:
                print(f"  ⚠️ Using strongest edge as fallback: {left_x}px")
                return left_x
            else:
                print(f"  ❌ No suitable edge found, using default")
                return 60  # Default fallback
        
        except Exception as e:
            print(f"  ❌ Edge detection failed: {e}")
            return 60  # Default fallback
    
    def _find_right_boundary(self, vertical_edge_strength: np.ndarray, max_width: int) -> Optional[int]:
        """Find the right boundary of message cards - restored simple version"""
        # Look for the rightmost significant vertical edge
        # This should be the right edge of the message cards
        
        # Calculate threshold as percentage of max edge strength
        max_strength = np.max(vertical_edge_strength)
        threshold = max_strength * 0.2  # 20% of max strength (restored from working version)
        
        # Find last position where edge strength exceeds threshold
        for x in range(len(vertical_edge_strength) - 1, -1, -1):
            if vertical_edge_strength[x] > threshold:
                return min(x, max_width - 1)
        
        return None
    
    def get_latest_screenshot(self, screenshot_dir: str = "pic/screenshots") -> Optional[str]:
        """
        Find the latest WeChat screenshot in the specified directory
        
        Args:
            screenshot_dir: Directory to search for screenshots
            
        Returns:
            Path to the latest screenshot file, or None if not found
        """
        if not os.path.exists(screenshot_dir):
            print(f"⚠️ Screenshot directory not found: {screenshot_dir}")
            return None
        
        # Look for files matching the WeChat screenshot pattern: YYYYMMDD_HHMMSS_WeChat.png
        screenshot_files = []
        for filename in os.listdir(screenshot_dir):
            if filename.endswith('_WeChat.png') and len(filename) >= 20:  # Minimum length check
                try:
                    # Verify the timestamp format
                    timestamp_part = filename.split('_WeChat.png')[0]
                    if len(timestamp_part) == 15 and timestamp_part.replace('_', '').isdigit():
                        screenshot_files.append(filename)
                except:
                    continue
        
        if not screenshot_files:
            print(f"⚠️ No WeChat screenshots found in {screenshot_dir}")
            return None
        
        # Sort by filename (which sorts by timestamp due to YYYYMMDD_HHMMSS format)
        latest_screenshot = sorted(screenshot_files)[-1]
        latest_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"📸 Using latest screenshot: {latest_screenshot}")
        return latest_path
    
    def create_width_visualization(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Create simple visualization showing ONLY the detected width boundaries
        No avatars, no zones, just two vertical lines and width value
        """
        # Detect width
        detection_result = self.detect_width(image_path)
        if detection_result is None:
            return None
        
        left_boundary, right_boundary, width = detection_result
        
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        result = img.copy()
        img_height = result.shape[0]
        
        # Draw simple vertical lines for boundaries
        cv2.line(result, (left_boundary, 0), (left_boundary, img_height), (0, 255, 0), 2)  # Green left line
        cv2.line(result, (right_boundary, 0), (right_boundary, img_height), (0, 255, 0), 2)  # Green right line
        
        # Add width text at the top
        width_text = f"Width: {width}px"
        cv2.putText(result, width_text, (left_boundary, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Add boundary labels
        cv2.putText(result, f"L:{left_boundary}", (left_boundary + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"R:{right_boundary}", (right_boundary - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_SimpleWidth_{width}px.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Simple width visualization saved: {output_path}")
        
        return os.path.basename(output_path)

if __name__ == "__main__":
    # Simple test - use latest screenshot
    detector = SimpleWidthDetector()
    
    # Get the latest WeChat screenshot
    test_image = detector.get_latest_screenshot()
    
    if test_image:
        print(f"🔍 Testing with: {test_image}")
        result = detector.detect_width(test_image)
        if result:
            left, right, width = result
            print(f"✅ Width: {width}px (boundaries: {left} to {right})")
            
            # Create visualization
            detector.create_width_visualization(test_image)
        else:
            print("❌ Width detection failed")
    else:
        print("❌ No WeChat screenshots found in pic/screenshots/")