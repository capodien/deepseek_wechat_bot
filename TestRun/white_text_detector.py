#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
White Text Detector for Username Extraction
Uses color-based detection to find white text (usernames) in WeChat message cards
"""

import cv2
import numpy as np
import easyocr
from typing import List, Dict, Tuple, Optional
from datetime import datetime
import os

class WhiteTextDetector:
    """Detect and extract white text (usernames) from message cards"""
    
    def __init__(self):
        # Initialize OCR reader
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        
        # White text detection parameters (more flexible)
        self.WHITE_THRESHOLD_MIN = 180  # Lower minimum brightness for "white" text
        self.WHITE_THRESHOLD_MAX = 255  # Maximum brightness
        
        # Morphological operations parameters
        self.MORPH_KERNEL_SIZE = (2, 2)  # Smaller kernel to preserve text details
        self.MIN_TEXT_AREA = 50   # Lower minimum area for text regions
        self.MAX_TEXT_AREA = 8000  # Higher maximum area for text regions
        
        # OCR parameters
        self.MIN_CONFIDENCE = 0.3
        self.MAX_USERNAME_LENGTH = 30
        
        print("✅ WhiteTextDetector initialized")
    
    def detect_white_text_in_card(self, img: np.ndarray, avatar_bounds: Tuple) -> Dict:
        """
        Detect white text (username) in a specific message card
        
        Args:
            img: Screenshot image
            avatar_bounds: (x, y, w, h) of avatar
            
        Returns:
            Dict with username detection results
        """
        try:
            x, y, w, h = avatar_bounds
            
            # Define search region to the right of avatar
            search_x = x + w + 5  # Start 5px right of avatar
            search_y = y
            search_w = min(300, img.shape[1] - search_x)  # Search up to 300px right
            search_h = int(h * 0.7)  # Use top 70% of card for username area
            
            if search_x >= img.shape[1] or search_w <= 0 or search_h <= 0:
                return {
                    'success': False,
                    'error': 'Invalid search region bounds',
                    'username': 'BOUNDS_ERROR'
                }
            
            # Extract search region
            search_region = img[search_y:search_y+search_h, search_x:search_x+search_w]
            
            # Detect white text areas
            white_mask = self._create_white_text_mask(search_region)
            
            if white_mask is None:
                return {
                    'success': False,
                    'error': 'Failed to create white text mask',
                    'username': 'MASK_ERROR'
                }
            
            # Find white text contours
            text_regions = self._find_text_regions(white_mask, search_region)
            
            if not text_regions:
                return {
                    'success': False,
                    'error': 'No white text regions found',
                    'username': 'NO_WHITE_TEXT',
                    'debug_info': {
                        'search_region': (search_x, search_y, search_w, search_h),
                        'white_pixels': np.sum(white_mask > 0)
                    }
                }
            
            # Perform OCR on white text regions
            username_result = self._extract_username_from_regions(text_regions, search_region)
            
            # Add region info for debugging
            username_result['search_region'] = (search_x, search_y, search_w, search_h)
            username_result['white_regions_count'] = len(text_regions)
            
            return username_result
            
        except Exception as e:
            return {
                'success': False,
                'error': f'White text detection error: {str(e)}',
                'username': 'DETECTION_ERROR'
            }
    
    def _create_white_text_mask(self, region: np.ndarray) -> Optional[np.ndarray]:
        """
        Create binary mask for white text detection
        """
        try:
            if len(region.shape) == 3:
                # Convert to grayscale
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()
            
            # Create binary mask for white pixels
            white_mask = cv2.threshold(gray, self.WHITE_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)[1]
            
            # Apply morphological operations to clean up text regions
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
            
            # Close small gaps in characters
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
            
            # Remove small noise
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_OPEN, kernel)
            
            return white_mask
            
        except Exception as e:
            print(f"⚠️ White mask creation error: {e}")
            return None
    
    def _find_text_regions(self, white_mask: np.ndarray, original_region: np.ndarray) -> List[Dict]:
        """
        Find potential text regions from white mask
        """
        text_regions = []
        
        try:
            # Find contours in white mask
            contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for i, contour in enumerate(contours):
                # Calculate contour properties
                area = cv2.contourArea(contour)
                
                # Filter by area (text should be reasonable size)
                if self.MIN_TEXT_AREA <= area <= self.MAX_TEXT_AREA:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    
                    # Filter by aspect ratio (text should be wider than tall, but be more flexible)
                    aspect_ratio = w / h if h > 0 else 0
                    if 0.5 <= aspect_ratio <= 20.0:  # More flexible text aspect ratios
                        
                        # Extract text region
                        text_roi = original_region[y:y+h, x:x+w]
                        
                        text_regions.append({
                            'region_id': i,
                            'bounds': (x, y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio,
                            'text_roi': text_roi,
                            'position_score': self._calculate_text_position_score(x, y, w, h)
                        })
            
            # Sort by position score (prefer left-side, top regions for usernames)
            text_regions.sort(key=lambda r: r['position_score'], reverse=True)
            
            return text_regions
            
        except Exception as e:
            print(f"⚠️ Text region detection error: {e}")
            return []
    
    def _calculate_text_position_score(self, x: int, y: int, w: int, h: int) -> float:
        """
        Calculate position-based score for text region
        Higher score for regions likely to contain usernames
        """
        score = 1.0
        
        # Prefer text near the left side (usernames start near avatar)
        if x <= 50:
            score += 0.5
        elif x <= 100:
            score += 0.3
        
        # Prefer text near the top (usernames are in top line)
        if y <= 20:
            score += 0.4
        elif y <= 40:
            score += 0.2
        
        # Prefer reasonable text sizes
        if 80 <= w <= 200 and 12 <= h <= 30:
            score += 0.3
        
        # Prefer wider text (usernames are typically longer)
        aspect_ratio = w / h if h > 0 else 0
        if 3.0 <= aspect_ratio <= 10.0:
            score += 0.2
        
        return score
    
    def _extract_username_from_regions(self, text_regions: List[Dict], search_region: np.ndarray) -> Dict:
        """
        Extract username text from detected white text regions
        """
        if not text_regions:
            return {
                'success': False,
                'username': 'NO_REGIONS',
                'confidence': 0.0,
                'method': 'white_text_detection'
            }
        
        # Try OCR on each region, starting with highest score
        for region_data in text_regions:
            try:
                text_roi = region_data['text_roi']
                
                # Preprocess region for better OCR
                processed_roi = self._preprocess_text_roi(text_roi)
                
                if processed_roi is None or processed_roi.size == 0:
                    continue
                
                # Perform OCR
                ocr_results = self.ocr_reader.readtext(processed_roi, detail=True, paragraph=False)
                
                # Process OCR results
                for result in ocr_results:
                    bbox, text, confidence = result
                    
                    # Clean and validate text
                    cleaned_text = self._clean_username_text(text)
                    
                    if (confidence >= self.MIN_CONFIDENCE and 
                        len(cleaned_text) > 0 and 
                        len(cleaned_text) <= self.MAX_USERNAME_LENGTH):
                        
                        return {
                            'success': True,
                            'username': cleaned_text,
                            'confidence': confidence,
                            'method': 'white_text_detection',
                            'region_info': {
                                'bounds': region_data['bounds'],
                                'area': region_data['area'],
                                'aspect_ratio': region_data['aspect_ratio'],
                                'position_score': region_data['position_score']
                            }
                        }
                
            except Exception as e:
                print(f"⚠️ OCR error on region: {e}")
                continue
        
        return {
            'success': False,
            'username': 'OCR_FAILED',
            'confidence': 0.0,
            'method': 'white_text_detection',
            'regions_tried': len(text_regions)
        }
    
    def _preprocess_text_roi(self, text_roi: np.ndarray) -> Optional[np.ndarray]:
        """
        Preprocess text region for better OCR results
        """
        try:
            if text_roi.size == 0:
                return None
            
            # Convert to grayscale if needed
            if len(text_roi.shape) == 3:
                gray = cv2.cvtColor(text_roi, cv2.COLOR_BGR2GRAY)
            else:
                gray = text_roi.copy()
            
            # Enhance contrast for white text
            processed = cv2.convertScaleAbs(gray, alpha=1.3, beta=20)
            
            # Apply slight blur to smooth text edges
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
            return processed
            
        except Exception as e:
            print(f"⚠️ Preprocessing error: {e}")
            return text_roi
    
    def _clean_username_text(self, text: str) -> str:
        """
        Clean extracted text to get better username
        """
        if not text:
            return ""
        
        # Remove extra whitespace
        cleaned = text.strip()
        cleaned = ' '.join(cleaned.split())
        
        # Remove common OCR artifacts
        artifacts = ['|', '_', '-', '=', '+', '*', '#', '~']
        for artifact in artifacts:
            if cleaned.count(artifact) > len(cleaned) // 3:
                cleaned = cleaned.replace(artifact, '')
        
        return cleaned.strip()
    
    def create_white_text_visualization(self, img_path: str, avatars: List[Dict], 
                                       detection_results: List[Dict], 
                                       output_path: str = None) -> str:
        """
        Create visualization showing white text detection results
        """
        try:
            img = cv2.imread(img_path)
            if img is None:
                return None
            
            result = img.copy()
            
            # Draw detection results for each avatar
            for i, (avatar, detection) in enumerate(zip(avatars, detection_results)):
                avatar_bounds = avatar['card_bounds']
                x, y, w, h = avatar_bounds
                
                # Draw avatar bounds (blue)
                cv2.rectangle(result, (x, y), (x + w, y + h), (255, 0, 0), 2)
                
                # Draw search region if available
                if 'search_region' in detection:
                    sx, sy, sw, sh = detection['search_region']
                    cv2.rectangle(result, (sx, sy), (sx + sw, sy + sh), (0, 165, 255), 2)  # Orange
                
                # Draw detected text region if available
                if detection.get('success') and 'region_info' in detection:
                    region_bounds = detection['region_info']['bounds']
                    rx, ry, rw, rh = region_bounds
                    
                    # Adjust coordinates to global image coordinates
                    global_rx = sx + rx
                    global_ry = sy + ry
                    
                    # Draw detected text region (green)
                    cv2.rectangle(result, (global_rx, global_ry), 
                                 (global_rx + rw, global_ry + rh), (0, 255, 0), 2)
                
                # Add text labels
                username = detection.get('username', 'FAILED')
                confidence = detection.get('confidence', 0.0)
                
                # Choose color based on success
                if detection.get('success', False):
                    text_color = (0, 255, 0)  # Green for success
                else:
                    text_color = (0, 0, 255)  # Red for failure
                
                # Add username label
                label_y = y - 10 if y > 20 else y + h + 20
                cv2.putText(result, f"#{i+1}: {username} ({confidence:.2f})", 
                           (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
            
            # Add title and legend
            title = f"White Text Detection: {sum(1 for d in detection_results if d.get('success', False))}/{len(detection_results)} Success"
            cv2.rectangle(result, (10, 10), (700, 100), (255, 255, 255), -1)
            cv2.rectangle(result, (10, 10), (700, 100), (0, 150, 255), 2)
            cv2.putText(result, title, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
            
            # Add legend
            legend_items = [
                "Blue Box = Avatar Region",
                "Orange Box = White Text Search Region", 
                "Green Box = Detected White Text",
                "Green Text = Successful Detection",
                "Red Text = Failed Detection"
            ]
            
            for i, item in enumerate(legend_items):
                cv2.putText(result, item, (20, 55 + i * 12),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Save result
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"pic/screenshots/white_text_detection_{timestamp}.png"
            
            cv2.imwrite(output_path, result)
            print(f"✅ White text detection visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None


def test_white_text_detection():
    """Test white text detection with latest screenshot"""
    import sys
    
    # Add current directory to path
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
    
    print(f"🔍 Testing white text detection with: {latest_screenshot}")
    
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
    
    # Test white text detection
    white_detector = WhiteTextDetector()
    detection_results = []
    
    print(f"\n🔤 White Text Detection Results:")
    for i, avatar in enumerate(avatars, 1):
        result = white_detector.detect_white_text_in_card(img, avatar['card_bounds'])
        detection_results.append(result)
        
        if result['success']:
            print(f"  ✅ Avatar #{i}: '{result['username']}' (confidence: {result['confidence']:.2f})")
            if 'region_info' in result:
                region = result['region_info']
                print(f"      📍 Region: {region['bounds']}, Score: {region['position_score']:.2f}")
        else:
            print(f"  ❌ Avatar #{i}: {result.get('error', 'Unknown error')}")
            if 'debug_info' in result:
                debug = result['debug_info']
                print(f"      🔍 Debug: white_pixels={debug.get('white_pixels', 0)}")
    
    # Create visualization
    visualization_path = white_detector.create_white_text_visualization(
        screenshot_path, avatars, detection_results
    )
    
    success_count = sum(1 for r in detection_results if r.get('success', False))
    print(f"\n📊 White Text Detection Summary:")
    print(f"   Success Rate: {success_count}/{len(avatars)} ({success_count/len(avatars)*100:.1f}%)")
    if visualization_path:
        print(f"   Visualization: {visualization_path}")


if __name__ == "__main__":
    test_white_text_detection()