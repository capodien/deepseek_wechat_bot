#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Username Extractor for Contact Cards
Extracts usernames from regions next to detected avatars using OCR
"""

import cv2
import numpy as np
import easyocr
import os
from typing import List, Dict, Optional
from datetime import datetime

class UsernameExtractor:
    """Extract usernames from contact card regions using OCR"""
    
    def __init__(self):
        # Initialize EasyOCR reader (reuse existing setup from main bot)
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        
        # Username region parameters (relative to avatar)
        self.USERNAME_OFFSET_X = 10    # Pixels right of avatar
        self.USERNAME_WIDTH = 200      # Default width (will be adaptive)
        self.USERNAME_HEIGHT_RATIO = 0.6  # Use top 60% of avatar height for username
        
        # OCR parameters
        self.MIN_CONFIDENCE = 0.3      # Lower threshold for usernames
        self.MAX_USERNAME_LENGTH = 30  # Reasonable username length limit
        
        # Import adaptive width calculator
        try:
            from TestRun.adaptive_width_calculator import AdaptiveWidthCalculator
            self.width_calculator = AdaptiveWidthCalculator()
            self.use_adaptive_width = True
            print("✅ Adaptive width calculation enabled")
        except ImportError:
            self.width_calculator = None
            self.use_adaptive_width = False
            print("⚠️ Using fixed width (adaptive calculator not available)")
        
    def extract_username_from_avatar(self, image_path: str, avatar_info: Dict) -> Dict:
        """
        Extract username from region next to a specific avatar
        
        Args:
            image_path: Path to screenshot
            avatar_info: Avatar detection result from OpenCVAdaptiveDetector
            
        Returns:
            Dict with username extraction results
        """
        try:
            # Load image
            img = cv2.imread(image_path)
            if img is None:
                return {'success': False, 'error': 'Failed to load image'}
            
            # Get avatar bounds
            avatar_bounds = avatar_info['card_bounds']  # (x, y, w, h)
            x, y, w, h = avatar_bounds
            
            # Calculate username region (adaptive or fixed width)
            if self.use_adaptive_width and self.width_calculator:
                regions = self.width_calculator.calculate_adaptive_regions(img, avatar_bounds)
                username_region = regions['username_region']
                username_x = username_region['x']
                username_y = username_region['y'] 
                username_w = username_region['width']
                username_h = username_region['height']
                print(f"  📏 Adaptive width: {username_w}px (method: {regions['adaptive_info']['method']})")
            else:
                # Fallback to fixed width calculation
                username_x = x + w + self.USERNAME_OFFSET_X
                username_y = y
                username_w = min(self.USERNAME_WIDTH, img.shape[1] - username_x)
                username_h = int(h * self.USERNAME_HEIGHT_RATIO)  # Top portion for username
                print(f"  📏 Fixed width: {username_w}px")
            
            # Validate region bounds
            if (username_x >= img.shape[1] or username_y >= img.shape[0] or 
                username_w <= 0 or username_h <= 0):
                return {'success': False, 'error': 'Invalid username region bounds'}
            
            # Extract username region
            username_region = img[username_y:username_y+username_h, username_x:username_x+username_w]
            
            # Preprocess region for better OCR
            processed_region = self.preprocess_username_region(username_region)
            
            # Perform OCR
            ocr_results = self.ocr_reader.readtext(processed_region, detail=True, paragraph=False)
            
            # Process OCR results to extract best username candidate
            username_result = self.process_ocr_results(ocr_results, avatar_info['card_id'])
            
            # Add region info for debugging
            username_result.update({
                'region_bounds': (username_x, username_y, username_w, username_h),
                'avatar_id': avatar_info['card_id']
            })
            
            return username_result
            
        except Exception as e:
            return {
                'success': False, 
                'error': f'Username extraction error: {str(e)}',
                'avatar_id': avatar_info.get('card_id', 'unknown')
            }
    
    def preprocess_username_region(self, region: np.ndarray) -> np.ndarray:
        """
        Preprocess username region for better OCR results
        Similar to theme detection in deal_chatbox.py
        """
        try:
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()
            
            # Detect if region is mostly dark (dark theme) or light (light theme)
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 127:
                # Dark theme: Invert for better text recognition
                processed = cv2.bitwise_not(gray)
            else:
                # Light theme: Use as is
                processed = gray
            
            # Apply slight blur to reduce noise
            processed = cv2.GaussianBlur(processed, (3, 3), 0)
            
            # Enhance contrast
            processed = cv2.convertScaleAbs(processed, alpha=1.2, beta=10)
            
            return processed
            
        except Exception as e:
            print(f"  ⚠️ Preprocessing error: {e}")
            return region
    
    def process_ocr_results(self, ocr_results: List, avatar_id: int) -> Dict:
        """
        Process OCR results to extract the most likely username
        """
        if not ocr_results:
            return {
                'success': False,
                'username': '',
                'confidence': 0.0,
                'method': 'ocr_failed'
            }
        
        # Filter and rank OCR results
        valid_candidates = []
        
        for result in ocr_results:
            bbox, text, confidence = result
            
            # Clean text
            cleaned_text = self.clean_username_text(text)
            
            # Apply filters
            if (confidence >= self.MIN_CONFIDENCE and 
                len(cleaned_text) > 0 and 
                len(cleaned_text) <= self.MAX_USERNAME_LENGTH):
                
                valid_candidates.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox,
                    'position_score': self.calculate_position_score(bbox)
                })
        
        if not valid_candidates:
            return {
                'success': False,
                'username': '',
                'confidence': 0.0,
                'method': 'no_valid_candidates',
                'raw_results_count': len(ocr_results)
            }
        
        # Select best candidate (highest confidence + position score)
        best_candidate = max(valid_candidates, 
                           key=lambda c: c['confidence'] * 0.7 + c['position_score'] * 0.3)
        
        print(f"  📝 Avatar {avatar_id} OCR: '{best_candidate['text']}' (confidence: {best_candidate['confidence']:.2f})")
        
        return {
            'success': True,
            'username': best_candidate['text'],
            'confidence': best_candidate['confidence'],
            'method': 'ocr_success',
            'candidates_count': len(valid_candidates)
        }
    
    def clean_username_text(self, text: str) -> str:
        """Clean extracted text to get better username"""
        if not text:
            return ""
        
        # Remove common OCR artifacts and extra whitespace
        cleaned = text.strip()
        cleaned = cleaned.replace('\n', ' ').replace('\r', ' ')
        
        # Remove multiple spaces
        while '  ' in cleaned:
            cleaned = cleaned.replace('  ', ' ')
        
        # Remove common punctuation that shouldn't be in usernames
        artifacts = ['|', '_', '-', '=', '+', '*', '#']
        for artifact in artifacts:
            if cleaned.count(artifact) > len(cleaned) // 3:  # Too many artifacts
                cleaned = cleaned.replace(artifact, '')
        
        return cleaned.strip()
    
    def calculate_position_score(self, bbox: List) -> float:
        """
        Calculate position-based score for OCR result
        Higher score for text that appears in typical username position
        """
        try:
            # bbox format: [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            # Calculate center and dimensions
            x_coords = [point[0] for point in bbox]
            y_coords = [point[1] for point in bbox]
            
            center_x = sum(x_coords) / 4
            center_y = sum(y_coords) / 4
            width = max(x_coords) - min(x_coords)
            height = max(y_coords) - min(y_coords)
            
            # Prefer text that is:
            # 1. Near top of region (usernames are typically at top)
            # 2. Has reasonable width/height ratio for text
            # 3. Not too small (likely artifacts)
            
            position_score = 1.0
            
            # Top position preference (0-50px from top gets highest score)
            if center_y <= 25:
                position_score += 0.3
            elif center_y <= 50:
                position_score += 0.1
            
            # Size preference (reasonable text size)
            if 10 <= width <= 150 and 8 <= height <= 30:
                position_score += 0.2
            
            # Aspect ratio preference (text should be wider than tall)
            if width > height * 1.5:
                position_score += 0.1
            
            return min(position_score, 2.0)  # Cap at 2.0
            
        except Exception:
            return 1.0  # Default score
    
    def extract_multiple_usernames(self, image_path: str, avatar_list: List[Dict]) -> List[Dict]:
        """
        Extract usernames for multiple avatars in one operation
        """
        results = []
        
        print(f"🔤 Extracting usernames for {len(avatar_list)} avatars...")
        
        for i, avatar_info in enumerate(avatar_list):
            result = self.extract_username_from_avatar(image_path, avatar_info)
            result['avatar_index'] = i
            results.append(result)
            
            if result['success']:
                print(f"  ✅ Avatar {avatar_info['card_id']}: '{result['username']}'")
            else:
                print(f"  ❌ Avatar {avatar_info['card_id']}: {result.get('error', 'Failed')}")
        
        return results
    
    def create_username_visualization(self, image_path: str, avatar_list: List[Dict], 
                                     username_results: List[Dict], output_path: str = None) -> str:
        """
        Create visualization showing username extraction regions and results
        """
        try:
            # Load original image
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            result = img.copy()
            
            # Draw username regions and results
            for avatar_info, username_result in zip(avatar_list, username_results):
                if not username_result.get('region_bounds'):
                    continue
                
                # Get region bounds
                region_x, region_y, region_w, region_h = username_result['region_bounds']
                
                # Draw username extraction region (blue rectangle)
                cv2.rectangle(result, (region_x, region_y), 
                             (region_x + region_w, region_y + region_h), 
                             (255, 165, 0), 2)  # Orange rectangle
                
                # Draw extracted username text
                username = username_result.get('username', 'FAILED')
                confidence = username_result.get('confidence', 0.0)
                
                # Choose color based on success/confidence
                if username_result.get('success', False) and confidence > 0.5:
                    text_color = (0, 255, 0)  # Green for high confidence
                elif username_result.get('success', False):
                    text_color = (0, 165, 255)  # Orange for low confidence
                else:
                    text_color = (0, 0, 255)  # Red for failed
                
                # Add text label
                label_y = region_y - 10 if region_y > 20 else region_y + region_h + 20
                cv2.putText(result, f"{username} ({confidence:.2f})", 
                           (region_x, label_y), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
                
                # Add region number
                cv2.putText(result, f"#{avatar_info['card_id']}", 
                           (region_x + region_w - 30, region_y + 15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add title and legend
            title = f"Username Extraction: {sum(1 for r in username_results if r.get('success', False))}/{len(username_results)} Success"
            cv2.rectangle(result, (10, 10), (600, 80), (255, 255, 255), -1)
            cv2.rectangle(result, (10, 10), (600, 80), (0, 150, 255), 2)
            cv2.putText(result, title, (20, 35),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 100, 200), 2)
            
            # Add legend
            legend_items = [
                "Orange Box = Username extraction region",
                "Green Text = High confidence (>0.5)", 
                "Orange Text = Low confidence",
                "Red Text = Extraction failed"
            ]
            
            for i, item in enumerate(legend_items):
                cv2.putText(result, item, (20, 55 + i * 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            
            # Save result
            if output_path is None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_path = f"pic/screenshots/username_extraction_{timestamp}.png"
            
            cv2.imwrite(output_path, result)
            print(f"✅ Username extraction visualization saved: {output_path}")
            
            return output_path
            
        except Exception as e:
            print(f"❌ Visualization error: {e}")
            return None

if __name__ == "__main__":
    # Test with latest screenshot
    from opencv_adaptive_detector import OpenCVAdaptiveDetector
    
    screenshot_dir = "pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) 
                  if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest)
        
        # Step 1: Detect avatars
        detector = OpenCVAdaptiveDetector()
        avatars = detector.detect_avatars(screenshot_path)
        
        if avatars:
            # Step 2: Extract usernames
            extractor = UsernameExtractor()
            username_results = extractor.extract_multiple_usernames(screenshot_path, avatars)
            
            # Create visualization
            output = extractor.create_username_visualization(screenshot_path, avatars, username_results)
            
            print(f"\n📊 Username Extraction Complete!")
            print(f"Avatars: {len(avatars)}, Successful extractions: {sum(1 for r in username_results if r.get('success'))}")
            if output:
                print(f"Visualization: {output}")
        else:
            print("❌ No avatars detected for username extraction")
    else:
        print("❌ No diagnostic screenshots found")