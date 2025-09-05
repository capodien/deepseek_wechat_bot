#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Message Card Extractor - Complete message card analysis with all properties
"""

import sys
import os
import time
import cv2
import numpy as np
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# Add current directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

try:
    from TestRun.opencv_adaptive_detector import OpenCVAdaptiveDetector
    from TestRun.username_extractor import UsernameExtractor
    import easyocr
    print("✅ Successfully imported message card modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class MessageCardExtractor:
    """
    Complete message card extractor that analyzes each detected avatar
    to extract all message card properties:
    - Card number (1, 2, 3, ...)
    - Avatar region
    - Username
    - Last message timestamp
    - Message preview text
    """
    
    def __init__(self):
        """Initialize the message card extractor"""
        self.avatar_detector = OpenCVAdaptiveDetector()
        self.username_extractor = UsernameExtractor()  # Use proven username extraction
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        
        # Region configuration for message cards (matched to successful UsernameExtractor params)
        self.USERNAME_OFFSET_X = 10      # Username starts 10px right of avatar
        self.USERNAME_WIDTH = 200        # Username region width (increased from 150)
        self.USERNAME_HEIGHT_RATIO = 0.6 # Top 60% of card for username (increased from 0.4)
        
        self.TIMESTAMP_OFFSET_X = 200    # Timestamp position from left
        self.TIMESTAMP_WIDTH = 100       # Timestamp region width
        self.TIMESTAMP_HEIGHT = 20       # Timestamp region height
        
        self.MESSAGE_OFFSET_X = 10       # Message preview starts 10px right of avatar
        self.MESSAGE_WIDTH = 250         # Message preview region width
        self.MESSAGE_HEIGHT_RATIO = 0.4  # Bottom 40% of card for message preview
        self.MESSAGE_OFFSET_Y_RATIO = 0.6 # Start message preview at 60% down the card
        
        # OCR Configuration (matched to UsernameExtractor)
        self.MIN_CONFIDENCE = 0.3        # Minimum OCR confidence threshold
        self.MAX_USERNAME_LENGTH = 30    # Maximum username length
        
        print("🎯 MessageCardExtractor initialized")
    
    def extract_message_cards(self, screenshot_path: str) -> List[Dict]:
        """
        Extract all message cards with complete information
        
        Returns:
            List of message card dictionaries with all properties
        """
        print(f"\n📋 Extracting message cards from: {screenshot_path}")
        start_time = time.time()
        
        # Step 1: Detect avatars (represents message cards)
        avatars = self.avatar_detector.detect_avatars(screenshot_path)
        print(f"📱 Detected {len(avatars)} message cards")
        
        if not avatars:
            return []
        
        # Step 2: Extract complete information for each message card
        message_cards = []
        
        for card_number, avatar_info in enumerate(avatars, 1):
            print(f"\n📋 Processing Message Card #{card_number}")
            
            card_data = self.extract_single_card(
                screenshot_path, card_number, avatar_info
            )
            
            message_cards.append(card_data)
            
            # Log card summary
            username = card_data.get('username', 'Unknown')
            timestamp = card_data.get('timestamp', 'Unknown')
            preview = card_data.get('message_preview', 'No preview')[:30]
            print(f"  ✅ Card #{card_number}: {username} | {timestamp} | {preview}...")
        
        processing_time = (time.time() - start_time) * 1000
        print(f"\n⚡ Extracted {len(message_cards)} message cards in {processing_time:.1f}ms")
        
        return message_cards
    
    def extract_single_card(self, screenshot_path: str, card_number: int, avatar_info: Dict) -> Dict:
        """
        Extract complete information from a single message card
        
        Args:
            screenshot_path: Path to screenshot
            card_number: Sequential card number (1, 2, 3, ...)
            avatar_info: Avatar detection results
            
        Returns:
            Complete message card data dictionary
        """
        try:
            # Load image
            img = cv2.imread(screenshot_path)
            if img is None:
                return self.create_error_card(card_number, "Failed to load image")
            
            # Get avatar/card bounds
            card_bounds = avatar_info.get('card_bounds')  # (x, y, w, h)
            if not card_bounds:
                return self.create_error_card(card_number, "No card bounds available")
            
            x, y, w, h = card_bounds
            
            # Extract all card components
            card_data = {
                'card_number': card_number,
                'avatar_info': avatar_info,
                'card_bounds': card_bounds,
                'click_coordinates': avatar_info.get('click_center', (x + w//2, y + h//2))
            }
            
            # Extract username using proven UsernameExtractor
            username_result = self.username_extractor.extract_username_from_avatar(screenshot_path, avatar_info)
            if username_result['success']:
                card_data.update({
                    'username': username_result['username'],
                    'username_confidence': username_result['confidence'],
                    'username_region': username_result.get('region_bounds', (0, 0, 0, 0))
                })
                print(f"    👤 Username: '{username_result['username']}' (confidence: {username_result['confidence']:.2f})")
            else:
                card_data.update({
                    'username': 'EXTRACTION_FAILED',
                    'username_confidence': 0.0,
                    'username_region': (0, 0, 0, 0)
                })
                print(f"    ❌ Username extraction failed: {username_result.get('error', 'Unknown error')}")
            
            # Extract timestamp
            card_data.update(self.extract_timestamp_region(img, x, y, w, h, card_number))
            
            # Extract message preview
            card_data.update(self.extract_message_preview_region(img, x, y, w, h, card_number))
            
            # Determine card status
            card_data.update(self.analyze_card_status(avatar_info))
            
            return card_data
            
        except Exception as e:
            print(f"  ❌ Card #{card_number} extraction error: {e}")
            return self.create_error_card(card_number, str(e))
    
    def extract_username_region(self, img: np.ndarray, x: int, y: int, w: int, h: int, card_number: int) -> Dict:
        """Extract username from the card"""
        try:
            # Calculate username region
            username_x = x + w + self.USERNAME_OFFSET_X
            username_y = y
            username_w = min(self.USERNAME_WIDTH, img.shape[1] - username_x)
            username_h = int(h * self.USERNAME_HEIGHT_RATIO)
            
            # Validate bounds
            if (username_x >= img.shape[1] or username_y >= img.shape[0] or 
                username_w <= 0 or username_h <= 0):
                return {'username': 'BOUNDS_ERROR', 'username_confidence': 0.0}
            
            # Extract and process region
            username_region = img[username_y:username_y+username_h, username_x:username_x+username_w]
            processed_region = self.preprocess_text_region(username_region)
            
            # Perform OCR
            ocr_results = self.ocr_reader.readtext(processed_region, detail=True, paragraph=False)
            
            # Find best username
            best_username, confidence = self.find_best_text_candidate(ocr_results, 'username')
            
            print(f"    👤 Username: '{best_username}' (confidence: {confidence:.2f})")
            
            return {
                'username': best_username,
                'username_confidence': confidence,
                'username_region': (username_x, username_y, username_w, username_h)
            }
            
        except Exception as e:
            print(f"    ❌ Username extraction error: {e}")
            return {'username': 'EXTRACTION_ERROR', 'username_confidence': 0.0}
    
    def extract_timestamp_region(self, img: np.ndarray, x: int, y: int, w: int, h: int, card_number: int) -> Dict:
        """Extract timestamp from the card"""
        try:
            # Calculate timestamp region (usually top-right of card)
            timestamp_x = x + w + self.TIMESTAMP_OFFSET_X
            timestamp_y = y
            timestamp_w = min(self.TIMESTAMP_WIDTH, img.shape[1] - timestamp_x)
            timestamp_h = self.TIMESTAMP_HEIGHT
            
            # Validate bounds
            if (timestamp_x >= img.shape[1] or timestamp_y >= img.shape[0] or 
                timestamp_w <= 0 or timestamp_h <= 0):
                return {'timestamp': 'BOUNDS_ERROR', 'timestamp_confidence': 0.0}
            
            # Extract and process region
            timestamp_region = img[timestamp_y:timestamp_y+timestamp_h, timestamp_x:timestamp_x+timestamp_w]
            processed_region = self.preprocess_text_region(timestamp_region)
            
            # Perform OCR
            ocr_results = self.ocr_reader.readtext(processed_region, detail=True, paragraph=False)
            
            # Find best timestamp
            best_timestamp, confidence = self.find_best_text_candidate(ocr_results, 'timestamp')
            
            print(f"    🕒 Timestamp: '{best_timestamp}' (confidence: {confidence:.2f})")
            
            return {
                'timestamp': best_timestamp,
                'timestamp_confidence': confidence,
                'timestamp_region': (timestamp_x, timestamp_y, timestamp_w, timestamp_h)
            }
            
        except Exception as e:
            print(f"    ❌ Timestamp extraction error: {e}")
            return {'timestamp': 'EXTRACTION_ERROR', 'timestamp_confidence': 0.0}
    
    def extract_message_preview_region(self, img: np.ndarray, x: int, y: int, w: int, h: int, card_number: int) -> Dict:
        """Extract message preview from the card"""
        try:
            # Calculate message preview region (lower portion of card)
            preview_x = x + w + self.MESSAGE_OFFSET_X
            preview_y = y + int(h * self.MESSAGE_OFFSET_Y_RATIO)
            preview_w = min(self.MESSAGE_WIDTH, img.shape[1] - preview_x)
            preview_h = int(h * self.MESSAGE_HEIGHT_RATIO)
            
            # Validate bounds
            if (preview_x >= img.shape[1] or preview_y >= img.shape[0] or 
                preview_w <= 0 or preview_h <= 0):
                return {'message_preview': 'BOUNDS_ERROR', 'message_confidence': 0.0}
            
            # Extract and process region
            preview_region = img[preview_y:preview_y+preview_h, preview_x:preview_x+preview_w]
            processed_region = self.preprocess_text_region(preview_region)
            
            # Perform OCR
            ocr_results = self.ocr_reader.readtext(processed_region, detail=True, paragraph=False)
            
            # Find best message preview
            best_preview, confidence = self.find_best_text_candidate(ocr_results, 'message')
            
            print(f"    💬 Preview: '{best_preview}' (confidence: {confidence:.2f})")
            
            return {
                'message_preview': best_preview,
                'message_confidence': confidence,
                'message_region': (preview_x, preview_y, preview_w, preview_h)
            }
            
        except Exception as e:
            print(f"    ❌ Message preview extraction error: {e}")
            return {'message_preview': 'EXTRACTION_ERROR', 'message_confidence': 0.0}
    
    def preprocess_text_region(self, region: np.ndarray) -> np.ndarray:
        """Preprocess text region for better OCR results"""
        try:
            # Convert to grayscale
            if len(region.shape) == 3:
                gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
            else:
                gray = region.copy()
            
            # Detect theme and adjust
            avg_brightness = np.mean(gray)
            
            if avg_brightness < 127:
                # Dark theme: Invert colors
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
            print(f"    ⚠️ Preprocessing error: {e}")
            return region
    
    def find_best_text_candidate(self, ocr_results: List, text_type: str) -> Tuple[str, float]:
        """Find the best text candidate from OCR results"""
        if not ocr_results:
            return f'NO_TEXT_{text_type.upper()}', 0.0
        
        # Filter and rank candidates
        candidates = []
        min_confidence = 0.3  # Minimum confidence threshold
        
        for result in ocr_results:
            bbox, text, confidence = result
            
            # Clean text
            cleaned_text = self.clean_extracted_text(text, text_type)
            
            # Apply type-specific filters
            if self.is_valid_text_for_type(cleaned_text, text_type, confidence):
                candidates.append({
                    'text': cleaned_text,
                    'confidence': confidence,
                    'bbox': bbox
                })
        
        if not candidates:
            return f'LOW_CONFIDENCE_{text_type.upper()}', 0.0
        
        # Return best candidate (highest confidence)
        best_candidate = max(candidates, key=lambda c: c['confidence'])
        return best_candidate['text'], best_candidate['confidence']
    
    def clean_extracted_text(self, text: str, text_type: str) -> str:
        """Clean extracted text based on type"""
        cleaned = text.strip()
        
        # Remove newlines and extra spaces
        cleaned = ' '.join(cleaned.split())
        
        # Type-specific cleaning
        if text_type == 'timestamp':
            # Keep common timestamp characters
            allowed_chars = '0123456789:-/ 上午下午昨天今天AMPMOCT月日年'
            cleaned = ''.join(c for c in cleaned if c in allowed_chars or c.isalpha())
        
        elif text_type == 'username':
            # Remove common OCR artifacts for usernames
            artifacts = ['|', '_', '-', '=', '+', '*', '#']
            for artifact in artifacts:
                if cleaned.count(artifact) > len(cleaned) // 3:
                    cleaned = cleaned.replace(artifact, '')
        
        return cleaned.strip()
    
    def is_valid_text_for_type(self, text: str, text_type: str, confidence: float) -> bool:
        """Validate text based on type-specific rules (matched to UsernameExtractor logic)"""
        if confidence < self.MIN_CONFIDENCE:
            return False
        
        if len(text) == 0:
            return False
        
        if text_type == 'username':
            return len(text) <= self.MAX_USERNAME_LENGTH and len(text) >= 1
        elif text_type == 'timestamp':
            return len(text) <= 20 and len(text) >= 1
        elif text_type == 'message':
            return len(text) <= 100  # Allow longer message previews
        
        return True
    
    def analyze_card_status(self, avatar_info: Dict) -> Dict:
        """Analyze message card status (red dot, unread, etc.)"""
        return {
            'has_red_dot': avatar_info.get('has_red_dot', False),
            'has_unread': avatar_info.get('has_unread', False),
            'card_priority': self.calculate_card_priority(avatar_info)
        }
    
    def calculate_card_priority(self, avatar_info: Dict) -> int:
        """Calculate card priority (1=highest, 10=lowest)"""
        priority = 5  # Default priority
        
        if avatar_info.get('has_red_dot', False):
            priority -= 2  # High priority for red dots
        
        if avatar_info.get('has_unread', False):
            priority -= 1  # Medium priority for unread
        
        return max(1, min(10, priority))  # Clamp between 1-10
    
    def create_error_card(self, card_number: int, error_message: str) -> Dict:
        """Create error card structure"""
        return {
            'card_number': card_number,
            'error': error_message,
            'username': 'ERROR',
            'timestamp': 'ERROR',
            'message_preview': 'ERROR',
            'username_confidence': 0.0,
            'timestamp_confidence': 0.0,
            'message_confidence': 0.0,
            'has_red_dot': False,
            'has_unread': False,
            'card_priority': 10
        }
    
    def create_card_visualization(self, screenshot_path: str, message_cards: List[Dict], output_path: str = None) -> str:
        """Create visualization showing all message card regions and extracted data"""
        try:
            # Load original image
            img = cv2.imread(screenshot_path)
            if img is None:
                return None
            
            # Draw card information
            for card in message_cards:
                card_num = card['card_number']
                
                # Draw card boundary
                if 'card_bounds' in card:
                    x, y, w, h = card['card_bounds']
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    
                    # Draw card number
                    cv2.putText(img, f"#{card_num}", (x - 20, y + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
                
                # Draw username region
                if 'username_region' in card:
                    ux, uy, uw, uh = card['username_region']
                    cv2.rectangle(img, (ux, uy), (ux + uw, uy + uh), (255, 0, 0), 1)
                    cv2.putText(img, card.get('username', 'N/A')[:15], (ux, uy - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
                
                # Draw timestamp region
                if 'timestamp_region' in card:
                    tx, ty, tw, th = card['timestamp_region']
                    cv2.rectangle(img, (tx, ty), (tx + tw, ty + th), (0, 0, 255), 1)
                    cv2.putText(img, card.get('timestamp', 'N/A'), (tx, ty - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                
                # Draw message region
                if 'message_region' in card:
                    mx, my, mw, mh = card['message_region']
                    cv2.rectangle(img, (mx, my), (mx + mw, my + mh), (0, 255, 255), 1)
                    cv2.putText(img, card.get('message_preview', 'N/A')[:20], (mx, my - 5), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Save visualization
            if output_path is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"pic/screenshots/message_cards_{timestamp}.png"
            
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            cv2.imwrite(output_path, img)
            
            print(f"🎨 Card visualization saved: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"❌ Visualization creation failed: {e}")
            return None

def test_message_card_extraction():
    """Test the message card extraction system"""
    print("🚀 Testing Message Card Extraction System")
    
    # Initialize extractor
    extractor = MessageCardExtractor()
    
    # Test with diagnostic screenshot
    test_screenshot = "pic/screenshots/diagnostic_test_20250904_120220.png"
    
    if not os.path.exists(test_screenshot):
        print(f"❌ Test screenshot not found: {test_screenshot}")
        print("💡 Run the diagnostic server first to generate test screenshots")
        return
    
    # Extract message cards
    message_cards = extractor.extract_message_cards(test_screenshot)
    
    # Display results
    print(f"\n📋 Message Card Extraction Results:")
    print(f"   Total Cards: {len(message_cards)}")
    
    for card in message_cards:
        print(f"\n📱 Message Card #{card['card_number']}:")
        print(f"   👤 Username: {card.get('username', 'N/A')} ({card.get('username_confidence', 0):.2f})")
        print(f"   🕒 Timestamp: {card.get('timestamp', 'N/A')} ({card.get('timestamp_confidence', 0):.2f})")
        print(f"   💬 Preview: {card.get('message_preview', 'N/A')} ({card.get('message_confidence', 0):.2f})")
        print(f"   🎯 Priority: {card.get('card_priority', 5)}")
        print(f"   📍 Click: {card.get('click_coordinates', 'N/A')}")
    
    # Create visualization
    visualization_path = extractor.create_card_visualization(test_screenshot, message_cards)
    
    print(f"\n✅ Message card extraction completed successfully!")
    if visualization_path:
        print(f"🎨 Visualization: {visualization_path}")
    
    return message_cards

if __name__ == "__main__":
    test_message_card_extraction()