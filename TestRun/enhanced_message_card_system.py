#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Enhanced Message Card System - Uses Step 2 output as base
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
    print("✅ Successfully imported enhanced message card modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class EnhancedMessageCardSystem:
    """
    Enhanced message card system that builds upon Step 2 output:
    - Step 2 creates base message cards (number + username)  
    - Step 3 enhances them (adds timestamp + message preview)
    """
    
    def __init__(self):
        """Initialize the enhanced system"""
        self.avatar_detector = OpenCVAdaptiveDetector()
        self.username_extractor = UsernameExtractor()
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        
        # Configuration for additional regions (fallback values)
        self.TIMESTAMP_OFFSET_X = 200    # Timestamp position from left
        self.TIMESTAMP_WIDTH = 100       # Timestamp region width  
        self.TIMESTAMP_HEIGHT = 20       # Timestamp region height
        
        self.MESSAGE_OFFSET_X = 10       # Message preview starts 10px right of avatar
        self.MESSAGE_WIDTH = 250         # Message preview region width
        self.MESSAGE_HEIGHT_RATIO = 0.4  # Bottom 40% of card for message preview
        self.MESSAGE_OFFSET_Y_RATIO = 0.6 # Start message preview at 60% down the card
        
        # OCR Configuration
        self.MIN_CONFIDENCE = 0.3        # Minimum OCR confidence threshold
        
        # Import adaptive width calculator
        try:
            from TestRun.adaptive_width_calculator import AdaptiveWidthCalculator
            self.width_calculator = AdaptiveWidthCalculator()
            self.use_adaptive_width = True
            print("🎯 EnhancedMessageCardSystem initialized with adaptive width")
        except ImportError:
            self.width_calculator = None
            self.use_adaptive_width = False
            print("🎯 EnhancedMessageCardSystem initialized with fixed width")
    
    def create_base_message_cards(self, screenshot_path: str) -> List[Dict]:
        """
        Step 2: Create base message cards with number and username
        This should be the main Step 2 output
        """
        print(f"\n📋 Step 2: Creating base message cards from: {screenshot_path}")
        start_time = time.time()
        
        # Detect avatars
        avatars = self.avatar_detector.detect_avatars(screenshot_path)
        print(f"👥 Detected {len(avatars)} avatars")
        
        if not avatars:
            return []
        
        # Create base message cards
        base_cards = []
        
        for card_number, avatar_info in enumerate(avatars, 1):
            print(f"\n📋 Creating Base Card #{card_number}")
            
            # Extract username using proven UsernameExtractor
            username_result = self.username_extractor.extract_username_from_avatar(screenshot_path, avatar_info)
            
            if username_result['success']:
                username = username_result['username']
                username_confidence = username_result['confidence']
                print(f"  👤 Username: '{username}' (confidence: {username_confidence:.2f})")
            else:
                username = 'EXTRACTION_FAILED'
                username_confidence = 0.0
                print(f"  ❌ Username extraction failed: {username_result.get('error', 'Unknown')}")
            
            # Create base card structure
            base_card = {
                'card_number': card_number,
                'username': username,
                'username_confidence': username_confidence,
                'avatar_info': avatar_info,
                'card_bounds': avatar_info.get('card_bounds'),
                'click_coordinates': avatar_info.get('click_center'),
                'has_red_dot': avatar_info.get('has_red_dot', False),
                'card_priority': self.calculate_base_priority(avatar_info, username_confidence),
                # Placeholders for Step 3 enhancements
                'timestamp': None,
                'timestamp_confidence': None,
                'message_preview': None,
                'message_confidence': None,
                'enhanced': False
            }
            
            base_cards.append(base_card)
            print(f"  ✅ Base Card #{card_number}: {username} | Priority: {base_card['card_priority']}")
        
        processing_time = (time.time() - start_time) * 1000
        print(f"\n⚡ Created {len(base_cards)} base message cards in {processing_time:.1f}ms")
        
        return base_cards
    
    def enhance_message_cards(self, base_cards: List[Dict], screenshot_path: str) -> List[Dict]:
        """
        Step 3: Enhance base message cards with timestamp and message preview
        """
        print(f"\n📋 Step 3: Enhancing {len(base_cards)} message cards")
        start_time = time.time()
        
        enhanced_cards = []
        
        for card in base_cards:
            print(f"\n🔧 Enhancing Card #{card['card_number']}: {card['username']}")
            
            # Create enhanced copy
            enhanced_card = card.copy()
            
            # Add timestamp
            timestamp_data = self.extract_timestamp_from_card(screenshot_path, card)
            enhanced_card.update(timestamp_data)
            
            # Add message preview
            message_data = self.extract_message_preview_from_card(screenshot_path, card)
            enhanced_card.update(message_data)
            
            # Mark as enhanced
            enhanced_card['enhanced'] = True
            
            # Update priority with new information
            enhanced_card['card_priority'] = self.calculate_enhanced_priority(enhanced_card)
            
            enhanced_cards.append(enhanced_card)
            
            print(f"  ✅ Enhanced Card #{card['card_number']}: {card['username']} | {timestamp_data.get('timestamp', 'No time')} | {message_data.get('message_preview', 'No preview')[:20]}...")
        
        processing_time = (time.time() - start_time) * 1000
        print(f"\n⚡ Enhanced {len(enhanced_cards)} message cards in {processing_time:.1f}ms")
        
        return enhanced_cards
    
    def extract_timestamp_from_card(self, screenshot_path: str, card: Dict) -> Dict:
        """Extract timestamp from a specific card"""
        try:
            # Load image
            img = cv2.imread(screenshot_path)
            if img is None:
                return {'timestamp': 'LOAD_ERROR', 'timestamp_confidence': 0.0}
            
            # Get card bounds
            card_bounds = card.get('card_bounds')
            if not card_bounds:
                return {'timestamp': 'NO_BOUNDS', 'timestamp_confidence': 0.0}
            
            x, y, w, h = card_bounds
            
            # Calculate timestamp region (adaptive or fixed)
            if self.use_adaptive_width and self.width_calculator:
                regions = self.width_calculator.calculate_adaptive_regions(img, card_bounds)
                timestamp_region = regions['timestamp_region']
                timestamp_x = timestamp_region['x']
                timestamp_y = timestamp_region['y']
                timestamp_w = timestamp_region['width']
                timestamp_h = timestamp_region['height']
                print(f"    📏 Adaptive timestamp region: {timestamp_w}px")
            else:
                # Fallback to fixed calculation
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
    
    def extract_message_preview_from_card(self, screenshot_path: str, card: Dict) -> Dict:
        """Extract message preview from a specific card"""
        try:
            # Load image
            img = cv2.imread(screenshot_path)
            if img is None:
                return {'message_preview': 'LOAD_ERROR', 'message_confidence': 0.0}
            
            # Get card bounds
            card_bounds = card.get('card_bounds')
            if not card_bounds:
                return {'message_preview': 'NO_BOUNDS', 'message_confidence': 0.0}
            
            x, y, w, h = card_bounds
            
            # Calculate message preview region (adaptive or fixed)
            if self.use_adaptive_width and self.width_calculator:
                regions = self.width_calculator.calculate_adaptive_regions(img, card_bounds)
                message_region = regions['message_region']
                preview_x = message_region['x']
                preview_y = message_region['y']
                preview_w = message_region['width']
                preview_h = message_region['height']
                print(f"    📏 Adaptive message region: {preview_w}px")
            else:
                # Fallback to fixed calculation
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
        
        return cleaned
    
    def is_valid_text_for_type(self, text: str, text_type: str, confidence: float) -> bool:
        """Validate text based on type-specific rules"""
        if confidence < self.MIN_CONFIDENCE:
            return False
        
        if len(text) == 0:
            return False
        
        if text_type == 'timestamp':
            return len(text) <= 20 and len(text) >= 1
        elif text_type == 'message':
            return len(text) <= 100  # Allow longer message previews
        
        return True
    
    def calculate_base_priority(self, avatar_info: Dict, username_confidence: float) -> int:
        """Calculate base card priority (1=highest, 10=lowest)"""
        priority = 5  # Default priority
        
        if avatar_info.get('has_red_dot', False):
            priority -= 2  # High priority for red dots
        
        if username_confidence > 0.8:
            priority -= 1  # Higher priority for confident username extraction
        
        return max(1, min(10, priority))
    
    def calculate_enhanced_priority(self, card: Dict) -> int:
        """Calculate enhanced card priority with timestamp and message info"""
        priority = self.calculate_base_priority(card.get('avatar_info', {}), card.get('username_confidence', 0))
        
        # Boost priority for recent messages or high-confidence previews
        if card.get('message_confidence', 0) > 0.8:
            priority -= 1
        
        return max(1, min(10, priority))

def test_enhanced_system():
    """Test the enhanced message card system"""
    print("🚀 Testing Enhanced Message Card System")
    
    # Initialize system
    system = EnhancedMessageCardSystem()
    
    # Test with diagnostic screenshot
    test_screenshot = "pic/screenshots/diagnostic_test_20250904_120220.png"
    
    if not os.path.exists(test_screenshot):
        print(f"❌ Test screenshot not found: {test_screenshot}")
        return
    
    # Step 2: Create base message cards
    base_cards = system.create_base_message_cards(test_screenshot)
    
    # Step 3: Enhance message cards
    enhanced_cards = system.enhance_message_cards(base_cards, test_screenshot)
    
    # Display results
    print(f"\n📋 Enhanced Message Card System Results:")
    print(f"   Base Cards Created: {len(base_cards)}")
    print(f"   Enhanced Cards: {len(enhanced_cards)}")
    
    for card in enhanced_cards:
        print(f"\n📱 Message Card #{card['card_number']}:")
        print(f"   👤 Username: {card.get('username', 'N/A')} ({card.get('username_confidence', 0):.2f})")
        print(f"   🕒 Timestamp: {card.get('timestamp', 'N/A')} ({card.get('timestamp_confidence', 0):.2f})")
        print(f"   💬 Preview: {card.get('message_preview', 'N/A')} ({card.get('message_confidence', 0):.2f})")
        print(f"   🎯 Priority: {card.get('card_priority', 5)}")
        print(f"   📍 Click: {card.get('click_coordinates', 'N/A')}")
        print(f"   🔧 Enhanced: {card.get('enhanced', False)}")
    
    print(f"\n✅ Enhanced message card system test completed!")
    
    return enhanced_cards

if __name__ == "__main__":
    test_enhanced_system()