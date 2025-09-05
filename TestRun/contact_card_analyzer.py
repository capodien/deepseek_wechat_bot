#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact Card Analyzer
Parses WeChat contact list into individual contact cards for precise detection
"""

import cv2
import numpy as np
import easyocr
import os
import json
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class ContactCard:
    """Represents a single WeChat contact card"""
    
    def __init__(self, bounds: Tuple[int, int, int, int], index: int):
        self.x, self.y, self.width, self.height = bounds
        self.index = index
        self.avatar_region = None
        self.name_region = None  
        self.message_region = None
        self.red_dot_region = None
        self.click_center = None
        self.contact_name = ""
        self.has_message = False
        self.has_red_dot = False
        
    def get_optimal_click_position(self) -> Tuple[int, int]:
        """Calculate optimal click position for this contact card"""
        if self.click_center:
            return self.click_center
        
        # Default to card center if not specifically calculated
        center_x = self.x + (self.width // 2)
        center_y = self.y + (self.height // 2)
        return (center_x, center_y)

class ContactCardAnalyzer:
    """Analyzes WeChat contact list by parsing individual contact cards"""
    
    def __init__(self, output_dir: str = "TestRun/contact_analysis"):
        self.output_dir = output_dir
        self.ocr_reader = None
        
        # WeChat contact card structure constants
        self.TYPICAL_CARD_HEIGHT = 75  # Approximate height of each contact card
        self.MIN_CARD_HEIGHT = 60      # Minimum card height to consider
        self.MAX_CARD_HEIGHT = 90      # Maximum card height to consider
        
        # Card component positions (relative to card bounds)
        self.AVATAR_SIZE = 50          # Avatar is typically 50x50px
        self.AVATAR_MARGIN = 12        # Left margin for avatar
        self.NAME_START_X = 70         # Name text starts after avatar
        self.MESSAGE_START_X = 70      # Message preview starts after avatar
        self.RED_DOT_X_OFFSET = -20    # Red dot appears left of avatar
        
        # Red dot detection parameters
        self.RED_DOT_COLORS = {
            'primary': np.array([84, 98, 227]),
            'windows': np.array([81, 81, 255]),
            'fallback': np.array([80, 100, 230])
        }
        self.COLOR_TOLERANCE = 15
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📋 Contact Card Analyzer initialized")
        print(f"📁 Output: {self.output_dir}")
    
    def analyze_contact_list(self, screenshot_path: str) -> List[ContactCard]:
        """Parse WeChat contact list into individual contact cards"""
        try:
            print(f"\n📋 Analyzing contact cards in: {os.path.basename(screenshot_path)}")
            
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"❌ Failed to load screenshot")
                return []
            
            # Step 1: Detect contact list boundaries
            contact_list_bounds = self._detect_contact_list_area(image)
            if not contact_list_bounds:
                print("❌ Could not detect contact list area")
                return []
            
            print(f"📍 Contact list detected: {contact_list_bounds}")
            
            # Step 2: Segment individual contact cards
            contact_cards = self._segment_contact_cards(image, contact_list_bounds)
            print(f"📋 Found {len(contact_cards)} contact cards")
            
            # Step 3: Analyze each contact card
            analyzed_cards = []
            for i, card in enumerate(contact_cards):
                analyzed_card = self._analyze_individual_card(image, card, i)
                if analyzed_card:
                    analyzed_cards.append(analyzed_card)
            
            # Step 4: Save analysis results
            self._save_analysis_results(analyzed_cards, screenshot_path)
            
            return analyzed_cards
            
        except Exception as e:
            print(f"❌ Contact list analysis error: {e}")
            return []
    
    def _detect_contact_list_area(self, image) -> Optional[Tuple[int, int, int, int]]:
        """Detect the overall contact list area in the screenshot"""
        try:
            height, width = image.shape[:2]
            
            # Basic heuristic: WeChat contact list typically occupies left portion
            # For now, use reasonable estimates - could be enhanced with UI detection
            
            # Estimate contact list bounds based on typical WeChat layout
            list_x = 60  # Left margin
            list_y = 100  # Top margin (below header)
            list_width = min(400, width // 3)  # Adaptive width
            list_height = height - 200  # Leave margin for bottom
            
            # Validate bounds
            if list_x + list_width > width:
                list_width = width - list_x - 20
            if list_y + list_height > height:
                list_height = height - list_y - 20
            
            return (list_x, list_y, list_width, list_height)
            
        except Exception as e:
            print(f"❌ Contact list detection error: {e}")
            return None
    
    def _segment_contact_cards(self, image, contact_list_bounds: Tuple[int, int, int, int]) -> List[ContactCard]:
        """Segment the contact list into individual contact cards"""
        try:
            list_x, list_y, list_width, list_height = contact_list_bounds
            
            # Extract contact list region
            contact_area = image[list_y:list_y+list_height, list_x:list_x+list_width]
            
            # Method 1: Use horizontal line detection to find card boundaries
            cards = self._detect_cards_by_horizontal_lines(contact_area, list_x, list_y)
            
            # Method 2: If line detection fails, use fixed height estimation
            if len(cards) < 2:
                print("📋 Using fixed height segmentation as fallback")
                cards = self._detect_cards_by_fixed_height(contact_area, list_x, list_y)
            
            return cards
            
        except Exception as e:
            print(f"❌ Card segmentation error: {e}")
            return []
    
    def _detect_cards_by_horizontal_lines(self, contact_area, offset_x: int, offset_y: int) -> List[ContactCard]:
        """Detect contact card boundaries using horizontal line detection"""
        try:
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(contact_area, cv2.COLOR_BGR2GRAY)
            
            # Detect horizontal edges
            horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
            horizontal_edges = cv2.morphologyEx(gray, cv2.MORPH_OPEN, horizontal_kernel)
            
            # Find horizontal lines (card separators)
            horizontal_lines = []
            for y in range(1, gray.shape[0] - 1):
                line_strength = np.sum(horizontal_edges[y, :])
                if line_strength > gray.shape[1] * 10:  # Threshold for significant horizontal line
                    horizontal_lines.append(y)
            
            # Group nearby lines and find card boundaries
            card_boundaries = []
            if horizontal_lines:
                # Remove lines that are too close together
                filtered_lines = [horizontal_lines[0]]
                for line in horizontal_lines[1:]:
                    if line - filtered_lines[-1] > self.MIN_CARD_HEIGHT:
                        filtered_lines.append(line)
                
                # Create cards from boundaries
                for i in range(len(filtered_lines) - 1):
                    y_start = filtered_lines[i]
                    y_end = filtered_lines[i + 1]
                    height = y_end - y_start
                    
                    if self.MIN_CARD_HEIGHT <= height <= self.MAX_CARD_HEIGHT:
                        card = ContactCard(
                            bounds=(offset_x, offset_y + y_start, contact_area.shape[1], height),
                            index=len(card_boundaries)
                        )
                        card_boundaries.append(card)
            
            return card_boundaries
            
        except Exception as e:
            print(f"❌ Horizontal line detection error: {e}")
            return []
    
    def _detect_cards_by_fixed_height(self, contact_area, offset_x: int, offset_y: int) -> List[ContactCard]:
        """Detect contact cards using fixed height estimation"""
        try:
            cards = []
            area_height = contact_area.shape[0]
            area_width = contact_area.shape[1]
            
            # Estimate number of cards based on typical card height
            estimated_cards = area_height // self.TYPICAL_CARD_HEIGHT
            actual_card_height = area_height // estimated_cards if estimated_cards > 0 else self.TYPICAL_CARD_HEIGHT
            
            print(f"📋 Estimating {estimated_cards} cards with height {actual_card_height}px")
            
            # Create cards with estimated positions
            for i in range(estimated_cards):
                y_start = i * actual_card_height
                y_end = min((i + 1) * actual_card_height, area_height)
                height = y_end - y_start
                
                if height >= self.MIN_CARD_HEIGHT:
                    card = ContactCard(
                        bounds=(offset_x, offset_y + y_start, area_width, height),
                        index=i
                    )
                    cards.append(card)
            
            return cards
            
        except Exception as e:
            print(f"❌ Fixed height segmentation error: {e}")
            return []
    
    def _analyze_individual_card(self, image, card: ContactCard, index: int) -> Optional[ContactCard]:
        """Analyze individual contact card for components and messages"""
        try:
            # Extract card region from image
            card_image = image[card.y:card.y+card.height, card.x:card.x+card.width]
            
            # Define component regions within the card
            self._define_card_regions(card)
            
            # Check for red notification dot
            card.has_red_dot = self._detect_red_dot_in_card(card_image, card)
            
            # Extract contact name using OCR
            card.contact_name = self._extract_contact_name(card_image, card)
            
            # Check for message preview
            card.has_message = self._detect_message_preview(card_image, card)
            
            # Calculate optimal click position
            card.click_center = self._calculate_optimal_click_position(card)
            
            print(f"📋 Card {index}: '{card.contact_name}' - Red dot: {card.has_red_dot} - Click: {card.click_center}")
            
            return card
            
        except Exception as e:
            print(f"❌ Card analysis error for card {index}: {e}")
            return None
    
    def _define_card_regions(self, card: ContactCard):
        """Define regions within a contact card for different components"""
        # Avatar region (left side)
        card.avatar_region = (
            self.AVATAR_MARGIN,
            (card.height - self.AVATAR_SIZE) // 2,
            self.AVATAR_SIZE,
            self.AVATAR_SIZE
        )
        
        # Name region (right of avatar, top half)
        card.name_region = (
            self.NAME_START_X,
            5,
            card.width - self.NAME_START_X - 10,
            card.height // 2
        )
        
        # Message region (right of avatar, bottom half)
        card.message_region = (
            self.MESSAGE_START_X,
            card.height // 2,
            card.width - self.MESSAGE_START_X - 10,
            card.height // 2 - 5
        )
        
        # Red dot region (left of avatar)
        dot_x = max(0, self.AVATAR_MARGIN + self.RED_DOT_X_OFFSET)
        card.red_dot_region = (
            dot_x,
            0,
            self.AVATAR_MARGIN - dot_x + 20,
            card.height
        )
    
    def _detect_red_dot_in_card(self, card_image, card: ContactCard) -> bool:
        """Detect red notification dot within this specific card"""
        try:
            if not card.red_dot_region:
                return False
            
            rx, ry, rw, rh = card.red_dot_region
            if rx + rw > card_image.shape[1] or ry + rh > card_image.shape[0]:
                return False
            
            dot_area = card_image[ry:ry+rh, rx:rx+rw]
            
            # Check for red dot colors
            for color_name, target_color in self.RED_DOT_COLORS.items():
                lower_bound = target_color - self.COLOR_TOLERANCE
                upper_bound = target_color + self.COLOR_TOLERANCE
                color_mask = np.all((lower_bound <= dot_area) & (dot_area <= upper_bound), axis=-1)
                
                if np.any(color_mask):
                    return True
            
            return False
            
        except Exception as e:
            print(f"❌ Red dot detection error: {e}")
            return False
    
    def _extract_contact_name(self, card_image, card: ContactCard) -> str:
        """Extract contact name from card using OCR"""
        try:
            if not card.name_region:
                return "Unknown"
            
            nx, ny, nw, nh = card.name_region
            if nx + nw > card_image.shape[1] or ny + nh > card_image.shape[0]:
                return "Unknown"
            
            name_area = card_image[ny:ny+nh, nx:nx+nw]
            
            # Use OCR to extract name
            reader = self._get_ocr_reader()
            results = reader.readtext(name_area)
            
            # Find the most prominent text (likely the name)
            if results:
                # Take the longest text as the name
                name_candidates = [result[1].strip() for result in results if len(result[1].strip()) > 1]
                if name_candidates:
                    return max(name_candidates, key=len)
            
            return "Unknown"
            
        except Exception as e:
            print(f"❌ Name extraction error: {e}")
            return "Unknown"
    
    def _detect_message_preview(self, card_image, card: ContactCard) -> bool:
        """Detect if there's a message preview in this card"""
        try:
            if not card.message_region:
                return False
            
            mx, my, mw, mh = card.message_region
            if mx + mw > card_image.shape[1] or my + mh > card_image.shape[0]:
                return False
            
            message_area = card_image[my:my+mh, mx:mx+mw]
            
            # Simple heuristic: if there's significant text content, there's likely a message
            reader = self._get_ocr_reader()
            results = reader.readtext(message_area)
            
            return len(results) > 0
            
        except Exception as e:
            return False
    
    def _calculate_optimal_click_position(self, card: ContactCard) -> Tuple[int, int]:
        """Calculate the optimal click position for this contact card"""
        # Click in the center-right area of the card (away from avatar, in the name/message area)
        click_x = card.x + self.NAME_START_X + 50  # 50px into the name area
        click_y = card.y + (card.height // 2)  # Vertical center of card
        
        return (click_x, click_y)
    
    def _get_ocr_reader(self):
        """Get OCR reader with lazy initialization"""
        if self.ocr_reader is None:
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        return self.ocr_reader
    
    def _save_analysis_results(self, cards: List[ContactCard], screenshot_path: str):
        """Save contact card analysis results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(self.output_dir, f'contact_analysis_{timestamp}.json')
            
            # Convert cards to serializable format
            cards_data = []
            for card in cards:
                card_data = {
                    'index': card.index,
                    'bounds': [card.x, card.y, card.width, card.height],
                    'contact_name': card.contact_name,
                    'has_red_dot': card.has_red_dot,
                    'has_message': card.has_message,
                    'click_center': list(card.click_center) if card.click_center else None,
                    'regions': {
                        'avatar': list(card.avatar_region) if card.avatar_region else None,
                        'name': list(card.name_region) if card.name_region else None,
                        'message': list(card.message_region) if card.message_region else None,
                        'red_dot': list(card.red_dot_region) if card.red_dot_region else None
                    }
                }
                cards_data.append(card_data)
            
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'source_screenshot': os.path.basename(screenshot_path),
                'total_cards': len(cards),
                'cards_with_red_dots': sum(1 for card in cards if card.has_red_dot),
                'cards_with_messages': sum(1 for card in cards if card.has_message),
                'contact_cards': cards_data
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Analysis results saved: {os.path.basename(results_file)}")
            
        except Exception as e:
            print(f"❌ Results saving error: {e}")
    
    def find_contacts_with_notifications(self, cards: List[ContactCard]) -> List[ContactCard]:
        """Find contact cards that have notifications (red dots)"""
        return [card for card in cards if card.has_red_dot]
    
    def get_best_click_target(self, cards: List[ContactCard]) -> Optional[ContactCard]:
        """Get the best contact card to click (prioritize red dots, then recent messages)"""
        # First priority: cards with red notification dots
        notification_cards = self.find_contacts_with_notifications(cards)
        if notification_cards:
            # Return the first card with notification (topmost in list)
            return notification_cards[0]
        
        # Second priority: cards with message previews
        message_cards = [card for card in cards if card.has_message]
        if message_cards:
            return message_cards[0]
        
        # Fallback: return first card
        return cards[0] if cards else None


if __name__ == '__main__':
    """Test the contact card analyzer"""
    analyzer = ContactCardAnalyzer()
    
    # Find latest screenshot
    screenshot_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest_screenshot = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"🧪 Testing contact card analysis on: {latest_screenshot}")
        
        # Analyze contact cards
        cards = analyzer.analyze_contact_list(screenshot_path)
        
        if cards:
            print(f"\n📊 Analysis Summary:")
            print(f"   Total Cards: {len(cards)}")
            print(f"   With Red Dots: {len(analyzer.find_contacts_with_notifications(cards))}")
            print(f"   With Messages: {len([c for c in cards if c.has_message])}")
            
            # Show best click target
            best_target = analyzer.get_best_click_target(cards)
            if best_target:
                print(f"\n🎯 Best Click Target:")
                print(f"   Contact: {best_target.contact_name}")
                print(f"   Position: {best_target.click_center}")
                print(f"   Has Red Dot: {best_target.has_red_dot}")
        else:
            print("❌ No contact cards detected")
    else:
        print("❌ No test screenshots found")