#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Improved Contact Card Analyzer
Uses avatar detection to accurately identify contact card boundaries
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
        self.avatar_bounds = None  # (x, y, width, height) of avatar
        self.name_region = None  
        self.message_region = None
        self.timestamp_region = None
        self.click_center = None
        self.contact_name = ""
        self.has_message = False
        self.has_red_dot = False
        self.message_text = ""
        self.timestamp_text = ""
        
    def get_optimal_click_position(self) -> Tuple[int, int]:
        """Calculate optimal click position for this contact card"""
        if self.click_center:
            return self.click_center
        
        # Click in the middle of the name/message area (not on avatar)
        if self.avatar_bounds:
            # Click to the right of avatar, in the text area
            click_x = self.avatar_bounds[0] + self.avatar_bounds[2] + 50
            click_y = self.y + (self.height // 2)
        else:
            # Fallback to card center
            click_x = self.x + (self.width // 2)
            click_y = self.y + (self.height // 2)
        
        return (click_x, click_y)

class ImprovedContactCardAnalyzer:
    """Analyzes WeChat contact list using avatar detection for accurate card boundaries"""
    
    def __init__(self, output_dir: str = "TestRun/contact_analysis"):
        self.output_dir = output_dir
        self.ocr_reader = None
        
        # WeChat contact card structure constants (based on actual UI)
        self.AVATAR_SIZE = 40  # Approximate avatar size in pixels
        self.CARD_HEIGHT = 70  # Approximate height of each contact card
        self.AVATAR_LEFT_MARGIN = 40  # Avatars start around x=40 (corrected based on analysis)
        self.CONTACT_LIST_WIDTH = 480  # Contact list is about 480px wide
        
        # Red dot/notification detection
        self.RED_NOTIFICATION_COLOR = np.array([84, 98, 227])  # Red color in BGR
        self.COLOR_TOLERANCE = 20
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📋 Improved Contact Card Analyzer initialized")
        print(f"📁 Output: {self.output_dir}")
    
    def analyze_contact_list(self, screenshot_path: str) -> List[ContactCard]:
        """Parse WeChat contact list into individual contact cards using avatar detection"""
        try:
            print(f"\n📋 Analyzing contact cards in: {os.path.basename(screenshot_path)}")
            
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"❌ Failed to load screenshot")
                return []
            
            # Step 1: Detect all avatars in the contact list
            avatars = self._detect_avatars(image)
            print(f"🖼️  Found {len(avatars)} avatar positions")
            
            # Step 2: Create contact cards based on avatar positions
            contact_cards = self._create_cards_from_avatars(image, avatars)
            print(f"📋 Created {len(contact_cards)} contact cards")
            
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
            import traceback
            traceback.print_exc()
            return []
    
    def _detect_avatars(self, image) -> List[Tuple[int, int, int, int]]:
        """Detect avatar positions in the contact list"""
        try:
            avatars = []
            height, width = image.shape[:2]
            
            # Convert to grayscale for edge detection
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Look for square regions that could be avatars
            # Avatars in WeChat are typically around x=40 and appear every ~70 pixels vertically
            x_start = self.AVATAR_LEFT_MARGIN  # Use the corrected avatar position
            x_end = x_start + self.AVATAR_SIZE + 10
            
            # Scan vertically for potential avatar positions
            y_positions = []
            y = 100  # Start from first potential avatar position (adjusted based on debug)
            
            while y < height - self.AVATAR_SIZE:
                # Check if there's likely an avatar at this position
                roi = gray[y:y+self.AVATAR_SIZE, x_start:x_end]
                
                # Check for sufficient variance (avatars have content, empty space doesn't)
                if roi.size > 0:
                    variance = np.var(roi)
                    if variance > 100:  # Threshold for detecting content
                        # Check if this looks like a distinct region (avatar)
                        edges = cv2.Canny(roi, 50, 150)
                        edge_density = np.sum(edges > 0) / edges.size
                        
                        if edge_density > 0.05:  # Has enough edges to be an avatar
                            avatars.append((x_start, y, self.AVATAR_SIZE, self.AVATAR_SIZE))
                            y_positions.append(y)
                            y += self.CARD_HEIGHT - 10  # Move to next likely avatar position
                        else:
                            y += 10  # Small increment to continue searching
                    else:
                        y += 10
                else:
                    y += 10
                
                # Safety check to prevent infinite loop
                if y > height - 50:
                    break
            
            # Refine avatar positions by ensuring consistent spacing
            if len(avatars) > 1:
                refined_avatars = self._refine_avatar_positions(avatars)
                return refined_avatars
            
            return avatars
            
        except Exception as e:
            print(f"❌ Avatar detection error: {e}")
            return []
    
    def _refine_avatar_positions(self, avatars: List[Tuple[int, int, int, int]]) -> List[Tuple[int, int, int, int]]:
        """Refine avatar positions to ensure consistent card heights"""
        try:
            if len(avatars) < 2:
                return avatars
            
            # Calculate average spacing between avatars
            spacings = []
            for i in range(1, len(avatars)):
                spacing = avatars[i][1] - avatars[i-1][1]
                if 60 <= spacing <= 80:  # Reasonable card height range
                    spacings.append(spacing)
            
            if spacings:
                avg_spacing = int(np.mean(spacings))
            else:
                avg_spacing = self.CARD_HEIGHT
            
            # Create refined list with consistent spacing
            refined = [avatars[0]]  # Keep first avatar
            expected_y = avatars[0][1]
            
            for avatar in avatars[1:]:
                expected_y += avg_spacing
                actual_y = avatar[1]
                
                # If avatar is close to expected position, adjust it
                if abs(actual_y - expected_y) < 20:
                    refined.append((avatar[0], expected_y, avatar[2], avatar[3]))
                    expected_y = expected_y  # Update for next iteration
                elif actual_y > expected_y + avg_spacing//2:
                    # There might be a gap, add the avatar at its actual position
                    refined.append(avatar)
                    expected_y = avatar[1]
            
            return refined
            
        except Exception as e:
            print(f"⚠️  Avatar refinement error: {e}")
            return avatars
    
    def _create_cards_from_avatars(self, image, avatars: List[Tuple[int, int, int, int]]) -> List[ContactCard]:
        """Create contact cards based on detected avatar positions"""
        try:
            cards = []
            height, width = image.shape[:2]
            
            for i, avatar in enumerate(avatars):
                ax, ay, aw, ah = avatar
                
                # Calculate card boundaries based on avatar position
                card_x = 0  # Cards span full width of contact list
                card_y = ay - 15  # Card starts a bit above avatar
                card_width = min(self.CONTACT_LIST_WIDTH, width)
                
                # Calculate card height
                if i < len(avatars) - 1:
                    # Use distance to next avatar
                    next_y = avatars[i + 1][1]
                    card_height = (next_y - 15) - card_y
                else:
                    # Last card - use standard height
                    card_height = self.CARD_HEIGHT
                
                # Ensure reasonable card dimensions
                if card_height < 60:
                    card_height = 60
                elif card_height > 90:
                    card_height = 90
                
                # Ensure card doesn't go beyond image bounds
                if card_y < 0:
                    card_y = 0
                if card_y + card_height > height:
                    card_height = height - card_y
                
                # Create contact card
                card = ContactCard(
                    bounds=(card_x, card_y, card_width, card_height),
                    index=i
                )
                card.avatar_bounds = avatar
                cards.append(card)
            
            return cards
            
        except Exception as e:
            print(f"❌ Card creation error: {e}")
            return []
    
    def _analyze_individual_card(self, image, card: ContactCard, index: int) -> Optional[ContactCard]:
        """Analyze individual contact card for components and content"""
        try:
            # Extract card region from image
            card_region = image[card.y:card.y+card.height, card.x:card.x+card.width]
            
            if card_region.size == 0:
                return None
            
            # Define component regions based on avatar position
            if card.avatar_bounds:
                ax, ay, aw, ah = card.avatar_bounds
                
                # Name region: to the right of avatar, top portion
                card.name_region = (
                    ax + aw + 10,  # Start after avatar
                    card.y + 5,
                    200,  # Width for name
                    30  # Height for name
                )
                
                # Message region: below name
                card.message_region = (
                    ax + aw + 10,
                    card.y + 35,
                    250,
                    30
                )
                
                # Timestamp region: right side
                card.timestamp_region = (
                    card.x + card.width - 100,
                    card.y + 5,
                    90,
                    25
                )
            
            # Check for red notification dot/badge
            card.has_red_dot = self._detect_red_notification(image, card)
            
            # Extract contact name using OCR
            card.contact_name = self._extract_contact_name(image, card)
            
            # Check for message preview
            card.has_message, card.message_text = self._extract_message_preview(image, card)
            
            # Extract timestamp
            card.timestamp_text = self._extract_timestamp(image, card)
            
            # Calculate optimal click position
            card.click_center = card.get_optimal_click_position()
            
            print(f"📋 Card {index}: '{card.contact_name}' - Red: {card.has_red_dot} - Msg: {card.has_message} - Click: {card.click_center}")
            
            return card
            
        except Exception as e:
            print(f"❌ Card analysis error for card {index}: {e}")
            return None
    
    def _detect_red_notification(self, image, card: ContactCard) -> bool:
        """Detect red notification badge on or near avatar"""
        try:
            if not card.avatar_bounds:
                return False
            
            ax, ay, aw, ah = card.avatar_bounds
            
            # Check area around avatar for red notification
            # Notification badges often appear on top-right of avatar
            check_region = image[max(0, ay-10):ay+ah+10, ax:ax+aw+20]
            
            if check_region.size == 0:
                return False
            
            # Look for red pixels
            lower_red = self.RED_NOTIFICATION_COLOR - self.COLOR_TOLERANCE
            upper_red = self.RED_NOTIFICATION_COLOR + self.COLOR_TOLERANCE
            
            # Also check for actual red color (in BGR: blue, green, RED)
            lower_red2 = np.array([0, 0, 200])
            upper_red2 = np.array([50, 50, 255])
            
            mask1 = cv2.inRange(check_region, lower_red, upper_red)
            mask2 = cv2.inRange(check_region, lower_red2, upper_red2)
            
            combined_mask = cv2.bitwise_or(mask1, mask2)
            
            # If we find enough red pixels, there's likely a notification
            red_pixel_count = np.sum(combined_mask > 0)
            
            return red_pixel_count > 20  # Threshold for detecting notification
            
        except Exception as e:
            print(f"⚠️  Red notification detection error: {e}")
            return False
    
    def _extract_contact_name(self, image, card: ContactCard) -> str:
        """Extract contact name from card"""
        try:
            if not card.name_region:
                return "Unknown"
            
            nx, ny, nw, nh = card.name_region
            
            # Ensure bounds are within image
            ny_end = min(ny + nh, image.shape[0])
            nx_end = min(nx + nw, image.shape[1])
            
            if nx >= image.shape[1] or ny >= image.shape[0]:
                return "Unknown"
            
            name_area = image[ny:ny_end, nx:nx_end]
            
            if name_area.size == 0:
                return "Unknown"
            
            # Use OCR to extract name
            reader = self._get_ocr_reader()
            results = reader.readtext(name_area)
            
            if results:
                # Take the first/most prominent text as the name
                return results[0][1].strip()
            
            return "Unknown"
            
        except Exception as e:
            print(f"⚠️  Name extraction error: {e}")
            return "Unknown"
    
    def _extract_message_preview(self, image, card: ContactCard) -> Tuple[bool, str]:
        """Extract message preview from card"""
        try:
            if not card.message_region:
                return False, ""
            
            mx, my, mw, mh = card.message_region
            
            # Ensure bounds are within image
            my_end = min(my + mh, image.shape[0])
            mx_end = min(mx + mw, image.shape[1])
            
            if mx >= image.shape[1] or my >= image.shape[0]:
                return False, ""
            
            message_area = image[my:my_end, mx:mx_end]
            
            if message_area.size == 0:
                return False, ""
            
            # Use OCR to extract message
            reader = self._get_ocr_reader()
            results = reader.readtext(message_area)
            
            if results:
                message_text = " ".join([r[1].strip() for r in results])
                return True, message_text
            
            return False, ""
            
        except Exception as e:
            return False, ""
    
    def _extract_timestamp(self, image, card: ContactCard) -> str:
        """Extract timestamp from card"""
        try:
            if not card.timestamp_region:
                return ""
            
            tx, ty, tw, th = card.timestamp_region
            
            # Ensure bounds are within image
            ty_end = min(ty + th, image.shape[0])
            tx_end = min(tx + tw, image.shape[1])
            
            if tx >= image.shape[1] or ty >= image.shape[0]:
                return ""
            
            timestamp_area = image[ty:ty_end, tx:tx_end]
            
            if timestamp_area.size == 0:
                return ""
            
            # Use OCR to extract timestamp
            reader = self._get_ocr_reader()
            results = reader.readtext(timestamp_area)
            
            if results:
                return results[0][1].strip()
            
            return ""
            
        except Exception as e:
            return ""
    
    def _get_ocr_reader(self):
        """Get OCR reader with lazy initialization"""
        if self.ocr_reader is None:
            self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        return self.ocr_reader
    
    def _save_analysis_results(self, cards: List[ContactCard], screenshot_path: str):
        """Save contact card analysis results"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            results_file = os.path.join(self.output_dir, f'improved_analysis_{timestamp}.json')
            
            # Convert cards to serializable format
            cards_data = []
            for card in cards:
                card_data = {
                    'index': card.index,
                    'bounds': [card.x, card.y, card.width, card.height],
                    'avatar_bounds': list(card.avatar_bounds) if card.avatar_bounds else None,
                    'contact_name': card.contact_name,
                    'has_red_dot': bool(card.has_red_dot),  # Ensure bool conversion
                    'has_message': bool(card.has_message),  # Ensure bool conversion
                    'message_text': card.message_text,
                    'timestamp': card.timestamp_text,
                    'click_center': list(card.click_center) if card.click_center else None
                }
                cards_data.append(card_data)
            
            analysis_results = {
                'timestamp': datetime.now().isoformat(),
                'source_screenshot': os.path.basename(screenshot_path),
                'total_cards': len(cards),
                'cards_with_notifications': sum(1 for card in cards if card.has_red_dot),
                'cards_with_messages': sum(1 for card in cards if card.has_message),
                'contact_cards': cards_data
            }
            
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_results, f, indent=2, ensure_ascii=False)
            
            print(f"💾 Analysis results saved: {os.path.basename(results_file)}")
            
        except Exception as e:
            print(f"❌ Results saving error: {e}")
    
    def find_contacts_with_notifications(self, cards: List[ContactCard]) -> List[ContactCard]:
        """Find contact cards that have notifications"""
        return [card for card in cards if card.has_red_dot]
    
    def get_best_click_target(self, cards: List[ContactCard]) -> Optional[ContactCard]:
        """Get the best contact card to click"""
        # Priority: red notifications > messages > first card
        notification_cards = self.find_contacts_with_notifications(cards)
        if notification_cards:
            return notification_cards[0]
        
        message_cards = [card for card in cards if card.has_message]
        if message_cards:
            return message_cards[0]
        
        return cards[0] if cards else None


if __name__ == '__main__':
    """Test the improved contact card analyzer"""
    analyzer = ImprovedContactCardAnalyzer()
    
    # Find latest screenshot
    screenshot_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest_screenshot = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"🧪 Testing improved contact card analysis on: {latest_screenshot}")
        
        # Analyze contact cards
        cards = analyzer.analyze_contact_list(screenshot_path)
        
        if cards:
            print(f"\n📊 Analysis Summary:")
            print(f"   Total Cards: {len(cards)}")
            print(f"   With Red Notifications: {len(analyzer.find_contacts_with_notifications(cards))}")
            print(f"   With Messages: {len([c for c in cards if c.has_message])}")
            
            # Show best click target
            best_target = analyzer.get_best_click_target(cards)
            if best_target:
                print(f"\n🎯 Best Click Target:")
                print(f"   Contact: {best_target.contact_name}")
                print(f"   Position: {best_target.click_center}")
                print(f"   Has Red Dot: {best_target.has_red_dot}")
                print(f"   Message: {best_target.message_text[:50] if best_target.message_text else 'None'}")
        else:
            print("❌ No contact cards detected")
    else:
        print("❌ No test screenshots found")