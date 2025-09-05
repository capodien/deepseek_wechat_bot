#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OpenCV Adaptive Thresholding Avatar Detector
Improved design based on proven methodology with WeChat-specific optimizations
"""

import cv2
import numpy as np
import os
from typing import List, Dict
from datetime import datetime

class OpenCVAdaptiveDetector:
    """Improved avatar detection using adaptive thresholding and aspect ratio filtering"""
    
    def __init__(self):
        # Avatar constraints - relaxed to catch all avatars
        self.MIN_AVATAR_SIZE = 25      # Reduced from 35 to catch smaller avatars
        self.MAX_AVATAR_SIZE = 120     # Increased from 100 to catch larger avatars
        self.MIN_ASPECT_RATIO = 0.7    # Expanded from 0.85 for rectangular avatars
        self.MAX_ASPECT_RATIO = 1.4    # Expanded from 1.15 for rectangular avatars
        
        # WeChat-specific improvements
        self.AVATAR_ZONE_WIDTH = 120   # Slightly wider than reference
        self.CLICK_OFFSET_X = 70       # Click position in text area
        self.MIN_VERTICAL_SPACING = 40 # Reduced from 50 to allow closer avatars
        
        # Message card analysis parameters
        self.CARD_RIGHT_MARGIN = 20    # Margin from right edge of screen
        self.COMPONENT_SPACING = 10    # Spacing between components
        
        # Dynamic timestamp width ranges based on content:
        # "Monday" = ~60px, "Tuesday" = ~70px, "Yesterday 16:22" = ~120px
        self.MIN_TIMESTAMP_WIDTH = 60   # For very short like "Monday"
        self.MAX_TIMESTAMP_WIDTH = 140  # For very long like "Yesterday 16:22"
        
        # Adaptive threshold parameters - fine-tuned
        self.ADAPTIVE_BLOCK_SIZE = 9   # Reduced from 11 for finer detail
        self.ADAPTIVE_C = 2            # Reduced from 3 for more sensitive detection
        
        # Gaussian blur parameters
        self.BLUR_KERNEL_SIZE = (3, 3) # Reduced from (5,5) to preserve detail
        
    def detect_message_cards(self, image_path: str) -> List[Dict]:
        """
        Step 1: Detect message card boundaries first (user's suggested approach)
        Find the rectangular message cards using edge detection and contours
        """
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return []
        
        print(f"🎯 Card-first detection in: {os.path.basename(image_path)}")
        print(f"📐 Image: {img.shape[1]}x{img.shape[0]}")
        
        # Focus on the conversation area (exclude sidebar)
        img_height, img_width = img.shape[:2]
        conversation_width = int(img_width * 0.65)  # Focus on left 65% where cards are
        conversation_area = img[:, :conversation_width]
        
        print(f"  💬 Conversation area: {conversation_area.shape[1]}x{conversation_area.shape[0]}")
        
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(conversation_area, cv2.COLOR_BGR2GRAY)
        
        # Apply gentle blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        
        # Use edge detection to find card boundaries
        # WeChat cards have distinct horizontal lines between them
        edges = cv2.Canny(blurred, 30, 100, apertureSize=3)
        
        # Dilate to connect broken edges
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        edges = cv2.dilate(edges, kernel, iterations=1)
        
        # Find contours for card detection
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  📊 Found {len(contours)} edge contours")
        
        # Filter for card-like rectangles
        card_candidates = []
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            
            # Cards should be:
            # - Wide (much wider than tall)
            # - Reasonable size
            # - Positioned in conversation area
            aspect_ratio = w / float(h) if h > 0 else 0
            area = w * h
            
            if (w > 300 and  # Cards are wide
                h > 40 and h < 120 and  # Cards are not too tall or short
                aspect_ratio > 3 and  # Cards are much wider than tall
                area > 15000 and  # Cards have reasonable area
                x > 20):  # Not at very edge
                
                card_candidates.append({
                    'bbox': (x, y, w, h),
                    'aspect_ratio': aspect_ratio,
                    'area': area
                })
        
        print(f"  📇 Filtered to {len(card_candidates)} card candidates")
        
        # Sort top to bottom
        card_candidates = sorted(card_candidates, key=lambda c: c['bbox'][1])
        
        # Remove overlapping cards (prefer larger ones)
        filtered_cards = []
        for i, candidate in enumerate(card_candidates):
            cx, cy, cw, ch = candidate['bbox']
            
            # Check for overlap with already accepted cards
            overlap = False
            for accepted in filtered_cards:
                ax, ay, aw, ah = accepted['bbox']
                
                # Check vertical overlap
                if not (cy + ch < ay or ay + ah < cy):
                    overlap = True
                    break
            
            if not overlap:
                filtered_cards.append(candidate)
        
        print(f"  ✅ Final count: {len(filtered_cards)} message cards")
        
        return filtered_cards

    def detect_avatars(self, image_path: str) -> List[Dict]:
        """
        Step 1: First detect message cards, then find avatars within each card
        This follows the user's suggested approach: boundaries first, details second
        """
        
        # 1. Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return []
        
        print(f"🎯 Card-first OpenCV detection in: {os.path.basename(image_path)}")
        print(f"📐 Image: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Detect message card boundaries first
        message_cards = self.detect_message_cards(image_path)
        
        if not message_cards:
            print("❌ No message cards detected")
            return []
        
        # Step 2: For each message card, find the avatar within it
        contact_cards = []
        for i, card in enumerate(message_cards):
            card_x, card_y, card_w, card_h = card['bbox']
            
            # Look for avatar within this card (left portion)
            # Avatar should be in the first ~120px of the card
            avatar_search_area = img[card_y:card_y+card_h, card_x:min(card_x+120, card_x+card_w)]
            
            # Convert to grayscale and find circular/square avatar shapes
            gray_avatar = cv2.cvtColor(avatar_search_area, cv2.COLOR_BGR2GRAY)
            blurred_avatar = cv2.GaussianBlur(gray_avatar, (3, 3), 0)
            
            # Use adaptive threshold to find avatar shapes within the card
            adaptive_thresh = cv2.adaptiveThreshold(blurred_avatar, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
            
            # Find contours for avatar detection within this card
            contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Look for avatar-like shapes (circular or square)
            potential_avatars = []
            for contour in contours:
                area = cv2.contourArea(contour)
                if 200 < area < 3000:  # Avatar size range
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    # Check if shape is square-ish (avatar-like)
                    if 0.7 < aspect_ratio < 1.3:  # Roughly square
                        # Adjust coordinates back to full image
                        full_x = card_x + x
                        full_y = card_y + y
                        
                        potential_avatars.append({
                            'contour': contour,
                            'bbox': (full_x, full_y, w, h),
                            'area': area,
                            'aspect_ratio': aspect_ratio
                        })
            
            # Select best avatar candidate for this card (largest area)
            if potential_avatars:
                best_avatar = max(potential_avatars, key=lambda x: x['area'])
                avatar_x, avatar_y, avatar_w, avatar_h = best_avatar['bbox']
                
                # Calculate avatar center for clicking
                avatar_center_x = avatar_x + avatar_w // 2
                avatar_center_y = avatar_y + avatar_h // 2
                
                # Calculate username region (right of avatar within card)
                username_x = avatar_x + avatar_w + 10  # Small gap after avatar
                username_y = card_y + 5  # Top of card with small margin
                username_w = card_w - (avatar_w + 20)  # Rest of card width
                username_h = max(25, card_h // 3)  # Username height
                
                # Calculate timestamp region (below username)
                timestamp_x = username_x
                timestamp_y = username_y + username_h + 2
                timestamp_w = username_w
                timestamp_h = max(20, card_h // 4)
                
                # Calculate message content region (below timestamp)
                message_x = username_x
                message_y = timestamp_y + timestamp_h + 2
                message_w = username_w
                message_h = card_y + card_h - message_y
                
                # Create contact card with all regions relative to the detected card boundary
                contact_card = {
                    'card_bounds': (card_x, card_y, card_w, card_h),
                    'avatar_bounds': (avatar_x, avatar_y, avatar_w, avatar_h),
                    'click_position': (avatar_center_x, avatar_center_y),
                    'username_region': (username_x, username_y, username_w, username_h),
                    'timestamp_region': (timestamp_x, timestamp_y, timestamp_w, timestamp_h),
                    'message_region': (message_x, message_y, message_w, message_h),
                    'card_index': i
                }
                
                contact_cards.append(contact_card)
                print(f"  ✅ Card {i+1}: Avatar at ({avatar_center_x}, {avatar_center_y}), Card: {card_w}x{card_h}")
            else:
                print(f"  ⚠️ Card {i+1}: No avatar found within card boundary")
        
        print(f"🎯 Created {len(contact_cards)} contact cards from {len(message_cards)} message cards")
        return contact_cards

    def get_contact_regions(self, image_path: str) -> List[Dict]:
        """Get contact card regions with click coordinates (API compatibility)"""
        contact_cards = self.detect_avatars(image_path)
        
        # Convert to the expected API format
        formatted_cards = []
        for i, card in enumerate(contact_cards):
            formatted_card = {
                'card_id': i + 1,
                'card_bounds': card['card_bounds'],
                'avatar_bounds': card['avatar_bounds'], 
                'avatar_center': card['click_position'],  # Use click position as center
                'click_center': card['click_position'],
                'click_position': card['click_position'],
                'has_red_dot': False,  # Not implemented yet
                'aspect_ratio': card['card_bounds'][2] / card['card_bounds'][3] if card['card_bounds'][3] > 0 else 1.0,
                'area': card['card_bounds'][2] * card['card_bounds'][3],
                'username_region': card['username_region'],
                'timestamp_region': card['timestamp_region'], 
                'message_region': card['message_region'],
                'card_index': card['card_index']
            }
            formatted_cards.append(formatted_card)
        
        return formatted_cards

    def create_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create visualization with card-first approach showing:
        1. Message card boundaries (blue rectangles)
        2. Avatar detection within cards (green rectangles) 
        3. Username, timestamp, message regions (colored overlays)
        4. Click positions (red circles)
        """
        
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        print(f"🎨 Creating card-first visualization for: {os.path.basename(image_path)}")
        
        # Get detections using the new card-first approach
        contact_cards = self.detect_avatars(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Draw each detected contact card with all regions
        for i, card in enumerate(contact_cards):
            card_x, card_y, card_w, card_h = card['card_bounds']
            avatar_x, avatar_y, avatar_w, avatar_h = card['avatar_bounds']
            click_x, click_y = card['click_position']
            username_x, username_y, username_w, username_h = card['username_region']
            timestamp_x, timestamp_y, timestamp_w, timestamp_h = card['timestamp_region']
            message_x, message_y, message_w, message_h = card['message_region']
            
            # 1. Draw message card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            cv2.putText(result, f"Card {i+1}", (card_x, card_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            
            # 2. Draw avatar detection area (green rectangle)
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            cv2.putText(result, "Avatar", (avatar_x, avatar_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            
            # 3. Draw click position (red circle)
            cv2.circle(result, (click_x, click_y), 5, (0, 0, 255), -1)
            cv2.putText(result, f"Click ({click_x},{click_y})", (click_x + 8, click_y - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
            
            # 4. Draw username region (cyan overlay)
            cv2.rectangle(result, (username_x, username_y), (username_x + username_w, username_y + username_h), (255, 255, 0), 1)
            cv2.putText(result, "Username", (username_x, username_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
            
            # 5. Draw timestamp region (magenta overlay)
            cv2.rectangle(result, (timestamp_x, timestamp_y), (timestamp_x + timestamp_w, timestamp_y + timestamp_h), (255, 0, 255), 1)
            cv2.putText(result, "Time", (timestamp_x, timestamp_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 0, 255), 1)
            
            # 6. Draw message region (yellow overlay)
            cv2.rectangle(result, (message_x, message_y), (message_x + message_w, message_y + message_h), (0, 255, 255), 1)
            cv2.putText(result, "Message", (message_x, message_y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
        
        # Add legend showing color coding
        legend_y = 30
        cv2.putText(result, "Legend:", (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, "Blue=Card  Green=Avatar  Red=Click  Cyan=Username  Magenta=Time  Yellow=Message", 
                   (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection summary
        summary = f"Detected: {len(contact_cards)} message cards with avatars"
        cv2.putText(result, summary, (10, legend_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Generate output filename with timestamp
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/opencv_adaptive_result_{timestamp}.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Card-first visualization saved: {output_path}")
        
        return os.path.basename(output_path)
