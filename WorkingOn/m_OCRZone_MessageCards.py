#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 Module: Step 2.5 - OCR Zone Boundaries Module
 File:   m_OCRZone_MessageCards.py
=====================================================================

📌 Version Info
- Version:        v1.0.0
- Created:        2024-09-04
- Last Modified:  2024-09-04
- Author:         AI Assistant (WeChat Bot System)

📌 Module Position
- Pipeline Step:  Step 2.5: OCR Zone Definition
- Previous Step:  Step 2.0 - Message Card Detection (opencv_adaptive_detector.py)
- Next Step:      Step 3.0 - OCR Processing (deal_chatbox.py)

📌 Input Contract
- define_ocr_zones() Parameters:
    message_cards: List[Dict] - Detected message cards with avatar coordinates
        Format: [{"card_number": int, "avatar_center": (x, y), "avatar_bounds": (x, y, w, h), ...}]
    screenshot_path: str - Path to screenshot image for validation
    adaptive_sizing: bool (default: True) - Whether to use adaptive zone sizing
- OCRZoneMessageCards() Constructor:
    enable_visual_validation: bool (default: True) - Generate overlay images
    zone_padding: int (default: 5) - Padding pixels around each zone
- System Prerequisites:
    OpenCV for image processing
    Detected message cards from avatar detection system
    Screenshot image file for validation

📌 Output Contract
- Produced State(s):
    Enhanced message cards with precise OCR zone boundaries
    Visual overlay images for validation (if enabled)
    Zone definition metrics and performance data
- Data Format:
    List[Dict] with added zone definitions:
    {
        "card_number": int,
        "avatar_zone": {"x": int, "y": int, "width": int, "height": int},
        "username_zone": {"x": int, "y": int, "width": int, "height": int},
        "timestamp_zone": {"x": int, "y": int, "width": int, "height": int},
        "message_preview_zone": {"x": int, "y": int, "width": int, "height": int},
        "zone_confidence": float,
        "validation_status": str
    }
- Performance Targets:
    Zone calculation: <100ms for 10 message cards
    Overlay generation: <200ms per image
    Memory usage: <50MB for typical operations

📌 Key Features
- Adaptive zone sizing based on card dimensions
- Cross-platform coordinate accuracy
- Visual validation through overlay generation
- Performance-optimized zone calculations
- Comprehensive error handling and edge case support
- Integration with existing WDC diagnostic interface

📌 Core Dependencies
- cv2 (OpenCV): Image processing and overlay generation
- numpy: Numerical operations and array handling
- typing: Type hints for better code maintainability
- os, time: File system and performance monitoring
- TestRun.opencv_adaptive_detector: Avatar detection integration

=====================================================================
"""

# Original module documentation preserved above
# WeChat desktop automation - OCR Zone Boundaries Module
# Defines precise OCR zones for each component within WeChat message cards

import cv2
import numpy as np
import os
import time
from typing import List, Dict, Optional, Tuple
from datetime import datetime

# Import enhanced boundary detection
try:
    from WorkingOn.m_CardBoundaryDetection import CardBoundaryDetector
    BOUNDARY_DETECTION_AVAILABLE = True
except ImportError:
    BOUNDARY_DETECTION_AVAILABLE = False
    print("⚠️ Enhanced boundary detection not available - using legacy mode")

class OCRZoneMessageCards:
    """
    OCR Zone Boundaries Module for WeChat Message Cards
    
    Defines precise OCR zones for each component within detected message cards:
    - Avatar Zone: Circular profile picture area
    - Username Zone: Contact name text area  
    - Timestamp Zone: Time/date display area
    - Message Preview Zone: Last message content area
    """
    
    def __init__(self, enable_visual_validation: bool = True, zone_padding: int = 5):
        """
        Initialize OCR Zone MessageCards module
        
        Args:
            enable_visual_validation: Generate overlay images for validation
            zone_padding: Padding pixels around each zone
        """
        self.enable_visual_validation = enable_visual_validation
        self.zone_padding = zone_padding
        
        # Zone dimension configuration (adaptive defaults)
        self.AVATAR_SIZE = 50           # Standard avatar diameter
        self.USERNAME_WIDTH = 200       # Username text area width
        self.USERNAME_HEIGHT = 25       # Username text area height
        self.TIMESTAMP_WIDTH = 100      # Timestamp area width
        self.TIMESTAMP_HEIGHT = 20      # Timestamp area height
        self.MESSAGE_PREVIEW_WIDTH = 250 # Message preview area width
        self.MESSAGE_PREVIEW_HEIGHT = 20 # Message preview area height
        
        # Positioning offsets (relative to avatar)
        self.USERNAME_OFFSET_X = 10     # Username starts 10px right of avatar
        self.USERNAME_OFFSET_Y = -5     # Username slightly above avatar center
        self.TIMESTAMP_OFFSET_X = 200   # Timestamp position from avatar
        self.TIMESTAMP_OFFSET_Y = -10   # Timestamp above avatar center
        self.MESSAGE_OFFSET_X = 10      # Message preview starts 10px right of avatar
        self.MESSAGE_OFFSET_Y = 15      # Message preview below avatar center
        
        # Adaptive sizing parameters
        self.MIN_CARD_WIDTH = 300       # Minimum card width for sizing
        self.MAX_CARD_WIDTH = 600       # Maximum card width for sizing
        self.CARD_HEIGHT_THRESHOLD = 80 # Height threshold for layout adjustments
        
        # Visual validation colors (BGR format for OpenCV)
        self.COLORS = {
            'avatar': (255, 100, 100),      # Light blue for avatar
            'username': (100, 255, 100),    # Light green for username
            'timestamp': (100, 100, 255),   # Light red for timestamp
            'message_preview': (255, 255, 100), # Light cyan for message preview
            'card_boundary': (255, 255, 255) # White for card boundaries
        }
        
        print(f"🎯 OCRZoneMessageCards initialized (validation: {enable_visual_validation})")
    
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
    
    def define_ocr_zones(self, message_cards: List[Dict], screenshot_path: Optional[str] = None, 
                        adaptive_sizing: bool = True) -> List[Dict]:
        """
        Define precise OCR zone boundaries for each message card
        
        Args:
            message_cards: List of detected message cards with avatar coordinates
            screenshot_path: Path to screenshot for validation (if None, uses latest from pic/screenshots)
            adaptive_sizing: Whether to use adaptive zone sizing
            
        Returns:
            Enhanced message cards with OCR zone definitions
        """
        print(f"\n📐 Defining OCR zones for {len(message_cards)} message cards")
        start_time = time.time()
        
        if not message_cards:
            print("⚠️ No message cards provided for zone definition")
            return []
        
        # Auto-detect latest screenshot if no path provided
        if screenshot_path is None:
            screenshot_path = self.get_latest_screenshot()
            if screenshot_path is None:
                print("⚠️ No screenshot available for visual validation")
        
        # Load screenshot for validation if visual mode enabled
        screenshot_image = None
        if self.enable_visual_validation and screenshot_path and os.path.exists(screenshot_path):
            screenshot_image = cv2.imread(screenshot_path)
            if screenshot_image is None:
                print(f"⚠️ Could not load screenshot: {screenshot_path}")
        elif self.enable_visual_validation and screenshot_path:
            print(f"⚠️ Screenshot file not found: {screenshot_path}")
        
        # Process each message card to define OCR zones
        enhanced_cards = []
        for card in message_cards:
            try:
                enhanced_card = self._define_single_card_zones(card, adaptive_sizing)
                enhanced_cards.append(enhanced_card)
                
                # Log zone summary
                card_num = enhanced_card.get('card_number', 'Unknown')
                confidence = enhanced_card.get('zone_confidence', 0.0)
                status = enhanced_card.get('validation_status', 'Unknown')
                print(f"  ✅ Card #{card_num}: Zones defined (confidence: {confidence:.2f}, status: {status})")
                
            except Exception as e:
                print(f"❌ Error defining zones for card {card.get('card_number', 'Unknown')}: {e}")
                # Add error card with minimal data
                error_card = card.copy()
                error_card.update({
                    'zone_confidence': 0.0,
                    'validation_status': 'error',
                    'error_message': str(e)
                })
                enhanced_cards.append(error_card)
        
        # Generate visual validation overlay if enabled
        if self.enable_visual_validation and screenshot_image is not None:
            self._generate_zone_overlay(enhanced_cards, screenshot_image, screenshot_path)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"⚡ OCR zones defined in {processing_time:.1f}ms")
        
        return enhanced_cards
    
    def _define_single_card_zones(self, card: Dict, adaptive_sizing: bool) -> Dict:
        """
        Define OCR zones for a single message card
        
        Args:
            card: Single message card data with avatar coordinates
            adaptive_sizing: Whether to use adaptive sizing
            
        Returns:
            Enhanced card with OCR zone definitions
        """
        enhanced_card = card.copy()
        
        # Extract avatar information
        avatar_center = card.get('avatar_center')
        avatar_bounds = card.get('avatar_bounds')
        card_number = card.get('card_number', 0)
        
        if not avatar_center:
            raise ValueError(f"Card {card_number}: Missing avatar_center coordinates")
        
        avatar_x, avatar_y = avatar_center
        
        # Calculate card dimensions for adaptive sizing
        card_width = self.MAX_CARD_WIDTH  # Default width
        card_height = self.CARD_HEIGHT_THRESHOLD  # Default height
        
        if avatar_bounds and len(avatar_bounds) >= 4:
            # Use avatar bounds to estimate card dimensions
            _, _, bounds_w, bounds_h = avatar_bounds
            card_height = max(bounds_h + 40, self.CARD_HEIGHT_THRESHOLD)  # Add padding
        
        # Apply adaptive sizing adjustments
        width_factor = 1.0
        height_factor = 1.0
        
        if adaptive_sizing:
            # Adjust zone sizes based on card dimensions
            if card_width > self.MAX_CARD_WIDTH:
                width_factor = 1.2
            elif card_width < self.MIN_CARD_WIDTH:
                width_factor = 0.8
                
            if card_height > self.CARD_HEIGHT_THRESHOLD:
                height_factor = 1.1
        
        # Define Avatar Zone (circular area around detected avatar)
        avatar_zone = {
            "x": max(0, int(avatar_x - self.AVATAR_SIZE // 2)),
            "y": max(0, int(avatar_y - self.AVATAR_SIZE // 2)),
            "width": int(self.AVATAR_SIZE * width_factor),
            "height": int(self.AVATAR_SIZE * height_factor)
        }
        
        # Define Username Zone (right of avatar, top area)
        username_zone = {
            "x": int(avatar_x + self.USERNAME_OFFSET_X),
            "y": max(0, int(avatar_y + self.USERNAME_OFFSET_Y)),
            "width": int(self.USERNAME_WIDTH * width_factor),
            "height": int(self.USERNAME_HEIGHT * height_factor)
        }
        
        # Define Timestamp Zone (right side of card)
        timestamp_zone = {
            "x": int(avatar_x + self.TIMESTAMP_OFFSET_X * width_factor),
            "y": max(0, int(avatar_y + self.TIMESTAMP_OFFSET_Y)),
            "width": int(self.TIMESTAMP_WIDTH * width_factor),
            "height": int(self.TIMESTAMP_HEIGHT * height_factor)
        }
        
        # Define Message Preview Zone (right of avatar, bottom area)
        message_preview_zone = {
            "x": int(avatar_x + self.MESSAGE_OFFSET_X),
            "y": int(avatar_y + self.MESSAGE_OFFSET_Y),
            "width": int(self.MESSAGE_PREVIEW_WIDTH * width_factor),
            "height": int(self.MESSAGE_PREVIEW_HEIGHT * height_factor)
        }
        
        # Add padding to all zones
        for zone in [avatar_zone, username_zone, timestamp_zone, message_preview_zone]:
            zone["x"] = max(0, zone["x"] - self.zone_padding)
            zone["y"] = max(0, zone["y"] - self.zone_padding)
            zone["width"] += 2 * self.zone_padding
            zone["height"] += 2 * self.zone_padding
        
        # Calculate zone confidence based on positioning validity
        confidence = self._calculate_zone_confidence(
            avatar_center, [avatar_zone, username_zone, timestamp_zone, message_preview_zone]
        )
        
        # Update enhanced card with zone definitions
        enhanced_card.update({
            'avatar_zone': avatar_zone,
            'username_zone': username_zone,
            'timestamp_zone': timestamp_zone,
            'message_preview_zone': message_preview_zone,
            'zone_confidence': confidence,
            'validation_status': 'success' if confidence > 0.7 else 'warning',
            'adaptive_factors': {'width_factor': width_factor, 'height_factor': height_factor}
        })
        
        return enhanced_card
    
    def _calculate_zone_confidence(self, avatar_center: Tuple[int, int], zones: List[Dict]) -> float:
        """
        Calculate confidence score for zone definitions
        
        Args:
            avatar_center: Avatar center coordinates
            zones: List of defined zones
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        confidence_factors = []
        
        # Check if zones are reasonable relative to avatar
        avatar_x, avatar_y = avatar_center
        
        for zone in zones:
            zone_x = zone.get('x', 0)
            zone_y = zone.get('y', 0)
            zone_w = zone.get('width', 0)
            zone_h = zone.get('height', 0)
            
            # Factor 1: Zone has reasonable dimensions
            size_factor = 1.0 if (zone_w > 10 and zone_h > 10 and zone_w < 400 and zone_h < 100) else 0.5
            
            # Factor 2: Zone is positioned reasonably relative to avatar
            distance_from_avatar = ((zone_x - avatar_x) ** 2 + (zone_y - avatar_y) ** 2) ** 0.5
            position_factor = 1.0 if distance_from_avatar < 300 else 0.7
            
            # Factor 3: Zone coordinates are positive
            coordinate_factor = 1.0 if (zone_x >= 0 and zone_y >= 0) else 0.3
            
            zone_confidence = (size_factor + position_factor + coordinate_factor) / 3.0
            confidence_factors.append(zone_confidence)
        
        # Overall confidence is average of all zone confidences
        return sum(confidence_factors) / len(confidence_factors) if confidence_factors else 0.0
    
    def _generate_zone_overlay(self, enhanced_cards: List[Dict], screenshot_image: np.ndarray, 
                              screenshot_path: str) -> Optional[str]:
        """
        Generate visual overlay showing all OCR zones
        
        Args:
            enhanced_cards: Message cards with OCR zone definitions
            screenshot_image: Screenshot image array
            screenshot_path: Original screenshot path
            
        Returns:
            Path to generated overlay image, or None if failed
        """
        try:
            print(f"🎨 Generating OCR zone overlay for {len(enhanced_cards)} cards")
            
            # Create overlay image (copy of original)
            overlay_img = screenshot_image.copy()
            
            # Draw zones for each card
            for card in enhanced_cards:
                card_number = card.get('card_number', 0)
                
                # Draw each zone type with different colors
                zone_types = ['avatar_zone', 'username_zone', 'timestamp_zone', 'message_preview_zone']
                
                for zone_type in zone_types:
                    zone = card.get(zone_type)
                    if zone and all(k in zone for k in ['x', 'y', 'width', 'height']):
                        self._draw_zone_rectangle(overlay_img, zone, zone_type, card_number)
                
                # Draw card number label at avatar center
                avatar_center = card.get('avatar_center')
                if avatar_center:
                    cv2.putText(overlay_img, f"#{card_number}", 
                              (avatar_center[0] - 10, avatar_center[1] - 60),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Add legend
            self._add_overlay_legend(overlay_img)
            
            # Save overlay image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overlay_filename = f"OCRZones_Overlay_{timestamp}.png"
            overlay_dir = os.path.dirname(screenshot_path)
            overlay_path = os.path.join(overlay_dir, overlay_filename)
            
            cv2.imwrite(overlay_path, overlay_img)
            print(f"💾 OCR zones overlay saved: {overlay_filename}")
            
            return overlay_path
            
        except Exception as e:
            print(f"❌ Error generating zone overlay: {e}")
            return None
    
    def _draw_zone_rectangle(self, img: np.ndarray, zone: Dict, zone_type: str, card_number: int):
        """Draw a single zone rectangle on the overlay image"""
        x, y = zone['x'], zone['y']
        w, h = zone['width'], zone['height']
        
        # Get color for this zone type
        color = self.COLORS.get(zone_type, (128, 128, 128))
        
        # Draw rectangle with semi-transparent fill
        overlay = img.copy()
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
        cv2.addWeighted(overlay, 0.3, img, 0.7, 0, img)
        
        # Draw border
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
        
        # Add zone type label
        label = zone_type.replace('_zone', '').replace('_', ' ').title()
        cv2.putText(img, label, (x + 2, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
    
    def _add_overlay_legend(self, img: np.ndarray):
        """Add color legend to overlay image"""
        legend_x, legend_y = 20, 30
        line_height = 25
        
        # Background rectangle for legend
        cv2.rectangle(img, (legend_x - 5, legend_y - 20), 
                     (legend_x + 200, legend_y + len(self.COLORS) * line_height), 
                     (0, 0, 0), -1)
        
        # Add title
        cv2.putText(img, "OCR Zone Types:", (legend_x, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Add each zone type with its color
        for i, (zone_type, color) in enumerate(self.COLORS.items()):
            if zone_type != 'card_boundary':  # Skip card boundary in legend
                y_pos = legend_y + (i + 1) * line_height
                
                # Draw color square
                cv2.rectangle(img, (legend_x, y_pos - 8), (legend_x + 15, y_pos + 2), color, -1)
                
                # Add label
                label = zone_type.replace('_', ' ').title()
                cv2.putText(img, label, (legend_x + 20, y_pos), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    def define_ocr_zones_enhanced(self, screenshot_path: Optional[str] = None, 
                                 row_height: int = 80, adaptive_sizing: bool = True) -> List[Dict]:
        """
        Enhanced OCR zone definition using 1-D vertical-edge projection boundary detection
        
        This method uses the new CardBoundaryDetector to get precise message card boundaries,
        then calculates OCR zones within those boundaries for maximum accuracy.
        
        Args:
            screenshot_path: Path to screenshot image (if None, uses latest from pic/screenshots)
            row_height: Expected height of message card rows
            adaptive_sizing: Whether to use adaptive zone sizing
            
        Returns:
            Enhanced message cards with precise OCR zone definitions
        """
        print(f"\n📐 Enhanced OCR zone definition with 1-D projection boundary detection")
        start_time = time.time()
        
        # Auto-detect latest screenshot if no path provided
        if screenshot_path is None:
            screenshot_path = self.get_latest_screenshot()
            if screenshot_path is None:
                print("⚠️ No screenshot available for enhanced zone definition")
                return []
        
        # Check if enhanced boundary detection is available
        if not BOUNDARY_DETECTION_AVAILABLE:
            print("⚠️ Enhanced boundary detection not available, falling back to legacy method")
            # Use existing method with empty message cards (will auto-detect)
            return self.define_ocr_zones([], screenshot_path, adaptive_sizing)
        
        # Use enhanced boundary detection
        boundary_detector = CardBoundaryDetector(
            enable_visual_validation=False,  # We'll create our own overlay
            adaptive_threshold_factor=2.5
        )
        
        # Detect precise card boundaries
        card_boundaries = boundary_detector.detect_card_boundaries(
            screenshot_path, row_height, left_margin=0
        )
        
        if not card_boundaries:
            print("❌ No card boundaries detected with enhanced method")
            return []
        
        print(f"  🎯 Detected {len(card_boundaries)} precise card boundaries")
        
        # Convert boundary data to message card format for zone calculation
        message_cards = []
        for boundary in card_boundaries:
            # Calculate avatar position within the detected boundary
            # Assume avatar is positioned at left side of card with some margin
            avatar_margin = 20  # Distance from left edge to avatar center
            avatar_x = boundary['left_x'] + avatar_margin + 25  # 25px avatar radius
            avatar_y = boundary['top_y'] + boundary['height'] // 2  # Vertical center
            
            message_cards.append({
                "card_number": boundary['card_number'],
                "avatar_center": (avatar_x, avatar_y),
                "avatar_bounds": (avatar_x - 25, avatar_y - 25, 50, 50),
                "detection_confidence": boundary['confidence'],
                "boundary_data": boundary,  # Keep original boundary data
                "detection_method": "enhanced_1d_projection"
            })
        
        # Define OCR zones using the precise boundaries
        enhanced_cards = self.define_ocr_zones(message_cards, screenshot_path, adaptive_sizing)
        
        # Enhance the results with boundary information
        for i, card in enumerate(enhanced_cards):
            if i < len(card_boundaries):
                boundary = card_boundaries[i]
                card.update({
                    'precise_boundaries': {
                        'left_x': boundary['left_x'],
                        'right_x': boundary['right_x'],
                        'top_y': boundary['top_y'],
                        'bottom_y': boundary['bottom_y'],
                        'panel_width': boundary['width'],
                        'card_height': boundary['height']
                    },
                    'boundary_confidence': boundary['confidence'],
                    'edge_strength': boundary['edge_strength'],
                    'enhanced_detection': True
                })
        
        processing_time = (time.time() - start_time) * 1000
        print(f"⚡ Enhanced OCR zones defined in {processing_time:.1f}ms")
        
        return enhanced_cards
    
    def define_ocr_zones_avatar_first(self, avatar_positions: List[Tuple[int, int]], 
                                    screenshot_path: Optional[str] = None, 
                                    adaptive_sizing: bool = True) -> List[Dict]:
        """
        Avatar-first OCR zone definition with dynamic width detection
        
        This method uses avatar positions to dynamically detect each card's width,
        then calculates OCR zones within those precise boundaries. Height calculation
        can be determined later based on content or avatar size.
        
        Args:
            avatar_positions: List of (x, y) avatar center coordinates
            screenshot_path: Path to screenshot image (if None, uses latest from pic/screenshots)
            adaptive_sizing: Whether to use adaptive zone sizing
            
        Returns:
            Message cards with precise OCR zone definitions based on dynamic widths
        """
        print(f"\n🎯 Avatar-first OCR zone definition for {len(avatar_positions)} avatars")
        start_time = time.time()
        
        # Auto-detect latest screenshot if no path provided
        if screenshot_path is None:
            screenshot_path = self.get_latest_screenshot()
            if screenshot_path is None:
                print("⚠️ No screenshot available for avatar-first zone definition")
                return []
        
        # Check if enhanced boundary detection is available
        if not BOUNDARY_DETECTION_AVAILABLE:
            print("⚠️ Enhanced boundary detection not available, using fallback method")
            return []
        
        # Use dynamic width detection from avatar positions
        from WorkingOn.m_CardBoundaryDetection import detect_dynamic_card_widths
        card_widths = detect_dynamic_card_widths(screenshot_path, avatar_positions)
        
        if not card_widths:
            print("❌ No dynamic widths detected from avatar positions")
            return []
        
        print(f"  🎯 Detected {len(card_widths)} dynamic card widths")
        
        # Convert width data to message card format for zone calculation
        message_cards = []
        for width_data in card_widths:
            avatar_x, avatar_y = width_data['avatar_position']
            
            # Calculate estimated card height based on avatar size (can be refined later)
            estimated_height = 80  # Default height, can be calculated dynamically later
            
            message_cards.append({
                "card_number": width_data['card_number'],
                "avatar_center": (avatar_x, avatar_y),
                "avatar_bounds": (avatar_x - 25, avatar_y - 25, 50, 50),
                "detection_confidence": width_data['confidence'],
                "dynamic_width_data": width_data,  # Keep original width data
                "detection_method": "avatar_first_dynamic_width"
            })
        
        # Define OCR zones using the dynamic width boundaries
        enhanced_cards = []
        
        for i, card in enumerate(message_cards):
            width_data = card['dynamic_width_data']
            avatar_x, avatar_y = card['avatar_center']
            
            # Use detected width boundaries for zone calculation
            left_x = width_data['left_x']
            right_x = width_data['right_x']
            card_width = width_data['width']
            
            # Calculate zones based on dynamic width
            zones = self._calculate_dynamic_zones(
                avatar_x, avatar_y, left_x, right_x, card_width, adaptive_sizing
            )
            
            # Create enhanced card with dynamic width zones
            enhanced_card = {
                "card_number": card['card_number'],
                "avatar_center": card['avatar_center'],
                "avatar_bounds": card['avatar_bounds'],
                "detection_method": card['detection_method'],
                "detection_confidence": card['detection_confidence'],
                
                # OCR Zones
                "avatar_zone": zones['avatar_zone'],
                "username_zone": zones['username_zone'], 
                "timestamp_zone": zones['timestamp_zone'],
                "message_preview_zone": zones['message_preview_zone'],
                
                # Dynamic width information
                "dynamic_boundaries": {
                    'left_x': left_x,
                    'right_x': right_x,
                    'width': card_width,
                    'confidence': width_data['confidence']
                },
                
                "zone_confidence": min(1.0, width_data['confidence'] + 0.2),
                "validation_status": "avatar_first_success",
                "enhanced_detection": True
            }
            
            enhanced_cards.append(enhanced_card)
            print(f"  ✅ Card #{enhanced_card['card_number']}: Dynamic zones defined (width: {card_width}px, confidence: {width_data['confidence']:.2f})")
        
        # Generate visual validation if enabled
        if self.enable_visual_validation:
            self._generate_avatar_first_overlay(screenshot_path, enhanced_cards)
        
        processing_time = (time.time() - start_time) * 1000
        print(f"⚡ Avatar-first OCR zones defined in {processing_time:.1f}ms")
        
        return enhanced_cards
    
    def _calculate_dynamic_zones(self, avatar_x: int, avatar_y: int, 
                               left_x: int, right_x: int, card_width: int, 
                               adaptive_sizing: bool) -> Dict:
        """
        Calculate OCR zones within dynamically detected card boundaries
        
        Args:
            avatar_x, avatar_y: Avatar center coordinates
            left_x, right_x: Dynamic card boundaries
            card_width: Detected card width
            adaptive_sizing: Whether to use adaptive sizing
            
        Returns:
            Dictionary with zone definitions
        """
        # Avatar zone (fixed around detected avatar)
        avatar_zone = {
            "x": avatar_x - 25,
            "y": avatar_y - 25,
            "width": 50,
            "height": 50
        }
        
        # Username zone (adaptive width based on card size)
        username_width = min(200, int(card_width * 0.4))  # Up to 40% of card width
        username_zone = {
            "x": avatar_x + 35,  # Right of avatar with margin
            "y": avatar_y - 30,  # Above avatar center
            "width": username_width,
            "height": 25
        }
        
        # Timestamp zone (dynamic positioning from right edge)
        timestamp_width = min(120, int(card_width * 0.25))  # Up to 25% of card width
        if adaptive_sizing and card_width > 450:
            timestamp_width = 140  # Wider for large cards
        elif card_width < 350:
            timestamp_width = 80   # Narrower for small cards
            
        timestamp_zone = {
            "x": right_x - timestamp_width - 20,  # From right edge with margin
            "y": avatar_y - 30,
            "width": timestamp_width,
            "height": 20
        }
        
        # Message preview zone (remaining space)
        preview_start_x = username_zone["x"] + username_zone["width"] + 10
        preview_end_x = timestamp_zone["x"] - 10
        preview_width = max(100, preview_end_x - preview_start_x)  # Ensure minimum width
        
        message_preview_zone = {
            "x": preview_start_x,
            "y": avatar_y + 5,  # Below avatar center
            "width": preview_width,
            "height": 20
        }
        
        return {
            'avatar_zone': avatar_zone,
            'username_zone': username_zone,
            'timestamp_zone': timestamp_zone, 
            'message_preview_zone': message_preview_zone
        }
    
    def _generate_avatar_first_overlay(self, screenshot_path: str, 
                                     enhanced_cards: List[Dict]) -> Optional[str]:
        """
        Generate visual overlay showing avatar-first dynamic width zones
        
        Args:
            screenshot_path: Path to original screenshot
            enhanced_cards: List of cards with dynamic zones
            
        Returns:
            Path to generated overlay image, or None if failed
        """
        try:
            import cv2
            
            print(f"  🎨 Generating avatar-first overlay for {len(enhanced_cards)} cards")
            
            img = cv2.imread(screenshot_path)
            if img is None:
                return None
            
            overlay_img = img.copy()
            
            # Draw dynamic boundaries and zones for each card
            for card in enhanced_cards:
                # Draw dynamic card boundary
                boundaries = card['dynamic_boundaries']
                left_x = boundaries['left_x']
                right_x = boundaries['right_x']
                avatar_y = card['avatar_center'][1]
                
                # Card boundary rectangle (yellow for dynamic width)
                cv2.rectangle(overlay_img, (left_x, avatar_y - 40), (right_x, avatar_y + 40), 
                             (0, 255, 255), 3)  # Yellow
                
                # Draw individual OCR zones with distinct colors
                zones = [
                    (card['avatar_zone'], self.COLORS['avatar'], 'Avatar'),
                    (card['username_zone'], self.COLORS['username'], 'Username'),
                    (card['timestamp_zone'], self.COLORS['timestamp'], 'Timestamp'),
                    (card['message_preview_zone'], self.COLORS['message_preview'], 'Message')
                ]
                
                for zone, color, label in zones:
                    x, y, w, h = zone['x'], zone['y'], zone['width'], zone['height']
                    cv2.rectangle(overlay_img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(overlay_img, f"{label} ({w}x{h})", (x + 2, y + 15),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                
                # Add card info
                card_info = f"Card #{card['card_number']}: {boundaries['width']}px (conf: {boundaries['confidence']:.2f})"
                cv2.putText(overlay_img, card_info, (left_x + 5, avatar_y - 50),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            
            # Add title
            title = f"Avatar-First Dynamic Width Detection: {len(enhanced_cards)} cards"
            cv2.putText(overlay_img, title, (20, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            # Save overlay image
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            overlay_filename = f"AvatarFirst_DynamicWidth_{timestamp}.png"
            overlay_dir = os.path.dirname(screenshot_path)
            overlay_path = os.path.join(overlay_dir, overlay_filename)
            
            cv2.imwrite(overlay_path, overlay_img)
            print(f"  💾 Avatar-first overlay saved: {overlay_filename}")
            
            return overlay_path
            
        except Exception as e:
            print(f"  ❌ Error generating avatar-first overlay: {e}")
            return None


# Utility functions for integration with existing system
def create_ocr_zone_processor(enable_validation: bool = True, padding: int = 5) -> OCRZoneMessageCards:
    """
    Factory function to create OCR Zone processor instance
    
    Args:
        enable_validation: Enable visual validation overlays
        padding: Zone padding in pixels
        
    Returns:
        OCRZoneMessageCards instance
    """
    return OCRZoneMessageCards(enable_visual_validation=enable_validation, zone_padding=padding)


def process_message_cards_zones(message_cards: List[Dict], screenshot_path: Optional[str] = None, 
                               adaptive: bool = True) -> List[Dict]:
    """
    Convenience function to process message cards and define OCR zones
    
    Args:
        message_cards: Detected message cards
        screenshot_path: Path to screenshot image (if None, uses latest from pic/screenshots)
        adaptive: Use adaptive zone sizing
        
    Returns:
        Message cards with OCR zone definitions
    """
    processor = create_ocr_zone_processor()
    return processor.define_ocr_zones(message_cards, screenshot_path, adaptive)


# Main execution for standalone testing
if __name__ == "__main__":
    print("🧪 OCR Zone MessageCards - Standalone Testing")
    
    # Example test data
    test_cards = [
        {
            "card_number": 1,
            "avatar_center": (50, 100),
            "avatar_bounds": (25, 75, 50, 50),
            "detection_confidence": 0.95
        },
        {
            "card_number": 2,
            "avatar_center": (50, 200),
            "avatar_bounds": (25, 175, 50, 50),
            "detection_confidence": 0.88
        }
    ]
    
    # Test zone definition
    processor = create_ocr_zone_processor()
    enhanced_cards = processor.define_ocr_zones(test_cards, "", adaptive_sizing=True)
    
    print(f"\n📊 Test Results:")
    for card in enhanced_cards:
        print(f"Card #{card['card_number']}:")
        print(f"  Avatar Zone: {card.get('avatar_zone', 'Not defined')}")
        print(f"  Username Zone: {card.get('username_zone', 'Not defined')}")
        print(f"  Confidence: {card.get('zone_confidence', 0.0):.2f}")