#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Card Boundary Detector
Detects complete boundaries (left, right, top, bottom) of WeChat message cards
Builds upon SimpleWidthDetector for horizontal boundaries and adds vertical detection
"""

import cv2
import numpy as np
import os
import sys
from typing import Optional, List, Dict, Tuple
from datetime import datetime

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import width detector for left/right boundaries
from modules.m_Card_Processing import SimpleWidthDetector

# Import avatar detector
try:
    from modules.card_avatar_detector import CardAvatarDetector
    AVATAR_DETECTION_AVAILABLE = True
except ImportError:
    AVATAR_DETECTION_AVAILABLE = False
    print("⚠️ Avatar detection not available - will use basic edge detection only")

def find_horizontal_edge_y(img, x0=0, x1=None, y0=0, y1=None, bottommost=True):
    """
    Find horizontal edges (card separators) in the image
    Similar to find_vertical_edge_x but for horizontal edges
    """
    if img.ndim == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    
    H, W = gray.shape
    x1 = W if x1 is None else x1
    y1 = H if y1 is None else y1
    roi = gray[y0:y1, x0:x1]
    
    # Denoise for better edge detection
    roi = cv2.bilateralFilter(roi, d=5, sigmaColor=25, sigmaSpace=25)
    
    # Compute vertical gradient (horizontal edges) → 1D profile
    sobely = cv2.Sobel(roi, cv2.CV_32F, 0, 1, ksize=3)
    prof = np.mean(np.abs(sobely), axis=1)  # average over columns → shape (roiH,)
    
    # Smooth & pick peak
    prof = cv2.GaussianBlur(prof.reshape(-1, 1), (7, 1), 0).ravel()
    
    if bottommost:
        idx = int(np.argmax(prof[::-1]))  # strongest from bottom
        yr = (y1 - 1) - idx
    else:
        yr = int(np.argmax(prof)) + y0
    
    # Confidence scoring
    peak = prof[(yr - y0)]
    med = float(np.median(prof))
    mad = float(np.median(np.abs(prof - med)) + 1e-6)
    conf = max(0.0, min(1.0, (peak - med) / (6*mad)))
    
    return yr, conf, prof

class CardBoundaryDetector:
    """Detects complete boundaries of WeChat message cards"""
    
    def __init__(self):
        # Initialize width detector for left/right boundaries
        self.width_detector = SimpleWidthDetector()
        
        # Initialize avatar detector if available
        if AVATAR_DETECTION_AVAILABLE:
            self.avatar_detector = CardAvatarDetector()
        else:
            self.avatar_detector = None
        
        # Card height constraints
        self.MIN_CARD_HEIGHT = 50   # Minimum reasonable card height
        self.MAX_CARD_HEIGHT = 200  # Maximum reasonable card height
        self.TYPICAL_CARD_HEIGHT = 100  # Typical card height for estimation
        
        # Edge detection parameters
        self.EDGE_THRESHOLD_LOW = 30
        self.EDGE_THRESHOLD_HIGH = 100
        self.MIN_EDGE_STRENGTH = 50  # Minimum strength for valid edge
        
    def detect_card_boundaries(self, image_path: str) -> List[Dict]:
        """
        Detect complete boundaries of all message cards in the screenshot
        Returns list of card dictionaries with boundary information
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return []
        
        print(f"🎯 Card Boundary Detection: {os.path.basename(image_path)}")
        print(f"📐 Image size: {img.shape[1]}x{img.shape[0]}")
        
        # Step 1: Get left and right boundaries from width detector
        width_result = self.width_detector.detect_width(image_path)
        if width_result is None:
            print("❌ Failed to detect width boundaries")
            return []
        
        left_boundary, right_boundary, width = width_result
        print(f"  📏 Width boundaries: left={left_boundary}px, right={right_boundary}px, width={width}px")
        
        # Step 2: Get avatar positions if available
        avatar_positions = []
        if self.avatar_detector:
            try:
                avatar_positions = self.avatar_detector.get_avatar_positions(image_path)
                print(f"  👤 Found {len(avatar_positions)} avatars")
            except Exception as e:
                print(f"  ⚠️ Avatar detection failed: {e}")
        
        # Step 3: Detect horizontal edges (card separators)
        horizontal_edges = self._detect_horizontal_edges(img, left_boundary, right_boundary)
        print(f"  📊 Found {len(horizontal_edges)} horizontal edges")
        
        # Step 4: Assemble cards from edges and avatars
        if avatar_positions:
            print(f"  🎯 Using avatar-centric detection with {len(avatar_positions)} detected avatars")
            # Use avatar-centric boundary calculation (preferred method)
            cards = self._calculate_avatar_centric_boundaries(
                avatar_positions, left_boundary, right_boundary, img.shape[0]
            )
        else:
            print(f"  🎯 Avatar detection failed, trying to estimate avatar positions from edges")
            # Try to estimate avatar positions from edge-based cards
            estimated_avatars = self._estimate_avatar_positions_from_edges(
                horizontal_edges, left_boundary, right_boundary, img.shape[0]
            )
            
            if estimated_avatars:
                print(f"  🎯 Using avatar-centric detection with {len(estimated_avatars)} estimated avatars")
                cards = self._calculate_avatar_centric_boundaries(
                    estimated_avatars, left_boundary, right_boundary, img.shape[0]
                )
            else:
                print(f"  🎯 Falling back to edge-only detection")
                # Fall back to edge-only detection
                cards = self._assemble_cards_from_edges(
                    horizontal_edges, left_boundary, right_boundary, img.shape[0]
                )
        
        print(f"  ✅ Detected {len(cards)} message cards")
        
        return cards
    
    def _detect_horizontal_edges(self, img: np.ndarray, left: int, right: int) -> List[Tuple[int, float]]:
        """
        Detect horizontal edges (card separators) between left and right boundaries
        Returns list of (y_position, confidence) tuples
        """
        img_height, img_width = img.shape[:2]
        
        # Focus on the message area between left and right boundaries
        message_area = img[:, left:right]
        
        # Convert to grayscale
        gray = cv2.cvtColor(message_area, cv2.COLOR_BGR2GRAY)
        
        # Apply blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 1), 0)  # Horizontal blur
        
        # Detect horizontal edges using Sobel
        sobely = cv2.Sobel(blurred, cv2.CV_64F, 0, 1, ksize=3)
        abs_sobely = np.abs(sobely)
        
        # Sum across columns to get edge strength profile
        edge_profile = np.mean(abs_sobely, axis=1)
        
        # Smooth the profile
        edge_profile = cv2.GaussianBlur(edge_profile.reshape(-1, 1), (7, 1), 0).ravel()
        
        # Find peaks in edge profile
        edges = []
        mean_strength = np.mean(edge_profile)
        threshold = mean_strength * 1.5  # Edges should be stronger than average
        
        # Find all peaks above threshold
        for y in range(1, len(edge_profile) - 1):
            if (edge_profile[y] > edge_profile[y-1] and 
                edge_profile[y] > edge_profile[y+1] and
                edge_profile[y] > threshold):
                
                # Calculate confidence
                local_region = edge_profile[max(0, y-10):min(len(edge_profile), y+11)]
                local_med = np.median(local_region)
                local_mad = np.median(np.abs(local_region - local_med)) + 1e-6
                confidence = min(1.0, (edge_profile[y] - local_med) / (6 * local_mad))
                
                edges.append((y, confidence))
        
        # Sort by y position
        edges.sort(key=lambda x: x[0])
        
        # Filter out edges that are too close together
        filtered_edges = []
        min_separation = 30  # Minimum pixels between edges
        
        for edge in edges:
            if not filtered_edges or edge[0] - filtered_edges[-1][0] >= min_separation:
                filtered_edges.append(edge)
        
        return filtered_edges
    
    def _assemble_cards_with_avatars(self, edges: List[Tuple[int, float]], 
                                     avatars: List[Tuple[int, int]],
                                     left: int, right: int, img_height: int) -> List[Dict]:
        """
        Assemble message cards using both edges and avatar positions
        Each avatar should be within a card bounded by edges
        """
        cards = []
        
        # Add virtual edges at top and bottom
        all_edges = [(0, 1.0)] + edges + [(img_height, 1.0)]
        
        # For each avatar, find the bounding edges
        for i, (avatar_x, avatar_y) in enumerate(avatars):
            # Find the edge above this avatar
            upper_edge = 0
            for edge_y, _ in all_edges:
                if edge_y < avatar_y:
                    upper_edge = edge_y
                else:
                    break
            
            # Find the edge below this avatar
            lower_edge = img_height
            for edge_y, _ in reversed(all_edges):
                if edge_y > avatar_y:
                    lower_edge = edge_y
                else:
                    break
            
            # Validate card height
            card_height = lower_edge - upper_edge
            if card_height < self.MIN_CARD_HEIGHT or card_height > self.MAX_CARD_HEIGHT:
                print(f"    ⚠️ Card {i+1}: Invalid height {card_height}px, skipping")
                continue
            
            # Create card dictionary
            card = {
                'card_id': i + 1,
                'boundaries': {
                    'left': left,
                    'top': upper_edge,
                    'right': right,
                    'bottom': lower_edge
                },
                'dimensions': {
                    'width': right - left,
                    'height': card_height
                },
                'avatar_position': (avatar_x, avatar_y),
                'confidence': 0.9,  # High confidence when using avatars
                'detection_method': 'avatar_guided'
            }
            
            cards.append(card)
            print(f"    📇 Card {i+1}: y={upper_edge}-{lower_edge} (height={card_height}px), avatar at y={avatar_y}")
        
        return cards
    
    def _assemble_cards_from_edges(self, edges: List[Tuple[int, float]], 
                                   left: int, right: int, img_height: int) -> List[Dict]:
        """
        Assemble message cards using only edge detection
        Pairs consecutive edges as card boundaries
        """
        cards = []
        
        # Add virtual edge at top if needed
        if not edges or edges[0][0] > 50:
            edges = [(0, 1.0)] + edges
        
        # Add virtual edge at bottom if needed
        if not edges or edges[-1][0] < img_height - 50:
            edges = edges + [(img_height, 1.0)]
        
        # Create cards from consecutive edge pairs
        for i in range(len(edges) - 1):
            upper_edge = edges[i][0]
            lower_edge = edges[i + 1][0]
            
            # Validate card height
            card_height = lower_edge - upper_edge
            if card_height < self.MIN_CARD_HEIGHT or card_height > self.MAX_CARD_HEIGHT:
                continue
            
            # Estimate avatar position (center of card, left side)
            avatar_x = left + 50  # Approximate avatar position
            avatar_y = upper_edge + card_height // 2
            
            card = {
                'card_id': len(cards) + 1,
                'boundaries': {
                    'left': left,
                    'top': upper_edge,
                    'right': right,
                    'bottom': lower_edge
                },
                'dimensions': {
                    'width': right - left,
                    'height': card_height
                },
                'avatar_position': (avatar_x, avatar_y),  # Estimated
                'confidence': min(edges[i][1], edges[i+1][1]),
                'detection_method': 'edge_only'
            }
            
            cards.append(card)
            print(f"    📇 Card {len(cards)}: y={upper_edge}-{lower_edge} (height={card_height}px)")
        
        return cards
    
    def _estimate_avatar_positions_from_edges(self, edges: List[Tuple[int, float]], 
                                            left: int, right: int, img_height: int) -> List[Tuple[int, int]]:
        """
        Estimate avatar positions when avatar detection fails
        Uses edge-based card detection and assumes avatars are in typical positions
        """
        if not edges:
            return []
        
        print(f"  🔍 Estimating avatar positions from {len(edges)} horizontal edges")
        
        # Create cards from edges (similar to existing method)
        all_edges = [(0, 1.0)] + edges + [(img_height, 1.0)]
        estimated_avatars = []
        
        # For each pair of consecutive edges, create a card and estimate avatar position
        for i in range(len(all_edges) - 1):
            upper_edge = all_edges[i][0]
            lower_edge = all_edges[i + 1][0]
            
            card_height = lower_edge - upper_edge
            
            # Skip cards that are too small or too large
            if card_height < self.MIN_CARD_HEIGHT or card_height > self.MAX_CARD_HEIGHT:
                continue
            
            # Estimate avatar position: left side of card, vertically centered
            avatar_x = left + 50  # Typical avatar position from left edge
            avatar_y = upper_edge + card_height // 2  # Center of card vertically
            
            estimated_avatars.append((avatar_x, avatar_y))
            print(f"    🔹 Estimated avatar {len(estimated_avatars)}: ({avatar_x}, {avatar_y}) in card y={upper_edge}-{lower_edge}")
        
        return estimated_avatars
    
    def _calculate_avatar_centric_boundaries(self, avatars: List[Tuple[int, int]], 
                                           left: int, right: int, img_height: int) -> List[Dict]:
        """
        Calculate card boundaries using avatar positions as anchors
        Places card edges exactly in the middle between adjacent avatars
        """
        if not avatars:
            print("  ⚠️ No avatars provided for avatar-centric detection")
            return []
        
        print(f"  🎯 Avatar-centric boundary calculation with {len(avatars)} avatars")
        
        # Sort avatars by Y position (top to bottom)
        sorted_avatars = sorted(avatars, key=lambda a: a[1])
        avatar_y_positions = [avatar[1] for avatar in sorted_avatars]
        
        print(f"  📍 Avatar Y positions: {avatar_y_positions}")
        
        # Calculate boundaries as midpoints between adjacent avatars
        boundaries = []
        
        # First card: from top of image to midpoint between first and second avatar
        if len(avatar_y_positions) == 1:
            # Single avatar case - create reasonable boundaries around it
            avatar_y = avatar_y_positions[0]
            card_height = 80  # Reasonable default height
            boundaries.append((max(0, avatar_y - card_height // 2), min(img_height, avatar_y + card_height // 2)))
        else:
            # Multiple avatars - use midpoints
            # First card: top to midpoint between first two avatars
            first_midpoint = (avatar_y_positions[0] + avatar_y_positions[1]) // 2
            boundaries.append((0, first_midpoint))
            
            # Middle cards: midpoint to midpoint
            for i in range(1, len(avatar_y_positions) - 1):
                prev_midpoint = (avatar_y_positions[i-1] + avatar_y_positions[i]) // 2
                next_midpoint = (avatar_y_positions[i] + avatar_y_positions[i+1]) // 2
                boundaries.append((prev_midpoint, next_midpoint))
            
            # Last card: midpoint to bottom
            if len(avatar_y_positions) > 1:
                last_midpoint = (avatar_y_positions[-2] + avatar_y_positions[-1]) // 2
                boundaries.append((last_midpoint, img_height))
        
        # Create card dictionaries
        cards = []
        for i, ((avatar_x, avatar_y), (top, bottom)) in enumerate(zip(sorted_avatars, boundaries)):
            card_height = bottom - top
            
            # Validate card height
            if card_height < self.MIN_CARD_HEIGHT:
                print(f"    ⚠️ Card {i+1}: Height too small ({card_height}px), adjusting")
                # Expand the card to minimum height, centered on avatar
                center_y = avatar_y
                top = max(0, center_y - self.MIN_CARD_HEIGHT // 2)
                bottom = min(img_height, top + self.MIN_CARD_HEIGHT)
                card_height = bottom - top
                
            elif card_height > self.MAX_CARD_HEIGHT:
                print(f"    ⚠️ Card {i+1}: Height too large ({card_height}px), adjusting")
                # Shrink the card to maximum height, centered on avatar
                center_y = avatar_y
                top = max(0, center_y - self.MAX_CARD_HEIGHT // 2)
                bottom = min(img_height, top + self.MAX_CARD_HEIGHT)
                card_height = bottom - top
            
            card = {
                'card_id': i + 1,
                'boundaries': {
                    'left': left,
                    'top': int(top),
                    'right': right,
                    'bottom': int(bottom)
                },
                'dimensions': {
                    'width': right - left,
                    'height': int(card_height)
                },
                'avatar_position': (avatar_x, avatar_y),
                'confidence': 0.95,  # High confidence for avatar-guided detection
                'detection_method': 'avatar_centric'
            }
            
            cards.append(card)
            
            # Check avatar centering
            avatar_offset_from_top = avatar_y - top
            card_center = card_height // 2
            centering_error = abs(avatar_offset_from_top - card_center)
            
            print(f"    📇 Card {i+1}: y={int(top)}-{int(bottom)} (height={int(card_height)}px)")
            print(f"         Avatar at y={avatar_y}, offset from top={avatar_offset_from_top}px, centering error={centering_error}px")
        
        return cards
    
    def create_visualization(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Create visualization showing detected card boundaries
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        # Detect boundaries
        cards = self.detect_card_boundaries(image_path)
        if not cards:
            print("❌ No cards detected for visualization")
            return None
        
        # Create visualization
        result = img.copy()
        
        # Draw each card
        for card in cards:
            left = card['boundaries']['left']
            top = card['boundaries']['top']
            right = card['boundaries']['right']
            bottom = card['boundaries']['bottom']
            avatar_x, avatar_y = card['avatar_position']
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (left, top), (right, bottom), (255, 0, 0), 2)
            
            # Draw avatar position (green circle)
            cv2.circle(result, (avatar_x, avatar_y), 8, (0, 255, 0), 2)
            
            # Add card label
            label = f"Card {card['card_id']}: {card['dimensions']['width']}x{card['dimensions']['height']}px"
            cv2.putText(result, label, (left + 5, top + 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add confidence score
            conf_text = f"Conf: {card['confidence']:.2f}"
            cv2.putText(result, conf_text, (left + 5, top + 40),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            # Add detection method
            method_text = f"Method: {card['detection_method']}"
            cv2.putText(result, method_text, (left + 5, top + 55),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 0), 1)
        
        # Draw left and right boundaries (vertical green lines)
        if cards:
            left_bound = cards[0]['boundaries']['left']
            right_bound = cards[0]['boundaries']['right']
            cv2.line(result, (left_bound, 0), (left_bound, img.shape[0]), (0, 255, 0), 1)
            cv2.line(result, (right_bound, 0), (right_bound, img.shape[0]), (0, 255, 0), 1)
            
            # Add boundary labels
            cv2.putText(result, f"L:{left_bound}", (left_bound + 5, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"R:{right_bound}", (right_bound - 50, 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        
        # Add legend
        legend_y = img.shape[0] - 60
        cv2.putText(result, "Legend: Blue=Card Boundary, Green=Avatar/Width, Yellow=Method",
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Total Cards: {len(cards)}",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_CardBoundaries_{len(cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Card boundary visualization saved: {output_path}")
        
        return os.path.basename(output_path)

if __name__ == "__main__":
    # Test the detector
    detector = CardBoundaryDetector()
    
    # Test with latest screenshot
    test_image = "pic/screenshots/20250904_235942_WeChat.png"
    if os.path.exists(test_image):
        print(f"🔍 Testing with: {test_image}")
        
        # Detect boundaries
        cards = detector.detect_card_boundaries(test_image)
        
        if cards:
            print(f"\n✅ Successfully detected {len(cards)} message cards:")
            for card in cards:
                print(f"  Card {card['card_id']}:")
                print(f"    Boundaries: ({card['boundaries']['left']}, {card['boundaries']['top']}) to ({card['boundaries']['right']}, {card['boundaries']['bottom']})")
                print(f"    Dimensions: {card['dimensions']['width']} × {card['dimensions']['height']} px")
                print(f"    Avatar at: {card['avatar_position']}")
                print(f"    Confidence: {card['confidence']:.2f}")
                print(f"    Method: {card['detection_method']}")
            
            # Create visualization
            vis_path = detector.create_visualization(test_image)
            if vis_path:
                print(f"\n🎨 Visualization saved: {vis_path}")
        else:
            print("❌ No cards detected")
    else:
        print(f"❌ Test image not found: {test_image}")