#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Card Processing Module (m_Card_Processing.py)

This module consolidates three card processing functionalities:
1. SimpleWidthDetector - Detects message card width boundaries
2. CardAvatarDetector - Detects avatar positions within cards  
3. CardBoundaryDetector - Detects individual card boundaries

Each functionality is implemented as a separate class for modular usage.
"""

import cv2
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import screenshot capture functionality
try:
    from .m_ScreenShot_WeChatWindow import capture_screenshot
    SCREENSHOT_AVAILABLE = True
except ImportError:
    # Fallback import for direct execution
    try:
        from m_ScreenShot_WeChatWindow import capture_screenshot
        SCREENSHOT_AVAILABLE = True
    except ImportError:
        print("⚠️  Screenshot module not available. Live capture disabled.")
        SCREENSHOT_AVAILABLE = False


# =============================================================================
# 1. SIMPLE WIDTH DETECTOR
# =============================================================================

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


# =============================================================================
# 1.5. RIGHT BOUNDARY DETECTOR
# =============================================================================

class RightBoundaryDetector:
    """Right boundary detector for WeChat message cards using pre-processed high-contrast images"""
    
    def __init__(self):
        # Optimized Photoshop levels parameters for high contrast preprocessing
        self.INPUT_BLACK_POINT = 32     # Optimized input black point (27-36 range) for distinctive edges
        self.INPUT_WHITE_POINT = 107    # Photoshop input white point
        self.GAMMA = 0.67               # Photoshop gamma value
        
        # Detection parameters
        self.EDGE_THRESHOLD = 0.10      # 10% threshold for internal preprocessing
        self.PREPROCESSED_THRESHOLD = 0.005  # 0.5% threshold for white-to-black transitions
        self.SMOOTHING_SIZE = 5         # Smoothing kernel size (reduced for sharper transitions)
        self.MIN_BOUNDARY_PX = 800      # Minimum boundary position for message content
        
    def _apply_level_adjustment(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Photoshop-style levels adjustment using your exact method with gamma correction
        Creates high contrast white cards on black background for clear boundary detection
        """
        print(f"  🎨 Applying Photoshop-style levels adjustment with gamma correction...")
        
        # Apply your exact method with gamma correction
        in_black, in_white = self.INPUT_BLACK_POINT, self.INPUT_WHITE_POINT
        gamma = self.GAMMA
        
        # Step 1: Normalize to 0-1 range and clip
        arr = np.clip((gray - in_black) / (in_white - in_black), 0, 1)
        
        # Step 2: Apply gamma correction and scale to 0-255
        arr = (arr ** (1/gamma)) * 255
        
        # Convert to uint8
        scaled = arr.astype(np.uint8)
        
        # Save the result
        self._save_preprocessing_image(scaled, "02_photoshop_levels_gamma.png")
        
        print(f"  🎨 Photoshop levels with gamma applied:")
        print(f"    - Input black: {in_black}")
        print(f"    - Input white: {in_white}")
        print(f"    - Gamma: {gamma}")
        print(f"    - Step 1: arr = clip((pixel - {in_black}) / ({in_white} - {in_black}), 0, 1)")
        print(f"    - Step 2: arr = (arr ** (1/{gamma})) * 255")
        print(f"  📸 Image saved: 02_photoshop_levels_gamma.png")
        
        return scaled
    
    def _save_preprocessing_image(self, img: np.ndarray, filename: str):
        """Save preprocessing step images for visualization"""
        import os
        from datetime import datetime
        
        try:
            # Create screenshot folder if it doesn't exist
            screenshot_dir = "pic/screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(screenshot_dir, f"{timestamp}_{filename}")
            
            # Save grayscale image
            import cv2
            cv2.imwrite(filepath, img)
            
        except Exception as e:
            print(f"  ⚠️ Could not save preprocessing image: {e}")
    
    def detect_right_boundary(self, img: np.ndarray = None, img_width: int = None, preprocessed_image_path: str = None) -> int:
        """
        Simplified right boundary detection using horizontal pixel difference visualization
        
        Based on the insight that horizontal pixel differences directly show boundaries as blue regions,
        this method:
        1. Creates horizontal pixel differences (like your visualization)
        2. Detects blue regions (strong negative transitions < -100)
        3. Finds rightmost boundary from visual pattern
        
        Args:
            img: Original image (optional, used for fallback preprocessing)
            img_width: Width of the original image (optional)
            preprocessed_image_path: Path to preprocessed level-adjusted image
            
        Returns:
            int: boundary_position_px
        """
        print(f"  🎯 Simplified Visual Pattern Boundary Detection")
        
        # Step 1: Load and prepare image
        if preprocessed_image_path and os.path.exists(preprocessed_image_path):
            print(f"  📸 Loading preprocessed image: {os.path.basename(preprocessed_image_path)}")
            adjusted = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            if img is None:
                raise ValueError("Either preprocessed_image_path or img must be provided")
            print(f"  🎨 Using original image")
            adjusted = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            
        img_width = img_width or adjusted.shape[1]
        
        # Step 2: Create horizontal pixel differences (like your visualization)
        print(f"  📊 Creating horizontal pixel difference pattern")
        diff_x = np.diff(adjusted.astype(np.int16), axis=1)
        
        # Step 3: Detect blue regions (strong negative transitions)
        blue_threshold = -100  # Blue regions in your visualization
        blue_regions = diff_x < blue_threshold
        
        print(f"  🔵 Detecting blue regions (transitions < {blue_threshold})")
        blue_pixel_count = np.sum(blue_regions)
        print(f"  📍 Found {blue_pixel_count} blue pixels representing strong transitions")
        
        # Step 4: Find rightmost boundary from blue regions
        # Count blue pixels per column (vertical projection)
        blue_column_intensity = np.sum(blue_regions, axis=0)
        
        # Search in reasonable boundary range
        search_start = int(img_width * 0.4)  # Skip left sidebar area
        search_end = int(img_width * 0.95)   # Skip right edge artifacts
        
        print(f"  🔍 Searching for rightmost blue region in {search_start}-{search_end}px")
        
        # Find columns with significant blue intensity (boundary regions)
        min_blue_intensity = 2  # Minimum blue pixels per column to be considered a boundary
        boundary_candidates = []
        
        for x in range(search_start, min(search_end, len(blue_column_intensity))):
            if blue_column_intensity[x] >= min_blue_intensity:
                boundary_candidates.append((x, blue_column_intensity[x]))
        
        print(f"  🎯 Found {len(boundary_candidates)} boundary candidates with blue regions")
        
        # Step 5: Select rightmost boundary
        if boundary_candidates:
            # Sort by position (rightmost first)
            boundary_candidates.sort(key=lambda b: b[0], reverse=True)
            
            # Select the rightmost boundary above minimum threshold
            for x, intensity in boundary_candidates:
                if x >= self.MIN_BOUNDARY_PX:
                    print(f"  ✅ Rightmost blue boundary detected: {x}px (blue intensity: {intensity})")
                    return x
            
            # Fallback to best available
            x, intensity = boundary_candidates[0]
            print(f"  ⚠️ Using best blue boundary (below min): {x}px (intensity: {intensity})")
            return x
        
        # Final fallback
        fallback = int(img_width * 0.8)
        print(f"  ❌ No blue regions found, using geometric fallback: {fallback}px")
        return fallback


class SimpleWidthDetector:
    """Enhanced width detector with dual-boundary coordination and visual marker integration"""
    
    def __init__(self):
        # Simple parameters for width detection only
        self.CONVERSATION_WIDTH_RATIO = 0.65  # Focus on left 65% of screen where cards are
        self.EDGE_THRESHOLD_LOW = 30
        self.EDGE_THRESHOLD_HIGH = 100
        
        # Initialize right boundary detector with enhanced capabilities
        self.right_detector = RightBoundaryDetector()
        
        # Dual-boundary coordination storage for blue line visualization integration
        self._boundary_markers = {
            'left': {'position': None, 'confidence': None, 'method': None},
            'right': {'position': None, 'confidence': None, 'method': None}
        }
        
    def detect_width(self, image_path: str, preprocessed_image_path: str = None) -> Optional[Tuple[int, int, int]]:
        """
        Enhanced width detection with dual-boundary coordination and blue line visualization integration
        
        Implements synchronized left and right boundary detection with confidence scoring
        and visual marker data compatible with horizontal pixel difference analysis
        
        Args:
            image_path: Path to the original WeChat screenshot
            preprocessed_image_path: Path to pre-processed level-adjusted image (recommended)
            
        Returns: (left_boundary, right_boundary, width) or None if failed
        """
        print(f"🎯 Enhanced Dual-Boundary Width Detection: {os.path.basename(image_path)}")
        
        # Phase 1: Image Loading and Preparation
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        print(f"📐 Image dimensions: {img.shape[1]}×{img.shape[0]}")
        img_height, img_width = img.shape[:2]
        
        # Phase 2: Left Boundary Detection with Confidence Scoring
        print(f"🔍 Phase 2: Left Boundary Detection")
        conversation_width = int(img_width * self.CONVERSATION_WIDTH_RATIO)
        conversation_area = img[:, :conversation_width]
        print(f"  💬 Conversation search area: {conversation_area.shape[1]}×{conversation_area.shape[0]}")
        
        left_boundary, left_confidence = self._detect_left_boundary_with_confidence(conversation_area)
        
        # Store left boundary marker data
        self._boundary_markers['left'] = {
            'position': left_boundary,
            'confidence': left_confidence,
            'method': 'edge_based_sidebar'
        }
        
        # Phase 3: Right Boundary Detection (Enhanced Method)
        print(f"🔍 Phase 3: Right Boundary Detection")
        right_boundary = self.right_detector.detect_right_boundary(
            img=img,
            img_width=img_width,
            preprocessed_image_path=preprocessed_image_path
        )
        
        # Extract right boundary confidence and method from detector
        right_confidence = 0.5  # Default fallback
        right_method = 'unknown'
        if hasattr(self.right_detector, '_boundary_visualization_data'):
            viz_data = self.right_detector._boundary_visualization_data
            right_confidence = viz_data.get('confidence', 0.5)
            right_method = viz_data.get('method', 'unknown')
        
        # Store right boundary marker data
        self._boundary_markers['right'] = {
            'position': right_boundary,
            'confidence': right_confidence,
            'method': right_method
        }
        
        # Phase 4: Dual-Boundary Validation and Coordination
        if left_boundary is None or right_boundary is None:
            print("❌ Dual-boundary detection failed")
            return None
        
        width = right_boundary - left_boundary
        
        # Phase 5: Boundary Relationship Analysis
        self._analyze_boundary_relationships(left_boundary, right_boundary, width, img_width)
        
        # Phase 6: Enhanced Visual Output with Blue Line Integration
        self._create_enhanced_visual_result(img, image_path)
        
        return left_boundary, right_boundary, width
    
    def _detect_left_boundary_with_confidence(self, conversation_area: np.ndarray) -> tuple:
        """Enhanced left boundary detection with confidence scoring"""
        # Use existing edge-based detection but add confidence calculation
        left_boundary = self._find_left_boundary_edge_based(conversation_area)
        
        if left_boundary is not None:
            # Calculate confidence based on edge strength and position consistency
            confidence = 0.8  # High confidence for sidebar edge detection
            print(f"  ✅ Left boundary: {left_boundary}px, confidence: {confidence:.3f}")
        else:
            confidence = 0.0
            print(f"  ❌ Left boundary detection failed")
        
        return left_boundary, confidence
    
    def _analyze_boundary_relationships(self, left: int, right: int, width: int, img_width: int):
        """Analyze relationships between detected boundaries for validation"""
        print(f"🔍 Phase 4: Boundary Relationship Analysis")
        
        # Calculate metrics
        left_ratio = left / img_width if img_width > 0 else 0
        right_ratio = right / img_width if img_width > 0 else 0
        width_ratio = width / img_width if img_width > 0 else 0
        
        # Relationship validation
        relationship_quality = "excellent"
        if width < 200:
            relationship_quality = "suspicious_narrow"
        elif width > img_width * 0.9:
            relationship_quality = "suspicious_wide"
        elif left < -50:
            relationship_quality = "left_overflow"
        elif right > img_width - 20:
            relationship_quality = "right_overflow"
        
        print(f"  📊 Boundary Metrics:")
        print(f"    Left position: {left}px ({left_ratio:.2%} of image)")
        print(f"    Right position: {right}px ({right_ratio:.2%} of image)")
        print(f"    Detected width: {width}px ({width_ratio:.2%} of image)")
        print(f"    Relationship quality: {relationship_quality}")
        
        # Store for visualization
        self._boundary_markers['relationship'] = {
            'width': width,
            'quality': relationship_quality,
            'ratios': {'left': left_ratio, 'right': right_ratio, 'width': width_ratio}
        }
    
    def _create_enhanced_visual_result(self, img: np.ndarray, image_path: str):
        """Create enhanced visual result with blue line markers like your visualization"""
        left_data = self._boundary_markers['left']
        right_data = self._boundary_markers['right']
        
        if left_data['position'] is None or right_data['position'] is None:
            return
        
        # Create enhanced visualization
        result_img = img.copy()
        img_height, img_width = img.shape[:2]
        
        # Draw blue vertical lines (like your horizontal pixel difference visualization)
        left_pos = left_data['position']
        right_pos = right_data['position']
        
        # Left boundary - blue line
        cv2.line(result_img, (left_pos, 0), (left_pos, img_height), (255, 100, 0), 3)  # Blue
        
        # Right boundary - blue line  
        cv2.line(result_img, (right_pos, 0), (right_pos, img_height), (255, 100, 0), 3)  # Blue
        
        # Enhanced info panel
        panel_height = 120
        cv2.rectangle(result_img, (left_pos, 20), (right_pos, 20 + panel_height), (255, 255, 255), -1)
        cv2.rectangle(result_img, (left_pos, 20), (right_pos, 20 + panel_height), (0, 0, 0), 2)
        
        # Text annotations
        font = cv2.FONT_HERSHEY_SIMPLEX
        width = right_pos - left_pos
        
        # Main width annotation
        cv2.putText(result_img, f"Width: {width}px", 
                   (left_pos + 10, 45), font, 0.8, (0, 0, 0), 2)
        
        # Boundary positions
        cv2.putText(result_img, f"L:{left_pos}px (conf:{left_data['confidence']:.2f})", 
                   (left_pos + 10, 70), font, 0.6, (0, 100, 200), 2)
        cv2.putText(result_img, f"R:{right_pos}px (conf:{right_data['confidence']:.2f})", 
                   (left_pos + 10, 90), font, 0.6, (0, 100, 200), 2)
        
        # Method annotations
        cv2.putText(result_img, f"Methods: {left_data['method']} + {right_data['method']}", 
                   (left_pos + 10, 115), font, 0.5, (100, 100, 100), 1)
        
        # Blue line markers at top and bottom (like your visualization)
        marker_size = 8
        # Top markers
        cv2.circle(result_img, (left_pos, 10), marker_size, (255, 100, 0), -1)
        cv2.circle(result_img, (right_pos, 10), marker_size, (255, 100, 0), -1)
        # Bottom markers
        cv2.circle(result_img, (left_pos, img_height - 10), marker_size, (255, 100, 0), -1)
        cv2.circle(result_img, (right_pos, img_height - 10), marker_size, (255, 100, 0), -1)
        
        # Save enhanced result
        screenshot_dir = "pic/screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_EnhancedDualBoundary_{width}px.png"
        output_path = os.path.join(screenshot_dir, output_filename)
        
        cv2.imwrite(output_path, result_img)
        print(f"  🎨 Enhanced dual-boundary visualization: {output_path}")
    
    def get_boundary_markers(self) -> dict:
        """Get boundary marker data for integration with blue line visualizations"""
        return self._boundary_markers
    
    def _save_visual_result(self, img: np.ndarray, left_boundary: int, right_boundary: int, width: int, image_path: str):
        """Save visual result showing detected boundaries to screenshots folder"""
        try:
            # Create output image with boundary overlays
            result_img = img.copy()
            img_height, img_width = img.shape[:2]
            
            # Draw left boundary line (green)
            cv2.line(result_img, (left_boundary, 0), (left_boundary, img_height), (0, 255, 0), 3)
            
            # Draw right boundary line (red)  
            cv2.line(result_img, (right_boundary, 0), (right_boundary, img_height), (0, 0, 255), 3)
            
            # Draw width measurement box at top
            cv2.rectangle(result_img, (left_boundary, 20), (right_boundary, 80), (255, 255, 255), -1)
            cv2.rectangle(result_img, (left_boundary, 20), (right_boundary, 80), (0, 0, 0), 2)
            
            # Add text showing measurements
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Width: {width}px"
            cv2.putText(result_img, text, (left_boundary + 10, 45), font, 0.8, (0, 0, 0), 2)
            
            left_text = f"L:{left_boundary}"
            cv2.putText(result_img, left_text, (left_boundary + 10, 65), font, 0.6, (0, 150, 0), 2)
            
            right_text = f"R:{right_boundary}"
            cv2.putText(result_img, right_text, (right_boundary - 80, 65), font, 0.6, (0, 0, 150), 2)
            
            # Create screenshot folder if it doesn't exist
            screenshot_dir = "pic/screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Generate output filename with timestamp and measurements
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_SimpleWidth_{width}px.png"
            output_path = os.path.join(screenshot_dir, output_filename)
            
            # Save the result
            cv2.imwrite(output_path, result_img)
            print(f"  📸 Visual result saved: {output_path}")
            
        except Exception as e:
            print(f"  ⚠️ Could not save visual result: {e}")
    
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


# =============================================================================
# 2. CARD AVATAR DETECTOR
# =============================================================================

class CardAvatarDetector:
    """
    Advanced avatar detector using gradient projection and geometric filtering
    Specifically optimized for WeChat message card layouts
    """
    
    def __init__(self):
        # Initialize width detector for dynamic boundary detection
        self.width_detector = SimpleWidthDetector()
        
        # Avatar size constraints (adjustable based on DPI)
        self.MIN_AVATAR_SIZE = 25       # Minimum avatar dimension
        self.MAX_AVATAR_SIZE = 140      # Maximum avatar dimension
        self.MIN_ASPECT_RATIO = 0.85    # Nearly square avatars
        self.MAX_ASPECT_RATIO = 1.15    # Nearly square avatars
        
        # WeChat-specific panel parameters (fallback values if width detection fails)
        self.EXPECTED_CONVERSATION_AREA_RATIO = 0.65  # Conversation area is ~65% of screen width
        self.AVATAR_COLUMN_WIDTH = 120  # Avatar column is ~120px wide (fallback)
        self.AVATAR_COLUMN_START = 60   # Avatar column starts around x=60px (fallback)
        
        # Edge detection parameters
        self.BILATERAL_D = 5            # Bilateral filter diameter
        self.BILATERAL_SIGMA_COLOR = 35 # Color sigma for bilateral filter
        self.BILATERAL_SIGMA_SPACE = 35 # Space sigma for bilateral filter
        self.CANNY_LOW = 50            # Lower threshold for Canny
        self.CANNY_HIGH = 120          # Upper threshold for Canny
        
        # Morphology parameters
        self.DILATE_KERNEL_SIZE = (3, 3)  # Dilation kernel
        self.DILATE_ITERATIONS = 1        # Dilation iterations
        
        # NMS parameters
        self.NMS_IOU_THRESHOLD = 0.2    # IoU threshold for non-maximum suppression
        
        # Solidity threshold (filled vs hollow shapes)
        self.MIN_SOLIDITY = 0.6         # Minimum solidity for avatar shapes
        
    def find_panel_right_edge(self, gray: np.ndarray, xL: int = 0, xR: Optional[int] = None, 
                              y0: int = 0, y1: Optional[int] = None) -> int:
        """
        1-D x-gradient projection to locate the right border of the list panel.
        Uses gradient analysis to find the strongest vertical edge (panel boundary)
        """
        H, W = gray.shape
        xR = W if xR is None else xR
        y1 = H if y1 is None else y1
        
        # Extract the search band
        band = gray[y0:y1, xL:xR]
        
        # Compute horizontal gradient (vertical edges)
        diff = np.abs(band[:, 1:].astype(np.int16) - band[:, :-1].astype(np.int16))
        
        # Create 1D profile by averaging across height
        prof = diff.mean(axis=0).astype(np.float32)
        
        # Smooth the profile to reduce noise
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 7), 0).ravel()
        
        # Find the strongest vertical edge
        xr_local = int(np.argmax(prof)) + 1
        
        return xL + xr_local

    def nms_boxes(self, boxes: List[List[int]], iou_thresh: float = 0.3) -> List[List[int]]:
        """
        Non-maximum suppression on axis-aligned boxes [x,y,w,h].
        Eliminates overlapping detections to keep only the best ones.
        """
        if not boxes:
            return []
        
        b = np.array(boxes, dtype=np.float32)
        x1, y1 = b[:, 0], b[:, 1]
        x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        scores = b[:, 2] * b[:, 3]  # Use area as score
        
        # Sort by score (area) in descending order
        idxs = scores.argsort()[::-1]
        keep = []
        
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / ((x2[i] - x1[i]) * (y2[i] - y1[i]) + 
                          (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]]) - 
                          inter + 1e-6)
            
            # Keep boxes with IoU below threshold
            idxs = idxs[1:][iou < iou_thresh]
        
        return [boxes[i] for i in keep]

    def detect_avatars(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Detect avatars using gradient projection and geometric filtering.
        
        Returns:
            Tuple of (avatar_results, detection_info)
            - avatar_results: List of detected avatars with bbox and center
            - detection_info: Diagnostic information about the detection process
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
        
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"🎯 Advanced Avatar Detection: {os.path.basename(image_path)}")
        print(f"📐 Image size: {W}x{H}")
        
        # Step 1: Use dynamic width detection to determine card boundaries
        print(f"  🔍 Detecting dynamic card boundaries...")
        width_result = self.width_detector.detect_width(image_path)
        
        if width_result is not None:
            left_boundary, right_boundary, detected_width = width_result
            # Use dynamic left boundary + fixed avatar column width (avatars are only in leftmost ~120px)
            x0 = left_boundary
            wR = self.AVATAR_COLUMN_WIDTH  # Keep avatar column width, only left boundary is dynamic
            avatar_search_area = (x0, 0, wR, H)
            boundary_source = "dynamic"
            print(f"  ✅ Using dynamic left boundary: {left_boundary}px, avatar column: {wR}px wide (search area: {x0}-{x0+wR}px)")
        else:
            # Fallback to hardcoded values
            conversation_width = int(W * self.EXPECTED_CONVERSATION_AREA_RATIO)
            x0 = self.AVATAR_COLUMN_START
            wR = self.AVATAR_COLUMN_WIDTH
            avatar_search_area = (x0, 0, wR, H)
            boundary_source = "fallback"
            print(f"  ⚠️ Width detection failed, using fallback boundaries: x={x0}, width={wR}px")
        
        # Extract avatar search region from the detected/fallback boundaries
        y0, hR = 0, H
        panel = img[y0:y0+hR, x0:x0+wR]
        gray_p = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        
        print(f"  📊 Avatar search area: x={x0}-{x0+wR}px ({boundary_source} left boundary, fixed avatar column width)")
        
        # Step 2: Edge detection with morphology to reveal rounded-square thumbnails
        blur = cv2.bilateralFilter(gray_p, self.BILATERAL_D, 
                                  self.BILATERAL_SIGMA_COLOR, 
                                  self.BILATERAL_SIGMA_SPACE)
        
        edges = cv2.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE), 
                         iterations=self.DILATE_ITERATIONS)
        
        # Step 3: Find contours and apply geometric filters
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  🔍 Found {len(cnts)} contours")
        
        candidates = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
            area = w * h
            
            # Size filter
            if area < self.MIN_AVATAR_SIZE**2 or area > self.MAX_AVATAR_SIZE**2:
                continue
            
            # Aspect ratio filter (nearly square)
            ar = w / float(h)
            if ar < self.MIN_ASPECT_RATIO or ar > self.MAX_ASPECT_RATIO:
                continue
            
            # Position filter (avatars should be within the avatar search area)
            # Since we're already searching in the avatar column, all detections are valid positions
            # Additional filter: ensure avatars are not at the very edge
            if x < 5 or x > wR - 10:
                continue
            
            # Solidity filter (prefer filled thumbnails vs thin edges)
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cv2.contourArea(c) / hull_area
                if solidity < self.MIN_SOLIDITY:
                    continue
            
            candidates.append([x, y, w, h])
        
        print(f"  ✅ Filtered to {len(candidates)} avatar candidates")
        
        # Step 4: Apply non-maximum suppression
        boxes = self.nms_boxes(candidates, iou_thresh=self.NMS_IOU_THRESHOLD)
        print(f"  🎯 After NMS: {len(boxes)} final avatars")
        
        # Step 5: Sort by y-coordinate (top to bottom) and convert to results format
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        results = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Convert back to full-image coordinates
            X, Y = x + x0, y + y0
            cx, cy = X + w // 2, Y + h // 2
            
            result = {
                "bbox": [int(X), int(Y), int(w), int(h)],
                "center": [int(cx), int(cy)],
                "avatar_id": i + 1,
                "area": int(w * h),
                "aspect_ratio": round(w / float(h), 3),
                "position_in_panel": round(x / float(wR), 3)
            }
            results.append(result)
            
            print(f"    👤 Avatar {i+1}: {w}×{h}px at ({X}, {Y}), center=({cx}, {cy})")
        
        # Detection info for diagnostics
        detection_info = {
            "avatar_search_area": avatar_search_area,
            "boundary_source": boundary_source,
            "total_contours": len(cnts),
            "candidates_after_filtering": len(candidates),
            "final_avatars": len(boxes),
            "processing_steps": [
                f"Dynamic width detection ({boundary_source} boundaries)",
                "Avatar region definition",
                "Bilateral filtering",
                "Canny edge detection", 
                "Morphological dilation",
                "Contour analysis",
                "Geometric filtering",
                "Non-maximum suppression"
            ]
        }
        
        # Add width detection specific info if dynamic boundaries were used
        if width_result is not None:
            detection_info["width_detection"] = {
                "left_boundary": width_result[0],
                "right_boundary": width_result[1],
                "detected_width": width_result[2]
            }
        else:
            detection_info["width_detection"] = {
                "failed": True,
                "fallback_start": self.AVATAR_COLUMN_START,
                "fallback_width": self.AVATAR_COLUMN_WIDTH
            }
        
        return results, detection_info

    def create_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive visualization showing the detection process.
        Includes panel boundary, contours, filtered candidates, and final results.
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating advanced avatar detection visualization...")
        
        # Get detection results
        avatars, info = self.detect_avatars(image_path)
        
        # Create visualization overlay
        result = img.copy()
        H, W = result.shape[:2]
        
        # Draw avatar search area (rectangle)
        search_area = info.get("avatar_search_area", (60, 0, 120, H))
        x0, y0, wR, hR = search_area
        cv2.rectangle(result, (x0, y0), (x0 + wR, y0 + hR), (255, 0, 0), 2)
        cv2.putText(result, f"Avatar Area: {x0}-{x0+wR}px", (x0 + 5, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw avatar detections
        for i, avatar in enumerate(avatars):
            x, y, w, h = avatar["bbox"]
            cx, cy = avatar["center"]
            
            # Draw bounding box (green)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point (red circle)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw avatar ID
            cv2.putText(result, f"A{avatar['avatar_id']}", (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center coordinates
            cv2.putText(result, f"({cx},{cy})", (cx + 5, cy - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Add legend and stats
        legend_y = 60
        cv2.putText(result, "Advanced Avatar Detection Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Avatar Search Area  Green=Avatar Box  Red=Center", 
                  (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection statistics
        stats_y = legend_y + 50
        cv2.putText(result, f"Detected: {len(avatars)} avatars", 
                  (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Search area: {x0}-{x0+wR}px (width={wR}px)", 
                  (10, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Contours: {info.get('total_contours', 0)} -> "
                          f"Candidates: {info.get('candidates_after_filtering', 0)} -> "
                          f"Final: {info.get('final_avatars', 0)}", 
                  (10, stats_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_advanced_avatar_detection.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Advanced avatar detection visualization saved: {output_path}")
        
        return os.path.basename(output_path)

    def get_avatar_positions(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Get avatar center positions for integration with boundary detection.
        Returns list of (x, y) center coordinates.
        """
        avatars, _ = self.detect_avatars(image_path)
        return [tuple(avatar["center"]) for avatar in avatars]


# =============================================================================
# 3. CARD BOUNDARY DETECTOR  
# =============================================================================

class CardBoundaryDetector:
    """
    Card Boundary Detector using avatar-centric approach
    Uses avatar positions to determine message card boundaries
    """
    
    def __init__(self):
        self.avatar_detector = CardAvatarDetector()
        self.width_detector = SimpleWidthDetector()
        
        # Card boundary parameters
        self.MIN_CARD_HEIGHT = 60       # Minimum height for a card
        self.MAX_CARD_HEIGHT = 300      # Maximum height for a card
        self.CARD_PADDING = 10          # Padding around card boundaries
        
        # Coordinate validation parameters
        self.VALIDATION_TOLERANCE = 5   # Pixels tolerance for boundary validation
        self.ENABLE_VALIDATION = True   # Enable/disable coordinate validation
        
    def detect_cards(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Detect individual message card boundaries using avatar positions
        
        Returns:
            Tuple of (card_results, detection_info)
        """
        print(f"🎯 Card Boundary Detection: {os.path.basename(image_path)}")
        
        # Step 1: Get width boundaries
        width_result = self.width_detector.detect_width(image_path)
        if width_result is None:
            print("❌ Width detection failed")
            return [], {}
        
        left_boundary, right_boundary, width = width_result
        print(f"  📏 Card width: {width}px (left: {left_boundary}, right: {right_boundary})")
        
        # Step 2: Get complete avatar data (not just positions)
        avatars, avatar_info = self.avatar_detector.detect_avatars(image_path)
        if not avatars:
            print("❌ No avatars detected")
            return [], {}
        
        print(f"  👥 Found {len(avatars)} avatars with complete boundary data")
        
        # Step 3: Calculate card boundaries using midpoint approach with complete avatar data
        cards = []
        for i, avatar in enumerate(avatars):
            avatar_x, avatar_y = avatar["center"]  # Get center from complete avatar data
            
            # Calculate vertical boundaries (midpoint between adjacent avatars)
            if i == 0:
                # First card: start from top
                top = max(0, avatar_y - 40)
            else:
                # Midpoint between current and previous avatar
                prev_avatar_y = avatars[i-1]["center"][1]
                top = (prev_avatar_y + avatar_y) // 2
            
            if i == len(avatars) - 1:
                # Last card: extend to reasonable bottom
                bottom = avatar_y + 40
            else:
                # Midpoint between current and next avatar
                next_avatar_y = avatars[i+1]["center"][1]
                bottom = (avatar_y + next_avatar_y) // 2
            
            # Create card boundary with complete avatar data
            avatar_bbox = avatar["bbox"]  # [x, y, w, h]
            card = {
                "card_id": i + 1,
                "bbox": [left_boundary, top, width, bottom - top],
                "avatar": {
                    "bbox": avatar_bbox,
                    "center": avatar["center"],
                    "avatar_id": avatar["avatar_id"],
                    "area": avatar["area"],
                    "aspect_ratio": avatar["aspect_ratio"],
                    "position_in_panel": avatar.get("position_in_panel", 0.0)
                },
                "boundaries": {
                    "card": {
                        "left": left_boundary,
                        "right": right_boundary, 
                        "top": top,
                        "bottom": bottom
                    },
                    "avatar": {
                        "left": avatar_bbox[0],
                        "right": avatar_bbox[0] + avatar_bbox[2],
                        "top": avatar_bbox[1], 
                        "bottom": avatar_bbox[1] + avatar_bbox[3]
                    }
                }
            }
            cards.append(card)
            
            # Enhanced card information display
            avatar_dims = f"{avatar_bbox[2]}×{avatar_bbox[3]}px"
            print(f"    📄 Card {i+1}: {width}×{bottom-top}px at ({left_boundary}, {top}) | Avatar: {avatar_dims} at ({avatar_bbox[0]}, {avatar_bbox[1]})")
        
        detection_info = {
            "total_cards": len(cards),
            "card_width": width,
            "width_boundaries": (left_boundary, right_boundary),
            "avatar_count": len(avatars),
            "avatar_detection_info": avatar_info,
            "enhanced_data": True  # Flag indicating this includes complete avatar boundaries
        }
        
        # Step 4: Validate coordinates if enabled
        if self.ENABLE_VALIDATION:
            validated_cards, validation_report = self._validate_card_avatar_coordinates(cards)
            detection_info["coordinate_validation"] = validation_report
            if validation_report["validation_passed"]:
                print(f"  ✅ Coordinate validation passed: All {len(cards)} cards have avatars within boundaries")
            else:
                print(f"  ⚠️  Coordinate validation warnings: {validation_report['warnings_count']} issues found")
                for warning in validation_report["warnings"][:3]:  # Show first 3 warnings
                    print(f"    ⚠️  {warning}")
                if len(validation_report["warnings"]) > 3:
                    print(f"    ... and {len(validation_report['warnings']) - 3} more warnings")
            cards = validated_cards

        return cards, detection_info

    def _validate_card_avatar_coordinates(self, cards: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Validate that avatars are positioned within their corresponding card boundaries
        
        Args:
            cards: List of card dictionaries with avatar data
            
        Returns:
            Tuple of (validated_cards, validation_report)
        """
        validated_cards = []
        warnings = []
        corrections_made = 0
        
        for i, card in enumerate(cards):
            card_bbox = card["bbox"]  # [x, y, w, h]
            avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
            
            # Card boundaries
            card_left = card_bbox[0]
            card_right = card_bbox[0] + card_bbox[2]
            card_top = card_bbox[1] 
            card_bottom = card_bbox[1] + card_bbox[3]
            
            # Avatar boundaries
            avatar_left = avatar_bbox[0]
            avatar_right = avatar_bbox[0] + avatar_bbox[2]
            avatar_top = avatar_bbox[1]
            avatar_bottom = avatar_bbox[1] + avatar_bbox[3]
            
            # Validation checks with tolerance
            violations = []
            corrected_avatar = list(avatar_bbox)  # Copy for potential corrections
            
            # Check left boundary
            if avatar_left < card_left - self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar left ({avatar_left}) extends beyond card left ({card_left})")
                corrected_avatar[0] = card_left  # Correct x position
                
            # Check right boundary
            if avatar_right > card_right + self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar right ({avatar_right}) extends beyond card right ({card_right})")
                # Keep avatar width, adjust position if needed
                if corrected_avatar[0] + corrected_avatar[2] > card_right:
                    corrected_avatar[0] = card_right - corrected_avatar[2]
                    
            # Check top boundary
            if avatar_top < card_top - self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar top ({avatar_top}) extends beyond card top ({card_top})")
                corrected_avatar[1] = card_top  # Correct y position
                
            # Check bottom boundary  
            if avatar_bottom > card_bottom + self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar bottom ({avatar_bottom}) extends beyond card bottom ({card_bottom})")
                # Keep avatar height, adjust position if needed
                if corrected_avatar[1] + corrected_avatar[3] > card_bottom:
                    corrected_avatar[1] = card_bottom - corrected_avatar[3]
            
            # Create validated card
            validated_card = card.copy()
            
            if violations:
                # Apply corrections and warn
                warning_msg = f"Card {card['card_id']}: {'; '.join(violations)}"
                warnings.append(warning_msg)
                
                # Update avatar bbox and recalculate center
                validated_card["avatar"]["bbox"] = corrected_avatar
                new_center = [
                    corrected_avatar[0] + corrected_avatar[2] // 2,
                    corrected_avatar[1] + corrected_avatar[3] // 2
                ]
                validated_card["avatar"]["center"] = new_center
                
                # Update boundaries dict
                validated_card["boundaries"]["avatar"] = {
                    "left": corrected_avatar[0],
                    "right": corrected_avatar[0] + corrected_avatar[2],
                    "top": corrected_avatar[1], 
                    "bottom": corrected_avatar[1] + corrected_avatar[3]
                }
                
                corrections_made += 1
            
            validated_cards.append(validated_card)
        
        # Generate validation report
        validation_report = {
            "validation_passed": len(warnings) == 0,
            "total_cards_checked": len(cards),
            "violations_found": len(warnings),
            "corrections_made": corrections_made,
            "warnings_count": len(warnings),
            "warnings": warnings,
            "tolerance_used": self.VALIDATION_TOLERANCE
        }
        
        return validated_cards, validation_report

    def create_card_visualization(self, image_path: str, output_path: str = None) -> str:
        """Create visualization showing detected card boundaries"""
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating card boundary visualization...")
        
        # Get detection results
        cards, detection_info = self.detect_cards(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Draw enhanced card boundaries with avatar boundaries
        for card in cards:
            # Card boundary data
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center across card width
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)  # Yellow line
            
            # Draw card ID and dimensions
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(result, f"Card: {card_w}×{card_h}px", (card_x + 10, card_y + card_h - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw avatar ID and dimensions
            cv2.putText(result, f"A{avatar_data['avatar_id']}", (avatar_x, avatar_y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Avatar: {avatar_w}×{avatar_h}px", (card_x + 10, card_y + card_h - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add enhanced legend and stats with validation info
        legend_y = 30
        cv2.putText(result, "Enhanced Card & Avatar Boundary Detection:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add validation status
        validation_info = detection_info.get("coordinate_validation", {})
        if validation_info:
            validation_status = "✅ VALIDATED" if validation_info.get("validation_passed", False) else "⚠️ CORRECTED"
            cv2.putText(result, f"Total Cards: {len(cards)} | {validation_status} | Complete Dataset",
                       (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if validation_info.get("corrections_made", 0) > 0:
                cv2.putText(result, f"Corrections Applied: {validation_info['corrections_made']} cards adjusted",
                           (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(result, f"Total Cards: {len(cards)} | Complete Coordinate Dataset",
                       (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_Enhanced_Card_Avatar_Boundaries_{len(cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Enhanced card & avatar boundary visualization saved: {output_path}")
        
        return os.path.basename(output_path)


# =============================================================================
# 4. CONTACT NAME BOUNDARY DETECTOR  
# =============================================================================

class ContactNameBoundaryDetector:
    """
    Contact Name Boundary Detector for visual detection of name regions
    Detects white text boundaries above avatar center lines without OCR
    """
    
    def __init__(self):
        self.card_boundary_detector = CardBoundaryDetector()
        
        # White text detection parameters
        self.WHITE_THRESHOLD_MIN = 180      # Minimum brightness for white text
        self.WHITE_THRESHOLD_MAX = 255      # Maximum brightness
        
        # Name size constraints (typical contact name dimensions)
        self.MIN_NAME_WIDTH = 16            # Minimum name width in pixels
        self.MAX_NAME_WIDTH = 200           # Maximum name width in pixels  
        self.MIN_NAME_HEIGHT = 8            # Minimum name height in pixels
        self.MAX_NAME_HEIGHT = 24           # Maximum name height in pixels
        
        # Morphological operation parameters
        self.MORPH_KERNEL_SIZE = (2, 2)     # Kernel for connecting text pixels
        self.MORPH_ITERATIONS = 1           # Number of morphological iterations
        
        # Search region parameters
        self.SEARCH_MARGIN_LEFT = 5         # Margin from avatar right edge
        self.SEARCH_MARGIN_RIGHT = 10       # Margin from card right edge
        self.SEARCH_MARGIN_TOP = 5          # Margin from card top
        
        # Detection confidence parameters
        self.MIN_WHITE_PIXEL_RATIO = 0.1    # Minimum ratio of white pixels in region
        
    def detect_name_boundaries(self, image_path: str, cards_with_times: List[Dict] = None) -> Tuple[List[Dict], Dict]:
        """
        Detect contact name boundaries using visual detection only, avoiding time conflict areas
        
        Args:
            image_path: Path to screenshot image
            cards_with_times: Optional cards with time box data (for conflict avoidance)
            
        Returns:
            Tuple of (enhanced_cards_with_names, detection_info)
        """
        print(f"🎯 Contact Name Boundary Detection: {os.path.basename(image_path)}")
        
        # Step 1: Get card data (use cards_with_times if provided, otherwise detect fresh)
        if cards_with_times:
            cards = cards_with_times
            card_detection_info = {"reused_cards_with_times": True}
            print(f"  📄 Using {len(cards)} cards with time box data")
        else:
            cards, card_detection_info = self.card_boundary_detector.detect_cards(image_path)
            if not cards:
                print("❌ No cards available for name detection")
                return [], {}
            
        # Step 2: Load image for processing
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
            
        print(f"  📄 Processing {len(cards)} cards for name boundary detection")
        
        # Step 3: Process each card for name boundaries
        enhanced_cards = []
        total_names_detected = 0
        
        for card in cards:
            enhanced_card = self._detect_name_boundary_for_card(img, card)
            enhanced_cards.append(enhanced_card)
            
            if enhanced_card.get("name_boundary"):
                total_names_detected += 1
                card_id = enhanced_card["card_id"]
                name_bbox = enhanced_card["name_boundary"]["bbox"]
                print(f"    📝 Card {card_id}: Name boundary {name_bbox[2]}×{name_bbox[3]}px at ({name_bbox[0]}, {name_bbox[1]})")
        
        # Step 4: Generate detection summary
        detection_info = {
            "total_cards_processed": len(cards),
            "names_detected": total_names_detected,
            "detection_success_rate": total_names_detected / len(cards) if cards else 0,
            "card_detection_info": card_detection_info,
            "detection_method": "visual_boundary_only"
        }
        
        print(f"  ✅ Name boundary detection complete: {total_names_detected}/{len(cards)} cards have detected names")
        
        return enhanced_cards, detection_info

    def _detect_name_boundary_for_card(self, img: np.ndarray, card: Dict) -> Dict:
        """
        Detect name boundary for a single card
        
        Args:
            img: Original image as numpy array
            card: Card dictionary with boundary and avatar data
            
        Returns:
            Enhanced card dictionary with name boundary information
        """
        enhanced_card = card.copy()
        
        try:
            # Create adaptive search region based on card and avatar boundaries
            search_region = self._create_adaptive_search_region(card)
            
            if search_region is None:
                return enhanced_card
            
            # Detect white text regions in the search area
            white_text_regions = self._detect_white_text_regions(img, search_region)
            
            if not white_text_regions:
                return enhanced_card
            
            # Filter and extract name-sized boundaries
            name_boundary = self._filter_and_extract_boundaries(white_text_regions, search_region)
            
            if name_boundary:
                enhanced_card["name_boundary"] = {
                    "bbox": name_boundary,  # [x, y, w, h]
                    "search_region": search_region,
                    "detection_method": "white_text_visual",
                    "confidence": self._calculate_boundary_confidence(img, name_boundary)
                }
        
        except Exception as e:
            print(f"    ⚠️ Card {card['card_id']}: Name detection failed - {e}")
        
        return enhanced_card
    
    def _create_adaptive_search_region(self, card: Dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Create adaptive search region based on card width, avatar position, and detected time boxes
        
        Args:
            card: Card dictionary with boundary data (may include time_box)
            
        Returns:
            Search region as (x, y, w, h) or None if invalid
        """
        card_bbox = card["bbox"]  # [x, y, w, h]
        avatar_center = card["avatar"]["center"]  # [x, y]
        avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
        
        card_x, card_y, card_w, card_h = card_bbox
        avatar_center_x, avatar_center_y = avatar_center
        avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
        
        # Start search region below avatar center line (times are detected above)
        search_top = avatar_center_y + 2  # Start just below avatar center line
        search_bottom = card_y + card_h - self.SEARCH_MARGIN_TOP  # End before card bottom
        
        # Horizontal bounds: right of avatar to card edge  
        search_left = avatar_x + avatar_w + self.SEARCH_MARGIN_LEFT  # Start right of avatar
        search_right = card_x + card_w - self.SEARCH_MARGIN_RIGHT    # End before card edge
        
        # Adjust for detected time boxes to avoid conflicts
        if card.get("time_box"):
            time_bbox = card["time_box"]["bbox"]  # [x, y, w, h]
            time_x, time_y, time_w, time_h = time_bbox
            time_bottom = time_y + time_h
            
            # If time box extends into our search area, adjust search top
            if time_bottom > search_top:
                search_top = max(search_top, time_bottom + 2)  # Small gap below time box
        
        # Validate search region dimensions
        if search_right <= search_left or search_bottom <= search_top:
            return None
            
        search_width = search_right - search_left
        search_height = search_bottom - search_top
        
        # Ensure minimum search area
        if search_width < self.MIN_NAME_WIDTH or search_height < self.MIN_NAME_HEIGHT:
            return None
        
        return (search_left, search_top, search_width, search_height)
    
    def _detect_white_text_regions(self, img: np.ndarray, search_region: Tuple[int, int, int, int]) -> List[Tuple[int, int, int, int]]:
        """
        Detect white text regions using brightness thresholding
        
        Args:
            img: Original image
            search_region: Region to search as (x, y, w, h)
            
        Returns:
            List of detected text regions as (x, y, w, h) tuples
        """
        search_x, search_y, search_w, search_h = search_region
        
        # Extract search region from image
        roi = img[search_y:search_y + search_h, search_x:search_x + search_w]
        
        # Convert to grayscale for brightness analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply brightness threshold to detect white text
        _, white_mask = cv2.threshold(gray_roi, self.WHITE_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
        
        # Apply morphological operations to connect text pixels
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=self.MORPH_ITERATIONS)
        
        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Convert contours to bounding rectangles (relative to search region)
        text_regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if region has sufficient white pixels
            region_mask = white_mask[y:y+h, x:x+w]
            white_pixel_ratio = np.sum(region_mask > 0) / (w * h)
            
            if white_pixel_ratio >= self.MIN_WHITE_PIXEL_RATIO:
                # Convert to absolute coordinates
                abs_x = search_x + x
                abs_y = search_y + y
                text_regions.append((abs_x, abs_y, w, h))
        
        return text_regions
    
    def _filter_and_extract_boundaries(self, text_regions: List[Tuple[int, int, int, int]], 
                                     search_region: Tuple[int, int, int, int]) -> Optional[List[int]]:
        """
        Filter text regions by size constraints and extract the best name boundary
        
        Args:
            text_regions: List of detected text regions as (x, y, w, h)
            search_region: Original search region as (x, y, w, h)
            
        Returns:
            Best name boundary as [x, y, w, h] or None if no suitable region found
        """
        if not text_regions:
            return None
        
        # Filter by size constraints
        valid_regions = []
        for x, y, w, h in text_regions:
            if (self.MIN_NAME_WIDTH <= w <= self.MAX_NAME_WIDTH and 
                self.MIN_NAME_HEIGHT <= h <= self.MAX_NAME_HEIGHT):
                valid_regions.append((x, y, w, h))
        
        if not valid_regions:
            return None
        
        # If multiple valid regions, choose the largest one (most likely to be name)
        if len(valid_regions) == 1:
            return list(valid_regions[0])
        else:
            # Sort by area (width * height) and take the largest
            valid_regions.sort(key=lambda region: region[2] * region[3], reverse=True)
            return list(valid_regions[0])
    
    def _calculate_boundary_confidence(self, img: np.ndarray, name_boundary: List[int]) -> float:
        """
        Calculate confidence score for detected name boundary
        
        Args:
            img: Original image
            name_boundary: Name boundary as [x, y, w, h]
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        x, y, w, h = name_boundary
        
        # Extract name region
        name_roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics for confidence
        # 1. Average brightness (higher for white text)
        avg_brightness = np.mean(gray_roi) / 255.0
        
        # 2. Contrast (difference between brightest and darkest pixels)
        contrast = (np.max(gray_roi) - np.min(gray_roi)) / 255.0
        
        # 3. Size appropriateness (penalty for very small or very large regions)
        size_score = min(1.0, w / 60.0)  # Normalize width to typical name width
        if w > 150:  # Penalty for very wide regions
            size_score *= 0.7
        
        # 4. Aspect ratio score (names are typically wider than tall)
        aspect_ratio = w / h
        if 2.0 <= aspect_ratio <= 8.0:  # Typical name aspect ratios
            aspect_score = 1.0
        else:
            aspect_score = 0.6
        
        # Combine metrics (weighted average)
        confidence = (avg_brightness * 0.4 + contrast * 0.3 + size_score * 0.2 + aspect_score * 0.1)
        
        return min(1.0, confidence)  # Cap at 1.0

    def create_name_boundary_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create visualization showing detected name boundaries with orange rectangles
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for visualization
            
        Returns:
            Filename of generated visualization
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating name boundary visualization...")
        
        # Get detection results
        enhanced_cards, detection_info = self.detect_name_boundaries(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Track visualization statistics
        names_visualized = 0
        search_regions_drawn = 0
        
        # Draw enhanced visualization with name boundaries
        for card in enhanced_cards:
            # Draw existing card and avatar boundaries (from CardBoundaryDetector)
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)
            
            # Draw name boundary if detected (orange rectangle)
            if card.get("name_boundary"):
                name_bbox = card["name_boundary"]["bbox"]
                name_x, name_y, name_w, name_h = name_bbox
                confidence = card["name_boundary"]["confidence"]
                
                # Draw name boundary (orange rectangle)
                cv2.rectangle(result, (name_x, name_y), (name_x + name_w, name_y + name_h), (0, 165, 255), 2)
                
                # Draw confidence score
                cv2.putText(result, f"{confidence:.2f}", (name_x, name_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                names_visualized += 1
                
                # Optionally draw search region (light gray rectangle)
                if card["name_boundary"].get("search_region"):
                    search_region = card["name_boundary"]["search_region"]
                    search_x, search_y, search_w, search_h = search_region
                    cv2.rectangle(result, (search_x, search_y), (search_x + search_w, search_y + search_h), (128, 128, 128), 1)
                    search_regions_drawn += 1
            
            # Draw card ID
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add enhanced legend and statistics
        legend_y = 30
        cv2.putText(result, "Contact Name Boundary Detection Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers  Orange=Names",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection statistics
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        cv2.putText(result, f"Names Detected: {names_visualized}/{detection_info['total_cards_processed']} cards ({success_rate:.1f}%)",
                   (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Detection Method: {detection_info['detection_method']} | Gray=Search Regions",
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_ContactName_Boundaries_{names_visualized}names_{len(enhanced_cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Contact name boundary visualization saved: {output_path}")
        print(f"   📊 Visualization stats: {names_visualized} names, {search_regions_drawn} search regions drawn")
        
        return os.path.basename(output_path)


# =============================================================================
# 5. TIME BOX DETECTOR  
# =============================================================================

class TimeBoxDetector:
    """
    Time Box Detector using visual density patterns (non-OCR approach)
    Detects timestamp regions using column projection and statistical analysis
    """
    
    def __init__(self):
        self.card_boundary_detector = CardBoundaryDetector()
        
        # WeChat-optimized layout parameters for upper-center time detection
        self.CARD_WIDTH_FRAC = 0.7      # Search up to 70% of card width
        self.VPAD_RATIO = 0.12          # Vertical padding ratio for text box sizing
        self.AVATAR_TIME_MARGIN = 15    # Gap between avatar and time box (pixels)
        self.MIN_W_PX = 10              # Minimum timestamp width (pixels) - reduced for smaller times
        
        # Image processing parameters
        self.BILATERAL_D = 5            # Bilateral filter diameter
        self.BILATERAL_SIGMA_COLOR = 35 # Color sigma for bilateral filter
        self.BILATERAL_SIGMA_SPACE = 35 # Space sigma for bilateral filter
        self.CLAHE_CLIP = 3.0          # CLAHE clip limit - increased for better gray text contrast
        self.CLAHE_GRID = 6            # CLAHE tile grid size - smaller tiles for finer contrast adjustment
        self.ADAPT_BLOCK = 21          # Adaptive threshold block size
        self.ADAPT_C = 8               # Adaptive threshold constant
        
        # Statistical analysis parameters
        self.K_MAD = 1.5               # MAD threshold multiplier - lowered for better detection of faint text
        self.K_MAD_FALLBACK = 1.0      # Even lower threshold for second pass if first fails
        
        # Timestamp-specific parameters (more lenient for small regions)
        self.K_MAD_TIMESTAMP = 1.0     # Lower threshold for timestamp region analysis
        self.K_MAD_TIMESTAMP_FALLBACK = 0.5  # More aggressive fallback for timestamps
        
    def detect_time_boundaries(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Detect timestamp boundaries using visual density patterns
        
        Args:
            image_path: Path to screenshot image
            
        Returns:
            Tuple of (enhanced_cards_with_times, detection_info)
        """
        print(f"🎯 Time Box Detection: {os.path.basename(image_path)}")
        
        # Step 1: Get validated card and avatar data
        cards, card_detection_info = self.card_boundary_detector.detect_cards(image_path)
        if not cards:
            print("❌ No cards available for time detection")
            return [], {}
            
        # Step 2: Load image for processing
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
            
        # Step 3: Get panel right boundary from width detection
        width_boundaries = card_detection_info.get("width_boundaries")
        if not width_boundaries:
            print("❌ No width boundaries available for time detection")
            return [], {}
            
        # width_boundaries is a tuple (left_boundary, right_boundary)
        left_boundary, right_boundary = width_boundaries
        x_panel_right = right_boundary
        print(f"  📏 Using panel right boundary: {x_panel_right}px")
        print(f"  📄 Processing {len(cards)} cards for time boundary detection")
        
        # Step 4: Process each card for time boundaries
        enhanced_cards = []
        total_times_detected = 0
        
        for card in cards:
            enhanced_card = self._detect_time_box_for_card(img, card, x_panel_right)
            enhanced_cards.append(enhanced_card)
            
            if enhanced_card.get("time_box"):
                total_times_detected += 1
                card_id = enhanced_card["card_id"]
                time_bbox = enhanced_card["time_box"]["bbox"]
                density_score = enhanced_card["time_box"]["density_score"]
                print(f"    ⏰ Card {card_id}: Time box {time_bbox[2]}×{time_bbox[3]}px at ({time_bbox[0]}, {time_bbox[1]}) | density={density_score:.1f}")
        
        # Step 5: Generate detection summary
        detection_info = {
            "total_cards_processed": len(cards),
            "times_detected": total_times_detected,
            "detection_success_rate": total_times_detected / len(cards) if cards else 0,
            "card_detection_info": card_detection_info,
            "detection_method": "visual_density_pattern",
            "panel_right_boundary": x_panel_right
        }
        
        print(f"  ✅ Time boundary detection complete: {total_times_detected}/{len(cards)} cards have detected timestamps")
        
        return enhanced_cards, detection_info

    def _detect_time_box_for_card(self, img: np.ndarray, card: Dict, x_panel_right: int) -> Dict:
        """
        Detect time box for a single card using upper region density pattern analysis
        
        Args:
            img: Original image as numpy array
            card: Card dictionary with boundary and avatar data
            x_panel_right: Right boundary of the panel (kept for compatibility)
            
        Returns:
            Enhanced card dictionary with time box information
        """
        enhanced_card = card.copy()
        
        try:
            # Get card and avatar boundaries
            card_bbox = card["bbox"]  # [x, y, w, h]
            avatar_data = card["avatar"]  # Complete avatar information
            
            # Initialize debug info collection if requested
            debug_info = {} if hasattr(self, '_collect_debug') and self._collect_debug else None
            
            # Apply density-based time box detection in upper region
            time_result = self._upper_density_time_box(img, card_bbox, avatar_data, debug_info)
            
            if time_result:
                tx, ty, tw, th, density_score = time_result
                enhanced_card["time_box"] = {
                    "bbox": [tx, ty, tw, th],
                    "density_score": density_score,
                    "detection_method": "upper_region_density",
                    "search_region_info": {
                        "avatar_right": avatar_data["bbox"][0] + avatar_data["bbox"][2],
                        "avatar_center_y": avatar_data["center"][1],
                        "card_width_frac": self.CARD_WIDTH_FRAC,
                        "card_width": card_bbox[2]
                    }
                }
                
                # Add success info to debug data
                if debug_info is not None:
                    debug_info['detection_successful'] = True
            else:
                # Mark as failed if debug info exists
                if debug_info is not None:
                    debug_info['detection_successful'] = False
            
            # Store debug information in the card
            if debug_info is not None:
                enhanced_card["time_detection_debug"] = debug_info
        
        except Exception as e:
            print(f"    ⚠️ Card {card['card_id']}: Time detection failed - {e}")
            if hasattr(self, '_collect_debug') and self._collect_debug:
                enhanced_card["time_detection_debug"] = {
                    'detection_failed': True,
                    'failure_reason': f'exception: {str(e)}',
                    'detection_successful': False
                }
        
        return enhanced_card

    def _upper_density_time_box(self, bgr: np.ndarray, row_box: List[int], avatar_data: Dict, debug_info: Dict = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Core density-based time box detection algorithm for upper card region
        
        Args:
            bgr: Original BGR image
            row_box: Card boundaries as [x, y, w, h] 
            avatar_data: Avatar information including bbox and center
            
        Returns:
            Tuple of (x, y, w, h, density_score) or None if not found
        """
        rx, ry, rw, rh = map(int, row_box)
        avatar_bbox = avatar_data["bbox"]
        avatar_center = avatar_data["center"]
        
        # Calculate search region (upper portion, after avatar)
        avatar_right = avatar_bbox[0] + avatar_bbox[2]
        avatar_center_y = avatar_center[1]
        
        # Horizontal bounds: from avatar right edge to full card width (expanded coverage)
        search_left = avatar_right + 5  # Smaller margin for better coverage
        search_right = rx + rw - 5      # Full card width with small right margin
        
        # Vertical bounds: upper portion above avatar center line (divider)
        search_top = ry + 5  # Small margin from card top
        search_bottom = avatar_center_y  # Stop at avatar center (divider line)
        
        # Validate search region
        if search_left >= search_right or search_top >= search_bottom:
            return None
        if search_right - search_left < self.MIN_W_PX or search_bottom - search_top < 6:
            return None

        # Extract ROI for processing
        roi = bgr[search_top:search_bottom, search_left:search_right]
        
        # Store debug information if requested
        if debug_info is not None:
            debug_info.update({
                'search_region': {
                    'left': search_left, 'right': search_right,
                    'top': search_top, 'bottom': search_bottom,
                    'width': search_right - search_left,
                    'height': search_bottom - search_top
                },
                'roi_original': roi.copy(),
                'card_boundaries': {'x': rx, 'y': ry, 'w': rw, 'h': rh},
                'avatar_info': {'bbox': avatar_bbox, 'center': avatar_center}
            })
        
        # Simple color-based detection: Names are white/light, timestamps are gray
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for gray timestamps (the key distinguishing feature)
        gray_lower = np.array([0, 0, 80])      # Lower bound for gray text
        gray_upper = np.array([180, 60, 180])  # Upper bound for gray text
        
        # Create mask for gray timestamp text
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Optional: Clean up the mask
        kernel = np.ones((2,2), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
        
        # Store preprocessing results in debug info
        if debug_info is not None:
            debug_info.update({
                'roi_original_hsv': hsv.copy(),
                'roi_gray_mask': gray_mask.copy(),
                'color_detection_method': 'HSV_gray_text_segmentation'
            })
        
        # Find contours in the gray timestamp mask
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug_info is not None:
                debug_info.update({
                    'detection_failed': True,
                    'failure_reason': 'no_valid_timestamp_contour',
                    'contours_found': 0
                })
            return None
        
        # Find the best timestamp contour (rightmost, reasonable size)
        best_contour = None
        best_score = -1
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Validate timestamp characteristics
            if w < 8 or h < 6:  # Too small
                continue
            if w > search_right - search_left - 10:  # Too wide  
                continue
            if h > (search_bottom - search_top) * 0.8:  # Too tall
                continue
                
            # Prefer rightmost contours (timestamps are typically on the right)
            position_score = x / max(1, search_right - search_left)
            size_score = min(w * h / 100, 1.0)  # Size bonus but capped
            
            total_score = position_score * 0.7 + size_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_contour = contour
        
        if best_contour is None:
            if debug_info is not None:
                debug_info.update({
                    'detection_failed': True,
                    'failure_reason': 'no_valid_timestamp_contour',
                    'contours_found': len(contours)
                })
            return None
        
        # Get bounding box of best timestamp contour
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Convert to absolute coordinates
        tx0 = search_left + x
        ty = search_top + y
        
        # Extend width to card right boundary for full timestamp capture
        card_right = rx + rw - 5
        tw = max(w, card_right - tx0)  # At least contour width, but extend to card edge
        th = h
        
        # Calculate density score for compatibility
        contour_area = cv2.contourArea(best_contour)
        density_score = contour_area / max(1, w * h) * 100  # Percentage fill
        
        if debug_info is not None:
            debug_info.update({
                'detection_failed': False,
                'detection_successful': True,
                'contours_found': len(contours),
                'best_contour_score': best_score,
                'timestamp_bbox': [tx0, ty, tw, th],
                'density_score': density_score
            })
        
        return (int(tx0), int(ty), int(tw), int(th), float(density_score))
    
    def _find_name_timestamp_boundary(self, roi: np.ndarray, column_projection: np.ndarray) -> int:
        """
        Find the boundary between name (left, larger/darker) and timestamp (right, smaller/lighter)
        
        Args:
            roi: Original ROI image (BGR)
            column_projection: Column-wise text density projection
            
        Returns:
            Boundary column index separating names from timestamps
        """
        if roi.shape[1] < 20:  # Too narrow for boundary detection
            return roi.shape[1] // 2
            
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Analyze text characteristics across columns
        roi_width = roi.shape[1]
        window_size = max(5, roi_width // 10)  # Sliding window for analysis
        
        color_scores = []  # Lower = darker text, Higher = lighter text
        size_scores = []   # Text density variations
        
        for col in range(0, roi_width - window_size, 2):  # Step by 2 for efficiency
            window = gray[:, col:col + window_size]
            
            # Color analysis: average intensity of text pixels
            text_pixels = window[window < 200]  # Assume text is darker than 200
            avg_intensity = text_pixels.mean() if len(text_pixels) > 0 else 255
            color_scores.append(avg_intensity)
            
            # Size analysis: variation in text density
            col_densities = column_projection[col:col + window_size]
            size_variation = col_densities.std() if len(col_densities) > 0 else 0
            size_scores.append(size_variation)
        
        if len(color_scores) < 3:
            return roi_width // 2
            
        # Find the transition point where text changes from dark to light
        color_scores = np.array(color_scores)
        
        # Look for the point where intensity increases significantly (dark → light)
        intensity_changes = np.diff(color_scores)
        
        # Find the largest positive change (transition to lighter text)
        max_change_idx = np.argmax(intensity_changes)
        
        # Convert back to column coordinate
        boundary_col = max_change_idx * 2 + window_size // 2
        
        # Ensure boundary is reasonable (not too close to edges)
        boundary_col = max(roi_width // 4, min(roi_width * 3 // 4, boundary_col))
        
        return boundary_col

    def create_time_box_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive visualization showing detected time boxes with purple rectangles
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for visualization
            
        Returns:
            Filename of generated visualization
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating comprehensive time box visualization...")
        
        # Get detection results
        enhanced_cards, detection_info = self.detect_time_boundaries(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Track visualization statistics
        times_visualized = 0
        search_regions_drawn = 0
        
        # Draw comprehensive visualization with all elements
        for card in enhanced_cards:
            # Draw existing card and avatar boundaries (from CardBoundaryDetector)
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)
            
            # Draw name boundary if detected (orange rectangle)
            if card.get("name_boundary"):
                name_bbox = card["name_boundary"]["bbox"]
                name_x, name_y, name_w, name_h = name_bbox
                cv2.rectangle(result, (name_x, name_y), (name_x + name_w, name_y + name_h), (0, 165, 255), 2)
            
            # Draw time box if detected (purple rectangle)
            if card.get("time_box"):
                time_bbox = card["time_box"]["bbox"]
                time_x, time_y, time_w, time_h = time_bbox
                density_score = card["time_box"]["density_score"]
                
                # Draw time box boundary (purple rectangle)
                cv2.rectangle(result, (time_x, time_y), (time_x + time_w, time_y + time_h), (128, 0, 128), 2)
                
                # Draw density score
                cv2.putText(result, f"{density_score:.0f}", (time_x, time_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
                
                times_visualized += 1
                
                # Optionally draw search region (light gray rectangle) for debugging
                if card["time_box"].get("search_region_info"):
                    search_info = card["time_box"]["search_region_info"]
                    avatar_right = search_info["avatar_right"]
                    avatar_center_y = search_info["avatar_center_y"]
                    card_width_frac = search_info["card_width_frac"]
                    card_width = search_info["card_width"]
                    
                    # Calculate upper region search bounds
                    search_left = avatar_right + 15  # AVATAR_TIME_MARGIN
                    search_right = card_x + int(card_width * card_width_frac)
                    search_top = card_y + 5  # Small margin from card top
                    search_bottom = avatar_center_y  # Above divider line
                    
                    cv2.rectangle(result, (search_left, search_top), (search_right, search_bottom), 
                                (192, 192, 192), 1)  # Light gray
                    search_regions_drawn += 1
            
            # Draw card ID
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add comprehensive legend and statistics
        legend_y = 30
        cv2.putText(result, "Complete WeChat Card Analysis Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers  Orange=Names  Purple=Times",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection statistics
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        cv2.putText(result, f"Times Detected: {times_visualized}/{detection_info['total_cards_processed']} cards ({success_rate:.1f}%)",
                   (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Method: {detection_info['detection_method']} | Panel Right: {detection_info['panel_right_boundary']}px",
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_Complete_Analysis_{times_visualized}times_{len(enhanced_cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Complete card analysis visualization saved: {output_path}")
        print(f"   📊 Visualization stats: {times_visualized} time boxes, {search_regions_drawn} search regions drawn")
        
        return os.path.basename(output_path)


    def create_debug_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive debug visualization showing all detection details for every card
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for debug visualization
            
        Returns:
            Filename of generated debug visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        
        # Load original image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ Failed to load image for debug visualization: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"🔍 Creating debug visualization with detailed analysis...")
        
        # Enable debug collection and get detection results
        self._collect_debug = True
        enhanced_cards, detection_info = self.detect_time_boundaries(image_path)
        self._collect_debug = False
        
        # Separate successful and failed detections
        successful_cards = [card for card in enhanced_cards if card.get("time_box")]
        failed_cards = [card for card in enhanced_cards if not card.get("time_box")]
        
        print(f"  📊 Debug Analysis: {len(successful_cards)} successful, {len(failed_cards)} failed detections")
        
        # Create comprehensive matplotlib figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main image with all search regions
        ax_main = fig.add_subplot(gs[0:2, 0:3])
        ax_main.imshow(img_rgb)
        ax_main.set_title(f'Time Detection Overview - {len(enhanced_cards)} Cards Analyzed', fontsize=14, fontweight='bold')
        ax_main.set_xlabel(f'✅ {len(successful_cards)} Successful  ❌ {len(failed_cards)} Failed')
        
        # Draw search regions for ALL cards
        for i, card in enumerate(enhanced_cards):
            debug_info = card.get("time_detection_debug", {})
            search_region = debug_info.get("search_region", {})
            
            if search_region:
                left, right = search_region['left'], search_region['right']
                top, bottom = search_region['top'], search_region['bottom']
                
                # Color code: green for success, red for failure
                color = 'lime' if card.get("time_box") else 'red'
                alpha = 0.3 if card.get("time_box") else 0.5
                
                # Draw search region rectangle
                rect = patches.Rectangle((left, top), right-left, bottom-top, 
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
                ax_main.add_patch(rect)
                
                # Add card number
                ax_main.text(left+5, top+15, f'Card {card["card_id"]}', 
                           fontsize=10, color='white', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        
        ax_main.set_xlim(0, img_rgb.shape[1])
        ax_main.set_ylim(img_rgb.shape[0], 0)
        
        # ROI previews for first few cards
        roi_cols = 3
        roi_start_row = 2
        for i, card in enumerate(enhanced_cards[:6]):  # Show first 6 cards
            debug_info = card.get("time_detection_debug", {})
            roi_original = debug_info.get("roi_original")
            
            if roi_original is not None:
                ax_roi = fig.add_subplot(gs[roi_start_row + i//roi_cols, i%roi_cols])
                roi_rgb = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB) if len(roi_original.shape) == 3 else roi_original
                ax_roi.imshow(roi_rgb, cmap='gray' if len(roi_original.shape) == 2 else None)
                
                success = card.get("time_box") is not None
                status = "✅" if success else "❌"
                ax_roi.set_title(f'{status} Card {card["card_id"]} ROI', fontsize=10)
                ax_roi.set_xticks([])
                ax_roi.set_yticks([])
        
        # Binary processing results
        for i, card in enumerate(enhanced_cards[:3]):  # Show first 3 cards' processing
            debug_info = card.get("time_detection_debug", {})
            roi_binary = debug_info.get("roi_binary")
            
            if roi_binary is not None:
                ax_bin = fig.add_subplot(gs[roi_start_row + 1, i + 3])
                ax_bin.imshow(roi_binary, cmap='gray')
                ax_bin.set_title(f'Card {card["card_id"]} Binary', fontsize=10)
                ax_bin.set_xticks([])
                ax_bin.set_yticks([])
        
        # Column projection histograms for failed cards
        failed_count = 0
        for card in failed_cards:
            if failed_count >= 3:  # Show max 3 failed cards
                break
                
            debug_info = card.get("time_detection_debug", {})
            column_projection = debug_info.get("column_projection")
            stats = debug_info.get("statistics", {})
            
            if column_projection is not None:
                ax_hist = fig.add_subplot(gs[3, failed_count])
                ax_hist.bar(range(len(column_projection)), column_projection, alpha=0.7, color='red')
                
                # Draw threshold lines
                if 'threshold_primary' in stats:
                    ax_hist.axhline(y=stats['threshold_primary'], color='orange', linestyle='--', 
                                  label=f'Primary: {stats["threshold_primary"]:.1f}')
                if stats.get('threshold_fallback'):
                    ax_hist.axhline(y=stats['threshold_fallback'], color='red', linestyle=':', 
                                  label=f'Fallback: {stats["threshold_fallback"]:.1f}')
                
                ax_hist.set_title(f'❌ Card {card["card_id"]} Column Projection', fontsize=10)
                ax_hist.set_xlabel('Column Position')
                ax_hist.set_ylabel('Pixel Density')
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
                
                failed_count += 1
        
        # Add overall statistics
        fig.suptitle(f'Time Detection Debug Analysis - {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        
        # Generate output filename
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_Debug_TimeDetection_{len(successful_cards)}success_{len(failed_cards)}failed.png"
        
        # Save debug visualization
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Debug visualization saved: {output_path}")
        print(f"   📊 Analysis complete: {len(successful_cards)}/{len(enhanced_cards)} successful detections")
        
        # Print detailed debug info for failed cards
        for card in failed_cards:
            debug_info = card.get("time_detection_debug", {})
            stats = debug_info.get("statistics", {})
            search_region = debug_info.get("search_region", {})
            
            print(f"   ❌ Card {card['card_id']} Debug Info:")
            print(f"      Search region: {search_region.get('width', 0)}×{search_region.get('height', 0)}px")
            print(f"      Max projection: {stats.get('max_projection', 0)}")
            primary_thr = stats.get('threshold_primary', 0) or 0
            fallback_thr = stats.get('threshold_fallback', 0) or 0
            print(f"      Threshold: {primary_thr:.1f} (primary), {fallback_thr:.1f} (fallback)")
            print(f"      Reason: {debug_info.get('failure_reason', 'unknown')}")
        
        return os.path.basename(output_path)


# =============================================================================
# ENHANCED API FUNCTIONS FOR LIVE PROCESSING
# =============================================================================

def capture_and_process_screenshot(output_dir: str = "pic/screenshots", 
                                  custom_filename: str = None) -> Optional[Tuple[str, Dict]]:
    """
    Capture a fresh WeChat screenshot and process it for card analysis
    
    Args:
        output_dir: Directory to save screenshot and visualizations
        custom_filename: Custom filename for screenshot, auto-generated if None
        
    Returns:
        Tuple of (screenshot_path, analysis_results) or None if failed
        
    Usage:
        screenshot_path, results = capture_and_process_screenshot()
        if results:
            print(f"Found {results['cards_detected']} cards")
    """
    if not SCREENSHOT_AVAILABLE:
        print("❌ Screenshot capture not available. Install required module.")
        return None
        
    try:
        print("\n🎯 Live WeChat Screenshot & Card Processing")
        print("=" * 50)
        
        # Step 1: Capture fresh screenshot
        print("\n📸 Step 1: Capturing WeChat screenshot...")
        screenshot_path = capture_screenshot(output_dir=output_dir, filename=custom_filename)
        
        if not screenshot_path:
            print("❌ Failed to capture screenshot")
            return None
            
        print(f"✅ Screenshot captured: {os.path.basename(screenshot_path)}")
        
        # Step 2: Process with card analysis
        print("\n🔍 Step 2: Processing card analysis...")
        results = process_screenshot_file(screenshot_path)
        
        if results:
            print(f"\n✅ Analysis complete:")
            print(f"   Width: {results.get('width_detected')}px")  
            print(f"   Avatars: {results.get('avatars_detected')}")
            print(f"   Cards: {results.get('cards_detected')}")
            return screenshot_path, results
        else:
            print("❌ Analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error in capture_and_process_screenshot: {e}")
        return None

def process_screenshot_file(image_path: str) -> Optional[Dict]:
    """
    Process a screenshot file and return comprehensive analysis results
    
    Args:
        image_path: Path to screenshot file to analyze
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        # Initialize processors
        width_detector = SimpleWidthDetector()
        avatar_detector = CardAvatarDetector() 
        boundary_detector = CardBoundaryDetector()
        
        results = {}
        
        # 1. Width Detection
        width_result = width_detector.detect_width(image_path)
        if width_result:
            left, right, width = width_result
            results['width_detected'] = width
            results['width_boundaries'] = {'left': left, 'right': right}
        else:
            results['width_detected'] = None
            results['width_boundaries'] = None
            
        # 2. Avatar Detection  
        avatars, avatar_info = avatar_detector.detect_avatars(image_path)
        results['avatars_detected'] = len(avatars)
        results['avatar_list'] = avatars
        results['avatar_detection_info'] = avatar_info
        
        # 3. Card Boundary Detection
        cards, card_info = boundary_detector.detect_cards(image_path)
        results['cards_detected'] = len(cards)
        results['card_list'] = cards  
        results['card_detection_info'] = card_info
        
        # 4. Summary
        results['processing_successful'] = True
        results['image_processed'] = os.path.basename(image_path)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing screenshot: {e}")
        return None

def process_current_wechat_window() -> Optional[Dict]:
    """
    Convenience function: Capture current WeChat window and analyze cards
    
    Returns:
        Dictionary with complete analysis results or None if failed
        
    Usage:
        results = process_current_wechat_window()
        if results:
            for i, card in enumerate(results['card_list'], 1):
                print(f"Card {i}: {card['width']}×{card['height']}px")
    """
    result = capture_and_process_screenshot()
    if result:
        screenshot_path, analysis = result
        return analysis
    return None

def get_live_card_analysis(include_visualizations: bool = True) -> Optional[Tuple[Dict, Dict]]:
    """
    Get comprehensive live card analysis with optional visualizations
    
    Args:
        include_visualizations: Whether to generate visualization files
        
    Returns:
        Tuple of (analysis_results, visualization_paths) or None if failed
    """
    result = capture_and_process_screenshot()
    if not result:
        return None
        
    screenshot_path, analysis = result
    
    visualization_paths = {}
    if include_visualizations:
        try:
            # Generate all visualizations
            width_detector = SimpleWidthDetector()  
            avatar_detector = CardAvatarDetector()
            boundary_detector = CardBoundaryDetector()
            
            # Width visualization
            if analysis.get('width_detected'):
                width_vis = width_detector.create_width_visualization(screenshot_path)
                visualization_paths['width'] = width_vis
                
            # Avatar visualization  
            if analysis.get('avatars_detected'):
                avatar_vis = avatar_detector.create_advanced_avatar_visualization(screenshot_path) 
                visualization_paths['avatars'] = avatar_vis
                
            # Card boundary visualization
            if analysis.get('cards_detected'):
                card_vis = boundary_detector.create_card_boundary_visualization(screenshot_path)
                visualization_paths['cards'] = card_vis
                
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            
    return analysis, visualization_paths


# =============================================================================
# MAIN DEMO SECTION
# =============================================================================

if __name__ == "__main__":
    
    print("🎯 Consolidated Card Processing Module (m_Card_Processing.py)")
    print("="*70)
    
    if SCREENSHOT_AVAILABLE:
        print("\n📸 LIVE WECHAT SCREENSHOT & CARD PROCESSING")
        print("=" * 50)
        
        print("\n📸 Step 1: Capturing fresh WeChat screenshot...")
        result = capture_and_process_screenshot()
        
        if result:
            screenshot_path, analysis = result
            print(f"\n✅ Live Analysis Results:")
            print(f"   Screenshot: {os.path.basename(screenshot_path)}")
            print(f"   Width: {analysis.get('width_detected')}px")
            print(f"   Avatars: {analysis.get('avatars_detected')}")
            print(f"   Cards: {analysis.get('cards_detected')}")
            
            # Generate comprehensive visualizations
            print(f"\n🎨 Step 2: Generating comprehensive visualizations...")
            try:
                # Initialize detectors for visualizations
                width_detector = SimpleWidthDetector()
                avatar_detector = CardAvatarDetector()
                boundary_detector = CardBoundaryDetector()
                name_detector = ContactNameBoundaryDetector()
                time_detector = TimeBoxDetector()
                
                viz_results = {}
                
                # Width visualization
                if analysis.get('width_detected'):
                    width_viz = width_detector.create_width_visualization(screenshot_path)
                    if width_viz:
                        viz_results['width'] = width_viz
                        print(f"   ✅ Width visualization: {width_viz}")
                    
                # Avatar visualization  
                if analysis.get('avatars_detected'):
                    avatar_viz = avatar_detector.create_visualization(screenshot_path)
                    if avatar_viz:
                        viz_results['avatars'] = avatar_viz
                        print(f"   ✅ Avatar visualization: {avatar_viz}")
                    
                # Card boundary visualization
                if analysis.get('cards_detected'):
                    card_viz = boundary_detector.create_card_visualization(screenshot_path)
                    if card_viz:
                        viz_results['cards'] = card_viz
                        print(f"   ✅ Card boundary visualization: {card_viz}")
                
                # Time box detection visualization (moved before names)
                if analysis.get('cards_detected'):
                    print(f"\n⏰ Step 3: Processing timestamp boundaries...")
                    time_viz = time_detector.create_time_box_visualization(screenshot_path)
                    if time_viz:
                        viz_results['times'] = time_viz
                        print(f"   ✅ Complete analysis visualization: {time_viz}")
                
                # Contact name boundary visualization (moved after times)
                if analysis.get('cards_detected'):
                    print(f"\n🔍 Step 4: Processing contact name boundaries...")
                    name_viz = name_detector.create_name_boundary_visualization(screenshot_path)
                    if name_viz:
                        viz_results['names'] = name_viz
                        print(f"   ✅ Name boundary visualization: {name_viz}")
                
                print(f"\n📊 COMPLETE PROCESSING SUMMARY:")
                print(f"   Original screenshot: {os.path.basename(screenshot_path)}")
                print(f"   Width detected: {analysis.get('width_detected')}px")
                print(f"   Avatars found: {analysis.get('avatars_detected')}")
                print(f"   Cards identified: {analysis.get('cards_detected')}")
                print(f"   Processing stages: Width → Avatars → Cards → Times → Names")
                print(f"   Visualizations generated: {len(viz_results)} files")
                
                print(f"\n✅ Complete WeChat card analysis pipeline finished!")
                print(f"   📁 Check pic/screenshots/ for all visualization outputs")
                print(f"   🎯 Final result: Complete analysis with all detected elements")
                    
            except Exception as e:
                print(f"⚠️  Visualization generation failed: {e}")
                print(f"   But basic analysis completed successfully.")
                
        else:
            print("❌ Live capture failed. Trying to process latest available screenshot...")
            
            # Fallback: Try to find and process the latest screenshot
            width_detector = SimpleWidthDetector()
            latest_screenshot = width_detector.get_latest_screenshot()
            
            if latest_screenshot:
                print(f"\n📁 Found latest screenshot: {os.path.basename(latest_screenshot)}")
                print("=" * 50)
                
                # Process the latest screenshot
                analysis = process_screenshot_file(latest_screenshot)
                if analysis:
                    print(f"\n✅ Analysis Results:")
                    print(f"   Screenshot: {os.path.basename(latest_screenshot)}")
                    print(f"   Width: {analysis.get('width_detected')}px")
                    print(f"   Avatars: {analysis.get('avatars_detected')}")
                    print(f"   Cards: {analysis.get('cards_detected')}")
                    
                    print(f"\n✅ Processing complete using latest screenshot!")
                else:
                    print("❌ Failed to process latest screenshot")
            else:
                print("❌ No screenshots found. Make sure WeChat is running and try again.")
    
    else:
        print("\n❌ SCREENSHOT CAPTURE NOT AVAILABLE")
        print("=" * 50)
        print("⚠️  Screenshot module not available.")
        print("💡 Please ensure the m_ScreenShot_WeChatWindow module is properly installed.")
        print("💡 Required dependencies: pyautogui, Quartz (macOS), easyocr")
        
        # Try to process existing screenshots as fallback
        width_detector = SimpleWidthDetector()
        latest_screenshot = width_detector.get_latest_screenshot()
        
        if latest_screenshot:
            print(f"\n📁 Fallback: Processing existing screenshot...")
            print(f"   Using: {os.path.basename(latest_screenshot)}")
            
            analysis = process_screenshot_file(latest_screenshot)
            if analysis:
                print(f"\n✅ Analysis Results (from existing file):")
                print(f"   Width: {analysis.get('width_detected')}px")
                print(f"   Avatars: {analysis.get('avatars_detected')}")
                print(f"   Cards: {analysis.get('cards_detected')}")
            else:
                print("❌ Failed to process existing screenshot")
        else:
            print("❌ No existing screenshots found either.")
    
    print(f"\n✅ Consolidated Card Processing Module execution complete!")
    print("="*70)