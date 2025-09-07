#!/usr/bin/env python3
"""
Timestamp Detection Module
Detects left boundary of grey timestamp text in WeChat screenshots.

Based on successful OpenCV analysis showing Sobel gradient detection 
provides most accurate results for timestamp boundary detection.
"""

import cv2
import numpy as np
from datetime import datetime
import os
from typing import Tuple, Optional, Dict, Any

class cTimestampDetector:
    """
    Timestamp left boundary detector using OpenCV computer vision techniques.
    
    Primary method: Sobel gradient detection with adaptive thresholding
    Fallback method: Canny edge detection
    """
    
    def __init__(self):
        """Initialize detector with optimized parameters."""
        # Sobel gradient parameters
        self.sobel_ksize = 3
        self.sobel_threshold_multiplier = 2.0
        
        # Search region parameters (focus on timestamp area)
        self.search_region_x_start = 0.6  # Start from 60% width
        self.search_region_height = 80    # Top 80 pixels
        
        # Canny edge parameters (fallback)
        self.canny_low_threshold = 50
        self.canny_high_threshold = 150
        self.gaussian_blur_ksize = 3
        
        # Confidence scoring
        self.min_boundary_candidates = 3
        self.max_std_dev_for_high_confidence = 5
        self.max_std_dev_for_medium_confidence = 15
        
    def detect_timestamp_boundary(self, image_path: str, save_debug: bool = False) -> Dict[str, Any]:
        """
        Detect timestamp left boundary using multiple methods.
        
        Args:
            image_path: Path to screenshot image
            save_debug: Whether to save debug visualization images
            
        Returns:
            Dictionary containing:
            - boundary_x: Detected x coordinate (int or None)
            - confidence: 'high', 'medium', 'low', or 'none'
            - method_used: 'gradient', 'edge', or 'none'
            - debug_info: Additional debugging information
            - processing_time_ms: Processing time in milliseconds
        """
        start_time = datetime.now()
        
        try:
            # Load and validate image
            img = cv2.imread(image_path)
            if img is None:
                return self._create_error_result("Failed to load image", start_time)
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            
            # Method 1: Sobel gradient detection (primary)
            gradient_result = self._detect_with_sobel_gradient(gray, width, height)
            
            # Method 2: Canny edge detection (fallback)
            edge_result = self._detect_with_canny_edges(gray, width, height)
            
            # Determine best result and confidence
            final_result = self._evaluate_results(gradient_result, edge_result)
            
            # Add processing time
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            final_result['processing_time_ms'] = round(processing_time, 2)
            
            # Save debug visualization if requested
            if save_debug and final_result['boundary_x'] is not None:
                debug_path = self._save_debug_visualization(
                    img, final_result, image_path
                )
                final_result['debug_image_path'] = debug_path
                
            return final_result
            
        except Exception as e:
            return self._create_error_result(f"Processing error: {str(e)}", start_time)
    
    def _detect_with_sobel_gradient(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Detect boundary using Sobel gradient method."""
        try:
            # Define search region (right portion, top area)
            search_height = min(self.search_region_height, height)
            search_x_start = int(width * self.search_region_x_start)
            timestamp_region = gray[:search_height, search_x_start:]
            
            # Apply Sobel X filter to detect vertical edges
            sobel_x = cv2.Sobel(timestamp_region, cv2.CV_64F, 1, 0, ksize=self.sobel_ksize)
            sobel_abs = np.absolute(sobel_x)
            
            # Calculate adaptive threshold
            threshold = (np.mean(sobel_abs) + 
                        self.sobel_threshold_multiplier * np.std(sobel_abs))
            
            # Find boundary candidates in each row
            boundary_candidates = []
            for row in range(sobel_abs.shape[0]):
                edge_positions = np.where(sobel_abs[row] > threshold)[0]
                if len(edge_positions) > 0:
                    # Convert back to full image coordinates
                    boundary_x = edge_positions[0] + search_x_start
                    boundary_candidates.append(boundary_x)
            
            if len(boundary_candidates) >= self.min_boundary_candidates:
                boundary_x = int(np.median(boundary_candidates))
                std_dev = np.std(boundary_candidates)
                
                return {
                    'boundary_x': boundary_x,
                    'method': 'gradient',
                    'candidates_count': len(boundary_candidates),
                    'std_dev': std_dev,
                    'threshold_used': threshold,
                    'search_region': (search_x_start, 0, width - search_x_start, search_height)
                }
            else:
                return {'boundary_x': None, 'method': 'gradient', 'candidates_count': len(boundary_candidates)}
                
        except Exception as e:
            return {'boundary_x': None, 'method': 'gradient', 'error': str(e)}
    
    def _detect_with_canny_edges(self, gray: np.ndarray, width: int, height: int) -> Dict[str, Any]:
        """Detect boundary using Canny edge detection method."""
        try:
            # Apply Gaussian blur and Canny edge detection
            blurred = cv2.GaussianBlur(gray, (self.gaussian_blur_ksize, self.gaussian_blur_ksize), 0)
            edges = cv2.Canny(blurred, self.canny_low_threshold, self.canny_high_threshold)
            
            # Define search region
            search_height = min(self.search_region_height, height)
            search_x_start = int(width * self.search_region_x_start)
            edge_region = edges[:search_height, search_x_start:]
            
            # Find leftmost edge in search region
            for col in range(edge_region.shape[1]):
                if np.any(edge_region[:, col]):
                    boundary_x = col + search_x_start
                    return {
                        'boundary_x': boundary_x,
                        'method': 'edge',
                        'search_region': (search_x_start, 0, width - search_x_start, search_height)
                    }
            
            return {'boundary_x': None, 'method': 'edge'}
            
        except Exception as e:
            return {'boundary_x': None, 'method': 'edge', 'error': str(e)}
    
    def _evaluate_results(self, gradient_result: Dict[str, Any], edge_result: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate results from multiple methods and determine best result."""
        
        gradient_boundary = gradient_result.get('boundary_x')
        edge_boundary = edge_result.get('boundary_x')
        
        # If gradient method succeeded, use it (most reliable based on analysis)
        if gradient_boundary is not None:
            confidence = self._calculate_confidence(gradient_result)
            
            return {
                'boundary_x': gradient_boundary,
                'confidence': confidence,
                'method_used': 'gradient',
                'debug_info': {
                    'gradient_result': gradient_result,
                    'edge_result': edge_result,
                    'agreement': abs(gradient_boundary - edge_boundary) < 20 if edge_boundary else False
                }
            }
        
        # Fallback to edge method
        elif edge_boundary is not None:
            return {
                'boundary_x': edge_boundary,
                'confidence': 'medium',  # Edge method is less precise
                'method_used': 'edge',
                'debug_info': {
                    'gradient_result': gradient_result,
                    'edge_result': edge_result,
                    'fallback_reason': 'gradient_method_failed'
                }
            }
        
        # No detection
        else:
            return {
                'boundary_x': None,
                'confidence': 'none',
                'method_used': 'none',
                'debug_info': {
                    'gradient_result': gradient_result,
                    'edge_result': edge_result,
                    'failure_reason': 'no_boundary_detected'
                }
            }
    
    def _calculate_confidence(self, gradient_result: Dict[str, Any]) -> str:
        """Calculate confidence level based on gradient detection results."""
        candidates_count = gradient_result.get('candidates_count', 0)
        std_dev = gradient_result.get('std_dev', float('inf'))
        
        if candidates_count >= self.min_boundary_candidates:
            if std_dev <= self.max_std_dev_for_high_confidence:
                return 'high'
            elif std_dev <= self.max_std_dev_for_medium_confidence:
                return 'medium'
            else:
                return 'low'
        else:
            return 'low'
    
    def _save_debug_visualization(self, img: np.ndarray, result: Dict[str, Any], 
                                 original_path: str) -> str:
        """Save debug visualization with detected boundary."""
        try:
            # Create visualization
            vis_img = img.copy()
            boundary_x = result['boundary_x']
            method = result['method_used']
            
            # Draw boundary line
            color = (0, 255, 0) if result['confidence'] == 'high' else (0, 165, 255)  # Green or Orange
            cv2.line(vis_img, (boundary_x, 0), (boundary_x, img.shape[0]), color, 3)
            
            # Add method label
            label = f"{method.upper()}: x={boundary_x} ({result['confidence']})"
            cv2.putText(vis_img, label, (boundary_x - 80, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw search region rectangle
            debug_info = result.get('debug_info', {})
            if 'gradient_result' in debug_info:
                gradient_info = debug_info['gradient_result']
                if 'search_region' in gradient_info:
                    x, y, w, h = gradient_info['search_region']
                    cv2.rectangle(vis_img, (x, y), (x + w, y + h), (255, 255, 0), 2)
            
            # Save with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(original_path))[0]
            debug_path = f"/Users/erli/coding/deepseek_wechat_bot/pic/screenshots/{timestamp}_timestamp_detection_{base_name}.png"
            
            cv2.imwrite(debug_path, vis_img)
            return debug_path
            
        except Exception as e:
            return f"Error saving debug image: {str(e)}"
    
    def _create_error_result(self, error_message: str, start_time: datetime) -> Dict[str, Any]:
        """Create standardized error result."""
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        return {
            'boundary_x': None,
            'confidence': 'none',
            'method_used': 'none',
            'error': error_message,
            'processing_time_ms': round(processing_time, 2),
            'debug_info': {'error': error_message}
        }

# Convenience function for direct usage
def fdetect_timestamp_boundary(image_path: str, save_debug: bool = False) -> Dict[str, Any]:
    """
    Convenience function to detect timestamp boundary.
    
    Args:
        image_path: Path to screenshot image
        save_debug: Whether to save debug visualization
        
    Returns:
        Detection result dictionary
    """
    detector = cTimestampDetector()
    return detector.detect_timestamp_boundary(image_path, save_debug)

# =============================================================================
# MANUAL CODE TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Code Testing - TIMESTAMP DETECTOR")
    print("=" * 60)
    print("🔍 [DEBUG] Smoke test ENTRY")
    
    try:
        # Simply instantiate the class
        print("   🔧 Testing cTimestampDetector...")
        detector = cTimestampDetector()
        print("   ✅ cTimestampDetector instantiated successfully")
        
        print("🏁 [DEBUG] Smoke test PASSED")
        
    except Exception as e:
        print(f"   ❌ [ERROR] Smoke test FAILED: {str(e)}")
        print("🏁 [DEBUG] Smoke test FAILED")