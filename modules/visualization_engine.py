#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization Engine Module (visualization_engine.py)

This module provides centralized visualization functionality for the WeChat bot.
Extracted from m_Card_Processing.py for better separation of concerns.

Features:
- Centralized visualization utilities
- Consistent styling and overlays
- Heatmap generation
- Debug visualization support
- Coordinate with detector classes for visualizations
"""

import cv2
import numpy as np
import os
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime


class VisualizationEngine:
    """
    Centralized visualization engine providing consistent visual overlays and debug outputs
    """
    
    def __init__(self):
        # Standard color scheme for consistent visualizations
        self.colors = {
            'boundary': (0, 255, 0),        # Green for boundaries
            'avatar': (255, 0, 0),          # Red for avatars
            'card': (0, 0, 255),            # Blue for cards  
            'name': (255, 255, 0),          # Cyan for names
            'time': (0, 255, 255),          # Yellow for timestamps
            'detection': (255, 0, 255),     # Magenta for detection points
            'text': (255, 255, 255),        # White for text
            'error': (0, 0, 255),           # Red for errors
            'success': (0, 255, 0),         # Green for success
        }
        
        # Text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2
    
    def create_base_overlay(self, image_path: str) -> Optional[np.ndarray]:
        """
        Create base image overlay for visualizations
        
        Args:
            image_path: Path to source image
            
        Returns:
            Base image array or None if failed
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            return img.copy()
        except Exception as e:
            print(f"❌ Error creating base overlay: {e}")
            return None
    
    def draw_vertical_line(self, img: np.ndarray, x: int, color: str = 'boundary', 
                          label: str = None, offset: int = 0) -> np.ndarray:
        """
        Draw vertical line with optional label
        
        Args:
            img: Image to draw on
            x: X coordinate for line
            color: Color key from self.colors
            label: Optional text label
            offset: Y offset for label positioning
        """
        if img is None:
            return img
            
        height = img.shape[0]
        color_bgr = self.colors.get(color, self.colors['boundary'])
        
        # Draw vertical line
        cv2.line(img, (x, 0), (x, height), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            y_pos = max(30 + offset, 30)
            cv2.putText(img, label, (x + 5, y_pos), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_horizontal_line(self, img: np.ndarray, y: int, x1: int = 0, 
                           x2: int = None, color: str = 'boundary', 
                           label: str = None) -> np.ndarray:
        """
        Draw horizontal line with optional label
        
        Args:
            img: Image to draw on
            y: Y coordinate for line
            x1: Start X coordinate
            x2: End X coordinate (image width if None)
            color: Color key from self.colors
            label: Optional text label
        """
        if img is None:
            return img
            
        if x2 is None:
            x2 = img.shape[1]
            
        color_bgr = self.colors.get(color, self.colors['boundary'])
        
        # Draw horizontal line
        cv2.line(img, (x1, y), (x2, y), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x1 + 5, y - 5), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_rectangle(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                      color: str = 'card', label: str = None, filled: bool = False) -> np.ndarray:
        """
        Draw rectangle with optional label
        
        Args:
            img: Image to draw on
            x, y: Top-left corner coordinates
            w, h: Width and height
            color: Color key from self.colors
            label: Optional text label
            filled: Whether to fill rectangle
        """
        if img is None:
            return img
            
        color_bgr = self.colors.get(color, self.colors['card'])
        
        # Draw rectangle
        if filled:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, -1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x + 5, y + 20), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_circle(self, img: np.ndarray, x: int, y: int, radius: int = 5,
                   color: str = 'detection', label: str = None) -> np.ndarray:
        """
        Draw circle with optional label
        
        Args:
            img: Image to draw on
            x, y: Center coordinates
            radius: Circle radius
            color: Color key from self.colors
            label: Optional text label
        """
        if img is None:
            return img
            
        color_bgr = self.colors.get(color, self.colors['detection'])
        
        # Draw circle
        cv2.circle(img, (x, y), radius, color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x + radius + 5, y + 5), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def add_info_panel(self, img: np.ndarray, info: Dict[str, Any], 
                      x: int = 10, y: int = 10) -> np.ndarray:
        """
        Add information panel to image
        
        Args:
            img: Image to draw on
            info: Dictionary of information to display
            x, y: Top-left position for panel
        """
        if img is None or not info:
            return img
            
        line_height = 25
        current_y = y
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(img, text, (x, current_y), self.font, 
                       self.font_scale, self.colors['text'], self.font_thickness)
            current_y += line_height
        
        return img
    
    def generate_heatmap(self, data: np.ndarray, width: int, height: int, 
                        title: str = "Heatmap") -> np.ndarray:
        """
        Generate heatmap visualization from 1D or 2D data
        
        Args:
            data: Data array to visualize
            width: Output width
            height: Output height  
            title: Heatmap title
            
        Returns:
            Heatmap image as BGR array
        """
        try:
            if data.ndim == 1:
                # Convert 1D to 2D by repeating rows
                data_2d = np.tile(data, (height // 4, 1))
            else:
                data_2d = data
            
            # Normalize to 0-255 range
            normalized = cv2.normalize(data_2d, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Resize to target dimensions
            if heatmap.shape[:2] != (height, width):
                heatmap = cv2.resize(heatmap, (width, height))
            
            # Add title
            cv2.putText(heatmap, title, (10, 30), self.font, 
                       0.8, self.colors['text'], self.font_thickness)
            
            return heatmap
            
        except Exception as e:
            print(f"❌ Error generating heatmap: {e}")
            # Return blank image on error
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank, f"Heatmap Error: {str(e)[:50]}", (10, height//2), 
                       self.font, 0.6, self.colors['error'], 1)
            return blank
    
    def save_visualization(self, img: np.ndarray, base_path: str, 
                          suffix: str = "visualization", 
                          output_dir: str = "pic/screenshots") -> Optional[str]:
        """
        Save visualization with timestamped filename
        
        Args:
            img: Image to save
            base_path: Base image path for naming
            suffix: Filename suffix
            output_dir: Output directory
            
        Returns:
            Saved filename or None if failed
        """
        try:
            if img is None:
                return None
                
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            filename = f"{timestamp}_{base_name}_{suffix}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"✅ Visualization saved: {filename}")
                return filename
            else:
                print(f"❌ Failed to save visualization: {output_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
            return None
    
    def create_composite_visualization(self, images: List[np.ndarray], 
                                     labels: List[str] = None,
                                     arrangement: str = 'horizontal') -> Optional[np.ndarray]:
        """
        Create composite visualization from multiple images
        
        Args:
            images: List of images to combine
            labels: Optional labels for each image
            arrangement: 'horizontal' or 'vertical'
            
        Returns:
            Composite image or None if failed
        """
        try:
            if not images:
                return None
                
            # Filter out None images
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                return None
            
            if arrangement == 'horizontal':
                composite = np.hstack(valid_images)
            else:  # vertical
                composite = np.vstack(valid_images)
            
            # Add labels if provided
            if labels and len(labels) >= len(valid_images):
                label_height = 30
                for i, label in enumerate(labels[:len(valid_images)]):
                    if arrangement == 'horizontal':
                        x = sum(img.shape[1] for img in valid_images[:i]) + 10
                        y = label_height
                    else:  # vertical
                        x = 10
                        y = sum(img.shape[0] for img in valid_images[:i]) + label_height
                    
                    cv2.putText(composite, label, (x, y), self.font, 
                               0.8, self.colors['text'], self.font_thickness)
            
            return composite
            
        except Exception as e:
            print(f"❌ Error creating composite visualization: {e}")
            return None


# Utility functions for backward compatibility and convenience
def create_width_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for width detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect width using the detector
    detection_result = detector.detect_width(image_path)
    if detection_result is None:
        return None
    
    left_boundary, right_boundary, width = detection_result
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw boundaries
    engine.draw_vertical_line(img, left_boundary, 'boundary', f"L:{left_boundary}")
    engine.draw_vertical_line(img, right_boundary, 'boundary', f"R:{right_boundary}", offset=30)
    
    # Add width info
    info = {'Width': f"{width}px"}
    engine.add_info_panel(img, info, left_boundary, 80)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"SimpleWidth_{width}px")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


def create_avatar_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for avatar detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect avatars using the detector
    avatars, avatar_info = detector.detect_avatars(image_path)
    if not avatars:
        return None
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw avatar detections
    for i, avatar in enumerate(avatars):
        x, y, w, h = avatar['x'], avatar['y'], avatar['width'], avatar['height']
        engine.draw_rectangle(img, x, y, w, h, 'avatar', f"Avatar {i+1}")
        
        # Mark center point
        center_x, center_y = x + w//2, y + h//2
        engine.draw_circle(img, center_x, center_y, 3, 'detection')
    
    # Add info panel
    info = {
        'Avatars': len(avatars),
        'Method': avatar_info.get('method', 'Unknown'),
        'Processing': f"{avatar_info.get('processing_time', 0):.2f}s"
    }
    engine.add_info_panel(img, info)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"Avatars_{len(avatars)}_detected")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


def create_card_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for card detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect cards using the detector
    cards, card_info = detector.detect_cards(image_path)
    if not cards:
        return None
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw card detections
    for i, card in enumerate(cards):
        x, y, w, h = card['x'], card['y'], card['width'], card['height']
        engine.draw_rectangle(img, x, y, w, h, 'card', f"Card {i+1}")
    
    # Add info panel
    info = {
        'Cards': len(cards),
        'Method': card_info.get('method', 'Unknown'),
        'Processing': f"{card_info.get('processing_time', 0):.2f}s"
    }
    engine.add_info_panel(img, info)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"Cards_{len(cards)}_detected")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


if __name__ == "__main__":
    print("🎨 Visualization Engine Module Test")
    print("=" * 40)
    
    # Test visualization engine
    engine = VisualizationEngine()
    print("✅ Visualization Engine initialized")
    
    # Test basic functionality
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw some test elements
    engine.draw_vertical_line(test_img, 150, 'boundary', "Left Boundary")
    engine.draw_vertical_line(test_img, 450, 'boundary', "Right Boundary", offset=30)
    engine.draw_rectangle(test_img, 200, 100, 100, 80, 'card', "Test Card")
    engine.draw_circle(test_img, 250, 140, 5, 'avatar', "Avatar")
    
    info = {'Test': 'Visualization Engine', 'Status': 'Working'}
    engine.add_info_panel(test_img, info, 10, 300)
    
    # Save test image
    test_path = engine.save_visualization(test_img, "test_image.png", "engine_test")
    if test_path:
        print(f"✅ Test visualization created: {test_path}")
    else:
        print("❌ Test visualization failed")