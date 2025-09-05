#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detection Region Visualizer
Creates annotated screenshots showing exactly what regions are being searched
"""

import cv2
import numpy as np
import os
from datetime import datetime

class RegionVisualizer:
    """Visualize detection regions on screenshots for debugging"""
    
    def __init__(self):
        # Current detection regions from message_detection_module.py
        self.contact_region = (60, 100, 320, 800)  # Contact list area
        self.red_dot_region = (60, 100, 380, 800)  # Red dot search area
        
        # Contact row parameters
        self.contact_row_height = 75
        self.click_offset_x = 200
        
        print("🎨 Region Visualizer initialized")
    
    def create_annotated_screenshot(self, input_path: str, output_path: str) -> bool:
        """Create annotated screenshot showing all detection regions"""
        try:
            # Load original screenshot
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ Failed to load image: {input_path}")
                return False
            
            # Create annotation overlay
            overlay = image.copy()
            
            # Draw contact region (blue rectangle)
            x, y, w, h = self.contact_region
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue
            cv2.putText(overlay, f'Contact Region {w}x{h}', (x, y - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            # Draw red dot search region (red rectangle)
            x2, y2, w2, h2 = self.red_dot_region
            cv2.rectangle(overlay, (x2, y2), (x2 + w2, y2 + h2), (0, 0, 255), 2)  # Red
            cv2.putText(overlay, f'Red Dot Search {w2}x{h2}', (x2 + 10, y2 + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw contact row lines (green)
            for row in range(0, h, self.contact_row_height):
                row_y = y + row
                if row_y < y + h:
                    cv2.line(overlay, (x, row_y), (x + w, row_y), (0, 255, 0), 1)  # Green
                    
                    # Show click position for this row
                    click_x = x + self.click_offset_x
                    click_y = row_y + (self.contact_row_height // 2)
                    cv2.circle(overlay, (click_x, click_y), 3, (0, 255, 255), -1)  # Yellow dot
            
            # Add legend
            legend_y = 30
            cv2.putText(overlay, 'Detection Regions:', (10, legend_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(overlay, '• Blue: Contact List Area', (10, legend_y + 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(overlay, '• Red: Red Dot Search Area', (10, legend_y + 45), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            cv2.putText(overlay, '• Green: Contact Row Boundaries', (10, legend_y + 65), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(overlay, '• Yellow: Click Target Centers', (10, legend_y + 85), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            # Blend overlay with original image
            alpha = 0.7
            annotated = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Save annotated image
            success = cv2.imwrite(output_path, annotated)
            
            if success:
                print(f"✅ Annotated screenshot saved: {output_path}")
                return True
            else:
                print(f"❌ Failed to save annotated screenshot")
                return False
                
        except Exception as e:
            print(f"❌ Annotation error: {e}")
            return False
    
    def analyze_red_dot_locations(self, input_path: str) -> list:
        """Analyze where red dots are actually located"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                return []
            
            # Red dot colors to search for
            red_dot_colors = {
                'primary': np.array([84, 98, 227]),
                'windows': np.array([81, 81, 255]),
                'fallback': np.array([80, 100, 230])
            }
            color_tolerance = 15
            
            found_dots = []
            
            for color_name, target_color in red_dot_colors.items():
                # Color matching
                lower_bound = target_color - color_tolerance
                upper_bound = target_color + color_tolerance
                color_mask = np.all((lower_bound <= image) & (image <= upper_bound), axis=-1)
                
                # Find all matching pixels
                y_coords, x_coords = np.where(color_mask)
                
                if len(x_coords) > 0:
                    # Group nearby pixels into clusters
                    for x, y in zip(x_coords, y_coords):
                        found_dots.append({
                            'color': color_name,
                            'x': int(x),
                            'y': int(y),
                            'in_search_region': self._is_in_search_region(int(x), int(y))
                        })
            
            return found_dots
            
        except Exception as e:
            print(f"❌ Red dot analysis error: {e}")
            return []
    
    def _is_in_search_region(self, x: int, y: int) -> bool:
        """Check if coordinates are within current search region"""
        x_start, y_start, width, height = self.red_dot_region
        return (x_start <= x <= x_start + width and 
                y_start <= y <= y_start + height)
    
    def suggest_optimal_regions(self, input_path: str) -> dict:
        """Analyze screenshot to suggest optimal detection regions"""
        try:
            image = cv2.imread(input_path)
            if image is None:
                return {}
            
            height, width = image.shape[:2]
            
            # Analyze image to detect WeChat interface elements
            # This is a simplified version - in practice you'd use more sophisticated detection
            
            suggestions = {
                'current_contact_region': self.contact_region,
                'current_red_dot_region': self.red_dot_region,
                'image_dimensions': (width, height),
                'suggested_contact_region': None,
                'suggested_red_dot_region': None,
                'analysis': []
            }
            
            # Basic analysis
            if width > 800:
                # Wider window - adjust regions
                new_contact_w = min(400, width // 3)
                suggestions['suggested_contact_region'] = (80, 120, new_contact_w, height - 200)
                suggestions['suggested_red_dot_region'] = (80, 120, new_contact_w + 50, height - 200)
                suggestions['analysis'].append("Detected wide window - suggested wider contact regions")
            
            if height > 1200:
                # Tall window - adjust regions
                suggestions['analysis'].append("Detected tall window - contact regions may need height adjustment")
            
            return suggestions
            
        except Exception as e:
            print(f"❌ Region analysis error: {e}")
            return {}


def create_region_visualization(screenshot_path: str) -> str:
    """Create annotated visualization of detection regions"""
    try:
        visualizer = RegionVisualizer()
        
        # Create output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"region_analysis_{timestamp}.png"
        output_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create annotated screenshot
        success = visualizer.create_annotated_screenshot(screenshot_path, output_path)
        
        if success:
            # Analyze red dot locations
            red_dots = visualizer.analyze_red_dot_locations(screenshot_path)
            print(f"\n📍 Red dot analysis:")
            for dot in red_dots[:10]:  # Show first 10
                status = "✅ IN" if dot['in_search_region'] else "❌ OUT"
                print(f"  {status} region: ({dot['x']}, {dot['y']}) - {dot['color']}")
            
            if len(red_dots) > 10:
                print(f"  ... and {len(red_dots) - 10} more")
            
            # Get region suggestions
            suggestions = visualizer.suggest_optimal_regions(screenshot_path)
            if suggestions.get('analysis'):
                print(f"\n💡 Suggestions:")
                for suggestion in suggestions['analysis']:
                    print(f"  • {suggestion}")
            
            return output_path
        else:
            return None
            
    except Exception as e:
        print(f"❌ Visualization creation error: {e}")
        return None


if __name__ == '__main__':
    # Find latest screenshot for testing
    screenshot_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest_screenshot = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"🎨 Creating region visualization for: {latest_screenshot}")
        result_path = create_region_visualization(screenshot_path)
        
        if result_path:
            print(f"✅ Region visualization created: {os.path.basename(result_path)}")
        else:
            print("❌ Failed to create visualization")
    else:
        print("❌ No test screenshots found")