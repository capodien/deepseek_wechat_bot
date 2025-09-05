#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Contact Card Visualizer
Creates annotated screenshots showing individual contact cards with their boundaries and details
"""

import cv2
import numpy as np
import os
import sys
from datetime import datetime

# Add current directory to Python path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from TestRun.improved_contact_card_analyzer import ImprovedContactCardAnalyzer

class ContactCardVisualizer:
    """Visualize individual contact cards on screenshots for debugging"""
    
    def __init__(self):
        self.analyzer = ImprovedContactCardAnalyzer()
        print("🎨 Contact Card Visualizer initialized")
    
    def create_contact_card_visualization(self, input_path: str, output_path: str) -> bool:
        """Create annotated screenshot showing individual contact cards"""
        try:
            print(f"🎨 Creating contact card visualization from: {os.path.basename(input_path)}")
            
            # Load original screenshot
            image = cv2.imread(input_path)
            if image is None:
                print(f"❌ Failed to load image: {input_path}")
                return False
            
            # Analyze contact cards
            contact_cards = self.analyzer.analyze_contact_list(input_path)
            
            if not contact_cards:
                print("❌ No contact cards found to visualize")
                return False
            
            print(f"📋 Visualizing {len(contact_cards)} contact cards")
            
            # Create visualization overlay
            overlay = image.copy()
            
            # Colors for different elements
            CARD_COLOR = (0, 255, 0)        # Green for card boundaries
            RED_DOT_COLOR = (0, 0, 255)     # Red for cards with red dots
            MESSAGE_COLOR = (255, 0, 0)     # Blue for cards with messages
            CLICK_COLOR = (0, 255, 255)     # Yellow for click positions
            TEXT_COLOR = (255, 255, 255)    # White for text
            
            # Draw each contact card
            for i, card in enumerate(contact_cards):
                # Choose card color based on status
                if card.has_red_dot:
                    card_color = RED_DOT_COLOR
                    card_thickness = 3
                elif card.has_message:
                    card_color = MESSAGE_COLOR  
                    card_thickness = 2
                else:
                    card_color = CARD_COLOR
                    card_thickness = 1
                
                # Draw card boundary
                cv2.rectangle(overlay, 
                            (card.x, card.y), 
                            (card.x + card.width, card.y + card.height), 
                            card_color, card_thickness)
                
                # Draw card index number
                cv2.putText(overlay, f'{i+1}', 
                          (card.x + 5, card.y + 20), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 2)
                
                # Draw contact name if available
                if card.contact_name and card.contact_name != "Unknown":
                    # Truncate long names
                    display_name = card.contact_name[:15] + "..." if len(card.contact_name) > 15 else card.contact_name
                    cv2.putText(overlay, display_name, 
                              (card.x + 25, card.y + 20), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, TEXT_COLOR, 1)
                
                # Draw click position
                if card.click_center:
                    cx, cy = card.click_center
                    # Convert to absolute coordinates if needed
                    if cx < card.x or cx > card.x + card.width:
                        cx = card.x + card.width // 2
                    if cy < card.y or cy > card.y + card.height:
                        cy = card.y + card.height // 2
                    
                    cv2.circle(overlay, (cx, cy), 4, CLICK_COLOR, -1)
                    cv2.circle(overlay, (cx, cy), 6, (0, 0, 0), 1)
                
                # Draw avatar and component regions within card
                self._draw_card_components(overlay, card)
            
            # Add legend
            self._draw_legend(overlay, contact_cards)
            
            # Add summary information
            self._draw_summary(overlay, contact_cards)
            
            # Blend overlay with original image
            alpha = 0.8
            annotated = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
            
            # Save annotated image
            success = cv2.imwrite(output_path, annotated)
            
            if success:
                print(f"✅ Contact card visualization saved: {output_path}")
                return True
            else:
                print(f"❌ Failed to save contact card visualization")
                return False
                
        except Exception as e:
            print(f"❌ Contact card visualization error: {e}")
            return False
    
    def _draw_card_components(self, overlay, card):
        """Draw internal components of a contact card"""
        try:
            # Draw avatar region (light blue) - now using avatar_bounds from improved analyzer
            if hasattr(card, 'avatar_bounds') and card.avatar_bounds:
                ax, ay, aw, ah = card.avatar_bounds
                cv2.rectangle(overlay, (ax, ay), (ax + aw, ay + ah), 
                            (255, 200, 100), 2)
                cv2.putText(overlay, 'Avatar', (ax + 2, ay - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 200, 100), 1)
            
            # Draw name region if available
            if hasattr(card, 'name_region') and card.name_region:
                nx, ny, nw, nh = card.name_region
                cv2.rectangle(overlay, (nx, ny), (nx + nw, ny + nh), 
                            (100, 255, 100), 1)
                cv2.putText(overlay, 'Name', (nx + 2, ny + 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 255, 100), 1)
            
            # Draw message region if available
            if hasattr(card, 'message_region') and card.message_region:
                mx, my, mw, mh = card.message_region
                cv2.rectangle(overlay, (mx, my), (mx + mw, my + mh), 
                            (100, 100, 255), 1)
                cv2.putText(overlay, 'Message', (mx + 2, my + 10), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (100, 100, 255), 1)
            
            # Draw red dot indicator if present
            if card.has_red_dot and hasattr(card, 'avatar_bounds') and card.avatar_bounds:
                ax, ay, aw, ah = card.avatar_bounds
                # Red dot is typically at top-right of avatar
                dot_x = ax + aw - 5
                dot_y = ay + 5
                cv2.circle(overlay, (dot_x, dot_y), 5, (0, 0, 255), -1)
                cv2.circle(overlay, (dot_x, dot_y), 6, (0, 0, 200), 1)
        
        except Exception as e:
            print(f"⚠️  Component drawing error: {e}")
    
    def _draw_legend(self, overlay, contact_cards):
        """Draw legend explaining the visualization"""
        try:
            legend_x = 10
            legend_y = 30
            line_height = 25
            
            # Background for legend
            cv2.rectangle(overlay, (legend_x - 5, legend_y - 20), 
                         (legend_x + 400, legend_y + line_height * 6), 
                         (0, 0, 0), -1)
            cv2.rectangle(overlay, (legend_x - 5, legend_y - 20), 
                         (legend_x + 400, legend_y + line_height * 6), 
                         (255, 255, 255), 1)
            
            # Title
            cv2.putText(overlay, 'Contact Card Visualization', 
                       (legend_x, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Legend items
            legend_items = [
                ('Red Border: Cards with red dot notifications', (0, 0, 255)),
                ('Blue Border: Cards with messages', (255, 0, 0)),
                ('Green Border: Regular contact cards', (0, 255, 0)),
                ('Yellow Dot: Optimal click position', (0, 255, 255)),
                ('Orange Box: Avatar detection', (255, 200, 100))
            ]
            
            for i, (text, color) in enumerate(legend_items):
                y_pos = legend_y + (i + 1) * line_height
                cv2.putText(overlay, text, (legend_x, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        except Exception as e:
            print(f"⚠️  Legend drawing error: {e}")
    
    def _draw_summary(self, overlay, contact_cards):
        """Draw summary statistics"""
        try:
            height, width = overlay.shape[:2]
            summary_x = width - 300
            summary_y = 30
            line_height = 20
            
            # Count different types of cards
            cards_with_red_dots = len([c for c in contact_cards if c.has_red_dot])
            cards_with_messages = len([c for c in contact_cards if c.has_message])
            
            # Background for summary
            cv2.rectangle(overlay, (summary_x - 5, summary_y - 15), 
                         (summary_x + 290, summary_y + line_height * 5), 
                         (0, 0, 0), -1)
            cv2.rectangle(overlay, (summary_x - 5, summary_y - 15), 
                         (summary_x + 290, summary_y + line_height * 5), 
                         (255, 255, 255), 1)
            
            # Summary items
            summary_items = [
                f'Total Contact Cards: {len(contact_cards)}',
                f'Cards with Red Dots: {cards_with_red_dots}',
                f'Cards with Messages: {cards_with_messages}',
                f'Regular Cards: {len(contact_cards) - cards_with_red_dots - cards_with_messages}'
            ]
            
            for i, text in enumerate(summary_items):
                y_pos = summary_y + i * line_height
                cv2.putText(overlay, text, (summary_x, y_pos), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        except Exception as e:
            print(f"⚠️  Summary drawing error: {e}")


def create_contact_card_visualization(screenshot_path: str) -> str:
    """Create contact card visualization of detection regions"""
    try:
        visualizer = ContactCardVisualizer()
        
        # Create output path
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_filename = f"contact_cards_{timestamp}.png"
        output_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
        output_path = os.path.join(output_dir, output_filename)
        
        # Create contact card visualization
        success = visualizer.create_contact_card_visualization(screenshot_path, output_path)
        
        if success:
            return output_path
        else:
            return None
            
    except Exception as e:
        print(f"❌ Contact card visualization creation error: {e}")
        return None


if __name__ == '__main__':
    # Find latest screenshot for testing
    screenshot_dir = "/Users/erliz/Library/CloudStorage/GoogleDrive-capodien@gmail.com/My Drive/Workspace/Coding/deepseek_wechat_bot/pic/screenshots"
    screenshots = [f for f in os.listdir(screenshot_dir) if f.startswith('diagnostic_test_') and f.endswith('.png')]
    
    if screenshots:
        latest_screenshot = sorted(screenshots)[-1]
        screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"🎨 Creating contact card visualization for: {latest_screenshot}")
        result_path = create_contact_card_visualization(screenshot_path)
        
        if result_path:
            print(f"✅ Contact card visualization created: {os.path.basename(result_path)}")
        else:
            print("❌ Failed to create contact card visualization")
    else:
        print("❌ No test screenshots found")