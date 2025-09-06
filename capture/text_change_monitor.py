#!/usr/bin/env python3
"""
Text Change Monitor - Alternative to red dot detection
Monitors contact list area for text changes indicating new messages
"""

import cv2
import numpy as np
import easyocr
import os
from datetime import datetime
import hashlib

class cTextChangeMonitor:
    def __init__(self):
        """Initialize the text change monitor"""
        self.ocr_reader = easyocr.Reader(['ch_sim', 'en'], gpu=True)
        self.previous_text_hash = None
        self.contact_region = (60, 100, 320, 800)  # (x, y, width, height) for contact list area
        
    def get_contact_list_text(self, screenshot_path):
        """Extract text from the contact list region"""
        try:
            # Load the screenshot
            image = cv2.imread(screenshot_path)
            if image is None:
                print(f"❌ Failed to load screenshot: {screenshot_path}")
                return None
                
            # Crop to contact list region
            x, y, w, h = self.contact_region
            cropped = image[y:y+h, x:x+w]
            
            # Use OCR to extract text
            results = self.ocr_reader.readtext(cropped)
            
            # Extract just the text content
            text_content = []
            for result in results:
                text = result[1].strip()
                if text:
                    text_content.append(text)
            
            return text_content
            
        except Exception as e:
            print(f"❌ Error extracting text: {str(e)}")
            return None
    
    def detect_text_changes(self, screenshot_path):
        """
        Detect if text has changed in the contact list area
        Returns: (has_changed, rio_old_position) where position is (x,y) if found
        """
        try:
            text_content = self.get_contact_list_text(screenshot_path)
            
            if text_content is None:
                return False, None
            
            # Create hash of current text content
            text_string = '|'.join(text_content)
            current_hash = hashlib.md5(text_string.encode()).hexdigest()
            
            # Check if text has changed
            has_changed = (self.previous_text_hash is not None and 
                          current_hash != self.previous_text_hash)
            
            # Update the stored hash
            self.previous_text_hash = current_hash
            
            # Look for Rio_Old in the text and estimate click position
            rio_old_position = self.find_rio_old_position(screenshot_path, text_content)
            
            if has_changed and rio_old_position:
                print(f"📝 Text change detected! Contact list text changed.")
                print(f"🎯 Rio_Old found at estimated position: {rio_old_position}")
                print(f"📋 Current text content: {text_content}")
                return True, rio_old_position
            elif has_changed:
                print(f"📝 Text change detected but Rio_Old not found in list")
                print(f"📋 Current text content: {text_content}")
                return False, None
            
            return False, None
            
        except Exception as e:
            print(f"❌ Error in text change detection: {str(e)}")
            return False, None
    
    def find_rio_old_position(self, screenshot_path, text_content):
        """Find the approximate click position for Rio_Old contact"""
        try:
            # Load the screenshot
            image = cv2.imread(screenshot_path)
            if image is None:
                return None
                
            # Crop to contact list region  
            x, y, w, h = self.contact_region
            cropped = image[y:y+h, x:x+w]
            
            # Use OCR with bounding boxes to find Rio_Old position
            results = self.ocr_reader.readtext(cropped)
            
            for result in results:
                text = result[1].strip()
                if 'Rio_Old' in text or 'Rio' in text:
                    # Get bounding box coordinates
                    bbox = result[0]
                    
                    # Calculate center position
                    center_x = int((bbox[0][0] + bbox[2][0]) / 2) + x  # Add offset back
                    center_y = int((bbox[0][1] + bbox[2][1]) / 2) + y  # Add offset back
                    
                    return (center_x, center_y)
            
            # If exact match not found, estimate based on text list position
            for i, text in enumerate(text_content):
                if 'Rio' in text:
                    # Estimate position based on list order (roughly 75px per contact)
                    estimated_x = x + w//2  # Middle of contact area
                    estimated_y = y + 150 + (i * 75)  # Start position + spacing
                    return (estimated_x, estimated_y)
            
            return None
            
        except Exception as e:
            print(f"❌ Error finding Rio_Old position: {str(e)}")
            return None

# Global monitor instance
text_monitor = cTextChangeMonitor()

def fdetect_new_message_by_text_change(screenshot_path):
    """
    Alternative detection method using text changes in contact list
    Returns: (x, y) if new message detected for Rio_Old, (None, None) otherwise
    """
    global text_monitor
    
    has_changed, position = text_monitor.detect_text_changes(screenshot_path)
    
    if has_changed and position:
        return position
    else:
        return (None, None)

# Test function
if __name__ == "__main__":
    import glob
    
    # Test on recent screenshots
    screenshots = glob.glob("../pic/screenshots/wechat_*.png")
    if screenshots:
        latest = max(screenshots, key=os.path.getmtime)
        print(f"Testing on: {latest}")
        
        monitor = TextChangeMonitor()
        text_content = monitor.get_contact_list_text(latest)
        print(f"Contact list text: {text_content}")
        
        position = monitor.find_rio_old_position(latest, text_content if text_content else [])
        print(f"Rio_Old position: {position}")