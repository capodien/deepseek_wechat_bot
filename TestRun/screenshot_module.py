#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone WeChat Screenshot Capture Module
Focus: Detect WeChat window + Take precise screenshots

Requirements:
1. Detect WeChat window size and location
2. Record coordinates for reuse
3. Take precise screenshot of WeChat window only
4. Validate screenshot quality
"""

import platform
import pyautogui
import time
import os
from datetime import datetime
from typing import Optional, Tuple, Dict
import json

class WeChatScreenshotCapture:
    """Focused WeChat screenshot capture module"""
    
    def __init__(self, output_dir: str = "TestRun/screenshots"):
        self.system = platform.system()
        self.output_dir = output_dir
        self.window_coords = None
        self.validation_enabled = True
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🖥️  System: {self.system}")
        print(f"📁 Output: {self.output_dir}")
    
    def detect_wechat_window(self, method: str = "native") -> Optional[Tuple[int, int, int, int]]:
        """
        Detect WeChat window location and size
        
        Args:
            method: "native" (OS-specific) or "ocr" (visual detection)
            
        Returns:
            (left, top, width, height) or None if not found
        """
        print(f"\n🔍 Detecting WeChat window using {method} method...")
        
        if method == "native" and self.system == "Darwin":
            coords = self._detect_macos_native()
        elif method == "ocr":
            coords = self._detect_ocr_based()
        else:
            print(f"❌ Method '{method}' not supported on {self.system}")
            return None
        
        if coords:
            self.window_coords = coords
            left, top, width, height = coords
            print(f"✅ WeChat window detected: ({left}, {top}, {width}x{height})")
            
            # Validate reasonable dimensions
            if self._validate_window_dimensions(coords):
                return coords
            else:
                self.window_coords = None
                return None
        else:
            print("❌ WeChat window not found")
            return None
    
    def _validate_window_dimensions(self, coords: Tuple[int, int, int, int]) -> bool:
        """Validate that detected window dimensions are reasonable for WeChat"""
        left, top, width, height = coords
        
        # Reasonable WeChat window constraints (adjusted for larger screens)
        MIN_WIDTH, MAX_WIDTH = 600, 2000  # Allow larger widths for high-res screens
        MIN_HEIGHT, MAX_HEIGHT = 400, 1200
        MIN_ASPECT, MAX_ASPECT = 0.8, 4.0  # Allow wider aspect ratios
        
        # Width validation
        if width < MIN_WIDTH or width > MAX_WIDTH:
            return False
        
        # Height validation  
        if height < MIN_HEIGHT or height > MAX_HEIGHT:
            return False
        
        # Aspect ratio validation
        aspect_ratio = width / height
        if aspect_ratio < MIN_ASPECT or aspect_ratio > MAX_ASPECT:
            return False
        
        return True
    
    def _detect_macos_native(self) -> Optional[Tuple[int, int, int, int]]:
        """Native macOS window detection using Quartz"""
        try:
            import Quartz
            
            # Get all visible windows
            window_list = Quartz.CGWindowListCopyWindowInfo(
                Quartz.kCGWindowListOptionOnScreenOnly,
                Quartz.kCGNullWindowID
            )
            
            print(f"📋 Scanning {len(window_list)} visible windows...")
            
            wechat_windows = []
            for window in window_list:
                owner_name = window.get('kCGWindowOwnerName', '')
                window_name = window.get('kCGWindowName', '')
                
                # Look for WeChat application
                if owner_name == 'WeChat' or 'WeChat' in window_name:
                    bounds = window.get('kCGWindowBounds', {})
                    if bounds:
                        coords = (
                            int(bounds['X']), 
                            int(bounds['Y']), 
                            int(bounds['Width']), 
                            int(bounds['Height'])
                        )
                        wechat_windows.append({
                            'coords': coords,
                            'name': window_name,
                            'owner': owner_name,
                            'layer': window.get('kCGWindowLayer', 0)
                        })
                        print(f"  📱 Found: {owner_name} - {window_name} - {coords}")
            
            if not wechat_windows:
                print("❌ No WeChat windows found")
                return None
            
            # Filter out unreasonably sized windows first
            valid_windows = []
            for window in wechat_windows:
                coords = window['coords']
                if self._validate_window_dimensions(coords):
                    valid_windows.append(window)
                    print(f"  ✅ Valid window: {window['name']} - {coords}")
                else:
                    print(f"  ❌ Invalid window: {window['name']} - {coords}")
            
            if not valid_windows:
                print("❌ No valid WeChat windows found")
                return None
            
            # Return the largest valid WeChat window
            main_window = max(valid_windows, key=lambda w: w['coords'][2] * w['coords'][3])
            print(f"🎯 Selected main window: {main_window['name']} - {main_window['coords']}")
            return main_window['coords']
            
        except ImportError:
            print("❌ Quartz not available, install pyobjc: pip install pyobjc")
            return None
        except Exception as e:
            print(f"❌ Native detection failed: {e}")
            return None
    
    def _detect_ocr_based(self) -> Optional[Tuple[int, int, int, int]]:
        """OCR-based detection as fallback"""
        try:
            import easyocr
            
            print("📖 Taking full screen screenshot for OCR analysis...")
            screenshot = pyautogui.screenshot()
            temp_path = os.path.join(self.output_dir, "temp_fullscreen.png")
            screenshot.save(temp_path)
            
            print("🔍 Analyzing screenshot with OCR...")
            reader = easyocr.Reader(['ch_sim', 'en'], gpu=False, verbose=False)
            results = reader.readtext(temp_path, width_ths=0.7, height_ths=0.7)
            
            # Look for WeChat-specific elements
            wechat_indicators = [
                '微信', 'WeChat', '文件传输助手', 'File Transfer',
                '聊天', 'Chat', '通讯录', 'Contacts'
            ]
            
            wechat_elements = []
            for (bbox, text, confidence) in results:
                if confidence > 0.7:  # High confidence only
                    for indicator in wechat_indicators:
                        if indicator in text.strip():
                            x_coords = [point[0] for point in bbox]
                            y_coords = [point[1] for point in bbox]
                            element_rect = (
                                min(x_coords), min(y_coords),
                                max(x_coords) - min(x_coords),
                                max(y_coords) - min(y_coords)
                            )
                            wechat_elements.append({
                                'text': text.strip(),
                                'rect': element_rect,
                                'confidence': confidence
                            })
                            print(f"  ✅ Found WeChat element: '{text.strip()}' (confidence: {confidence:.2f})")
                            break
            
            # Cleanup temp file
            try:
                os.remove(temp_path)
            except:
                pass
            
            if wechat_elements:
                return self._estimate_window_from_elements(wechat_elements, screenshot.size)
            else:
                print("❌ No WeChat elements found in screenshot")
                return None
                
        except ImportError:
            print("❌ EasyOCR not available, install: pip install easyocr")
            return None
        except Exception as e:
            print(f"❌ OCR detection failed: {e}")
            return None
    
    def _estimate_window_from_elements(self, elements, screen_size) -> Optional[Tuple[int, int, int, int]]:
        """Estimate WeChat window bounds from detected UI elements"""
        if not elements:
            return None
        
        # Get bounding box of all elements
        all_x = []
        all_y = []
        for element in elements:
            x, y, w, h = element['rect']
            all_x.extend([x, x + w])
            all_y.extend([y, y + h])
        
        # Add reasonable padding for window chrome
        padding = 50
        title_height = 30
        
        left = max(0, min(all_x) - padding)
        top = max(0, min(all_y) - title_height)
        right = min(screen_size[0], max(all_x) + padding)
        bottom = min(screen_size[1], max(all_y) + padding)
        
        width = right - left
        height = bottom - top
        
        return (left, top, width, height)
    
    def save_window_coordinates(self, filename: str = "wechat_window_coords.json") -> bool:
        """Save detected window coordinates to file"""
        if not self.window_coords:
            print("❌ No window coordinates to save")
            return False
        
        try:
            coord_data = {
                'coordinates': self.window_coords,
                'timestamp': datetime.now().isoformat(),
                'system': self.system,
                'validated': True
            }
            
            filepath = os.path.join(self.output_dir, filename)
            with open(filepath, 'w') as f:
                json.dump(coord_data, f, indent=2)
            
            print(f"💾 Coordinates saved to: {filepath}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to save coordinates: {e}")
            return False
    
    def load_window_coordinates(self, filename: str = "wechat_window_coords.json") -> bool:
        """Load window coordinates from file"""
        try:
            filepath = os.path.join(self.output_dir, filename)
            if not os.path.exists(filepath):
                print(f"⚠️  No saved coordinates found at: {filepath}")
                return False
            
            with open(filepath, 'r') as f:
                coord_data = json.load(f)
            
            # Check if coordinates are recent (within 1 hour)
            saved_time = datetime.fromisoformat(coord_data['timestamp'])
            age_hours = (datetime.now() - saved_time).total_seconds() / 3600
            
            if age_hours > 1:
                print(f"⚠️  Saved coordinates are {age_hours:.1f} hours old, detection recommended")
                return False
            
            self.window_coords = tuple(coord_data['coordinates'])
            print(f"📂 Loaded coordinates: {self.window_coords}")
            return True
            
        except Exception as e:
            print(f"❌ Failed to load coordinates: {e}")
            return False
    
    def capture_screenshot(self, save_as: str = None) -> Optional[str]:
        """
        Capture precise screenshot of WeChat window
        
        Args:
            save_as: Custom filename, auto-generated if None
            
        Returns:
            Path to saved screenshot or None if failed
        """
        if not self.window_coords:
            print("❌ No window coordinates available. Run detect_wechat_window() first.")
            return None
        
        try:
            left, top, width, height = self.window_coords
            
            print(f"📸 Capturing screenshot: ({left}, {top}, {width}, {height})")
            
            # Take screenshot of specific region
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Generate filename
            if not save_as:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_as = f"wechat_window_{timestamp}.png"
            
            filepath = os.path.join(self.output_dir, save_as)
            screenshot.save(filepath)
            
            print(f"✅ Screenshot saved: {filepath}")
            
            # Validate screenshot if enabled
            if self.validation_enabled:
                if self._validate_screenshot(filepath):
                    return filepath
                else:
                    print("❌ Screenshot validation failed")
                    return None
            
            return filepath
            
        except Exception as e:
            print(f"❌ Screenshot capture failed: {e}")
            return None
    
    def _validate_screenshot(self, filepath: str) -> bool:
        """Validate that screenshot contains WeChat content"""
        try:
            from PIL import Image
            
            # Basic validation - check if image was created and has reasonable size
            if not os.path.exists(filepath):
                return False
            
            with Image.open(filepath) as img:
                width, height = img.size
                file_size = os.path.getsize(filepath)
                
                print(f"📊 Screenshot validation: {width}x{height}px, {file_size/1024:.1f}KB")
                
                # Basic sanity checks
                if width < 500 or height < 300:
                    print("❌ Screenshot too small")
                    return False
                
                if file_size < 10000:  # Less than 10KB probably empty/black
                    print("❌ Screenshot file too small")
                    return False
                
                print("✅ Screenshot validation passed")
                return True
                
        except Exception as e:
            print(f"❌ Screenshot validation error: {e}")
            return False
    
    def test_capture_sequence(self) -> bool:
        """Run complete test sequence"""
        print("\n" + "="*50)
        print("🧪 WECHAT SCREENSHOT CAPTURE TEST")
        print("="*50)
        
        # Step 1: Try to load existing coordinates
        print("\n1. Loading saved coordinates...")
        if not self.load_window_coordinates():
            print("   No valid saved coordinates, will detect fresh")
        
        # Step 2: Detect window (if needed)
        if not self.window_coords:
            print("\n2. Detecting WeChat window...")
            # Try native first, fallback to OCR
            if not self.detect_wechat_window("native"):
                print("   Native detection failed, trying OCR...")
                if not self.detect_wechat_window("ocr"):
                    print("❌ All detection methods failed")
                    return False
        else:
            print("\n2. Using loaded coordinates")
        
        # Step 3: Save coordinates
        print("\n3. Saving window coordinates...")
        self.save_window_coordinates()
        
        # Step 4: Take test screenshot
        print("\n4. Capturing test screenshot...")
        screenshot_path = self.capture_screenshot("test_capture.png")
        
        if screenshot_path:
            print(f"\n✅ TEST PASSED")
            print(f"Screenshot saved: {screenshot_path}")
            print(f"Window coordinates: {self.window_coords}")
            return True
        else:
            print("\n❌ TEST FAILED")
            return False


def main():
    """Test the screenshot capture module"""
    print("🚀 WeChat Screenshot Capture Module Test")
    
    # Create capture instance
    capturer = WeChatScreenshotCapture()
    
    # Run complete test
    success = capturer.test_capture_sequence()
    
    if success:
        print(f"\n🎉 Module ready for integration!")
        print(f"📁 Check {capturer.output_dir} for screenshots")
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"   • Make sure WeChat desktop app is running and visible")
        print(f"   • Ensure WeChat window is not minimized or hidden")
        print(f"   • Check that screen recording permissions are granted")


if __name__ == "__main__":
    main()