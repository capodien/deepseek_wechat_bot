#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 Consolidated Screenshot Processing Module (screenshot_processor.py)
=====================================================================

📌 Version Info
- Version:        v2.0.1 - Fixed class structure
- Created:        2024-09-04
- Last Modified:  2025-01-07
- Author:         AI Assistant (WeChat Bot System)

📌 Module Position
- Pipeline Phase:  Phase 1.0: Screenshot Capture & Processing
- Previous Phase:  None (Entry point)
- Next Phase:      Phase 2.0 - New Message Detection

📌 Main Functions
- fcapture_screenshot() - Unified screenshot capture (modern + legacy API)
- fcapture_and_process_screenshot() - Live capture with analysis
- fprocess_screenshot_file() - Process existing screenshot files
- fprocess_current_wechat_window() - Convenience wrapper
- fget_live_card_analysis() - Comprehensive analysis with visualizations
"""

import platform
import pyautogui
import time
import os
from datetime import datetime
from typing import Optional, Tuple, Dict
import json

# Set availability flag
SCREENSHOT_AVAILABLE = True


# ============================================================================
# CORE SCREENSHOT CAPTURE CLASS
# ============================================================================

class cWeChatScreenshotCapture:
    """
    WeChat window screenshot capture with single-screenshot architecture and intelligent caching.
    
    Provides comprehensive screenshot capture functionality for WeChat desktop application
    with cross-platform window detection, quality validation, and performance optimization.
    """
    
    def __init__(self, output_dir: str = "pic/screenshots"):
        """Initialize the screenshot capture module with single-screenshot architecture"""
        self.system = platform.system()
        self.output_dir = output_dir
        self.window_coords = None
        self.validation_enabled = True
        
        # Screenshot caching system (Single Screenshot Architecture)
        self._cached_screenshot_pil = None      # PIL Image object cache
        self._cached_screenshot_path = None     # File path cache  
        self._cached_timestamp = None           # When screenshot was taken
        self._cache_expiry_seconds = 300        # Cache validity (5 minutes)
        self._processing_session_id = None      # Session tracking
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        print(f"🖥️  System: {self.system}")
        print(f"📁 Output: {self.output_dir}")
    
    def detect_wechat_window(self, method: str = "native") -> Optional[Tuple[int, int, int, int]]:
        """Detect WeChat window location and size using cross-platform detection methods"""
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
        
        # Reasonable WeChat window constraints
        MIN_WIDTH, MAX_WIDTH = 600, 2000
        MIN_HEIGHT, MAX_HEIGHT = 400, 2000
        MIN_ASPECT, MAX_ASPECT = 0.3, 4.0
        
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
            
            # Filter out unreasonably sized windows
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
            
            # Prefer actual WeChat windows over browser windows
            actual_wechat_windows = []
            for window in valid_windows:
                owner = window['owner']
                name = window['name']
                
                if owner == 'WeChat' and ('Weixin' in name or name == '' or 'WeChat' in name):
                    if 'Diagnostic' not in name and 'Step-by-Step' not in name and 'Bot' not in name:
                        actual_wechat_windows.append(window)
            
            # Select the best window
            if actual_wechat_windows:
                main_window = max(actual_wechat_windows, key=lambda w: w['coords'][2] * w['coords'][3])
                print(f"🎯 Selected WeChat app window: {main_window['name']} - {main_window['coords']}")
                return main_window['coords']
            else:
                main_window = max(valid_windows, key=lambda w: w['coords'][2] * w['coords'][3])
                print(f"🎯 Selected fallback window: {main_window['name']} - {main_window['coords']}")
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
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            temp_path = os.path.join(self.output_dir, f"{timestamp}_OCR_fullscreen.png")
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
                if confidence > 0.7:
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
    
    def capture_screenshot(self, save_as: str = None) -> Optional[str]:
        """Capture precise screenshot of WeChat window"""
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
                save_as = f"{timestamp}_WeChat.png"
            
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
        
        # Detect window
        print("\n1. Detecting WeChat window...")
        if not self.detect_wechat_window("native"):
            print("   Native detection failed, trying OCR...")
            if not self.detect_wechat_window("ocr"):
                print("❌ All detection methods failed")
                return False
        
        # Take test screenshot
        print("\n2. Capturing test screenshot...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        screenshot_path = self.capture_screenshot(f"{timestamp}_test.png")
        
        if screenshot_path:
            print(f"\n✅ TEST PASSED")
            print(f"Screenshot saved: {screenshot_path}")
            print(f"Window coordinates: {self.window_coords}")
            return True
        else:
            print("\n❌ TEST FAILED")
            return False


# ============================================================================
# UNIFIED API INTERFACE
# ============================================================================

# Global instance for efficient reuse
_global_capturer = None

def fget_capturer():
    """Get global capturer instance for efficient reuse"""
    global _global_capturer
    if _global_capturer is None:
        _global_capturer = cWeChatScreenshotCapture()
    return _global_capturer

def fcapture_screenshot(output_dir: str = "pic/screenshots", 
                      filename_screenshot: str = None,
                      detect_window: bool = True) -> Optional[str]:
    """Unified WeChat screenshot capture function"""
    try:
        capturer = fget_capturer()
        capturer.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect window if needed
        if detect_window and not capturer.window_coords:
            coords = capturer.detect_wechat_window()
            if not coords:
                print("❌ Failed to detect WeChat window")
                return None
        
        # Generate filename
        if not filename_screenshot:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename_screenshot = f"{timestamp}_WeChat.png"
        
        # Capture screenshot
        screenshot_path = capturer.capture_screenshot(filename_screenshot)
        
        if screenshot_path:
            print(f"✅ Screenshot captured: {screenshot_path}")
            return screenshot_path
        else:
            print("❌ Screenshot capture failed")
            return None
            
    except Exception as e:
        print(f"❌ Screenshot capture error: {e}")
        return None


# ============================================================================
# MODULE TEST
# ============================================================================

def main():
    """Test the screenshot capture module"""
    print("🚀 Screenshot Capture Module Test")
    print("="*60)
    
    # Initialize capturer
    capturer = cWeChatScreenshotCapture("pic/screenshots")
    
    # Run test sequence
    success = capturer.test_capture_sequence()
    
    if success:
        print("\n🎉 Module test complete!")
        print("📁 Check pic/screenshots for output files")
    else:
        print("\n❌ Module test failed")
        print("💡 Make sure WeChat is open and visible")
    
    return success

if __name__ == "__main__":
    main()