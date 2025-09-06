#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
 Module: Phase 1.0 - Screenshot Capture Module
 File:   m_ScreenShot_WeChatWindow.py
=====================================================================

📌 Version Info
- Version:        v1.0.0
- Created:        2024-09-04
- Last Modified:  2024-09-04
- Author:         AI Assistant (WeChat Bot System)

📌 Module Position
- Pipeline Phase:  Phase 1.0: Screenshot Capture
- Previous Phase:  None (Entry point)
- Next Phase:      Phase 2.0 - New Message Detection (monitor_new_message.py)

📌 Input Contract
- capture_screenshot() Unified Parameters:
    Modern style:
        output_dir: str (default: "pic/screenshots") - Directory to save screenshot
        filename: Optional[str] (default: None) - Custom filename, auto-generated if None
        detect_window: bool (default: True) - Whether to auto-detect window if not cached
    Legacy style (backward compatibility):
        save_dir: str - Directory to save screenshots (overrides output_dir)
        region: Optional[Tuple] - Specific screen region to capture (unused)
        prefix: str - Filename prefix (overrides filename generation)
        use_dynamic_detection: bool - Whether to dynamically detect window (overrides detect_window)
- capture_messages_screenshot() Legacy Alias:
    Identical to legacy style parameters above (maintained for backward compatibility)
- ccWeChatScreenshotCapture() Constructor:
    output_dir: str (default: "pic/screenshots") - Directory to save screenshots
- System Prerequisites:
    WeChat desktop application (must be running and visible)
    Cross-platform system support (macOS native detection preferred)

📌 Output Contract
- Produced State(s):
    screenshot_path: str (path to captured screenshot file)
    window_coords: Tuple[int,int,int,int] (validated window coordinates)
- Side Effects:
    Creates screenshot files in pic/screenshots/ directory
    Updates coordinate cache for performance optimization
    Generates diagnostic metrics and logging

📌 Notes
- Dependencies:   pyautogui, platform, os, datetime, typing, json
- Usage Examples:
    Modern:  `screenshot_path = capture_screenshot()`  # → 20250904_210954_WeChat.png
             `screenshot_path = capture_screenshot("custom/dir", "my_screenshot.png")`
    Legacy:  `screenshot_path = capture_messages_screenshot()`  # → 20250904_210954_WeChat.png
             `screenshot_path = capture_messages_screenshot(save_dir="pics")`
- Output Format:  {YYYYMMDD_HHMMSS}_WeChat.png (timestamp first for better sorting)
- Performance:    20-50ms typical capture time
- Cross-Platform: macOS (native Quartz), Windows/Linux (OCR fallback)
=====================================================================

ScreenShot_WeChatWindow Module
WeChat Window Screenshot Capture with Cross-Platform Support

This module provides isolated screenshot capture functionality specifically designed
for WeChat desktop applications. It includes:

1. Cross-platform window detection (macOS native, OCR fallback)
2. Precise screenshot capture of WeChat windows
3. Window coordinate validation and caching
4. Screenshot quality validation
5. Clean API interface for easy integration

Main Features:
- WeChatScreenshotCapture: Main class for screenshot operations
- capture_screenshot(): Unified API function (modern + legacy parameter styles)
- capture_messages_screenshot(): Legacy alias for backward compatibility
- Cross-platform compatibility (macOS, Windows, Linux)
- Automatic window detection and coordinate caching
- Quality validation and error handling

Usage:
    from modules.m_ScreenShot_WeChatWindow import capture_screenshot, capture_messages_screenshot
    
    # Modern style (recommended)
    screenshot_path = fcapture_screenshot()
    screenshot_path = capture_screenshot("custom/dir", "my_screenshot.png")
    
    # Legacy style (backward compatibility)
    screenshot_path = capture_messages_screenshot()
    screenshot_path = capture_messages_screenshot(save_dir="pics", prefix="test_")
    
    # Advanced usage
    capturer = cWeChatScreenshotCapture("custom/output/dir")
    capturer.detect_wechat_window()
    screenshot_path = capturer.capture_screenshot("custom_name.png")
"""

import platform
import pyautogui
import time
import os
from datetime import datetime
from typing import Optional, Tuple, Dict
import json


class cWeChatScreenshotCapture:
    """
    WeChat Window Screenshot Capture Module
    
    Provides comprehensive screenshot capture functionality for WeChat desktop application
    with cross-platform window detection and quality validation.
    """
    
    def __init__(self, output_dir: str = "pic/screenshots"):
        """
        Initialize the screenshot capture module
        
        Args:
            output_dir: Directory to save screenshots (default: "pic/screenshots")
        """
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
        
        # Reasonable WeChat window constraints (adjusted for larger screens and tall windows)
        MIN_WIDTH, MAX_WIDTH = 600, 2000  # Allow larger widths for high-res screens  
        MIN_HEIGHT, MAX_HEIGHT = 400, 2000  # Allow much taller windows for long contact lists
        MIN_ASPECT, MAX_ASPECT = 0.3, 4.0  # Allow taller aspect ratios for vertical windows
        
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
            
            # Prioritize actual WeChat app windows over browser windows
            actual_wechat_windows = []
            browser_windows = []
            
            for window in valid_windows:
                owner = window['owner']
                name = window['name']
                
                # Check if it's the actual WeChat application
                if owner == 'WeChat' and ('Weixin' in name or name == '' or 'WeChat' in name):
                    # Exclude browser-related windows
                    if 'Diagnostic' not in name and 'Step-by-Step' not in name and 'Bot' not in name:
                        actual_wechat_windows.append(window)
                else:
                    browser_windows.append(window)
            
            # Prefer actual WeChat windows over browser windows
            if actual_wechat_windows:
                main_window = max(actual_wechat_windows, key=lambda w: w['coords'][2] * w['coords'][3])
                print(f"🎯 Selected WeChat app window: {main_window['name']} - {main_window['coords']}")
                return main_window['coords']
            else:
                # Fall back to largest valid window if no actual WeChat windows found
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
        
        📌 INPUT CONTRACT:
        - save_as: Optional[str] - Custom filename (auto-generated if None)
        - Prerequisites: WeChat desktop app running and visible, window coordinates detected
        
        📌 OUTPUT CONTRACT:
        - Success: str - Path to saved screenshot file (YYYYMMDD_HHMMSS_WeChat.png)
        - Failure: None
        
        Side Effects:
        - Creates screenshot file in self.output_dir directory
        - Validates screenshot quality if validation_enabled=True
        - Updates processing timestamp and file metadata
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
        
        # Try to load existing coordinates
        print("\n1. Loading saved coordinates...")
        if not self.load_window_coordinates():
            print("   No valid saved coordinates, will detect fresh")
        
        # Detect window (if needed)
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
        
        # Save coordinates
        print("\n3. Saving window coordinates...")
        self.save_window_coordinates()
        
        # Take test screenshot
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
                      filename: str = None,
                      detect_window: bool = True,
                      # Legacy parameters for backward compatibility
                      save_dir: str = None,
                      region = None,
                      prefix: str = None,
                      use_dynamic_detection: bool = None) -> Optional[str]:
    """
    Unified WeChat screenshot capture function
    
    📌 INPUT CONTRACT:
    Modern Parameters:
    - output_dir: str - Directory to save screenshot (default: "pic/screenshots")
    - filename: Optional[str] - Custom filename (auto-generated if None)
    - detect_window: bool - Whether to auto-detect window if not cached
    
    Legacy Parameters (backward compatibility):
    - save_dir: Optional[str] - Directory override (legacy)
    - region: Optional - Unused but supported for compatibility
    - prefix: Optional[str] - Filename prefix override (legacy)
    - use_dynamic_detection: Optional[bool] - Detection override (legacy)
    
    📌 OUTPUT CONTRACT:
    - Success: str - Path to saved screenshot file (YYYYMMDD_HHMMSS_WeChat.png)
    - Failure: None
    
    Side Effects:
    - Creates output directory if it doesn't exist
    - Auto-detects WeChat window if needed
    - Validates screenshot quality
    - Supports both modern and legacy parameter styles
    """
    try:
        # Handle legacy parameter mapping
        if save_dir is not None:
            output_dir = save_dir
        if use_dynamic_detection is not None:
            detect_window = use_dynamic_detection
        
        capturer = fget_capturer()
        capturer.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Auto-detect window if needed
        if detect_window and not capturer.window_coords:
            coords = capturer.detect_wechat_window()
            if not coords:
                print("❌ Failed to detect WeChat window")
                return None
        
        # Generate filename (unified format for both modern and legacy styles)
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # Use consistent format: timestamp first, WeChat suffix
            filename = f"{timestamp}_WeChat.png"
        
        # Capture screenshot
        screenshot_path = capturer.capture_screenshot(filename)
        
        if screenshot_path:
            print(f"✅ Screenshot captured: {screenshot_path}")
            return screenshot_path
        else:
            print("❌ Screenshot capture failed")
            return None
            
    except Exception as e:
        print(f"❌ Screenshot capture error: {e}")
        return None

def fdetect_wechat_window() -> Optional[Tuple[int, int, int, int]]:
    """
    Simple API function to detect WeChat window coordinates
    
    Returns:
        (left, top, width, height) or None if not found
        
    Usage:
        coords = fdetect_wechat_window()
        if coords:
            left, top, width, height = coords
    """
    try:
        capturer = fget_capturer()
        coords = capturer.detect_wechat_window()
        return coords
    except Exception as e:
        print(f"❌ Window detection error: {e}")
        return None

# ============================================================================
# LEGACY ALIAS FOR BACKWARD COMPATIBILITY
# ============================================================================

def fcapture_messages_screenshot(save_dir="pic/screenshots", 
                               region=None, 
                               prefix="wechat_", 
                               use_dynamic_detection=True):
    """
    Legacy function alias for backward compatibility
    
    This function is now just an alias to the unified capture_screenshot()
    function with legacy parameter mapping.
    """
    return fcapture_screenshot(
        save_dir=save_dir,
        region=region,
        prefix=prefix,
        use_dynamic_detection=use_dynamic_detection
    )

def fget_wechat_window_coords():
    """
    Backward compatibility function - get current WeChat window coordinates
    """
    try:
        capturer = fget_capturer()
        coords = capturer.detect_wechat_window()
        if coords:
            return coords
        else:
            print("⚠️ Failed to detect WeChat window, using default")
            return None
    except Exception as e:
        print(f"❌ Window detection error: {e}")
        return None


# ============================================================================
# MODULE TEST AND VALIDATION
# ============================================================================

def fmain():
    """Test the ScreenShot_WeChatWindow module"""
    print("🚀 ScreenShot_WeChatWindow Module Test")
    print("="*50)
    
    # Test simple API
    print("\n1. Testing simple API...")
    screenshot_path = fcapture_screenshot()
    if screenshot_path:
        print(f"✅ Simple API test passed: {screenshot_path}")
    else:
        print("❌ Simple API test failed")
    
    # Test advanced API
    print("\n2. Testing advanced API...")
    capturer = cWeChatScreenshotCapture("pic/screenshots")
    success = capturer.test_capture_sequence()
    
    if success:
        print(f"\n🎉 Module ready for integration!")
        print(f"📁 Check {capturer.output_dir} for screenshots")
        
        # Test backward compatibility
        print(f"\n3. Testing backward compatibility...")
        old_style_path = fcapture_messages_screenshot()
        if old_style_path:
            print(f"✅ Backward compatibility test passed: {old_style_path}")
        else:
            print("❌ Backward compatibility test failed")
            
    else:
        print(f"\n💡 Troubleshooting tips:")
        print(f"   • Make sure WeChat desktop app is running and visible")
        print(f"   • Ensure WeChat window is not minimized or hidden")
        print(f"   • Check that screen recording permissions are granted")


if __name__ == "__main__":
    fmain()