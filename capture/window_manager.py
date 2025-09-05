#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Window Manager
Integrates dynamic window detection with existing bot functionality
"""

import time
import pyautogui
from typing import Optional, Tuple
from .dynamic_window_finder import DynamicWindowFinder
from Constants import Constants


class WindowManager:
    """Manages WeChat window detection and provides backward compatibility"""
    
    def __init__(self, use_dynamic_detection: bool = True):
        """
        Initialize window manager
        
        Args:
            use_dynamic_detection: Whether to use dynamic detection or fall back to static
        """
        self.use_dynamic_detection = use_dynamic_detection
        self.finder = DynamicWindowFinder() if use_dynamic_detection else None
        self.static_window = Constants.WECHAT_WINDOW
        self.last_detection_time = 0
        self.detection_interval = 30  # Re-detect every 30 seconds
        
    def get_wechat_window(self, force_refresh: bool = False) -> Tuple[int, int, int, int]:
        """
        Get current WeChat window coordinates
        
        Args:
            force_refresh: Force new detection even if cache is valid
            
        Returns:
            Tuple of (left, top, width, height)
        """
        if not self.use_dynamic_detection:
            return self.static_window
        
        current_time = time.time()
        
        # Use dynamic detection if enabled and conditions are met
        if (force_refresh or 
            current_time - self.last_detection_time > self.detection_interval):
            
            try:
                window_coords = self.finder.get_wechat_window(use_cache=not force_refresh)
                
                if window_coords:
                    self.last_detection_time = current_time
                    print(f"✅ 动态检测到窗口: {window_coords}")
                    return window_coords
                else:
                    print("⚠️ 动态检测失败，使用静态坐标")
                    
            except Exception as e:
                print(f"❌ 动态检测异常: {e}，回退到静态坐标")
        
        # Fallback to static coordinates
        return self.static_window
    
    def capture_window_screenshot(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Capture screenshot of WeChat window
        
        Args:
            save_path: Path to save screenshot, auto-generated if None
            
        Returns:
            Path of saved screenshot or None if failed
        """
        try:
            window_coords = self.get_wechat_window()
            left, top, width, height = window_coords
            
            # Take screenshot
            screenshot = pyautogui.screenshot(region=(left, top, width, height))
            
            # Generate save path if not provided
            if not save_path:
                timestamp = int(time.time() * 1000)
                save_path = f"{Constants.SCREENSHOTS_DIR}/wechat_window_{timestamp}.png"
            
            screenshot.save(save_path)
            print(f"📷 窗口截图已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ 窗口截图失败: {e}")
            return None
    
    def capture_message_area(self, save_path: Optional[str] = None) -> Optional[str]:
        """
        Capture screenshot of message area (chat content region)
        
        Args:
            save_path: Path to save screenshot, auto-generated if None
            
        Returns:
            Path of saved screenshot or None if failed
        """
        try:
            window_coords = self.get_wechat_window()
            window_left, window_top, window_width, window_height = window_coords
            
            # Calculate message area based on window coordinates
            # Adjust these offsets based on WeChat UI layout
            msg_area_left = window_left + 304  # Offset from left edge
            msg_area_top = window_top
            msg_area_width = window_width - 304  # Subtract left sidebar width
            msg_area_height = min(800, window_height)  # Limit height
            
            # Take screenshot of message area
            screenshot = pyautogui.screenshot(region=(
                msg_area_left, msg_area_top, msg_area_width, msg_area_height
            ))
            
            # Generate save path if not provided
            if not save_path:
                timestamp = int(time.time() * 1000)
                save_path = f"{Constants.MESSAGES_DIR}/message_area_{timestamp}.png"
            
            screenshot.save(save_path)
            print(f"📷 消息区域截图已保存: {save_path}")
            return save_path
            
        except Exception as e:
            print(f"❌ 消息区域截图失败: {e}")
            return None
    
    def is_wechat_window_active(self) -> bool:
        """
        Check if WeChat window is currently visible and active
        
        Returns:
            True if window appears to be active/visible
        """
        try:
            window_coords = self.get_wechat_window()
            if not window_coords:
                return False
            
            # Take a small screenshot to verify window exists
            left, top, width, height = window_coords
            test_screenshot = pyautogui.screenshot(region=(left, top, min(100, width), min(100, height)))
            
            # Basic check - if screenshot is captured without error, window likely exists
            return test_screenshot is not None
            
        except Exception:
            return False
    
    def wait_for_wechat_window(self, timeout: int = 30) -> bool:
        """
        Wait for WeChat window to become available
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if window becomes available, False if timeout
        """
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            if self.is_wechat_window_active():
                return True
            
            print("⏳ 等待 WeChat 窗口...")
            time.sleep(2)
        
        print(f"⏰ 超时 ({timeout}秒) - WeChat 窗口未找到")
        return False
    
    def refresh_window_detection(self) -> bool:
        """
        Force refresh window detection
        
        Returns:
            True if detection successful
        """
        try:
            window_coords = self.get_wechat_window(force_refresh=True)
            return window_coords is not None
            
        except Exception as e:
            print(f"❌ 刷新窗口检测失败: {e}")
            return False
    
    def get_window_info(self) -> dict:
        """
        Get detailed window information for debugging
        
        Returns:
            Dictionary with window details
        """
        window_coords = self.get_wechat_window()
        
        info = {
            "dynamic_detection": self.use_dynamic_detection,
            "window_coords": window_coords,
            "static_fallback": self.static_window,
            "last_detection": self.last_detection_time,
            "window_active": self.is_wechat_window_active()
        }
        
        return info
    
    def enable_dynamic_detection(self):
        """Enable dynamic window detection"""
        if not self.finder:
            self.finder = DynamicWindowFinder()
        self.use_dynamic_detection = True
        self.last_detection_time = 0  # Force refresh
        print("✅ 已启用动态窗口检测")
    
    def disable_dynamic_detection(self):
        """Disable dynamic window detection, use static coordinates"""
        self.use_dynamic_detection = False
        print("✅ 已禁用动态窗口检测，使用静态坐标")


# Global instance for backward compatibility
_window_manager = None

def get_window_manager(use_dynamic: bool = True) -> WindowManager:
    """Get global window manager instance"""
    global _window_manager
    if _window_manager is None:
        _window_manager = WindowManager(use_dynamic_detection=use_dynamic)
    return _window_manager

def get_wechat_window() -> Tuple[int, int, int, int]:
    """Backward compatibility function for existing code"""
    return get_window_manager().get_wechat_window()

def capture_wechat_screenshot(save_path: Optional[str] = None) -> Optional[str]:
    """Backward compatibility function for taking screenshots"""
    return get_window_manager().capture_window_screenshot(save_path)


if __name__ == "__main__":
    # Test the window manager
    print("🚀 测试窗口管理器")
    print("=" * 50)
    
    # Test with dynamic detection
    manager = WindowManager(use_dynamic_detection=True)
    
    print("📊 窗口信息:")
    info = manager.get_window_info()
    for key, value in info.items():
        print(f"  {key}: {value}")
    
    # Test screenshot
    if manager.wait_for_wechat_window(timeout=10):
        screenshot_path = manager.capture_window_screenshot()
        if screenshot_path:
            print("✅ 截图测试成功")
        else:
            print("❌ 截图测试失败")
    else:
        print("❌ WeChat 窗口未找到")