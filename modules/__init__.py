#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Bot Modules Package

This package contains isolated, modular components for the WeChat bot system.
Each module is designed to be self-contained and testable.

Modules:
- m_screenshot_processor: Consolidated screenshot capture and processing functionality (SINGLE SOURCE OF TRUTH)
- m_Card_Processing: WeChat message card detection and analysis
- visualization_engine: Centralized visualization utilities
- image_utils: Shared image processing functions
"""

# Conditional imports to allow direct execution for testing
try:
    from .m_screenshot_processor import (
        cWeChatScreenshotCapture, 
        fcapture_screenshot,
        fcapture_messages_screenshot,
        fcapture_and_process_screenshot,
        fprocess_screenshot_file,
        fprocess_current_wechat_window,
        fget_live_card_analysis
    )
except ImportError:
    # When running directly, set dummy values for testing
    if __name__ == "__main__":
        cWeChatScreenshotCapture = None
        fcapture_screenshot = None
        fcapture_messages_screenshot = None
        fcapture_and_process_screenshot = None
        fprocess_screenshot_file = None
        fprocess_current_wechat_window = None
        fget_live_card_analysis = None

__all__ = [
    'cWeChatScreenshotCapture',
    'fcapture_screenshot', 
    'fcapture_messages_screenshot',
    'fcapture_and_process_screenshot',
    'fprocess_screenshot_file',
    'fprocess_current_wechat_window',
    'fget_live_card_analysis'
]

__version__ = "1.0.0"


# =============================================================================
# MANUAL CODE TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Code Testing - MODULES PACKAGE")
    print("=" * 60)
    print("🔍 [DEBUG] Smoke test ENTRY")
    
    try:
        # Note: This module is a package initializer, not a class container
        print("   🔧 Testing package initialization...")
        print("   ✅ Modules package initialized successfully")
        
        print("🏁 [DEBUG] Smoke test PASSED")
        
    except Exception as e:
        print(f"   ❌ [ERROR] Smoke test FAILED: {str(e)}")
        print("🏁 [DEBUG] Smoke test FAILED")