#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Bot Modules Package

This package contains isolated, modular components for the WeChat bot system.
Each module is designed to be self-contained and testable.

Modules:
- m_ScreenShot_WeChatWindow: Screenshot capture functionality for WeChat windows
"""

from .m_ScreenShot_WeChatWindow import WeChatScreenshotCapture, capture_screenshot

__all__ = [
    'WeChatScreenshotCapture',
    'capture_screenshot'
]

__version__ = "1.0.0"