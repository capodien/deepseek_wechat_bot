#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
WeChat Bot Modules Package

This package contains isolated, modular components for the WeChat bot system.
Each module is designed to be self-contained and testable.

Modules:
- screenshot_processor: Consolidated screenshot capture and processing functionality
- m_Card_Processing: WeChat message card detection and analysis
- visualization_engine: Centralized visualization utilities
- image_utils: Shared image processing functions
"""

from .screenshot_processor import (
    cWeChatScreenshotCapture, 
    fcapture_screenshot,
    fcapture_messages_screenshot,
    fcapture_and_process_screenshot,
    fprocess_screenshot_file,
    fprocess_current_wechat_window,
    fget_live_card_analysis
)

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