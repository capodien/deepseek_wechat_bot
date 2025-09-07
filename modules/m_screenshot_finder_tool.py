#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
=====================================================================
WeChat Screenshot Finder Tool (m_screenshot_finder_tool.py)
=====================================================================

📌 Version Info
- Version:        v1.0.0 - Initial Tool Class Implementation
- Created:        2024-09-07
- Last Modified:  2024-09-07
- Author:         AI Assistant (WeChat Bot System)

📌 Tool Position
- Category:       TOOL - Reusable component for screenshot management
- Usage:          Called across multiple modules and phases
- Human Logic:    Repeatedly-called functionality = Tool Class

📌 Purpose
Dedicated tool for finding and managing WeChat screenshot files.
This tool implements consistent screenshot finding logic that can be
called repeatedly from any module, phase, or processing step.

📌 Human Structural Logic
- TOOL: Reusable component independent from detector/processor logic
- CROSS-MODULE: Used by m_photo_processor.py, WDC_Server.py, m_Card_Processing.py
- STATEFUL: Maintains cache for performance optimization
- SINGLE RESPONSIBILITY: Screenshot discovery and validation only

📌 Main Functions
- get_latest_screenshot() - Find most recent WeChat screenshot
- get_all_screenshots() - List all screenshots sorted by timestamp
- get_screenshot_by_timestamp() - Find specific screenshot by timestamp
- validate_screenshot() - Check file existence and format
- clear_cache() - Reset internal cache

📌 Usage Examples
Basic Usage:
    from modules.m_screenshot_finder_tool import cWeChat_Screenshot_Finder
    finder = cWeChat_Screenshot_Finder()
    latest = finder.get_latest_screenshot()

Cached Usage (Performance):
    finder = cWeChat_Screenshot_Finder(enable_cache=True)
    latest = finder.get_latest_screenshot()  # File system scan
    latest = finder.get_latest_screenshot()  # Cache hit (faster)

Custom Directory:
    finder = cWeChat_Screenshot_Finder("custom/screenshot/path")
    screenshots = finder.get_all_screenshots()

Diagnostic Integration:
    finder = cWeChat_Screenshot_Finder()
    info = finder.get_tool_info()  # For diagnostic display
=====================================================================
"""

import os
import time
from datetime import datetime
from typing import Optional, List, Dict, Tuple
import glob

class cWeChat_Screenshot_Finder:
    """
    Dedicated tool for finding and managing WeChat screenshot files.
    
    📋 PURPOSE:
    Centralized tool for screenshot discovery across all modules.
    This tool implements consistent screenshot finding logic that can be
    called repeatedly from any module, phase, or processing step.
    Follows the Human Structural Logic: Repeatedly-called functionality = Tool Class.
    
    📌 INPUT CONTRACT:
    - default_dir: str - Default directory to search (default: "pic/screenshots")
    - enable_cache: bool - Enable caching for performance (default: True)
    - cache_duration: int - Cache duration in seconds (default: 300)
    
    📌 OUTPUT CONTRACT:
    - Success: Initialized tool instance with screenshot finding capabilities
    - Failure: Exception raised if default directory creation fails
    - Instance provides screenshot discovery, validation, and cache management methods
    
    🔧 ALGORITHM:
    1. Initialize with default screenshot directory and caching system
    2. Set up screenshot pattern matching for WeChat format (YYYYMMDD_HHMMSS_WeChat.png)
    3. Configure performance optimization with intelligent caching
    4. Establish validation rules for screenshot format and existence
    5. Provide multiple access methods for different use cases
    
    📊 KEY PARAMETERS:
    - WECHAT_SCREENSHOT_PATTERN = "*_WeChat.png"  # File pattern for WeChat screenshots
    - TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"          # Expected timestamp format
    - cache_duration = 300                        # Cache validity (5 minutes)
    - MIN_TIMESTAMP_LENGTH = 15                   # Minimum timestamp length for validation
    
    🎨 VISUAL OUTPUT:
    - Debug mode generates console output showing found screenshots
    - Cache status display for performance monitoring
    - File validation results with detailed error messages
    - Performance metrics showing cache hits vs file system scans
    
    🔍 DEBUG VARIABLES:
    - Key variables used for debugging:
      • _cache: Dict - Cached screenshot listings with timestamps
      • _cache_timestamp: float - When cache was last updated
      • screenshot_files: List - Current list of valid screenshot files
      • latest_file: str - Most recent screenshot filename found
    - Debug triggers: Cache misses, file validation failures, directory access errors
    - Output format: Console messages with clear success/failure indicators
    
    ⚙️ DEPENDENCIES:
    - Required: os, time, datetime, typing, glob
    - Optional: No external dependencies beyond Python standard library
    - Integrates with: Any module requiring WeChat screenshot access
    """
    
    # Class constants for WeChat screenshot detection
    WECHAT_SCREENSHOT_PATTERN = "*_WeChat.png"
    TIMESTAMP_FORMAT = "%Y%m%d_%H%M%S"
    MIN_TIMESTAMP_LENGTH = 15
    
    def __init__(self, default_dir: str = "pic/screenshots", 
                 enable_cache: bool = True, cache_duration: int = 300):
        """
        Initialize the WeChat Screenshot Finder Tool
        
        📌 INPUT CONTRACT:
        - default_dir: str - Default directory to search for screenshots
        - enable_cache: bool - Enable performance caching (default: True)
        - cache_duration: int - Cache duration in seconds (default: 300/5 minutes)
        
        📌 OUTPUT CONTRACT:
        - Success: Initialized tool instance ready for screenshot finding
        - Failure: Exception raised if directory creation fails
        
        Side Effects:
        - Creates default directory if it doesn't exist
        - Initializes caching system for performance optimization
        - Sets up screenshot pattern matching and validation rules
        """
        self.default_dir = default_dir
        self.enable_cache = enable_cache
        self.cache_duration = cache_duration
        
        # Cache system for performance optimization
        self._cache = {}
        self._cache_timestamp = 0
        
        # Create default directory if it doesn't exist
        os.makedirs(default_dir, exist_ok=True)
        
        print(f"🔍 WeChat Screenshot Finder Tool initialized")
        print(f"   📁 Default directory: {self.default_dir}")
        print(f"   ⚡ Caching: {'Enabled' if self.enable_cache else 'Disabled'}")
        if self.enable_cache:
            print(f"   ⏱️  Cache duration: {self.cache_duration}s")
    
    def get_latest_screenshot(self, screenshot_dir: str = None) -> Optional[str]:
        """
        Find the latest WeChat screenshot in the specified directory
        
        📌 INPUT CONTRACT:
        - screenshot_dir: Optional[str] - Directory to search (uses default if None)
        
        📌 OUTPUT CONTRACT:
        - Success: str - Full path to the latest WeChat screenshot file
        - Failure: None - No screenshots found or directory doesn't exist
        - Format: YYYYMMDD_HHMMSS_WeChat.png (e.g., 20240907_143022_WeChat.png)
        
        Side Effects:
        - May update internal cache if caching enabled
        - Prints status messages about screenshot discovery
        - Validates file existence and format before returning
        """
        search_dir = screenshot_dir if screenshot_dir is not None else self.default_dir
        
        # Check cache first if enabled
        if self.enable_cache and self._is_cache_valid(search_dir):
            cached_files = self._cache.get(search_dir, [])
            if cached_files:
                latest_file = cached_files[-1]  # Last item is latest (sorted)
                print(f"♻️  Using cached result: {os.path.basename(latest_file)}")
                return latest_file
        
        # Perform file system scan
        screenshots = self._scan_directory(search_dir)
        if not screenshots:
            return None
        
        # Update cache if enabled
        if self.enable_cache:
            self._update_cache(search_dir, screenshots)
        
        # Return latest screenshot
        latest_screenshot = screenshots[-1]  # Last item is latest (sorted)
        print(f"📸 Latest screenshot found: {os.path.basename(latest_screenshot)}")
        return latest_screenshot
    
    def get_all_screenshots(self, screenshot_dir: str = None) -> List[str]:
        """
        Get all WeChat screenshots sorted by timestamp (oldest first)
        
        📌 INPUT CONTRACT:
        - screenshot_dir: Optional[str] - Directory to search (uses default if None)
        
        📌 OUTPUT CONTRACT:
        - Success: List[str] - Full paths to all WeChat screenshots, sorted by timestamp
        - Failure: Empty list - No screenshots found or directory doesn't exist
        - Sorting: Chronological order (oldest first, latest last)
        
        Side Effects:
        - May update internal cache if caching enabled
        - Validates all files before including in results
        """
        search_dir = screenshot_dir if screenshot_dir is not None else self.default_dir
        
        # Check cache first if enabled
        if self.enable_cache and self._is_cache_valid(search_dir):
            cached_files = self._cache.get(search_dir, [])
            if cached_files:
                print(f"♻️  Using cached results: {len(cached_files)} screenshots")
                return cached_files.copy()  # Return copy to prevent cache modification
        
        # Perform file system scan
        screenshots = self._scan_directory(search_dir)
        
        # Update cache if enabled
        if self.enable_cache and screenshots:
            self._update_cache(search_dir, screenshots)
        
        print(f"📂 Found {len(screenshots)} screenshots in {os.path.basename(search_dir)}")
        return screenshots
    
    def get_screenshot_by_timestamp(self, timestamp: str, 
                                   screenshot_dir: str = None) -> Optional[str]:
        """
        Find a specific screenshot by timestamp
        
        📌 INPUT CONTRACT:
        - timestamp: str - Timestamp in YYYYMMDD_HHMMSS format (e.g., "20240907_143022")
        - screenshot_dir: Optional[str] - Directory to search (uses default if None)
        
        📌 OUTPUT CONTRACT:
        - Success: str - Full path to the matching screenshot file
        - Failure: None - No screenshot found with that timestamp
        
        Side Effects:
        - Uses cached results if available for performance
        - Validates timestamp format before searching
        """
        if not self._validate_timestamp_format(timestamp):
            print(f"❌ Invalid timestamp format: {timestamp} (expected: YYYYMMDD_HHMMSS)")
            return None
        
        search_dir = screenshot_dir if screenshot_dir is not None else self.default_dir
        expected_filename = f"{timestamp}_WeChat.png"
        expected_path = os.path.join(search_dir, expected_filename)
        
        # Check if file exists directly (fastest method)
        if os.path.exists(expected_path):
            print(f"✅ Found screenshot by timestamp: {expected_filename}")
            return expected_path
        
        # Fallback: Search through all screenshots (slower but comprehensive)
        all_screenshots = self.get_all_screenshots(search_dir)
        for screenshot_path in all_screenshots:
            filename = os.path.basename(screenshot_path)
            if filename.startswith(timestamp):
                print(f"✅ Found screenshot by timestamp: {filename}")
                return screenshot_path
        
        print(f"❌ No screenshot found with timestamp: {timestamp}")
        return None
    
    def validate_screenshot(self, screenshot_path: str) -> Tuple[bool, str]:
        """
        Validate that a screenshot file exists and has correct format
        
        📌 INPUT CONTRACT:
        - screenshot_path: str - Full path to screenshot file to validate
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[True, ""] - File is valid
        - Failure: Tuple[False, error_message] - File is invalid with reason
        
        Validation Checks:
        - File exists on disk
        - Filename follows WeChat pattern (YYYYMMDD_HHMMSS_WeChat.png)
        - File size is reasonable (not empty, not too large)
        - Timestamp in filename is valid format
        """
        if not os.path.exists(screenshot_path):
            return False, f"File does not exist: {screenshot_path}"
        
        filename = os.path.basename(screenshot_path)
        
        # Check filename pattern
        if not filename.endswith('_WeChat.png'):
            return False, f"Invalid filename pattern: {filename} (must end with '_WeChat.png')"
        
        # Extract and validate timestamp
        try:
            timestamp_part = filename.split('_WeChat.png')[0]
            if not self._validate_timestamp_format(timestamp_part):
                return False, f"Invalid timestamp format in filename: {timestamp_part}"
        except:
            return False, f"Cannot extract timestamp from filename: {filename}"
        
        # Check file size
        try:
            file_size = os.path.getsize(screenshot_path)
            if file_size < 1000:  # Less than 1KB probably empty
                return False, f"File too small: {file_size} bytes"
            if file_size > 50 * 1024 * 1024:  # More than 50MB probably corrupted
                return False, f"File too large: {file_size / (1024*1024):.1f} MB"
        except:
            return False, f"Cannot read file size: {screenshot_path}"
        
        return True, ""
    
    def clear_cache(self) -> None:
        """
        Clear the internal cache to force fresh file system scans
        
        Use Cases:
        - After taking new screenshots
        - When file system changes outside the tool
        - For memory management in long-running processes
        - For testing and debugging
        """
        self._cache.clear()
        self._cache_timestamp = 0
        print("🗑️  Screenshot finder cache cleared")
    
    def get_tool_info(self) -> Dict:
        """
        Get information about the tool status for diagnostic purposes
        
        📌 OUTPUT CONTRACT:
        - Dict with tool status, cache info, and performance metrics
        
        Returns:
        {
            'default_dir': str,
            'caching_enabled': bool,
            'cache_entries': int,
            'cache_age_seconds': float,
            'cache_valid': bool,
            'total_screenshots_found': int,
            'performance_mode': str
        }
        """
        cache_age = time.time() - self._cache_timestamp if self._cache_timestamp > 0 else None
        total_screenshots = sum(len(files) for files in self._cache.values())
        
        return {
            'default_dir': self.default_dir,
            'caching_enabled': self.enable_cache,
            'cache_entries': len(self._cache),
            'cache_age_seconds': cache_age,
            'cache_valid': cache_age is not None and cache_age < self.cache_duration,
            'total_screenshots_found': total_screenshots,
            'performance_mode': 'cached' if self.enable_cache else 'direct_scan'
        }
    
    def _scan_directory(self, directory: str) -> List[str]:
        """
        Internal method to scan directory for WeChat screenshots
        
        Returns:
            List of full paths to valid screenshot files, sorted by timestamp
        """
        if not os.path.exists(directory):
            print(f"⚠️ Screenshot directory not found: {directory}")
            return []
        
        # Use glob to find WeChat screenshot files
        pattern = os.path.join(directory, self.WECHAT_SCREENSHOT_PATTERN)
        screenshot_files = []
        
        for filepath in glob.glob(pattern):
            filename = os.path.basename(filepath)
            
            # Validate filename format
            try:
                timestamp_part = filename.split('_WeChat.png')[0]
                if (len(timestamp_part) == self.MIN_TIMESTAMP_LENGTH and 
                    self._validate_timestamp_format(timestamp_part)):
                    screenshot_files.append(filepath)
            except:
                continue  # Skip invalid files
        
        # Sort by filename (which sorts by timestamp due to YYYYMMDD_HHMMSS format)
        screenshot_files.sort()
        
        return screenshot_files
    
    def _validate_timestamp_format(self, timestamp: str) -> bool:
        """
        Internal method to validate timestamp format
        
        Args:
            timestamp: String in YYYYMMDD_HHMMSS format
            
        Returns:
            True if valid format, False otherwise
        """
        if len(timestamp) != self.MIN_TIMESTAMP_LENGTH:
            return False
        
        if timestamp[8] != '_':  # Check underscore position
            return False
        
        date_part = timestamp[:8]
        time_part = timestamp[9:]
        
        # Check if all characters are digits (except underscore)
        if not date_part.isdigit() or not time_part.isdigit():
            return False
        
        # Validate date/time ranges (basic check)
        try:
            year = int(timestamp[:4])
            month = int(timestamp[4:6])
            day = int(timestamp[6:8])
            hour = int(timestamp[9:11])
            minute = int(timestamp[11:13])
            second = int(timestamp[13:15])
            
            if not (2020 <= year <= 2030):  # Reasonable year range
                return False
            if not (1 <= month <= 12):
                return False
            if not (1 <= day <= 31):
                return False
            if not (0 <= hour <= 23):
                return False
            if not (0 <= minute <= 59):
                return False
            if not (0 <= second <= 59):
                return False
                
            return True
        except:
            return False
    
    def _is_cache_valid(self, directory: str) -> bool:
        """
        Internal method to check if cache is valid for given directory
        
        Args:
            directory: Directory path to check cache for
            
        Returns:
            True if cache is valid and fresh, False otherwise
        """
        if not self.enable_cache:
            return False
        
        if directory not in self._cache:
            return False
        
        cache_age = time.time() - self._cache_timestamp
        return cache_age < self.cache_duration
    
    def _update_cache(self, directory: str, screenshots: List[str]) -> None:
        """
        Internal method to update cache with new screenshot list
        
        Args:
            directory: Directory path being cached
            screenshots: List of screenshot file paths to cache
        """
        self._cache[directory] = screenshots
        self._cache_timestamp = time.time()


# ============================================================================
# CONVENIENCE FUNCTIONS FOR EASY INTEGRATION
# ============================================================================

# Global instance for efficient reuse (similar to m_screenshot_processor pattern)
_global_finder = None

def get_screenshot_finder() -> cWeChat_Screenshot_Finder:
    """
    Get global screenshot finder instance for efficient reuse
    
    Returns:
        Shared instance of cWeChat_Screenshot_Finder
    """
    global _global_finder
    if _global_finder is None:
        _global_finder = cWeChat_Screenshot_Finder()
    return _global_finder

def find_latest_screenshot(screenshot_dir: str = None) -> Optional[str]:
    """
    Convenience function to find latest screenshot using global instance
    
    Args:
        screenshot_dir: Optional directory to search (uses default if None)
        
    Returns:
        Path to latest screenshot file or None if not found
        
    Usage:
        from modules.m_screenshot_finder_tool import find_latest_screenshot
        latest = find_latest_screenshot()
    """
    finder = get_screenshot_finder()
    return finder.get_latest_screenshot(screenshot_dir)

def find_all_screenshots(screenshot_dir: str = None) -> List[str]:
    """
    Convenience function to find all screenshots using global instance
    
    Args:
        screenshot_dir: Optional directory to search (uses default if None)
        
    Returns:
        List of all screenshot file paths, sorted by timestamp
    """
    finder = get_screenshot_finder()
    return finder.get_all_screenshots(screenshot_dir)


# ============================================================================
# MODULE TEST AND VALIDATION
# ============================================================================

def test_screenshot_finder_tool():
    """Test the screenshot finder tool functionality"""
    print("🧪 WECHAT SCREENSHOT FINDER TOOL TEST")
    print("=" * 60)
    
    # Test tool initialization
    print("\n1. Testing tool initialization...")
    finder = cWeChat_Screenshot_Finder(enable_cache=True)
    print("✅ Tool initialized successfully")
    
    # Test latest screenshot finding
    print("\n2. Testing latest screenshot finding...")
    latest = finder.get_latest_screenshot()
    if latest:
        print(f"✅ Latest screenshot found: {os.path.basename(latest)}")
        
        # Test screenshot validation
        print("\n3. Testing screenshot validation...")
        is_valid, error_msg = finder.validate_screenshot(latest)
        if is_valid:
            print("✅ Screenshot validation passed")
        else:
            print(f"❌ Screenshot validation failed: {error_msg}")
    else:
        print("⚠️  No screenshots found for testing")
        print("   Please capture a WeChat screenshot first")
    
    # Test all screenshots finding
    print("\n4. Testing all screenshots finding...")
    all_screenshots = finder.get_all_screenshots()
    print(f"✅ Found {len(all_screenshots)} screenshots total")
    
    # Test cache performance
    print("\n5. Testing cache performance...")
    start_time = time.time()
    finder.get_latest_screenshot()  # This should use cache
    cache_time = time.time() - start_time
    print(f"✅ Cache lookup completed in {cache_time*1000:.1f}ms")
    
    # Test tool info
    print("\n6. Testing tool information...")
    tool_info = finder.get_tool_info()
    print(f"✅ Tool info: {tool_info['performance_mode']} mode, "
          f"{tool_info['total_screenshots_found']} screenshots cached")
    
    print("\n🎉 Screenshot Finder Tool test completed!")
    print("💡 This tool can now be used across all modules for consistent screenshot access")

# =============================================================================
# MANUAL CODE TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Code Testing - SCREENSHOT FINDER TOOL")
    print("=" * 60)
    print("🔍 [DEBUG] Smoke test ENTRY")
    
    try:
        # Simply instantiate the class
        print("   🔧 Testing cWeChat_Screenshot_Finder...")
        finder = cWeChat_Screenshot_Finder()
        print("   ✅ cWeChat_Screenshot_Finder instantiated successfully")
        
        print("🏁 [DEBUG] Smoke test PASSED")
        
    except Exception as e:
        print(f"   ❌ [ERROR] Smoke test FAILED: {str(e)}")
        print("🏁 [DEBUG] Smoke test FAILED")