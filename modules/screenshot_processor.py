#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Screenshot Processing Module (screenshot_processor.py)

This module handles all screenshot capture and processing operations for the WeChat bot.
Extracted from m_Card_Processing.py for better separation of concerns.

Functions:
- capture_and_process_screenshot() - Live capture with analysis
- process_screenshot_file() - Process existing screenshot files  
- process_current_wechat_window() - Convenience wrapper
- get_live_card_analysis() - Comprehensive analysis with visualizations
"""

import os
from typing import Dict, Optional, Tuple

# Import screenshot capture functionality
try:
    from .m_ScreenShot_WeChatWindow import fcapture_screenshot
    SCREENSHOT_AVAILABLE = True
except ImportError:
    # Fallback import for direct execution
    try:
        from m_ScreenShot_WeChatWindow import fcapture_screenshot
        SCREENSHOT_AVAILABLE = True
    except ImportError:
        print("⚠️  Screenshot module not available. Live capture disabled.")
        SCREENSHOT_AVAILABLE = False


def fcapture_and_process_screenshot(output_dir: str = "pic/screenshots", 
                                  custom_filename: str = None) -> Optional[Tuple[str, Dict]]:
    """
    Capture a fresh WeChat screenshot and process it for card analysis
    
    Args:
        output_dir: Directory to save screenshot and visualizations
        custom_filename: Custom filename for screenshot, auto-generated if None
        
    Returns:
        Tuple of (screenshot_path, analysis_results) or None if failed
        
    Usage:
        screenshot_path, results = fcapture_and_process_screenshot()
        if results:
            print(f"Found {results['cards_detected']} cards")
    """
    if not SCREENSHOT_AVAILABLE:
        print("❌ Screenshot capture not available. Install required module.")
        return None
        
    try:
        print("\n🎯 Live WeChat Screenshot & Card Processing")
        print("=" * 50)
        
        # Step 1: Capture fresh screenshot
        print("\n📸 Step 1: Capturing WeChat screenshot...")
        screenshot_path = fcapture_screenshot(output_dir=output_dir, filename=custom_filename)
        
        if not screenshot_path:
            print("❌ Failed to capture screenshot")
            return None
            
        print(f"✅ Screenshot captured: {os.path.basename(screenshot_path)}")
        
        # Step 2: Process with card analysis
        print("\n🔍 Step 2: Processing card analysis...")
        results = fprocess_screenshot_file(screenshot_path)
        
        if results:
            print(f"\n✅ Analysis complete:")
            print(f"   Width: {results.get('width_detected')}px")  
            print(f"   Avatars: {results.get('avatars_detected')}")
            print(f"   Cards: {results.get('cards_detected')}")
            return screenshot_path, results
        else:
            print("❌ Analysis failed")
            return None
            
    except Exception as e:
        print(f"❌ Error in capture_and_process_screenshot: {e}")
        return None


def fprocess_screenshot_file(image_path: str) -> Optional[Dict]:
    """
    Process a screenshot file and return comprehensive analysis results
    
    Args:
        image_path: Path to screenshot file to analyze
        
    Returns:
        Dictionary with analysis results or None if failed
    """
    try:
        # Import complete analysis pipeline from main module
        try:
            from . import m_Card_Processing  # Relative import for module usage
        except ImportError:
            import m_Card_Processing  # Direct import for standalone execution
        
        # Use the complete pipeline with coordinate saving
        results = m_Card_Processing.fcomplete_card_analysis(image_path, debug_mode=False)
        
        # Save coordinates to wechat_ctx.json if analysis succeeded
        if results:
            m_Card_Processing.f_save_coordinates_to_context(image_path, results)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing screenshot: {e}")
        return None
        
        # 1. Width Detection
        width_result = width_detector.detect_width(image_path)
        if width_result:
            left, right, width = width_result
            results['width_detected'] = width
            results['width_boundaries'] = {'left': left, 'right': right}
        else:
            results['width_detected'] = None
            results['width_boundaries'] = None
            
        # 2. Avatar Detection  
        avatars, avatar_info = avatar_detector.detect_avatars(image_path)
        results['avatars_detected'] = len(avatars)
        results['avatar_list'] = avatars
        results['avatar_detection_info'] = avatar_info
        
        # 3. Card Boundary Detection
        cards, card_info = boundary_detector.detect_cards(image_path)
        results['cards_detected'] = len(cards)
        results['card_list'] = cards  
        results['card_detection_info'] = card_info
        
        # 4. Time Box Detection (MUST RUN FIRST - detects boundary)
        if results['card_list']:
            enhanced_cards_with_times, time_info = time_detector.detect_time_boundaries(image_path)
            results['cards_with_times'] = enhanced_cards_with_times
            results['time_detection_info'] = time_info
            results['times_detected'] = time_info.get('total_times_detected', 0)
            
            # 5. Contact Name Detection (RUNS SECOND - uses boundary from time detection)
            enhanced_cards_with_names, name_info = name_detector.detect_name_boundaries(
                image_path, 
                cards_with_times=enhanced_cards_with_times  # Pass time detection results
            )
            results['cards_with_names'] = enhanced_cards_with_names
            results['name_detection_info'] = name_info
            results['names_detected'] = name_info.get('total_names_detected', 0)
            
            # Update card_list with the most enhanced version
            results['card_list'] = enhanced_cards_with_names
        
        # 6. Summary
        results['processing_successful'] = True
        results['image_processed'] = os.path.basename(image_path)
        
        return results
        
    except Exception as e:
        print(f"❌ Error processing screenshot: {e}")
        return None


def fprocess_current_wechat_window() -> Optional[Dict]:
    """
    Convenience function: Capture current WeChat window and analyze cards
    
    Returns:
        Dictionary with complete analysis results or None if failed
        
    Usage:
        results = fprocess_current_wechat_window()
        if results:
            for i, card in enumerate(results['card_list'], 1):
                print(f"Card {i}: {card['width']}×{card['height']}px")
    """
    result = fcapture_and_process_screenshot()
    if result:
        screenshot_path, analysis = result
        return analysis
    return None


def fget_live_card_analysis(include_visualizations: bool = True) -> Optional[Tuple[Dict, Dict]]:
    """
    Get comprehensive live card analysis with optional visualizations
    
    Args:
        include_visualizations: Whether to generate visualization files
        
    Returns:
        Tuple of (analysis_results, visualization_paths) or None if failed
    """
    result = fcapture_and_process_screenshot()
    if not result:
        return None
        
    screenshot_path, analysis = result
    
    visualization_paths = {}
    if include_visualizations:
        try:
            # Import visualization functions from main module
            try:
                from . import m_Card_Processing  # Relative import for module usage
            except ImportError:
                import m_Card_Processing  # Direct import for standalone execution
            
            # Initialize detectors for visualizations
            width_detector = m_Card_Processing.BoundaryCoordinator()  
            avatar_detector = m_Card_Processing.CardAvatarDetector()
            boundary_detector = m_Card_Processing.CardBoundaryDetector()
            
            # Width visualization
            if analysis.get('width_detected'):
                width_vis = width_detector.create_width_visualization(screenshot_path)
                visualization_paths['width'] = width_vis
                
            # Avatar visualization  
            if analysis.get('avatars_detected'):
                avatar_vis = avatar_detector.create_visualization(screenshot_path) 
                visualization_paths['avatars'] = avatar_vis
                
            # Card boundary visualization
            if analysis.get('cards_detected'):
                card_vis = boundary_detector.create_card_visualization(screenshot_path)
                visualization_paths['cards'] = card_vis
                
        except Exception as e:
            print(f"⚠️  Visualization generation failed: {e}")
            
    return analysis, visualization_paths


def fis_screenshot_available() -> bool:
    """
    Check if screenshot capture functionality is available
    
    Returns:
        True if screenshot capture is available, False otherwise
    """
    return SCREENSHOT_AVAILABLE


if __name__ == "__main__":
    print("🎯 Screenshot Processor Module Test")
    print("=" * 40)
    
    if SCREENSHOT_AVAILABLE:
        print("✅ Screenshot capture available")
        
        # Test basic capture and processing
        result = fcapture_and_process_screenshot()
        if result:
            screenshot_path, analysis = result
            print(f"\n📊 Test Results:")
            print(f"   Screenshot: {os.path.basename(screenshot_path)}")
            print(f"   Cards detected: {analysis.get('cards_detected', 0)}")
        else:
            print("❌ Test failed - no results")
    else:
        print("❌ Screenshot capture not available")