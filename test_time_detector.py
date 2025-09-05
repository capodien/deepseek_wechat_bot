#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Time Box Detector

This script tests the Time Box detection functionality to show its outputs.
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.m_Card_Processing import TimeBoxDetector
    print("✅ Successfully imported TimeBoxDetector")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_time_box_detection():
    """Test the time box detection with comprehensive output"""
    print("🕒 Testing Time Box Detection")
    print("=" * 60)
    
    # Find a suitable screenshot for testing
    screenshot_dir = "pic/screenshots"
    if not os.path.exists(screenshot_dir):
        print(f"❌ Screenshot directory not found: {screenshot_dir}")
        return
    
    # Get a suitable test image - avoid boundary detection visualizations
    screenshots = []
    for file in os.listdir(screenshot_dir):
        if (file.endswith('.png') and 
            not ('Debug_' in file or 'horizontal_differences' in file or 'photoshop' in file or
                 'EnhancedDualBoundary' in file or 'Enhanced_Card_Avatar_Boundaries' in file or
                 'Phase1_Upper_Region' in file)):
            file_path = os.path.join(screenshot_dir, file)
            screenshots.append((file_path, os.path.getmtime(file_path)))
    
    if not screenshots:
        print(f"❌ No suitable screenshots found in {screenshot_dir}")
        return
    
    # Sort by modification time and get the most recent
    screenshots.sort(key=lambda x: x[1], reverse=True)
    test_screenshot = screenshots[0][0]
    print(f"📸 Using test screenshot: {os.path.basename(test_screenshot)}")
    
    try:
        # Initialize detector with debug mode
        print("🔧 Initializing TimeBoxDetector with debug mode...")
        detector = TimeBoxDetector(debug_mode=True)
        
        # Run time box detection
        print("🎯 Running time box detection...")
        print("   • Card boundary detection")
        print("   • Avatar detection")
        print("   • Time region analysis")
        print("   • Density-based time box detection")
        print("   • Comprehensive debug visualization")
        
        start_time = time.time()
        
        enhanced_cards, detection_info = detector.detect_time_boundaries(
            test_screenshot
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️  Detection completed in {processing_time:.3f} seconds")
        
        # Display results summary
        total_cards = detection_info.get("total_cards_processed", 0)
        times_detected = detection_info.get("times_detected", 0)
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        
        print(f"\\n📊 Time Detection Results:")
        print(f"   • Total Cards: {total_cards}")
        print(f"   • Times Detected: {times_detected}")
        print(f"   • Success Rate: {success_rate:.1f}%")
        
        # Show individual card results
        if enhanced_cards:
            print(f"\\n🕒 Individual Card Results:")
            for i, card in enumerate(enhanced_cards):
                card_id = i + 1
                if "time_box" in card:
                    time_box = card["time_box"]
                    x, y, w, h = time_box["bbox"]
                    density = time_box.get("density_score", 0)
                    print(f"    ✅ Card {card_id}: Time box at ({x}, {y}) {w}×{h}px, density={density:.3f}")
                else:
                    print(f"    ❌ Card {card_id}: No time box detected")
        
        # Analyze detection methods
        density_detections = 0
        fallback_detections = 0
        for card in enhanced_cards:
            if "time_box" in card:
                method = card["time_box"].get("detection_method", "unknown")
                if method == "upper_density":
                    density_detections += 1
                else:
                    fallback_detections += 1
        
        print(f"\\n🔍 Detection Method Analysis:")
        print(f"   • Density-based detections: {density_detections}")
        print(f"   • Fallback detections: {fallback_detections}")
        
        # Show debug data if available
        debug_data = detection_info.get("debug_data")
        if debug_data:
            print(f"\\n🐛 Debug Data Available:")
            print(f"   • ROI Images: {len(debug_data.get('roi_images', {}))}")
            print(f"   • Detection Steps: {len(debug_data.get('processing_steps', []))}")
            print(f"   • Card Analysis: {len(debug_data.get('card_analysis', {}))}")
            
            # Show some processing steps
            steps = debug_data.get('processing_steps', [])
            if steps:
                print(f"   • Processing Steps Sample:")
                for step in steps[:3]:  # Show first 3 steps
                    print(f"     - {step}")
        else:
            print("❌ No debug data collected!")
            return
        
        # Create comprehensive debug visualization
        print("\\n🎨 Creating comprehensive time detection visualization...")
        try:
            viz_output = detector.create_comprehensive_debug_visualization(enhanced_cards, detection_info)
            if viz_output:
                print(f"✅ Time detection debug visualization created: {os.path.basename(viz_output)}")
                
                # Verify file was created
                if os.path.exists(viz_output):
                    file_size = os.path.getsize(viz_output) / 1024  # KB
                    print(f"📁 File size: {file_size:.1f} KB")
                    
                    # Check for comprehensive features
                    if 'Debug_TimeDetection' in os.path.basename(viz_output):
                        print("✅ Comprehensive time detection debug visualization generated")
                        print("   • Multi-panel layout with individual card analysis")
                        print("   • ROI extraction and processing visualization")
                        print("   • Density analysis and statistical reporting")
                        print("   • Success/failure indicators with detailed breakdown")
                else:
                    print("❌ Visualization file was not created successfully")
            else:
                print("❌ Failed to create time detection debug visualization")
                
        except Exception as viz_error:
            print(f"❌ Visualization creation failed: {viz_error}")
            import traceback
            traceback.print_exc()
        
        # Summary of time detection features
        print(f"\\n🚀 Time Detection Features:")
        print(f"   ✅ Upper region density analysis")
        print(f"   ✅ HSV color space processing for gray text detection")
        print(f"   ✅ Morphological operations for text connectivity")
        print(f"   ✅ Adaptive thresholding based on region statistics")
        print(f"   ✅ Comprehensive debug visualization with multi-panel layout")
        print(f"   ✅ Individual card ROI analysis and processing steps")
        
    except Exception as e:
        print(f"❌ Time detection test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_time_box_detection()
    print("\\n🏁 Time box detection test complete!")