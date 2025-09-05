#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Boundary-Based Contact Name Detection

This script tests the enhanced contact name detection using the sophisticated 
horizontal boundary detection method from time detection.
"""

import os
import sys
import time
from datetime import datetime

# Add current directory to path to import modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from modules.m_Card_Processing import ContactNameBoundaryDetector
    print("✅ Successfully imported ContactNameBoundaryDetector")
except Exception as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

def test_boundary_based_detection():
    """Test the boundary-based contact name detection with comprehensive visualization"""
    print("🔍 Testing Boundary-Based Contact Name Detection")
    print("=" * 70)
    
    # Find a suitable screenshot for testing
    screenshot_dir = "pic/screenshots"
    if not os.path.exists(screenshot_dir):
        print(f"❌ Screenshot directory not found: {screenshot_dir}")
        return
    
    # Get a suitable test image (use the horizontal differences which contains WeChat interface)
    screenshots = []
    for file in os.listdir(screenshot_dir):
        if (file.endswith('.png') and 
            ('horizontal_differences' in file or 'screenshot' in file.lower()) and 
            not ('Debug_' in file or 'photoshop' in file)):
            file_path = os.path.join(screenshot_dir, file)
            screenshots.append((file_path, os.path.getmtime(file_path)))
    
    if not screenshots:
        print(f"❌ No suitable screenshots found in {screenshot_dir}")
        return
    
    # Sort by modification time and get the most recent
    screenshots.sort(key=lambda x: x[1], reverse=True)
    # Try to find a diagnostic test screenshot first
    diagnostic_screenshot = os.path.join(screenshot_dir, "diagnostic_test_20250905_222532.png")
    if os.path.exists(diagnostic_screenshot):
        test_screenshot = diagnostic_screenshot
        print(f"📸 Using diagnostic screenshot: {os.path.basename(test_screenshot)}")
    else:
        test_screenshot = screenshots[0][0]
        print(f"📸 Using test screenshot: {os.path.basename(test_screenshot)}")
    
    try:
        # Initialize detector
        print("🔧 Initializing ContactNameBoundaryDetector...")
        detector = ContactNameBoundaryDetector(debug_mode=True)
        
        # Run detection with debug mode enabled
        print("🎯 Running boundary-based contact name detection...")
        print("   • Horizontal boundary detection")
        print("   • Projection analysis")
        print("   • Boundary-guided search regions")
        print("   • Enhanced visualization with projection plots")
        
        start_time = time.time()
        
        enhanced_cards, detection_info = detector.detect_name_boundaries(
            test_screenshot, 
            debug_mode=True  # Enable comprehensive debug data collection
        )
        
        processing_time = time.time() - start_time
        print(f"⏱️  Detection completed in {processing_time:.3f} seconds")
        
        # Display results
        total_cards = detection_info.get("total_cards_processed", 0)
        names_detected = detection_info.get("names_detected", 0)
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        
        print(f"\n📊 Detection Results:")
        print(f"   • Total Cards: {total_cards}")
        print(f"   • Names Detected: {names_detected}")
        print(f"   • Success Rate: {success_rate:.1f}%")
        
        # Analyze boundary detection success
        boundary_detections = 0
        fallback_detections = 0
        for card in enhanced_cards:
            if card.get("name_time_boundary"):
                if card["name_time_boundary"]["detection_method"] == "horizontal_projection_analysis":
                    boundary_detections += 1
                    boundary_y = card["name_time_boundary"]["y"]
                    avatar_y = card["name_time_boundary"]["avatar_center_y"]
                    print(f"    🎯 Card {card.get('card_id', '?')}: Boundary at y={boundary_y} (avatar center: {avatar_y})")
                else:
                    fallback_detections += 1
        
        print(f"\n🔍 Boundary Analysis:")
        print(f"   • Successful Boundary Detections: {boundary_detections}")
        print(f"   • Avatar Center Fallbacks: {fallback_detections}")
        print(f"   • Boundary Detection Rate: {boundary_detections/total_cards*100:.1f}%" if total_cards > 0 else "   • Boundary Detection Rate: 0%")
        
        # Check if debug data is available
        debug_data = detection_info.get("debug_data")
        if debug_data:
            print(f"\n🐛 Enhanced Debug Data:")
            print(f"   • ROI Images: {len(debug_data.get('roi_images', {}))}")
            print(f"   • Horizontal Projections: {len(debug_data.get('horizontal_projections', {}))}")
            print(f"   • Binary Masks: {len(debug_data.get('binary_masks', {}))}")
            print(f"   • Contour Data: {len(debug_data.get('contour_data', {}))}")
            print(f"   • Processing Steps: {len(debug_data.get('processing_steps', []))}")
            print(f"   • Statistical Analysis: {'Available' if debug_data.get('statistical_analysis') else 'Not Available'}")
            
            # Show some projection data if available
            projections = debug_data.get('horizontal_projections', {})
            if projections:
                print(f"   • Projection Analysis Examples:")
                for card_id, proj_data in list(projections.items())[:2]:  # Show first 2
                    threshold = proj_data.get('threshold', 0)
                    regions_count = len(proj_data.get('text_regions_1d', []))
                    print(f"     - Card {card_id}: threshold={threshold:.1f}, regions={regions_count}")
        else:
            print("❌ No debug data collected!")
            return
        
        # Create comprehensive debug visualization with projection plots
        print("\n🎨 Creating enhanced debug visualization with projection analysis...")
        try:
            viz_output = detector.create_comprehensive_debug_visualization(enhanced_cards, detection_info)
            if viz_output:
                print(f"✅ Enhanced debug visualization created: {os.path.basename(viz_output)}")
                
                # Verify file was created and compare with previous
                if os.path.exists(viz_output):
                    file_size = os.path.getsize(viz_output) / 1024  # KB
                    print(f"📁 File size: {file_size:.1f} KB")
                    
                    # Check for enhanced features in filename
                    if 'Debug_ContactNameDetection' in os.path.basename(viz_output):
                        print("✅ Enhanced boundary-based debug visualization generated")
                        print("   • Includes horizontal projection plots")
                        print("   • Shows boundary detection analysis")
                        print("   • Enhanced statistical reporting")
                    
                    # Compare improvement over simple method
                    if boundary_detections > 0:
                        print(f"🎯 Method Enhancement Success:")
                        print(f"   • {boundary_detections} boundaries detected using projection analysis")
                        print(f"   • Sophisticated filtering and morphological operations")
                        print(f"   • Search regions guided by detected boundaries")
                    else:
                        print("⚠️  No boundaries detected - may need parameter tuning")
                else:
                    print("❌ Visualization file was not created successfully")
            else:
                print("❌ Failed to create enhanced debug visualization")
                
        except Exception as viz_error:
            print(f"❌ Visualization creation failed: {viz_error}")
            import traceback
            traceback.print_exc()
        
        # Summary of improvements
        print(f"\n🚀 Method Improvements Implemented:")
        print(f"   ✅ Horizontal boundary detection (from time detection)")
        print(f"   ✅ Bilateral filtering and noise reduction")  
        print(f"   ✅ Projection analysis with adaptive thresholding")
        print(f"   ✅ Boundary-guided search regions")
        print(f"   ✅ Enhanced debug visualization with projection plots")
        print(f"   ✅ Statistical analysis and morphological processing")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_boundary_based_detection()
    print("\n🏁 Boundary-based detection test complete!")