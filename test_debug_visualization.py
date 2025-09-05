#!/usr/bin/env python3
"""
Test script to generate debug visualization for time detection analysis
"""

import os
import sys
sys.path.append('modules')
from m_Card_Processing import TimeBoxDetector

def test_debug_visualization():
    """Test the debug visualization with the latest screenshot"""
    
    print("🔍 Time Detection Debug Visualization Test")
    print("=" * 50)
    
    # Find the latest screenshot
    screenshot_dir = "pic/screenshots"
    if not os.path.exists(screenshot_dir):
        print(f"❌ Screenshot directory not found: {screenshot_dir}")
        return
    
    # Get latest WeChat screenshot
    screenshots = [f for f in os.listdir(screenshot_dir) if f.endswith('_WeChat.png')]
    if not screenshots:
        print(f"❌ No WeChat screenshots found in {screenshot_dir}")
        # Try to find any screenshot file
        all_files = os.listdir(screenshot_dir)
        print(f"   Available files: {all_files}")
        return
    
    latest_screenshot = sorted(screenshots)[-1]
    screenshot_path = os.path.join(screenshot_dir, latest_screenshot)
    
    print(f"📸 Using screenshot: {latest_screenshot}")
    
    # Initialize time detector
    time_detector = TimeBoxDetector()
    
    # Generate debug visualization
    print(f"\n🎨 Generating debug visualization...")
    debug_viz_filename = time_detector.create_debug_visualization(screenshot_path)
    
    if debug_viz_filename:
        debug_path = os.path.join(screenshot_dir, debug_viz_filename)
        print(f"\n✅ Debug visualization complete!")
        print(f"   📁 Saved to: {debug_path}")
        print(f"   🔍 Open this file to see detailed analysis of Cards 1 & 2 failures")
        
        # Print summary
        print(f"\n📋 Next Steps:")
        print(f"   1. Open the debug visualization: {debug_viz_filename}")
        print(f"   2. Examine the red search regions for failed cards")
        print(f"   3. Look at the ROI (Region of Interest) images to see what the algorithm sees")
        print(f"   4. Check the column projection histograms to understand detection thresholds")
        print(f"   5. Use the printed debug info to identify specific failure reasons")
    else:
        print(f"❌ Debug visualization generation failed")

if __name__ == "__main__":
    test_debug_visualization()