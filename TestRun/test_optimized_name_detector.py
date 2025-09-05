#!/usr/bin/env python3
"""
Test optimized Contact Name Boundary Detector performance
This test runs the full Card Processing module to get fresh data and test the name detector optimizations.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.m_Card_Processing import process_card_boundaries

def test_optimized_name_detector():
    """Test the optimized Contact Name Boundary Detector with fresh data"""
    
    print("🎯 Testing Optimized Contact Name Boundary Detector")
    print("=" * 60)
    print("📋 Applied Optimizations:")
    print("  ✅ Search region: Above avatars (not below)")  
    print("  ✅ White threshold: 155 (reduced from 180)")
    print("  ✅ Search margins: Increased for better coverage")
    print("  ✅ White pixel ratio: 0.15 (more selective)")
    print()
    
    # Test with the latest visualization file (it contains the source image data)
    screenshot_path = "pic/screenshots/20250905_210551_04_Enhanced_Card_Avatar_Boundaries_9cards.png"
    
    if not os.path.exists(screenshot_path):
        print(f"❌ Screenshot not found: {screenshot_path}")
        return
    
    print(f"📸 Using screenshot: {os.path.basename(screenshot_path)}")
    print()
    
    try:
        # Run full processing to get fresh results
        print("🔄 Running full card processing...")
        results = process_card_boundaries(screenshot_path, debug_mode=True, sections_to_run=[5])
        
        if not results:
            print("❌ No results from card processing")
            return
            
        # Extract name detection results  
        name_results = results.get('name_boundary_detection')
        if not name_results:
            print("❌ No name boundary detection results")
            return
            
        enhanced_cards, detection_info = name_results
        
        print("📊 OPTIMIZATION TEST RESULTS")
        print("=" * 40)
        print(f"  • Cards processed: {detection_info.get('total_cards_processed', 'Unknown')}")
        print(f"  • Names detected: {detection_info.get('names_detected', 'Unknown')}")
        
        success_rate = detection_info.get('detection_success_rate', 0)
        print(f"  • Success rate: {success_rate:.1%}")
        
        if success_rate > 0.11:  # Better than original 11% (1/9)
            improvement = (success_rate - 0.11) / 0.11 * 100
            print(f"  • Improvement: +{improvement:.0f}% over original")
            print(f"  • Status: ✅ OPTIMIZATIONS SUCCESSFUL")
        else:
            print(f"  • Status: ⚠️ NEEDS FURTHER OPTIMIZATION")
        
        print(f"  • Detection method: {detection_info.get('detection_method', 'Unknown')}")
        print()
        
        # Show detailed results
        if enhanced_cards:
            print("🔍 DETAILED DETECTION RESULTS")
            print("=" * 35)
            for i, card in enumerate(enhanced_cards, 1):
                if card.get('name_boundary'):
                    nb = card['name_boundary']
                    bbox = nb['bbox']
                    conf = nb.get('confidence', 0)
                    method = nb.get('detection_method', 'unknown')
                    print(f"  📝 Card {i}: {bbox[2]}×{bbox[3]}px at ({bbox[0]}, {bbox[1]}) - conf: {conf:.2f} ({method})")
                else:
                    print(f"  ❌ Card {i}: No name boundary detected")
        
        # Check for visualization files
        viz_files = results.get('visualization_files', {})
        name_viz = viz_files.get('name_boundary_detection')
        if name_viz:
            print()
            print(f"🎨 Visualization created: {os.path.basename(name_viz)}")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_optimized_name_detector()