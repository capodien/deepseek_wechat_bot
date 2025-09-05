#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Integration module for click determination with main bot workflow
"""

import sys
import os
import time

# Add current directory to Python path
current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, current_dir)

try:
    from TestRun.opencv_adaptive_detector import OpenCVAdaptiveDetector
    from TestRun.username_extractor import UsernameExtractor
    print("✅ Successfully imported enhanced detection modules")
except ImportError as e:
    print(f"❌ Import error: {e}")
    sys.exit(1)

class IntegratedClickDetermination:
    """
    Integrated click determination system that combines:
    - Avatar detection using OpenCV
    - Username extraction using EasyOCR
    - Early filtering against monitoring list
    - Click coordinate determination
    """
    
    def __init__(self):
        """Initialize the integrated system"""
        self.avatar_detector = OpenCVAdaptiveDetector()
        self.username_extractor = UsernameExtractor()
        self.monitoring_list = self.load_monitoring_list()
        print(f"🎯 IntegratedClickDetermination initialized with {len(self.monitoring_list)} monitored contacts")
    
    def load_monitoring_list(self):
        """Load monitoring list from names.txt"""
        try:
            with open('names.txt', 'r', encoding='utf-8') as f:
                contacts = [line.strip() for line in f if line.strip()]
            print(f"📋 Loaded {len(contacts)} contacts: {contacts}")
            return contacts
        except Exception as e:
            print(f"❌ Failed to load monitoring list: {e}")
            return []
    
    def determine_relevant_clicks(self, screenshot_path):
        """
        Main integration method that replaces the old red dot detection
        Returns: List of click coordinates for relevant contacts
        
        Process:
        1. Detect avatars in screenshot
        2. Extract usernames from avatar regions
        3. Filter against monitoring list
        4. Return click coordinates for relevant contacts only
        """
        print(f"\n🔍 Processing screenshot: {screenshot_path}")
        start_time = time.time()
        
        # Step 1: Detect avatars
        avatar_start = time.time()
        avatars = self.avatar_detector.detect_avatars(screenshot_path)
        avatar_duration = (time.time() - avatar_start) * 1000
        print(f"👥 Detected {len(avatars)} avatars in {avatar_duration:.1f}ms")
        
        if not avatars:
            print("❌ No avatars detected")
            return []
        
        # Step 2: Extract and filter usernames
        filter_start = time.time()
        relevant_contacts = []
        
        for i, avatar_info in enumerate(avatars):
            # Extract username from avatar using the avatar_info dict format
            username_result = self.username_extractor.extract_username_from_avatar(
                screenshot_path, avatar_info
            )
            
            if username_result['success']:
                username = username_result['username']
                
                # Check if username is in monitoring list
                if username in self.monitoring_list:
                    # Get click coordinates from avatar info
                    click_coords = avatar_info.get('click_center', avatar_info.get('avatar_center'))
                    
                    relevant_contacts.append({
                        'username': username,
                        'click_coords': click_coords,
                        'avatar_region': avatar_info.get('card_bounds'),
                        'confidence': username_result['confidence']
                    })
                    print(f"✅ Found relevant contact: {username} at {click_coords}")
                else:
                    print(f"⏭️  Skipped {username} (not monitored)")
            else:
                print(f"⚠️  OCR failed for avatar {i+1}: {username_result.get('error', 'Unknown error')}")
        
        filter_duration = (time.time() - filter_start) * 1000
        total_duration = (time.time() - start_time) * 1000
        
        print(f"🎯 Filtering completed in {filter_duration:.1f}ms")
        print(f"⚡ Total processing: {total_duration:.1f}ms")
        print(f"📊 Efficiency: {len(relevant_contacts)}/{len(avatars)} contacts are relevant")
        
        if relevant_contacts:
            # Return click coordinates for the first relevant contact
            # In future iterations, could implement priority logic
            first_contact = relevant_contacts[0]
            print(f"🎯 Returning click coordinates for: {first_contact['username']}")
            return first_contact['click_coords']
        else:
            print("❌ No relevant contacts found")
            return None
    
    def get_contact_info(self, screenshot_path):
        """
        Enhanced method that returns detailed contact information
        Useful for diagnostic purposes and advanced workflow
        """
        print(f"\n📊 Getting detailed contact info from: {screenshot_path}")
        
        # Detect avatars
        avatars = self.avatar_detector.detect_avatars(screenshot_path)
        
        contact_info = []
        for i, avatar_info in enumerate(avatars):
            # Extract username using avatar_info dict format
            username_result = self.username_extractor.extract_username_from_avatar(
                screenshot_path, avatar_info
            )
            
            if username_result['success']:
                username = username_result['username']
                confidence = username_result['confidence']
            else:
                username = f"EXTRACTION_FAILED_{i+1}"
                confidence = 0.0
            
            # Determine if monitored
            is_monitored = username in self.monitoring_list
            
            # Get click coordinates
            click_coords = avatar_info.get('click_center', avatar_info.get('avatar_center'))
            
            contact_info.append({
                'index': i + 1,
                'username': username,
                'is_monitored': is_monitored,
                'click_coords': click_coords,
                'avatar_region': avatar_info.get('card_bounds'),
                'confidence': confidence
            })
        
        return contact_info
    
    def integration_test(self, screenshot_path):
        """Test the integration with detailed output"""
        print(f"\n🧪 Integration Test Starting...")
        print(f"📷 Screenshot: {screenshot_path}")
        print(f"📋 Monitoring: {self.monitoring_list}")
        
        # Test the main method
        click_coords = self.determine_relevant_clicks(screenshot_path)
        
        # Get detailed info
        contact_info = self.get_contact_info(screenshot_path)
        
        print(f"\n📊 Integration Test Results:")
        print(f"   Total contacts detected: {len(contact_info)}")
        print(f"   Monitoring list size: {len(self.monitoring_list)}")
        
        relevant_count = sum(1 for c in contact_info if c['is_monitored'])
        print(f"   Relevant contacts: {relevant_count}")
        
        if click_coords:
            print(f"   Recommended click: {click_coords}")
        else:
            print(f"   Recommended action: No clicks needed (no relevant contacts)")
        
        return {
            'click_coords': click_coords,
            'contact_info': contact_info,
            'total_detected': len(contact_info),
            'relevant_count': relevant_count
        }

def test_integration():
    """Test the integrated click determination system"""
    print("🚀 Testing Integrated Click Determination System")
    
    # Initialize the system
    integrated_system = IntegratedClickDetermination()
    
    # Test with diagnostic screenshot
    test_screenshot = "pic/screenshots/diagnostic_test_20250904_120220.png"
    
    if not os.path.exists(test_screenshot):
        print(f"❌ Test screenshot not found: {test_screenshot}")
        print("💡 Run the diagnostic server first to generate test screenshots")
        return
    
    # Run integration test
    results = integrated_system.integration_test(test_screenshot)
    
    print(f"\n✅ Integration test completed successfully!")
    return results

if __name__ == "__main__":
    test_integration()