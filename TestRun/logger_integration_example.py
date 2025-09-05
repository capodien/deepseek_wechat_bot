#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Logger Integration Example
Demonstrates how to integrate diagnostic_logger with existing modules
"""

import sys
import os
import time
from pathlib import Path

# Add parent directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

from diagnostic.diagnostic_logs.diagnostic_logger import get_logger, log_detection, log_ocr, log_performance, log_error


def example_avatar_detection_with_logging():
    """
    Example of how to integrate logging with avatar detection
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Simulate taking a screenshot
        screenshot_path = "pic/screenshots/test_screenshot.png"
        logger.log_event(
            event_type="screenshot",
            module="example_avatar_detection",
            data={"action": "capture_wechat_window", "target": "contact_list"},
            screenshot_path=screenshot_path if os.path.exists(screenshot_path) else None
        )
        
        # Simulate detection results
        mock_detections = [
            {
                "bbox": [100, 150, 150, 200],
                "center": (125, 175),
                "click_pos": (130, 180),
                "label": "Contact #1",
                "confidence": 0.95
            },
            {
                "bbox": [100, 220, 150, 270],
                "center": (125, 245),
                "click_pos": (130, 250),
                "label": "Contact #2", 
                "confidence": 0.87
            }
        ]
        
        # Log detection with visual overlay
        log_detection(
            module="example_avatar_detection",
            detections=mock_detections,
            image_path=screenshot_path if os.path.exists(screenshot_path) else None,
            method="opencv_adaptive",
            algorithm="threshold_contour_analysis"
        )
        
        # Log performance
        duration = (time.time() - start_time) * 1000
        log_performance(
            module="example_avatar_detection",
            operation="avatar_detection",
            duration_ms=duration,
            detections_found=len(mock_detections),
            avg_confidence=sum(d["confidence"] for d in mock_detections) / len(mock_detections)
        )
        
        print(f"✅ Avatar detection completed: {len(mock_detections)} contacts found in {duration:.1f}ms")
        return mock_detections
        
    except Exception as e:
        log_error("example_avatar_detection", e, {"operation": "avatar_detection"})
        raise


def example_ocr_with_logging():
    """
    Example of how to integrate logging with OCR processing
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Simulate OCR processing
        screenshot_path = "pic/screenshots/test_chat.png"
        extracted_text = "这是一条测试消息\nHello World\n如何使用OCR"
        confidence = 0.92
        
        # Create mock text regions for overlay
        text_regions = [
            {
                "bbox": [50, 100, 200, 130],
                "text": "这是一条测试消息",
                "confidence": 0.95,
                "label": "Chinese Text"
            },
            {
                "bbox": [50, 140, 150, 170],
                "text": "Hello World",
                "confidence": 0.98,
                "label": "English Text"  
            },
            {
                "bbox": [50, 180, 180, 210],
                "text": "如何使用OCR",
                "confidence": 0.87,
                "label": "Chinese Text"
            }
        ]
        
        # Log OCR results
        log_ocr(
            module="example_ocr",
            text=extracted_text,
            confidence=confidence,
            image_path=screenshot_path if os.path.exists(screenshot_path) else None,
            text_regions=text_regions,
            language_detected=["zh", "en"],
            processing_method="easyocr"
        )
        
        # Create visual overlay for OCR
        if os.path.exists(screenshot_path):
            overlay_path = logger.create_visual_overlay(
                screenshot_path, 
                text_regions, 
                "ocr"
            )
            logger.log_event(
                event_type="visual_overlay",
                module="example_ocr", 
                data={"overlay_created": overlay_path},
                screenshot_path=overlay_path
            )
        
        # Log performance
        duration = (time.time() - start_time) * 1000
        log_performance(
            module="example_ocr",
            operation="text_extraction",
            duration_ms=duration,
            characters_extracted=len(extracted_text),
            text_regions_found=len(text_regions)
        )
        
        print(f"✅ OCR completed: {len(extracted_text)} characters extracted in {duration:.1f}ms")
        return extracted_text
        
    except Exception as e:
        log_error("example_ocr", e, {"operation": "text_extraction"})
        raise


def example_message_detection_with_logging():
    """
    Example of how to integrate logging with message detection
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Simulate message detection
        screenshot_path = "pic/screenshots/test_messages.png"
        
        # Mock message detections (red dot notifications)
        message_detections = [
            {
                "bbox": [300, 150, 320, 170],
                "center": (310, 160),
                "label": "New Message",
                "confidence": 0.94,
                "color_match": {"r": 255, "g": 0, "b": 0}
            },
            {
                "bbox": [300, 250, 320, 270], 
                "center": (310, 260),
                "label": "Unread Count: 2",
                "confidence": 0.89,
                "color_match": {"r": 240, "g": 10, "b": 5}
            }
        ]
        
        # Log message detection
        log_detection(
            module="example_message_detection",
            detections=message_detections,
            image_path=screenshot_path if os.path.exists(screenshot_path) else None,
            detection_method="color_threshold",
            red_threshold=200,
            total_scanned_pixels=1920*1080
        )
        
        # Log performance
        duration = (time.time() - start_time) * 1000
        log_performance(
            module="example_message_detection",
            operation="red_dot_detection",
            duration_ms=duration,
            messages_found=len(message_detections),
            scan_resolution="1920x1080"
        )
        
        print(f"✅ Message detection completed: {len(message_detections)} notifications found in {duration:.1f}ms")
        return message_detections
        
    except Exception as e:
        log_error("example_message_detection", e, {"operation": "message_detection"})
        raise


def example_click_action_with_logging():
    """
    Example of how to log GUI automation actions
    """
    logger = get_logger()
    start_time = time.time()
    
    try:
        # Simulate click action
        target_pos = (150, 200)
        
        logger.log_event(
            event_type="click",
            module="example_gui_automation",
            data={
                "action": "click_contact",
                "coordinates": target_pos,
                "button": "left",
                "duration": "single_click"
            },
            status="success"
        )
        
        # Simulate brief processing delay
        time.sleep(0.1)
        
        # Log performance
        duration = (time.time() - start_time) * 1000
        log_performance(
            module="example_gui_automation",
            operation="mouse_click",
            duration_ms=duration,
            coordinates=target_pos,
            success=True
        )
        
        print(f"✅ Click action completed at {target_pos} in {duration:.1f}ms")
        return True
        
    except Exception as e:
        log_error("example_gui_automation", e, {"operation": "click_action", "target": target_pos})
        raise


def example_complete_workflow():
    """
    Example of logging a complete workflow with multiple steps
    """
    print("🚀 Starting complete workflow example with logging...")
    logger = get_logger()
    workflow_start = time.time()
    
    # Log workflow start
    logger.log_event(
        event_type="workflow_start",
        module="example_complete_workflow",
        data={"workflow": "message_response_cycle", "steps": 4},
        status="success"
    )
    
    try:
        # Step 1: Avatar Detection
        print("\n📋 Step 1: Avatar Detection")
        avatars = example_avatar_detection_with_logging()
        
        # Step 2: Message Detection  
        print("\n📋 Step 2: Message Detection")
        messages = example_message_detection_with_logging()
        
        # Step 3: Click Action
        print("\n📋 Step 3: Click Action")
        click_success = example_click_action_with_logging()
        
        # Step 4: OCR Processing
        print("\n📋 Step 4: OCR Processing")
        extracted_text = example_ocr_with_logging()
        
        # Log workflow completion
        total_duration = (time.time() - workflow_start) * 1000
        logger.log_event(
            event_type="workflow_complete",
            module="example_complete_workflow",
            data={
                "workflow": "message_response_cycle",
                "steps_completed": 4,
                "results": {
                    "avatars_found": len(avatars),
                    "messages_found": len(messages), 
                    "click_successful": click_success,
                    "text_extracted": len(extracted_text)
                }
            },
            metrics={"total_duration_ms": total_duration},
            status="success"
        )
        
        print(f"\n✅ Complete workflow finished successfully in {total_duration:.1f}ms")
        
        # Print summary
        print(f"\n📊 Summary:")
        print(f"  • Avatars detected: {len(avatars)}")
        print(f"  • Messages found: {len(messages)}")  
        print(f"  • Text extracted: {len(extracted_text)} characters")
        print(f"  • Total time: {total_duration:.1f}ms")
        
        # Show logger metrics
        metrics = logger.get_metrics_summary()
        print(f"\n📈 Logger Metrics:")
        print(f"  • Total events logged: {metrics['metrics']['total_events']}")
        print(f"  • Errors: {metrics['metrics']['errors']}")
        print(f"  • Warnings: {metrics['metrics']['warnings']}")
        print(f"  • Log directory: {metrics['log_directory']}")
        
        return True
        
    except Exception as e:
        # Log workflow failure
        logger.log_event(
            event_type="workflow_error",
            module="example_complete_workflow", 
            data={
                "workflow": "message_response_cycle",
                "error": str(e),
                "failed_at": "unknown_step"
            },
            status="error"
        )
        print(f"\n❌ Workflow failed: {e}")
        raise


if __name__ == "__main__":
    print("🔧 Diagnostic Logger Integration Example")
    print("=" * 50)
    
    try:
        # Run complete workflow example
        success = example_complete_workflow()
        
        if success:
            print(f"\n🎉 Example completed successfully!")
            print(f"\n📂 Check the diagnostic_logs/ directory for:")
            print(f"  • JSON log files with all events")  
            print(f"  • screenshots/ folder with captured images")
            print(f"  • overlays/ folder with annotated visualizations")
            print(f"\n🖥️  Open diagnostic_viewer.html to view the logs")
            
    except Exception as e:
        print(f"\n💥 Example failed: {e}")
        
    finally:
        # Cleanup
        logger = get_logger()
        logger.close()
        print(f"\n🔄 Logger closed gracefully")