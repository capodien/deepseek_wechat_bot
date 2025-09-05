#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log_Diagnostic_Console (LDC) Logger Module
Provides structured JSON logging for all diagnostic events with visual output support
"""

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List
import threading
from queue import Queue
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class DiagnosticLogger:
    """
    Thread-safe JSON logger for diagnostic events with visual annotation support
    """
    
    def __init__(self, log_dir: str = "diagnostic_logs", max_file_size_mb: int = 50):
        """
        Initialize the diagnostic logger
        
        Args:
            log_dir: Directory to store log files
            max_file_size_mb: Maximum size of each log file in MB before rotation
        """
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories for different types of outputs
        self.screenshots_dir = self.log_dir / "screenshots"
        self.overlays_dir = self.log_dir / "overlays"
        self.screenshots_dir.mkdir(exist_ok=True)
        self.overlays_dir.mkdir(exist_ok=True)
        
        self.max_file_size = max_file_size_mb * 1024 * 1024  # Convert to bytes
        self.current_log_file = None
        self.log_queue = Queue()
        self.lock = threading.Lock()
        
        # Start background writer thread
        self.writer_thread = threading.Thread(target=self._writer_loop, daemon=True)
        self.writer_thread.start()
        
        # Performance metrics
        self.metrics = {
            "total_events": 0,
            "errors": 0,
            "warnings": 0,
            "avg_processing_time": 0
        }
    
    def _get_current_log_file(self) -> Path:
        """Get or create current log file with rotation support"""
        today = datetime.now().strftime("%Y-%m-%d")
        base_filename = f"diagnostic_{today}"
        
        # Find existing files for today
        existing_files = sorted(self.log_dir.glob(f"{base_filename}_*.json"))
        
        if existing_files:
            latest_file = existing_files[-1]
            if latest_file.stat().st_size < self.max_file_size:
                return latest_file
            else:
                # Need to rotate
                counter = len(existing_files) + 1
        else:
            counter = 1
        
        new_filename = self.log_dir / f"{base_filename}_{counter:03d}.json"
        # Initialize with empty array
        with open(new_filename, 'w') as f:
            json.dump([], f)
        
        return new_filename
    
    def _writer_loop(self):
        """Background thread that writes log entries to file"""
        while True:
            try:
                entry = self.log_queue.get(timeout=1)
                if entry is None:  # Shutdown signal
                    break
                
                with self.lock:
                    log_file = self._get_current_log_file()
                    
                    # Read existing entries
                    with open(log_file, 'r') as f:
                        entries = json.load(f)
                    
                    # Append new entry
                    entries.append(entry)
                    
                    # Write back
                    with open(log_file, 'w') as f:
                        json.dump(entries, f, indent=2, ensure_ascii=False)
                    
                    # Update metrics
                    self.metrics["total_events"] += 1
                    if entry["status"] == "error":
                        self.metrics["errors"] += 1
                    elif entry["status"] == "warning":
                        self.metrics["warnings"] += 1
                        
            except Exception as e:
                print(f"Error in logger writer thread: {e}")
    
    def log_event(self, 
                  event_type: str,
                  module: str,
                  data: Dict[str, Any],
                  screenshot_path: Optional[str] = None,
                  status: str = "success",
                  metrics: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Log a diagnostic event
        
        Args:
            event_type: Type of event (screenshot, detection, ocr, click, error, etc.)
            module: Source module name
            data: Event-specific data
            screenshot_path: Path to associated screenshot
            status: success, warning, or error
            metrics: Performance metrics for this event
        
        Returns:
            The complete log entry that was created
        """
        timestamp = datetime.now().isoformat()
        
        # Handle screenshot if provided
        visual_output = None
        if screenshot_path and os.path.exists(screenshot_path):
            # Copy screenshot to our directory with timestamp
            screenshot_filename = f"{event_type}_{timestamp.replace(':', '-')}.png"
            saved_screenshot = self.screenshots_dir / screenshot_filename
            
            # Copy original
            import shutil
            shutil.copy2(screenshot_path, saved_screenshot)
            visual_output = str(saved_screenshot.relative_to(self.log_dir))
        
        # Create log entry
        entry = {
            "timestamp": timestamp,
            "event_type": event_type,
            "module": module,
            "data": data,
            "visual_output": visual_output,
            "metrics": metrics or {},
            "status": status
        }
        
        # Queue for writing
        self.log_queue.put(entry)
        
        return entry
    
    def create_visual_overlay(self,
                            image_path: str,
                            detections: List[Dict[str, Any]],
                            event_type: str) -> str:
        """
        Create an annotated overlay image with detection results
        
        Args:
            image_path: Path to original image
            detections: List of detection results with coordinates and labels
            event_type: Type of detection for styling
        
        Returns:
            Path to the generated overlay image
        """
        # Load image
        if isinstance(image_path, str):
            image = cv2.imread(image_path)
        else:
            image = image_path
        
        if image is None:
            raise ValueError(f"Cannot load image from {image_path}")
        
        overlay = image.copy()
        timestamp = datetime.now().isoformat().replace(':', '-')
        
        # Define color schemes for different event types
        color_schemes = {
            "avatar_detection": {
                "box": (0, 255, 0),      # Green
                "center": (0, 0, 255),    # Red
                "text": (255, 255, 0),    # Yellow
                "arrow": (255, 0, 255)    # Magenta
            },
            "message_detection": {
                "box": (0, 255, 0),       # Green
                "dot": (0, 0, 255),       # Red
                "text": (255, 255, 255),  # White
                "confidence": (0, 255, 255) # Cyan
            },
            "ocr": {
                "box": (255, 0, 0),       # Blue
                "text": (255, 255, 0),    # Yellow
                "confidence": (0, 255, 0)  # Green
            },
            "click": {
                "target": (0, 0, 255),    # Red
                "path": (0, 255, 0)       # Green
            }
        }
        
        colors = color_schemes.get(event_type, color_schemes["avatar_detection"])
        
        # Draw detections
        for i, detection in enumerate(detections):
            if "bbox" in detection:
                x1, y1, x2, y2 = detection["bbox"]
                # Draw bounding box
                cv2.rectangle(overlay, (x1, y1), (x2, y2), colors["box"], 2)
                
                # Draw label
                label = detection.get("label", f"#{i+1}")
                confidence = detection.get("confidence", None)
                
                if confidence:
                    text = f"{label} ({confidence:.2f})"
                else:
                    text = label
                
                # Add text background for visibility
                (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(overlay, (x1, y1 - text_height - 4), 
                            (x1 + text_width + 4, y1), (0, 0, 0), -1)
                cv2.putText(overlay, text, (x1 + 2, y1 - 2), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors["text"], 2)
            
            if "center" in detection:
                cx, cy = detection["center"]
                # Draw center point
                cv2.circle(overlay, (cx, cy), 5, colors["center"], -1)
                cv2.circle(overlay, (cx, cy), 7, colors["box"], 2)
            
            if "click_pos" in detection:
                cx, cy = detection["click_pos"]
                # Draw click target
                cv2.drawMarker(overlay, (cx, cy), colors["box"], 
                             cv2.MARKER_CROSS, 20, 2)
                
                # Draw arrow from center to click position if both exist
                if "center" in detection:
                    start = detection["center"]
                    end = detection["click_pos"]
                    cv2.arrowedLine(overlay, start, end, colors["arrow"], 2)
        
        # Add event info overlay
        info_text = [
            f"Event: {event_type}",
            f"Time: {datetime.now().strftime('%H:%M:%S')}",
            f"Detections: {len(detections)}"
        ]
        
        y_offset = 30
        for text in info_text:
            cv2.putText(overlay, text, (10, y_offset), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            y_offset += 30
        
        # Save overlay
        overlay_filename = f"overlay_{event_type}_{timestamp}.png"
        overlay_path = self.overlays_dir / overlay_filename
        cv2.imwrite(str(overlay_path), overlay)
        
        return str(overlay_path.relative_to(self.log_dir))
    
    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get summary of logging metrics"""
        return {
            "metrics": self.metrics,
            "log_directory": str(self.log_dir),
            "current_log": str(self._get_current_log_file().name) if self.current_log_file else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def close(self):
        """Shutdown the logger gracefully"""
        self.log_queue.put(None)
        self.writer_thread.join(timeout=5)


# Singleton instance for global access
_logger_instance = None

def get_logger() -> DiagnosticLogger:
    """Get or create the singleton logger instance"""
    global _logger_instance
    if _logger_instance is None:
        _logger_instance = DiagnosticLogger()
    return _logger_instance


# Convenience functions for common logging operations
def log_screenshot(module: str, screenshot_path: str, **kwargs) -> Dict[str, Any]:
    """Log a screenshot capture event"""
    return get_logger().log_event(
        event_type="screenshot",
        module=module,
        screenshot_path=screenshot_path,
        data=kwargs
    )


def log_detection(module: str, detections: List[Dict], image_path: str = None, **kwargs) -> Dict[str, Any]:
    """Log a detection event with visual overlay"""
    logger = get_logger()
    
    # Create overlay if image provided
    overlay_path = None
    if image_path and detections:
        try:
            overlay_path = logger.create_visual_overlay(image_path, detections, "avatar_detection")
        except Exception as e:
            print(f"Failed to create overlay: {e}")
    
    return logger.log_event(
        event_type="detection",
        module=module,
        data={"detections": detections, "overlay": overlay_path, **kwargs},
        screenshot_path=image_path
    )


def log_ocr(module: str, text: str, confidence: float = None, image_path: str = None, **kwargs) -> Dict[str, Any]:
    """Log an OCR event"""
    data = {"text": text, **kwargs}
    if confidence is not None:
        data["confidence"] = confidence
    
    return get_logger().log_event(
        event_type="ocr",
        module=module,
        data=data,
        screenshot_path=image_path
    )


def log_error(module: str, error: Exception, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """Log an error event"""
    return get_logger().log_event(
        event_type="error",
        module=module,
        data={
            "error_type": type(error).__name__,
            "error_message": str(error),
            "context": context or {}
        },
        status="error"
    )


def log_performance(module: str, operation: str, duration_ms: float, **kwargs) -> Dict[str, Any]:
    """Log a performance metric"""
    return get_logger().log_event(
        event_type="performance",
        module=module,
        data={"operation": operation, **kwargs},
        metrics={"duration_ms": duration_ms}
    )


if __name__ == "__main__":
    # Test the logger
    logger = get_logger()
    
    # Test logging various events
    print("Testing diagnostic logger...")
    
    # Log a simple event
    entry = logger.log_event(
        event_type="test",
        module="diagnostic_logger",
        data={"message": "Test event"},
        status="success"
    )
    print(f"Created log entry: {entry['timestamp']}")
    
    # Test convenience functions
    log_performance("test_module", "initialization", 125.5, items_processed=100)
    
    # Get metrics
    print(f"Metrics: {logger.get_metrics_summary()}")
    
    # Cleanup
    logger.close()
    print("Logger test completed!")