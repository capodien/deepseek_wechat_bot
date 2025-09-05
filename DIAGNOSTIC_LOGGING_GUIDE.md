# 🔍 Log_Diagnostic_Console (LDC) System Guide

## Overview

The new diagnostic logging system provides a **server-less, log-based approach** to diagnosing WeChat bot issues. Instead of running a web server, the system generates structured JSON logs that can be analyzed offline using a simple HTML viewer.

## Benefits Over Server-Based Approach

✅ **Simpler Deployment** - No web server required  
✅ **Better for Production** - Continuous monitoring without intervention  
✅ **Easy Sharing** - Send log files to others for analysis  
✅ **Offline Analysis** - Review logs without running the bot  
✅ **Historical Data** - Analyze past issues anytime  
✅ **Reduced Complexity** - No API synchronization or CORS issues  

## Quick Start

### 1. Basic Usage

```python
from diagnostic_logger import get_logger, log_detection, log_ocr, log_performance

# Simple logging
logger = get_logger()
logger.log_event(
    event_type="screenshot",
    module="my_module", 
    data={"action": "capture_window"},
    screenshot_path="path/to/screenshot.png"
)

# Convenience functions
log_detection("module_name", detections_list, "screenshot.png")
log_ocr("module_name", extracted_text, confidence=0.95)
log_performance("module_name", "operation_name", duration_ms=150.5)
```

### 2. View Logs

1. Open `diagnostic_viewer.html` in any web browser
2. Drag & drop JSON log files from `diagnostic_logs/` directory
3. Filter, search, and analyze events visually

### 3. Run Example

```bash
python3 TestRun/logger_integration_example.py
```

## System Architecture

```
Bot Modules → diagnostic_logger.py → JSON Files → diagnostic_viewer.html
     ↓              ↓                    ↓              ↓
  Log Events    Generate Logs      Store to Disk   View & Analyze
```

## File Structure

```
diagnostic_logs/
├── diagnostic_2025-09-04_001.json    # Main log file
├── screenshots/                      # Original screenshots  
└── overlays/                        # Annotated visualizations
```

## Log Entry Format

```json
{
  "timestamp": "2025-09-04T19:48:31.021906",
  "event_type": "detection|ocr|screenshot|click|error|performance",
  "module": "source_module_name",
  "data": {
    "detections": [...],
    "method": "opencv_adaptive",
    "custom_fields": "any_value"
  },
  "visual_output": "overlays/overlay_detection_2025-09-04T19-48-31.png",
  "metrics": {
    "duration_ms": 250.5,
    "accuracy": 0.95,
    "items_processed": 5
  },
  "status": "success|warning|error"
}
```

## Integration Examples

### Avatar Detection

```python
def detect_avatars_with_logging():
    start_time = time.time()
    
    try:
        # Your detection logic
        detections = find_avatars(screenshot_path)
        
        # Log results with visual overlay
        log_detection(
            module="avatar_detection",
            detections=detections,
            image_path=screenshot_path,
            method="opencv_adaptive",
            threshold_value=127
        )
        
        # Log performance
        duration = (time.time() - start_time) * 1000
        log_performance(
            module="avatar_detection",
            operation="detect_contacts", 
            duration_ms=duration,
            detections_found=len(detections)
        )
        
        return detections
        
    except Exception as e:
        log_error("avatar_detection", e, {"operation": "detect_contacts"})
        raise
```

### OCR Processing

```python
def extract_text_with_logging():
    try:
        # OCR logic
        text = ocr_reader.readtext(image)
        confidence = calculate_confidence(text)
        
        # Log OCR results
        log_ocr(
            module="ocr_processor",
            text=extracted_text,
            confidence=confidence,
            image_path=screenshot_path,
            language_detected=["zh", "en"]
        )
        
        return text
        
    except Exception as e:
        log_error("ocr_processor", e)
        raise
```

### Message Detection

```python
def detect_messages_with_logging():
    # Detection logic
    red_dots = find_red_dots(screenshot)
    
    # Log with custom data
    log_detection(
        module="message_detector",
        detections=red_dots,
        image_path=screenshot,
        red_threshold=(200, 10, 10),
        scan_area="full_window"
    )
    
    return red_dots
```

## Visual Overlay Features

The system automatically creates annotated images with:

🟢 **Green boxes** around detected objects  
🔴 **Red dots** marking centers  
🔵 **Blue crosses** showing click positions  
➡️ **Arrows** connecting centers to targets  
🏷️ **Labels** with confidence scores  
📊 **Metrics** overlaid on images  

## Viewer Features

### Filtering & Search
- **Event Type**: screenshot, detection, ocr, click, error, performance
- **Status**: success, warning, error  
- **Module**: Filter by source module
- **Search**: Full-text search in event data

### Metrics Dashboard
- Total events processed
- Success/warning/error counts
- Average processing duration
- Real-time statistics

### Timeline View
- Chronological event display
- Color-coded by status
- Click to view detailed information
- Visual thumbnails for screenshots

### Export Options
- Export filtered results as JSON
- Share specific event subsets
- Generate reports for analysis

## Performance Considerations

### Log Rotation
- Automatic daily log rotation
- Configurable file size limits (default: 50MB)
- Compressed historical logs

### Background Writing
- Thread-safe asynchronous logging
- Queue-based to prevent blocking
- Automatic cleanup on shutdown

### Resource Usage
- Minimal memory footprint (~5-10MB)
- Fast JSON serialization
- Efficient image overlay generation

## Production Usage

### Integration Steps

1. **Add to existing modules**:
```python
from diagnostic_logger import log_detection, log_ocr, log_error

# Replace print statements and manual logging
# with structured diagnostic events
```

2. **Configure log retention**:
```python
logger = DiagnosticLogger(
    log_dir="production_logs",
    max_file_size_mb=100  # Larger files for production
)
```

3. **Set up monitoring**:
- Daily log review process
- Automated error alerting
- Performance trend analysis

### Deployment Considerations

- **Log Directory**: Ensure writable permissions
- **Disk Space**: Monitor log directory size
- **Rotation**: Configure appropriate retention policies
- **Access**: Secure log files in production environments

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
ModuleNotFoundError: No module named 'cv2'
pip install opencv-python
```

**Permission Errors**:
```bash
mkdir -p diagnostic_logs
chmod 755 diagnostic_logs/
```

**Large Log Files**:
- Reduce `max_file_size_mb` parameter
- Implement more aggressive log rotation
- Use log filtering in viewer

### Debug Mode

```python
# Enable verbose logging
logger = get_logger()
logger.log_event(
    event_type="debug",
    module="troubleshooting",
    data={"debug_info": "detailed_context"},
    status="success"
)
```

## Migration from Server-Based System

### Advantages of Migration

| Server-Based | Log-Based |
|--------------|-----------|
| Real-time interaction | Historical analysis |
| Network dependencies | Offline capable |
| Complex setup | Simple files |
| Development focused | Production ready |
| API synchronization | Direct logging |

### Hybrid Approach

Keep both systems:
- **Server-based** for development and real-time debugging
- **Log-based** for production monitoring and historical analysis

```python
# Environment-based selection
if os.getenv("ENVIRONMENT") == "development":
    use_server_diagnostics()
else:
    use_log_based_diagnostics()
```

## Best Practices

### 1. Structured Logging
- Use consistent event types
- Include relevant context in data field
- Log both success and failure cases

### 2. Performance Monitoring
- Always log timing information
- Track resource usage metrics
- Monitor error rates and patterns

### 3. Visual Verification
- Generate overlays for all computer vision tasks
- Include confidence scores and detection details
- Save both original and annotated images

### 4. Error Handling
- Log exceptions with full context
- Include recovery attempts and outcomes
- Track error patterns for system improvements

### 5. Data Privacy
- Sanitize sensitive information from logs
- Use relative paths for screenshots
- Consider log encryption for sensitive environments

## Advanced Usage

### Custom Event Types

```python
logger.log_event(
    event_type="custom_workflow",
    module="advanced_module",
    data={
        "workflow_step": "user_verification",
        "verification_method": "face_recognition",
        "confidence_threshold": 0.85
    },
    metrics={
        "processing_time_ms": 1250.5,
        "faces_detected": 1,
        "verification_confidence": 0.92
    },
    status="success"
)
```

### Batch Operations

```python
# Log multiple related events
events = []
for detection in detections:
    events.append({
        "event_type": "individual_detection",
        "module": "batch_processor",
        "data": detection,
        "status": "success"
    })

# Efficient batch logging
for event in events:
    logger.log_event(**event)
```

### Performance Profiling

```python
import functools

def log_performance_decorator(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            duration = (time.time() - start_time) * 1000
            log_performance(
                module=func.__module__,
                operation=func.__name__,
                duration_ms=duration,
                success=True
            )
            return result
        except Exception as e:
            duration = (time.time() - start_time) * 1000
            log_performance(
                module=func.__module__,
                operation=func.__name__,
                duration_ms=duration,
                success=False,
                error=str(e)
            )
            raise
    return wrapper

# Usage
@log_performance_decorator
def expensive_operation():
    # Your code here
    pass
```

## Conclusion

The diagnostic logging system provides a robust, scalable solution for monitoring and debugging the WeChat bot. Its server-less architecture makes it ideal for production environments while maintaining the visual verification capabilities essential for computer vision debugging.

For questions or improvements, refer to the example code in `TestRun/logger_integration_example.py` or examine the generated logs in `diagnostic_logs/`.