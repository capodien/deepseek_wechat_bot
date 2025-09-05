# Diagnostic Tools Guide

Development tools, testing interfaces, and troubleshooting systems for the WeChat bot.

## Overview

The bot includes comprehensive diagnostic tools for testing, debugging, and visual verification of all computer vision and automation components.

## WDC Interface

### WDC (Web Diagnostic Console)
```bash
python step_diagnostic_server.py
# Access at: http://localhost:5001
```

**Features**:
- Visual verification with image overlays
- Real-time performance metrics
- Interactive coordinate testing
- Screenshot analysis tools
- API connection testing

## Core Diagnostic Functions

### 1. Avatar Detection Testing

**Button**: 🔍 **Test Avatar Detection**

**Functionality**:
- Captures current WeChat screenshot
- Runs OpenCV adaptive detection
- Generates visual overlay with bounding boxes, centers, click positions
- Returns detection accuracy and timing metrics

**API Endpoint**: `/api/analyze-contact-cards-opencv`

**Expected Output**:
```
🔍 OpenCV Adaptive Detection Complete
• Method: OpenCVAdaptiveDetector
• Module: TestRun/opencv_adaptive_detector.py  
• Technique: OpenCV Adaptive Thresholding with Contour Detection
• Parameters: Size 25-120px, Aspect 0.7-1.4, Block 9, C=2
• Performance: 11 avatars detected in 324ms
• Accuracy: 100% success rate
```

### 2. OCR Text Extraction Testing

**Button**: 📝 **Test OCR Processing**

**Functionality**:
- Captures message area screenshot
- Runs EasyOCR text extraction
- Shows extracted text with confidence scores
- Displays processing time and GPU/CPU usage

**API Endpoint**: `/api/test-ocr-extraction`

### 3. Message Detection Testing

**Button**: 🔴 **Test Message Detection**

**Functionality**:
- Monitors for red notification dots
- Shows pixel color analysis
- Tests notification detection accuracy
- Provides threshold adjustment recommendations

**API Endpoint**: `/api/test-message-detection`

### 4. Coordinate Verification

**Button**: 🎯 **Test Click Coordinates**

**Functionality**:
- Tests all predefined coordinates from `Constants.py`
- Shows visual markers on current screen
- Validates click position accuracy
- Provides calibration adjustment suggestions

**API Endpoint**: `/api/verify-coordinates`

### 5. API Connection Testing

**Button**: 🌐 **Test DeepSeek API**

**Functionality**:
- Tests API connection and authentication
- Measures response time and latency
- Validates API key configuration
- Shows available models and limits

**API Endpoint**: `/api/test-deepseek-connection`

## Visual Verification System

### Automatic Overlay Generation

**MANDATORY** for all computer vision features:

1. **Detection Results**: Green bounding boxes around detected elements
2. **Center Points**: Red dots marking geometric centers
3. **Click Targets**: Blue crosses showing precise click positions  
4. **Relationships**: Arrows connecting centers to click targets
5. **Labels**: Numbered identification tags (#1, #2, #3...)

### Overlay Storage
- **Location**: `pic/screenshots/`
- **Naming**: `{method}_{timestamp}.png` (e.g., `opencv_detection_20240904_083905.png`)
- **Access**: Direct download links in web interface
- **Comparison**: Side-by-side original vs. overlay view

### Quality Standards
- **Precision**: Sub-pixel accuracy for click coordinates
- **Completeness**: All detected elements must be visualized
- **Clarity**: Distinct colors and clear labeling
- **Documentation**: Comprehensive legends explaining all visual elements

## Development Testing Tools

### Component Testing Scripts
Located in `TestRun/` directory:

```bash
# OCR functionality testing
python TestRun/test_ocr_processing.py

# Message detection testing  
python TestRun/test_message_monitoring.py

# Contact name recognition testing
python TestRun/test_name_extraction.py

# Coordinate calibration testing
python TestRun/test_coordinate_accuracy.py
```

### Debug Utilities
```bash  
# Screenshot analysis
python TestRun/debug_screenshot_analysis.py

# Performance profiling
python TestRun/debug_performance_analysis.py

# API response analysis
python TestRun/debug_api_responses.py
```

## Troubleshooting Guide

### Common Diagnostic Failures

#### OCR Recognition Issues
**Symptoms**: Empty text extraction, garbled characters
**Diagnostic Steps**:
1. Run OCR test button to check GPU availability
2. Verify screenshot quality and resolution
3. Test with CPU fallback mode
4. Check language model configuration

**Solutions**:
```python
# GPU availability check
python -c "import easyocr; print(easyocr.Reader(['ch_sim'], gpu=True))"

# CPU fallback configuration
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)

# Image preprocessing improvement
img = cv2.resize(img, None, fx=2, fy=2)  # Upscale for better OCR
```

#### Avatar Detection Failures
**Symptoms**: No avatars detected, incorrect click positions
**Diagnostic Steps**:
1. Use avatar detection test to verify detection accuracy
2. Check visual overlay for detection quality
3. Analyze screenshot for lighting or UI changes
4. Verify WeChat window size and position

**Solutions**:
```python
# Adjust detection parameters
detector_params = {
    'size_range': (20, 130),      # Wider size range
    'aspect_ratio': (0.6, 1.5),   # More flexible aspect ratio
    'block_size': 11,             # Larger block size for adaptive threshold
    'C': 3                        # Higher threshold adjustment
}
```

#### Coordinate Misalignment  
**Symptoms**: Clicks missing targets, wrong regions captured
**Diagnostic Steps**:
1. Run coordinate verification test
2. Compare visual markers with expected positions
3. Check screen resolution and WeChat window size
4. Verify `Constants.py` values

**Solutions**:
```python
# Constants.py recalibration
WECHAT_WINDOW = (new_x, new_y, new_width, new_height)

# Use diagnostic tools for precise measurement
# Visual verification shows exact offset needed
```

#### API Connection Problems
**Symptoms**: DeepSeek API timeouts, authentication errors
**Diagnostic Steps**:
1. Run API connection test
2. Verify API key in `.env` file
3. Check network connectivity
4. Test API rate limits

**Solutions**:
```bash
# Environment variable verification
echo $DEEPSEEK_API_KEY

# Direct API testing
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models

# Rate limit checking
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/usage
```

### Performance Diagnosis

#### Slow OCR Processing
**Expected**: 200-400ms (GPU), 400-800ms (CPU)
**Investigation**:
- GPU memory availability
- Image resolution and quality
- Concurrent processing load
- Hardware specifications

#### High Memory Usage
**Expected**: 200-500MB baseline, up to 1GB during processing
**Investigation**:
- Screenshot cache size
- OpenCV memory leaks
- Unreleased image resources
- Database connection pooling

## Error Pattern Analysis

### Critical Failure Modes
1. **WeChat Process Not Found**: Application not running or hidden
2. **Screen Lock/Sleep**: System sleeping interrupts automation
3. **UI Language Changes**: OCR confusion with unexpected languages  
4. **Coordinate Drift**: Gradual misalignment due to window movement
5. **Network Interruption**: API connection failures during processing

### Logging Configuration
```python
import logging

# Comprehensive logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(module)s:%(lineno)d - %(message)s',
    handlers=[
        logging.FileHandler('logs/diagnostic_operations.log'),
        logging.StreamHandler()  # Console output for real-time monitoring
    ]
)

# Component-specific loggers
ocr_logger = logging.getLogger('ocr_processing')
detection_logger = logging.getLogger('avatar_detection')  
api_logger = logging.getLogger('api_integration')
```

### Health Monitoring
```python
# Health check endpoint for continuous monitoring
@app.route('/health')
def system_health():
    return {
        'status': 'healthy',
        'components': {
            'wechat_process': check_wechat_running(),
            'ocr_gpu': test_gpu_availability(),
            'api_connection': ping_deepseek_api(),
            'coordinate_accuracy': verify_click_positions(),
            'disk_space': get_available_storage(),
            'memory_usage': get_current_memory()
        },
        'last_successful_operation': get_last_success_timestamp(),
        'error_rate_24h': calculate_error_rate()
    }
```

## Development Standards

### Creating New Diagnostic Tools

#### Required Components
1. **Backend API Endpoint**:
```python
@app.route('/api/test-new-feature', methods=['POST'])
def test_new_feature():
    try:
        # Implementation with comprehensive error handling
        result = perform_feature_test()
        return jsonify({
            "success": True, 
            "data": result,
            "performance_ms": timing_data,
            "accuracy": accuracy_metrics
        })
    except Exception as e:
        return jsonify({
            "success": False, 
            "error": str(e),
            "diagnostic_info": gather_diagnostic_context()
        })
```

2. **Frontend Interface**:
```html
<button onclick="testNewFeature()">🔧 Test New Feature</button>
<div id="newFeatureResults"></div>
<script>
function testNewFeature() {
    showLoading('newFeatureResults');
    fetch('/api/test-new-feature', {method: 'POST'})
        .then(response => response.json())
        .then(data => displayResults('newFeatureResults', data));
}
</script>
```

3. **Visual Verification** (for CV features):
```python
def create_diagnostic_visualization(image, detections):
    overlay = image.copy()
    for i, detection in enumerate(detections):
        # Draw detection elements with distinct colors
        cv2.rectangle(overlay, detection.bbox, (0, 255, 0), 2)  # Green box
        cv2.circle(overlay, detection.center, 5, (0, 0, 255), -1)  # Red center
        cv2.putText(overlay, f"#{i+1}", detection.label_pos, font, 1, (255, 255, 255), 2)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"diagnostic_{feature_name}_{timestamp}.png"
    filepath = f"pic/screenshots/{filename}"
    cv2.imwrite(filepath, overlay)
    
    return filepath
```

### Quality Standards
- **User-Friendly**: Results understandable by non-technical users
- **Comprehensive**: All technical details and performance metrics
- **Visual**: Image overlays for computer vision components
- **Interactive**: Coordinate testing and validation capabilities
- **Robust**: Graceful error handling with detailed diagnostics

## Integration with Main System

### Diagnostic Data Flow
```
User Input → Frontend Interface → API Endpoint → Feature Test → 
Visual Overlay Generation → Results Display → Performance Logging
```

### Performance Monitoring Integration
- **Real-time Metrics**: Processing times, accuracy scores, resource usage
- **Historical Tracking**: Performance trends over time  
- **Anomaly Detection**: Automatic alerts for performance degradation
- **Optimization Suggestions**: AI-driven recommendations for improvements

### Continuous Integration
```bash
# Automated diagnostic testing
python -m pytest TestRun/test_diagnostic_coverage.py

# Performance benchmark verification  
python TestRun/benchmark_diagnostic_performance.py

# Visual regression testing
python TestRun/test_visual_verification_accuracy.py
```