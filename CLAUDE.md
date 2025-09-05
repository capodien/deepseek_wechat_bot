# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the WeChat automation bot codebase.

## Project Overview

WeChat desktop automation bot that uses computer vision, OCR, and DeepSeek AI to automatically respond to WeChat messages through a 6-stage processing pipeline.

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up environment
cp .env.example .env  # Add your DEEPSEEK_API_KEY

# Start diagnostic interface (recommended for development)
python step_diagnostic_server.py  # WDC at http://localhost:5001

# Run the bot (requires WeChat desktop open)
python app.py
```

## 1. System Health Indicators

Before development, verify these indicators to ensure the system is ready:

✅ **WeChat Desktop**: Running, visible, and logged into account  
✅ **WDC (Web Diagnostic Console)**: Accessible at http://localhost:5001  
✅ **Avatar Detection**: Finds ≥5 contact cards in contact list  
✅ **OCR Processing**: Correctly extracts Chinese text from screenshots  
✅ **API Connection**: DeepSeek API responds within 5 seconds  
✅ **Coordinate Accuracy**: No coordinate drift warnings in diagnostics

**Quick Health Check**:
```bash
python step_diagnostic_server.py
# Visit http://localhost:5001 and test each component
```

## 2. Documentation Structure

📋 **Core Guides**:
- **2.1** [Setup Guide](docs/SETUP.md) - Installation, dependencies, and configuration
- **2.2** [Architecture Guide](docs/ARCHITECTURE.md) - System design and technical architecture  
- **2.3** [Diagnostic Tools](docs/DIAGNOSTICS.md) - Development tools and troubleshooting
- **2.4** [Security Guide](docs/SECURITY.md) - Security vulnerabilities and fixes
- **2.5** [Maintenance Guide](docs/MAINTENANCE.md) - Best practices and workflows

📖 **Process Documentation**:
- **2.6** [process.md](process.md) - Definitive workflow and system operation guide

🛠️ **Development Tools**:
- **2.7** WDC (Web Diagnostic Console): `python step_diagnostic_server.py` → http://localhost:5001

## 3. Key System Information

### 3.1 6-Stage Processing Pipeline
```
Message Detection → Screenshot → OCR → AI Processing → GUI Automation → Data Storage
```

### 3.2 Performance Benchmarks
- **OCR**: 200-800ms (GPU: 200-400ms, CPU: 400-800ms)
- **Message Detection**: 50-150ms per cycle  
- **API Response**: 1-5 seconds (DeepSeek)
- **Memory**: 200-500MB baseline, up to 1GB processing

### 3.3 Critical Dependencies
```python
easyocr>=1.6.0          # Chinese/English OCR
opencv-python>=4.5.0     # Computer vision
pyautogui>=0.9.54        # GUI automation
openai>=1.0.0            # DeepSeek API
```

### 3.4 System Requirements
- **Python 3.8+**, **macOS 10.14+/Windows 10+**
- **GPU**: CUDA-compatible (optional, OCR acceleration)
- **WeChat Desktop 3.0+** running and visible

## 4. Project Structure

```
deepseek_wechat_bot/
├── app.py                    # Main application
├── Constants.py              # GUI coordinates (system-specific)
│
├── capture/                  # Computer vision
│   ├── monitor_new_message.py   # Red dot detection
│   ├── deal_chatbox.py          # OCR processing
│   └── get_name_free.py         # Contact recognition
│
├── deepseek/                 # AI integration
│   └── deepseekai.py            # API interface
│
├── db/                       # Database layer
│   └── db.py                    # SQLite operations
│
├── modules/                  # Modular components (NEW)
│   ├── __init__.py              # Module initialization
│   ├── m_Card_Processing.py     # Main orchestrator (detector classes)
│   ├── m_ScreenShot_WeChatWindow.py # Screenshot capture
│   ├── screenshot_processor.py  # Screenshot I/O operations
│   ├── visualization_engine.py  # Centralized visualization utilities
│   └── image_utils.py           # Shared image processing functions
│
├── docs/                     # Documentation
│   ├── SETUP.md                 # Installation guide
│   ├── ARCHITECTURE.md          # System design
│   ├── DIAGNOSTICS.md           # Development tools
│   ├── SECURITY.md              # Security analysis
│   └── MAINTENANCE.md           # Best practices
│
├── TestRun/                  # Development utilities (temporary)
├── pic/screenshots/          # Image storage
├── step_diagnostic_server.py # Enhanced diagnostics
└── process.md               # Definitive workflow guide
```

## 4.1 Modular Architecture (NEW)

The system now uses a **modular architecture** for better maintainability and separation of concerns:

### Core Modules:

**🧠 Main Orchestrator** (`modules/m_Card_Processing.py`)
- Contains all detector classes: `SimpleWidthDetector`, `CardAvatarDetector`, `CardBoundaryDetector`
- Core detection logic and algorithms remain centralized
- **Classes**: `RightBoundaryDetector`, `ContactNameBoundaryDetector`, `TimeBoxDetector`

**📸 Screenshot Processor** (`modules/screenshot_processor.py`)
- `capture_and_process_screenshot()` - Live capture with analysis
- `process_screenshot_file()` - Process existing screenshot files
- `get_live_card_analysis()` - Comprehensive analysis with visualizations
- **Clean separation** of I/O operations from detection logic

**🎨 Visualization Engine** (`modules/visualization_engine.py`)
- Centralized visualization utilities with consistent styling
- `VisualizationEngine` class for overlay generation
- Heatmap creation, composite visualizations, debug outputs
- **Standardized** color schemes and visual markers

**🛠️ Image Utilities** (`modules/image_utils.py`)
- `find_vertical_edge_x()` - Vertical edge detection with confidence scoring
- `apply_level_adjustment()` - Photoshop-style image preprocessing
- Common image processing functions and utilities
- **Shared** functionality across all detector classes

### Integration Benefits:
- ✅ **Backward Compatibility**: Existing code works unchanged
- ✅ **Modular Testing**: Each component can be tested independently
- ✅ **Maintainability**: Clear separation of concerns and responsibilities
- ✅ **Reusability**: Modules can be used by other components
- ✅ **Legacy Fallbacks**: Automatic fallback to inline implementations

### Usage Patterns:
```python
# Import main orchestrator (recommended)
from modules import m_Card_Processing

# Use detector classes as before
detector = m_Card_Processing.SimpleWidthDetector()
results = detector.detect_width("screenshot.png")

# Import individual modules (advanced)
from modules import screenshot_processor, visualization_engine
screenshot_path, analysis = screenshot_processor.capture_and_process_screenshot()
```

## 5. Key Technical Features

### 5.1 Avatar Detection System
**OpenCV Adaptive Thresholding** (`TestRun/opencv_adaptive_detector.py`)
- **Accuracy**: 100% on test datasets
- **Speed**: 200-400ms processing time
- **🎨 Visual Overlay Requirements**:
  - 🟢 Green bounding boxes around detected avatars
  - 🔴 Red dots marking avatar centers
  - 🔵 Blue crosses showing precise click positions
  - ➡️ Arrows connecting avatar centers to click targets
  - 🏷️ Numbered labels (#1, #2, #3...) for identification
- **Integration**: `/api/analyze-contact-cards-opencv` endpoint

### 5.2 Message Detection System
**Red Dot Notification Detection** (`capture/monitor_new_message.py`)
- **Speed**: 50-150ms per detection cycle
- **🎨 Visual Overlay Requirements**:
  - 🔴 Red circles highlighting detected notification dots
  - 📍 Coordinate markers showing exact pixel positions (x,y)
  - 🟢 Green boxes around active message areas
  - 📊 Color threshold visualization (RGB values: R,G,B displayed)
  - ⚡ Detection confidence scores (0-100%)
- **Integration**: `/api/test-message-detection` endpoint

### 5.3 OCR Processing System
**Text Extraction** (`capture/deal_chatbox.py`)
- **Speed**: 200-800ms processing time
- **🎨 Visual Overlay Requirements**:
  - 📝 Blue bounding boxes around detected text regions
  - 💯 Confidence scores per text block (0.0-1.0)
  - 🔤 Extracted text overlaid in yellow
  - 📏 Text orientation arrows
  - 🎯 Character-level detection boxes (detailed view)
- **Integration**: `/api/test-ocr-extraction` endpoint

### 5.4 Diagnostic Interface
**WDC (Web Diagnostic Console)** (`step_diagnostic_server.py`)
- Real-time computer vision testing with mandatory visual outputs
- Performance metrics and accuracy validation  
- Interactive coordinate verification with click testing
- Screenshot analysis with comprehensive overlays

## 10. Development Standards

### 10.1 Mandatory Requirements
🔧 **Every feature MUST have diagnostic interface** for human verification

**Standard Components**:
1. **Backend API endpoint** in `step_diagnostic_server.py`
2. **Frontend button** in `step_diagnostic.html`  
3. **🎨 Visual verification** (MANDATORY for all computer vision features)
4. **Error handling** with detailed diagnostics

**Visual Diagnostic Requirements for ALL CV Features**:
- **Automatic overlay generation** on original screenshots
- **Distinct color coding**: Different colors for different detection types
- **Comprehensive labeling**: Numbers, coordinates, confidence scores
- **Side-by-side display**: Original vs overlay comparison
- **Download capability**: Save annotated images for inspection
- **Real-time metrics**: Processing time, accuracy, detection count

**Output Format**:
```
🔍 [Feature] Analysis Complete
• Method: [Class/Function]
• Module: [path.py]
• Technique: [Algorithm description]
• Visual Overlay: [Generated filename with timestamp]
• Performance: [timing/accuracy]
• Status: ✅ Success / ❌ Failed
```

### 10.2 Development Workflow

**Pre-Development**:
1. Review [`process.md`](process.md) and relevant documentation
2. Verify WeChat setup and coordinate calibration
3. Check existing diagnostic tools for similar patterns

**Development Process**:
1. **Choose Architecture**: Decide between modular components or inline implementation
   - **Use modules**: For reusable functionality, complex features, or visual components
   - **Example**: `from modules import visualization_engine, screenshot_processor`
2. Implement feature with comprehensive error handling
3. Create diagnostic endpoint in `step_diagnostic_server.py`
4. Add frontend button in `step_diagnostic.html`
5. **🎨 MANDATORY: Implement visual overlay generation**:
   - **Option A**: Use `visualization_engine.VisualizationEngine` for consistent styling
   - **Option B**: Implement custom visualization within detector classes
   - Create visualization function that draws ALL detection elements
   - Use distinct colors and clear labeling
   - Save overlay image with descriptive timestamp filename
   - Return overlay path in API response for frontend display
6. Test error handling, edge cases, and visual accuracy
7. Verify overlay displays correctly in web interface

**Post-Development**:
1. **MANDATORY**: Update [`process.md`](process.md) with workflow changes
2. Move temporary files to `TestRun/` directory
3. Verify end-to-end diagnostic functionality
4. Update relevant documentation files

## 6. Critical System Limitations

⚠️ **Important Constraints**:
- **Platform-Specific Coordinates**: GUI coordinates in `Constants.py` must be calibrated per system
- **WeChat Window Dependency**: Requires WeChat in specific layout and size
- **Screenshot Dependencies**: Fragile to WeChat UI changes
- **Synchronous Processing**: No parallel message handling
- **Single Instance**: Cannot monitor multiple WeChat accounts

## 7. System Breaking Scenarios

🚫 **These actions will break the bot** (requires immediate recalibration):

- **WeChat Window Moved/Resized** → All coordinates become invalid
- **WeChat Language Changed** → OCR fails completely (expects Chinese/English)
- **Screen Locked/Minimized** → GUI automation stops working
- **Multiple WeChat Instances** → Wrong window targeted for automation
- **WeChat UI Theme Changed** → Color-based detection fails
- **WeChat Updates** → UI layout changes invalidate coordinates
- **Display Resolution Changed** → All pixel coordinates shift

**Recovery**: Use diagnostic tools to recalibrate coordinates and verify functionality after any of these scenarios.

### Archive Directory Protocol
🚫 **DO NOT TOUCH**: Contains archived/legacy code
- **NEVER modify, reference, or run anything in Archive/ directory**
- Archive is for historical purposes only
- All active development uses current modules in main directories
- Archive files may have outdated imports and broken dependencies

### TestRun Directory Protocol
⚠️ **DEVELOPMENT ONLY**: Contains temporary test files and experimental code
- Safe to delete without affecting bot operation
- Never move core functionality here
- Production code forbidden in this directory

## Working with This Codebase

### Critical Dependencies
The system requires several computer vision and GUI automation libraries:
- `easyocr`: Chinese/English OCR processing
- `opencv-python`: Image processing and color detection
- `pyautogui`: Cross-platform GUI automation
- `openai`: DeepSeek API integration (uses OpenAI-compatible interface)

### Development Environment Setup
1. WeChat desktop application must be running and visible
2. Screen coordinates in `Constants.py` must match your WeChat window layout
3. DeepSeek API key required in `.env` file
4. GPU support recommended for OCR performance

### Key System Limitations
- **Platform-Specific Coordinates**: GUI automation coordinates are hardcoded and system-specific
- **WeChat Window Requirements**: Requires WeChat to be in specific window layout and size
- **Screenshot Dependencies**: Heavy reliance on visual recognition makes system fragile to UI changes
- **Synchronous Processing**: All operations are synchronous, causing potential performance bottlenecks

## 11. Getting Started

### 11.1 For New Developers
1. **Read**: [`docs/SETUP.md`](docs/SETUP.md) for installation and configuration
2. **Understand**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for system design
3. **Review**: [`process.md`](process.md) for definitive workflow guide

### 11.2 For System Operations
1. **Security**: [`docs/SECURITY.md`](docs/SECURITY.md) for vulnerability analysis and fixes
2. **Diagnostics**: [`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md) for troubleshooting tools
3. **Maintenance**: [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) for operational procedures

### 11.3 Quick Commands
```bash
# Development diagnostics
python step_diagnostic_server.py     # WDC (Web Diagnostic Console)

# Component testing
python capture/deal_chatbox.py       # Test OCR
python capture/monitor_new_message.py # Test detection

# Module testing (NEW)
python modules/image_utils.py        # Test image utility functions
python modules/visualization_engine.py # Test visualization engine
python modules/screenshot_processor.py # Test screenshot processing

# Database inspection
sqlite3 history.db "SELECT * FROM conversations LIMIT 5;"
```

### 11.4 Modular Development Patterns (NEW)

**Using Detector Classes** (Recommended approach):
```python
from modules import m_Card_Processing

# Initialize detectors
width_detector = m_Card_Processing.SimpleWidthDetector()
avatar_detector = m_Card_Processing.CardAvatarDetector()

# Use as before - full backward compatibility
width_result = width_detector.detect_width("screenshot.png")
avatars, info = avatar_detector.detect_avatars("screenshot.png")
```

**Using Individual Modules** (Advanced/custom development):
```python
from modules import screenshot_processor, visualization_engine, image_utils

# Screenshot processing
screenshot_path, analysis = screenshot_processor.capture_and_process_screenshot()

# Visualization
engine = visualization_engine.VisualizationEngine()
overlay = engine.create_base_overlay(screenshot_path)
engine.draw_rectangle(overlay, x, y, w, h, 'card', 'Detected Card')

# Image utilities
x_pos, confidence, profile = image_utils.find_vertical_edge_x(img_array)
enhanced_img = image_utils.apply_level_adjustment(gray_img)
```

**Module Integration in Diagnostic Endpoints**:
```python
# In step_diagnostic_server.py
from modules import m_Card_Processing, visualization_engine

@app.route('/api/test-card-detection')
def test_card_detection():
    try:
        # Use main orchestrator for detection
        detector = m_Card_Processing.CardBoundaryDetector()
        cards, info = detector.detect_cards(image_path)
        
        # Use visualization engine for consistent overlays
        engine = visualization_engine.VisualizationEngine()
        overlay_path = engine.save_visualization(
            overlay_img, image_path, "card_detection"
        )
        
        return {'cards': cards, 'overlay': overlay_path}
    except Exception as e:
        return {'error': str(e)}
```

## 9. Security Alert

🚨 **CRITICAL VULNERABILITIES** - [Full Security Analysis](docs/SECURITY.md)

**Immediate Action Required**:
1. **SQL Injection Risk** - Database uses string interpolation instead of parameterized queries
2. **Input Validation Missing** - No sanitization of AI-processed messages  
3. **Credential Exposure** - API keys stored in plain text environment variables
4. **Data Encryption Missing** - Screenshots and database unencrypted

**Current Status**: ⚠️ **High Risk** - Production use not recommended until fixes applied

**Remediation**: See [Security Guide](docs/SECURITY.md) for detailed fixes and implementation roadmap

## Documentation Maintenance Requirements

### CRITICAL: Process Documentation Updates

**MANDATORY**: When making ANY changes to the bot's workflow, architecture, or core functionality, you MUST update `process.md` to reflect these changes. This includes:

#### Required Updates for process.md:
1. **Workflow Changes**: Any modifications to the 6-step execution cycle
2. **New Features**: Additional functionality, modes, or capabilities
3. **Architecture Changes**: Component additions, removals, or restructuring  
4. **Performance Optimizations**: Timing improvements, bottleneck fixes
5. **Safety/Security Updates**: New safety mechanisms, error handling
6. **Configuration Changes**: New settings, modes, or operational parameters
7. **API/Integration Changes**: DeepSeek API updates, new service integrations
8. **Database Schema Changes**: New tables, fields, or data structures

#### Update Process:
1. **Before making code changes**: Review current `process.md` sections that will be affected
2. **During development**: Note all changes that impact user-facing behavior or system architecture  
3. **After completing changes**: Update corresponding sections in `process.md` with:
   - New process flows or step modifications
   - Updated performance metrics and timing
   - New configuration options or requirements
   - Modified error handling or safety procedures
   - Any new dependencies or system requirements

#### Documentation Quality Standards:
- **Accuracy**: All technical details must match actual implementation
- **Completeness**: Cover all major system components and workflows
- **Clarity**: Use clear, technical language suitable for developers
- **Examples**: Include code snippets, configuration examples, and usage patterns
- **Performance Data**: Update timing analysis and resource usage information

#### Verification Requirements:
Before considering any development task complete, verify:
- [ ] `process.md` accurately reflects all system changes
- [ ] New functionality is properly documented with examples
- [ ] Performance implications are noted and quantified  
- [ ] Error handling and safety considerations are documented
- [ ] Any breaking changes or migration requirements are clearly stated


## 8. Common Issues & Quick Fixes

### 8.1 OCR Recognition Problems
```python
# Check GPU availability
python -c "import easyocr; print(easyocr.Reader(['ch_sim'], gpu=True))"

# CPU fallback if GPU fails
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)
```

### 8.2 Coordinate Misalignment
1. Use WDC (Web Diagnostic Console) to verify click positions: http://localhost:5001
2. Recalibrate coordinates in `Constants.py`
3. Check WeChat window size and position

### 8.3 Message Detection Failures
```python
# Adjust red dot detection thresholds in monitor_new_message.py
red_threshold = (0, 0, 200)
time.sleep(1.5)  # Increase polling interval
```

### 8.4 API Connection Issues
```bash
# Verify environment setup
echo $DEEPSEEK_API_KEY

# Test API connectivity
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models
```

**For comprehensive troubleshooting**, see [Diagnostic Tools Guide](docs/DIAGNOSTICS.md)

## 12. Documentation Update Requirements

### 12.1 MANDATORY Updates
When making **ANY** changes to bot functionality:

| Change Type | Required Updates |
|-------------|------------------|
| **Workflow changes** | [`process.md`](process.md) |
| **New features** | [`process.md`](process.md) + relevant docs/ files |
| **Security changes** | [`docs/SECURITY.md`](docs/SECURITY.md) |
| **Performance changes** | Update benchmarks in relevant files |
| **Architecture changes** | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |

### 12.2 Quality Standards
- **Technical accuracy**: All code examples must work
- **Performance data**: Keep benchmarks current  
- **Security awareness**: Document security implications
- **Completeness**: Cover all major system components

---

**Note**: This main `CLAUDE.md` provides overview and navigation. Detailed information is in the specialized documentation files in the [`docs/`](docs/) directory.