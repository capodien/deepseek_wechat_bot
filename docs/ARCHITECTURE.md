# Architecture Guide

System design and technical architecture of the WeChat automation bot.

## System Overview

The WeChat bot uses computer vision, OCR, and AI to automatically respond to WeChat messages through a 6-stage processing pipeline.

## Core System Flow

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Message        │    │  Screenshot     │    │  OCR           │
│  Detection      │───▶│  Capture        │───▶│  Processing    │
│                 │    │                 │    │                │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                                              │
         │                                              ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  GUI            │    │  Data           │    │  AI             │
│  Automation     │◀───│  Persistence    │◀───│  Integration    │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

### Stage Descriptions

1. **Message Detection**: `capture/monitor_new_message.py` monitors for red notification dots using pixel color detection
2. **Screenshot Capture**: Takes targeted screenshots of message areas and chat regions
3. **OCR Processing**: `capture/deal_chatbox.py` extracts Chinese/English text using EasyOCR
4. **AI Integration**: `deepseek/deepseekai.py` generates responses via DeepSeek API with streaming
5. **GUI Automation**: Uses pyautogui for cross-platform clicking and typing
6. **Data Persistence**: SQLite databases store conversation history and message logs

## Project Structure

```
deepseek_wechat_bot/
├── app.py                    # Main application orchestration
├── Constants.py              # GUI coordinate constants (system-specific)
├── 
├── capture/                  # Computer vision and detection
│   ├── monitor_new_message.py    # Red dot notification detection
│   ├── deal_chatbox.py           # OCR text extraction
│   ├── get_name_free.py          # Contact name recognition
│   └── text_change_monitor.py    # Text change detection
├── 
├── deepseek/                 # AI integration layer
│   └── deepseekai.py             # DeepSeek API interface with streaming
├── 
├── db/                       # Database layer
│   └── db.py                     # SQLite operations and schema
├── 
├── modules/                  # Modular components (NEW ARCHITECTURE)
│   ├── __init__.py               # Module initialization  
│   ├── m_Card_Processing.py      # Main orchestrator with all detector classes
│   ├── m_ScreenShot_WeChatWindow.py # Screenshot capture functionality
│   ├── screenshot_processor.py   # Screenshot I/O operations and processing
│   ├── visualization_engine.py   # Centralized visualization utilities
│   └── image_utils.py            # Shared image processing functions
├── 
├── Diagnostic/               # Enhanced diagnostic functionality
│   ├── *.py                      # Advanced diagnostic tools
│   └── *.png                     # Analysis screenshots
├── 
├── TestRun/                  # Development utilities (temporary files only)
│   ├── test_*.py                 # Component testing scripts
│   ├── debug_*.py                # Development debugging tools
│   └── README.md                 # Development utilities documentation
├── 
├── pic/                      # Screenshot storage
│   ├── screenshots/              # Main window captures + diagnostic overlays
│   ├── message/                  # Message area captures
│   └── chatname/                 # Contact name captures
├── step_diagnostic_server.py # WDC (Web Diagnostic Console) server
└── step_diagnostic.html      # WDC interface UI
```

## Key Architectural Patterns

### Event-Driven Monitoring
The main loop in `app.py` uses continuous polling with nested state machines:
- **Outer loop**: Monitor for new message indicators (red dots)
- **Inner loop**: Process conversation until no new messages detected
- **State management**: Tracks conversation context and user interaction

### Platform Abstraction
Cross-platform GUI operations handled via platform detection:
```python
import platform

if platform.system() == 'Darwin':  # macOS
    pyautogui.hotkey('command', 'v')
elif platform.system() == 'Windows':
    pyautogui.hotkey('ctrl', 'v')
```

### Performance Instrumentation
Detailed timing analysis throughout critical paths:
```python
import time

@timing_monitor
def process_screenshot(image_path):
    start_time = time.time()
    # OCR processing, API calls, image operations
    processing_time = (time.time() - start_time) * 1000
    logger.info(f"Screenshot processing: {processing_time:.2f}ms")
```

### Modular Architecture (NEW)
The system now employs a **modular architecture** for improved maintainability and separation of concerns:

#### Design Principles
- **Single Responsibility**: Each module handles a specific domain (I/O, visualization, utilities)
- **Loose Coupling**: Modules interact through well-defined interfaces
- **High Cohesion**: Related functionality grouped within modules
- **Backward Compatibility**: Legacy code continues to work unchanged

#### Module Responsibilities

**Main Orchestrator** (`modules/m_Card_Processing.py`)
- **Purpose**: Central coordination of all detector classes
- **Components**: `SimpleWidthDetector`, `CardAvatarDetector`, `CardBoundaryDetector`, `ContactNameBoundaryDetector`, `TimeBoxDetector`, `RightBoundaryDetector`
- **Pattern**: Facade pattern providing unified interface to detection subsystems

**Screenshot Processor** (`modules/screenshot_processor.py`)
- **Purpose**: I/O operations and screenshot lifecycle management
- **Functions**: `capture_and_process_screenshot()`, `process_screenshot_file()`, `get_live_card_analysis()`
- **Pattern**: Service layer abstracting screenshot operations

**Visualization Engine** (`modules/visualization_engine.py`)
- **Purpose**: Centralized visualization utilities and consistent styling
- **Features**: `VisualizationEngine` class, standardized overlays, heatmap generation
- **Pattern**: Strategy pattern for different visualization types

**Image Utilities** (`modules/image_utils.py`)
- **Purpose**: Shared image processing functions and algorithms
- **Functions**: `find_vertical_edge_x()`, `apply_level_adjustment()`, edge detection utilities
- **Pattern**: Utility/Helper pattern for common operations

#### Integration Mechanisms
```python
# Automatic module detection and fallback
if MODULAR_IMPORTS_AVAILABLE:
    find_vertical_edge_x = image_utils.find_vertical_edge_x
    capture_and_process_screenshot = screenshot_processor.capture_and_process_screenshot
else:
    # Legacy inline implementations for backward compatibility
    def find_vertical_edge_x(img, x0=0, x1=None, y0=0, y1=None, rightmost=True):
        # Original implementation preserved
```

#### Benefits Achieved
- **Maintainability**: Clear separation enables independent module updates
- **Testability**: Each module can be unit tested in isolation
- **Reusability**: Modules can be imported and used by other components
- **Performance**: Selective imports reduce memory footprint
- **Development**: Parallel development on different modules

## Computer Vision System

### Avatar Detection Methods

#### Primary: OpenCV Adaptive Detection
- **Class**: `OpenCVAdaptiveDetector` (`TestRun/opencv_adaptive_detector.py`)
- **Technique**: Adaptive thresholding + contour detection + geometric filtering
- **Performance**: 100% accuracy on test datasets, 200-400ms processing time

```python
# Core detection parameters
detection_params = {
    'size_range': (25, 120),      # Avatar size constraints (pixels)
    'aspect_ratio': (0.7, 1.4),   # Shape validation
    'block_size': 9,              # Adaptive threshold neighborhood
    'C': 2,                       # Threshold adjustment constant
    'blur_kernel': 3,             # Noise reduction filter
    'click_offset': 70            # Text area targeting offset
}
```

#### Visual Verification System
- **Function**: `create_visualization()` (same module)
- **Output Elements**: Detection boxes, center points, click targets, numbered labels
- **Integration**: Automatic overlay generation via `/api/analyze-contact-cards-opencv`

## Database Architecture

### Schema Design
```sql
-- history.db: Conversation context storage
CREATE TABLE conversations (
    id INTEGER PRIMARY KEY,
    user_id TEXT,
    role TEXT,           -- 'user' or 'assistant'
    content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
);

-- messages.db: Message exchange logs
CREATE TABLE messages (
    id INTEGER PRIMARY KEY,
    contact_name TEXT,
    message_content TEXT,
    response_content TEXT,
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    processing_time_ms INTEGER
);
```

### Data Flow
1. **Incoming messages** → OCR extraction → `messages.db`
2. **Conversation context** → AI processing → `history.db`
3. **Response generation** → Streaming output → Database logging
4. **Performance metrics** → Processing times → Analytics

## Critical Configuration Points

### Coordinate Configuration
All GUI automation coordinates defined in `Constants.py` - **MUST** be calibrated per system:
```python
# System-specific coordinates (require calibration)
WECHAT_WINDOW = (x, y, width, height)  # Main WeChat window bounds
MESSAGE_REGION = (x1, y1, x2, y2)     # Chat message area
CONTACT_LIST = (x1, y1, x2, y2)       # Contact list coordinates
```

### OCR Configuration
EasyOCR initialization with language and GPU settings:
```python
# GPU acceleration (recommended)
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# CPU fallback (slower but universal)
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)
```

### Contact Management
Monitored contacts defined in `names.txt`:
```
张三
李四
王五
```

## Performance Characteristics

### Timing Benchmarks
- **OCR Processing**: 200-800ms per screenshot (GPU: 200-400ms, CPU: 400-800ms)
- **Message Detection**: 50-150ms per polling cycle
- **API Response**: 1-5 seconds (varies with DeepSeek load)
- **Screenshot Capture**: 100-300ms
- **Total Response Time**: 2-8 seconds from message to reply

### Resource Usage
- **Memory**: 200-500MB baseline, up to 1GB during processing
- **CPU**: 10-30% average, spikes to 60-80% during OCR
- **GPU**: 200-500MB VRAM when GPU acceleration enabled
- **Disk**: 10-50MB per hour for screenshots and logs

### Scalability Limitations
- **Sequential Processing**: No parallel message handling
- **Single WeChat Instance**: Cannot monitor multiple accounts
- **Coordinate Dependency**: Breaks with WeChat UI changes
- **Platform Specific**: Requires recalibration per system

## System Limitations & Constraints

### Technical Limitations
- **Platform-Specific Coordinates**: GUI automation coordinates are hardcoded and system-specific
- **WeChat Window Requirements**: Requires WeChat in specific window layout and size
- **Screenshot Dependencies**: Heavy reliance on visual recognition makes system fragile to UI changes
- **Synchronous Processing**: All operations synchronous, causing potential performance bottlenecks
- **Single Language OCR**: Optimized for Chinese/English, may struggle with other languages

### Operational Constraints
- **Requires Human Supervision**: No autonomous error recovery
- **WeChat App Dependency**: Must have WeChat desktop running and visible
- **Network Dependency**: Requires stable internet for DeepSeek API
- **Screen Unlock Requirement**: Cannot operate when screen is locked
- **UI Stability Requirement**: Breaks with WeChat updates that change UI layout

## Integration Points

### External Dependencies
- **DeepSeek API**: AI response generation (OpenAI-compatible interface)
- **EasyOCR**: Chinese/English text recognition
- **OpenCV**: Image processing and computer vision
- **PyAutoGUI**: Cross-platform GUI automation
- **SQLite**: Local data persistence

### Extension Opportunities
- **Multi-Account Support**: Monitor multiple WeChat instances
- **Cloud Integration**: Remote monitoring and control
- **Advanced AI**: Context-aware response personalization
- **Mobile Support**: Integration with WeChat mobile API
- **Analytics Dashboard**: Real-time performance monitoring

## Security Architecture

### Current Security Model
- **Local Processing**: All data processed locally (no cloud storage)
- **Environment Variables**: API keys stored in `.env` file
- **Database Encryption**: None (plaintext SQLite files)
- **Input Validation**: Minimal (see [Security Guide](SECURITY.md) for issues)

### Trust Boundaries
- **User Environment**: Trusted (local machine, user's WeChat account)
- **DeepSeek API**: External trust (encrypted HTTPS communication)
- **File System**: Trusted (local screenshot and database storage)
- **Network**: Untrusted (requires secure API communication)

For detailed security analysis and remediation steps, see [Security Guide](SECURITY.md).