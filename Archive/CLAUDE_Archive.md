# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with the WeChat automation bot codebase.

## Project Overview

WeChat desktop automation bot that uses computer vision, OCR, and DeepSeek AI to automatically respond to WeChat messages through a 6-stage processing pipeline.

## Table of Contents

### Quick Start & Overview
- [🚨 KISS PRINCIPLE - MANDATORY DEVELOPMENT PHILOSOPHY](#-kiss-principle---mandatory-development-philosophy)
- [Quick Start](#quick-start)
- [1. System Health Indicators](#1-system-health-indicators)
- [2. Documentation Structure](#2-documentation-structure)
- [3. Key System Information](#3-key-system-information)

### Architecture & Components  
- [4. Project Structure](#4-project-structure)
- [4.1 Modular Architecture](#41-modular-architecture-existing---do-not-expand-without-permission)
- [5. Key Technical Features](#5-key-technical-features)
- [5.5 Photo Processor Architecture](#55-photo-processor-architecture-new)

### System Operations
- [6. Critical System Limitations](#6-critical-system-limitations)
- [7. System Breaking Scenarios](#7-system-breaking-scenarios)  
- [8. Common Issues & Quick Fixes](#8-common-issues--quick-fixes)
- [9. Security Alert](#9-security-alert)

### Development Guide
- [10. Getting Started](#10-getting-started)
- [11. Development Standards](#11-development-standards)
- [12. Documentation Maintenance Requirements](#12-documentation-maintenance-requirements)

---

## 🚨 KISS PRINCIPLE - MANDATORY DEVELOPMENT PHILOSOPHY

### Core Principle: Keep It Simple, Stupid (KISS)

**PRIMARY DIRECTIVE**: Keep code SIMPLE and HUMAN-READABLE at all times.

### ⛔ MODULE CREATION POLICY

**STRICT RULE**: Claude MUST NOT create new modules, classes, or complex abstractions without EXPLICIT user approval.

**Why This Matters**:
- **Human Readability**: Code must be immediately understandable by humans
- **Maintainability**: Simple code is easier to debug and modify
- **Tracking**: User must be able to track all new classes and modules
- **Avoiding Over-Engineering**: Most features can be implemented inline

### 📋 SIMPLICITY CHECKLIST (MANDATORY)

Before implementing ANY feature, Claude MUST verify:

1. ✅ **Can this be done with existing code?** (Try this FIRST)
2. ✅ **Is inline implementation sufficient?** (Default approach)
3. ✅ **Will adding complexity actually help?** (Usually NO)
4. ✅ **Is this the SIMPLEST possible solution?** (If not, simplify)

### 🛑 REQUIRES EXPLICIT USER APPROVAL

The following actions REQUIRE explicit user permission:
- Creating new module files (`.py` files in `modules/` or elsewhere)
- Creating new classes (even within existing files)
- Adding complex abstractions or design patterns
- Implementing inheritance hierarchies
- Creating factory patterns or complex decorators

### ✅ PREFERRED APPROACHES

**ALWAYS prefer these simple solutions**:
```python
# ✅ GOOD: Simple function in existing file
def process_data(data):
    return data.upper()

# ❌ BAD: Creating unnecessary class
class DataProcessor:
    def process(self, data):
        return data.upper()
```

**EXTEND existing modules** rather than creating new ones:
```python
# ✅ GOOD: Add function to existing module
# In existing m_Card_Processing.py
def new_simple_function():
    pass

# ❌ BAD: Create new module for one function
# Creating new_module.py
```

### 🎯 WHEN SIMPLICITY MATTERS MOST

- **Bug Fixes**: Use the simplest fix that works
- **Small Features**: Implement inline first
- **Prototypes**: Start simple, complexity only if proven necessary
- **Refactoring**: Simplify, don't add abstraction layers

### 📊 COMPLEXITY THRESHOLD

Only consider modularization when:
- Feature is used in 3+ different places
- Code block exceeds 100 lines
- Clear separation of concerns exists
- AND user explicitly approves the added complexity

**Remember**: It's easier to add complexity later than to remove it. Start simple, stay simple.

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
│   ├── m_screenshot_processor.py # Screenshot capture and processing (CONSOLIDATED)
│   ├── visualization_engine.py  # Centralized visualization engine (NEW)
│   ├── diagnostic_templates.py  # Standardized visualization templates (NEW)
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

## 4.1 Modular Architecture (EXISTING - DO NOT EXPAND WITHOUT PERMISSION)

⚠️ **KISS WARNING**: The existing modules are SUFFICIENT for most tasks. DO NOT create new modules without explicit user approval.

The system uses a **LIMITED modular architecture** - keep it that way:

### 🚨 CRITICAL SCREENSHOT PROCESSING POLICY

**MANDATORY**: All screenshot and image processing operations must use `modules/m_screenshot_processor.py` as the single source of truth.

**Why This Matters**:
- **Consolidated Architecture**: All screenshot functionality has been consolidated from the previous dual-module approach (`m_ScreenShot_WeChatWindow.py` + `screenshot_processor.py`) into a single unified module
- **Eliminated Complexity**: Removed confusing dual-approach patterns that caused import conflicts and inconsistencies
- **Single Source of Truth**: One module contains all screenshot capture and processing functionality
- **Backward Compatibility**: Legacy function names and APIs preserved for existing code

**Implementation Requirements**:
```python
# ✅ CORRECT - Use consolidated module
from modules import m_screenshot_processor
screenshot = m_screenshot_processor.fcapture_messages_screenshot()

# ❌ INCORRECT - These modules no longer exist
from modules import m_ScreenShot_WeChatWindow  # DELETED
from modules import screenshot_processor  # DELETED (old wrapper)
```

**All Functions Available**:
- `fcapture_screenshot()` - Basic screenshot capture
- `fcapture_messages_screenshot()` - WeChat message area capture
- `fcapture_and_process_screenshot()` - Live capture with card analysis
- `fprocess_screenshot_file()` - Process existing screenshot files
- `fget_live_card_analysis()` - Comprehensive analysis with visualizations
- `cWeChatScreenshotCapture` - Full screenshot capture class

### 🛑 MODULE CREATION RULES

**BEFORE creating ANY new module, Claude MUST**:
1. Try adding functions to existing modules FIRST
2. Prove that existing modules cannot handle the feature
3. Get EXPLICIT user approval with justification
4. Document why simpler approaches won't work

**Preferred approach for new features**:
```python
# ✅ GOOD: Extend existing module
# In modules/m_Card_Processing.py or appropriate existing module
def new_feature_function():
    """Simple inline implementation"""
    pass

# ❌ BAD: Create new module
# modules/new_feature.py  # DON'T DO THIS WITHOUT PERMISSION
```

### Core Modules (USE THESE, DON'T CREATE NEW ONES):

**🧠 Main Orchestrator** (`modules/m_Card_Processing.py`)
- Contains all detector classes: `SimpleWidthDetector`, `CardAvatarDetector`, `CardBoundaryDetector`
- Core detection logic and algorithms remain centralized
- **Classes**: `RightBoundaryDetector`, `ContactNameBoundaryDetector`, `TimeBoxDetector`

**📸 Screenshot Processor** (`modules/m_screenshot_processor.py`)
- `fcapture_screenshot()` - Basic screenshot capture
- `fcapture_messages_screenshot()` - Message area capture for bot operations
- `fcapture_and_process_screenshot()` - Live capture with analysis
- `fprocess_screenshot_file()` - Process existing screenshot files
- `fget_live_card_analysis()` - Comprehensive analysis with visualizations
- **CONSOLIDATED**: All screenshot and image processing functionality in single module

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

**🎨 Photo Processor** (`modules/m_photo_processor.py`) **NEW**
- Consolidated photo and image processing module eliminating code duplication
- `PhotoshopProcessor` - Photoshop-style levels adjustment with gamma correction
- `EdgeDetector` - Unified Sobel edge detection (vertical/horizontal) with debug visualization
- `BinaryProcessor` - Adaptive thresholding and morphological operations
- **Consolidates** functions from m_Card_Processing, image_utils, and timestamp_detector
- **Backward Compatible** module-level functions: `apply_photoshop_levels()`, `find_vertical_edge_x()`, `create_gradient_profile()`
- **Debug Mode** generates comprehensive visualizations with performance metrics

### Integration Benefits (BUT DON'T ADD MORE MODULES):
- ✅ **Backward Compatibility**: Existing code works unchanged
- ✅ **KISS Compliance**: Simple, human-readable interfaces
- ✅ **Maintainability**: Keep it simple to maintain easily
- ✅ **Reusability**: Use existing modules, don't create new ones
- ✅ **No Over-Engineering**: Resist the urge to abstract everything

### Usage Patterns (KEEP IT SIMPLE):
```python
# ✅ PREFERRED: Use existing modules (KISS approach)
from modules import m_Card_Processing

# Use detector classes - simple and direct
detector = m_Card_Processing.SimpleWidthDetector()
results = detector.detect_width("screenshot.png")

# ⚠️ ONLY when necessary: Import specific modules (don't create new ones)
from modules import m_screenshot_processor, visualization_engine, m_photo_processor
screenshot_path, analysis = m_screenshot_processor.fcapture_and_process_screenshot()

# NEW: Use photo processor for image enhancement
from modules.m_photo_processor import PhotoshopProcessor, EdgeDetector, BinaryProcessor
photoshop = PhotoshopProcessor(debug_mode=True)
enhanced_image = photoshop.apply_photoshop_levels(grayscale_image)

# NEW: Use consolidated edge detection  
edge_detector = EdgeDetector(debug_mode=True)
x_pos, confidence, profile = edge_detector.detect_vertical_edges(image, x0=0, x1=300)

# NEW: Backward compatible module-level functions
from modules.m_photo_processor import apply_photoshop_levels, find_vertical_edge_x
enhanced = apply_photoshop_levels(image, gamma=0.67, debug_mode=True)
edge_x, conf, prof = find_vertical_edge_x(image, rightmost=False)
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

## 5.5 Photo Processor Architecture (NEW)

**Unified Image Processing Module** (`modules/m_photo_processor.py`)

### 🎯 **Architecture Benefits**
- **Eliminates Code Duplication**: Consolidates 3+ duplicate Photoshop levels implementations
- **Single Source of Truth**: One module for all image processing operations  
- **Consistent Parameters**: Standardized thresholds and settings across all functions
- **Enhanced Debug Capabilities**: Comprehensive visualizations with performance metrics
- **Backward Compatibility**: Existing code continues to work unchanged

### 📊 **Consolidation Results**
- **Lines of Code Reduced**: ~800 lines removed from scattered modules
- **Functions Consolidated**: 15+ image processing functions unified
- **Debug Files Standardized**: Consistent naming `YYYYMMDD_HHMMSS_PhotoProcessor_##_operation.png`
- **Performance Improved**: Centralized parameter management and caching

### 🔧 **Three-Class Architecture**

**PhotoshopProcessor**:
- Photoshop-compatible levels adjustment with gamma correction
- Input/output black/white point mapping
- Debug visualization with histograms and gamma curves
- Statistics tracking (min/max/mean values)

**EdgeDetector**:
- Unified Sobel edge detection (vertical/horizontal)
- Adaptive thresholding with confidence scoring
- ROI-based detection with coordinate conversion
- Comprehensive debug visualization with profile plots

**BinaryProcessor**:
- Adaptive thresholding (Gaussian/Mean)
- Morphological operations (opening/closing/gradient)
- Connected component analysis
- Before/after comparison visualizations

### 📌 **Module-Level Compatibility Functions**
```python
# Drop-in replacements for existing code
from modules.m_photo_processor import (
    apply_photoshop_levels,     # Replaces m_Card_Processing._apply_level_adjustment
    find_vertical_edge_x,       # Replaces image_utils.ffind_vertical_edge_x  
    create_gradient_profile,    # Replaces image_utils.fcreate_gradient_profile
    apply_adaptive_threshold,   # Consolidates timestamp_detector thresholding
    detect_edges_canny         # Replaces image_utils.fdetect_edges_canny
)
```

### 🎨 **Enhanced Debug System**
- **Automatic Visualization**: Debug mode generates comprehensive analysis images
- **Performance Metrics**: Processing time tracking and optimization insights
- **Parameter Display**: All settings and thresholds shown in visualizations
- **Comparison Views**: Before/after side-by-side analysis
- **Confidence Scoring**: Quality metrics for detection algorithms

## 11. Development Standards

### 📋 Development Quick Reference

**KISS Checklist** (Before writing ANY code):
1. ✅ Can existing code handle this?
2. ✅ Is inline implementation sufficient?  
3. ✅ Is this the simplest solution?
4. ✅ Do I need user approval for complexity?

**Prohibited Without Permission**:
- Creating new `.py` files
- Creating new classes  
- Complex abstractions

**Every Feature Must Have**:
- Diagnostic interface (`step_diagnostic_server.py`)
- Input/Output contracts
- Visual verification (for CV features)
- Error handling

**File Naming**: `filepath_screenshot`, `filename_config` (descriptive + type)

---

### 11.1 Documentation Accuracy Requirements

**CRITICAL RULE**: Claude MUST verify and update documentation accuracy after EVERY code change.

**Immediate Validation Required**:
- **Comments in Code**: Match actual implementation
- **CLAUDE.md Examples**: All imports and function calls work
- **File References**: All paths and filenames are current
- **Module Names**: No references to deleted/renamed modules
- **Function Signatures**: Parameters and return types are accurate

**Failure to Validate = Incomplete Task**: Any development work is considered INCOMPLETE until documentation accuracy is verified.

### 11.2 Human Structural Logic for Code Organization

**Core Principle**: "Human cognition organizes code into Modules → Tools → Steps"

In human cognitive structure, code follows a clear three-tier hierarchy for optimal comprehension and maintenance:

#### **Three-Tier Architecture**
- **📦 Modules**: High-level functional areas (m_Card_Processing.py, m_photo_processor.py, m_screenshot_processor.py)
- **🔧 Tools**: Reusable components called across multiple phases and modules
- **🎯 Steps**: Individual operations within specific processes

#### Tool Classification Rules
**MANDATORY**: If functionality meets any of these criteria, it MUST be implemented as a Tool Class:

1. **Repeated Usage Rule**: Called from 3+ different locations → Tool Class
2. **Cross-Module Usage**: Used across different modules → Tool Class  
3. **Frequency Rule**: Called repeatedly in a session → Tool Class
4. **Independence Rule**: Functionality independent from specific detector/processor logic → Tool Class

#### Tool Implementation Standards
```python
# ✅ CORRECT: Tool Class Pattern
class cWeChat_Screenshot_Finder:
    """
    📋 PURPOSE: Centralized tool for screenshot discovery
    🔧 HUMAN LOGIC: Repeatedly-called functionality = Tool Class
    🎯 USAGE: Called across multiple modules and phases
    """
    
    def get_latest_screenshot(self) -> Optional[str]:
        """Primary tool function"""
    
    def get_all_screenshots(self) -> List[str]:
        """Extended tool functionality"""

# ✅ USAGE: Clean tool integration
from modules.m_screenshot_finder_tool import cWeChat_Screenshot_Finder
finder = cWeChat_Screenshot_Finder()
latest = finder.get_latest_screenshot()
```

#### **🗂️ Tool Organization Guidelines**
- **File Location**: `modules/m_*_tool.py` (e.g., `m_screenshot_finder_tool.py`)
- **Class Naming**: `cTool_Name` convention (e.g., `cWeChat_Screenshot_Finder`)
- **Global Instances**: Provide convenience functions for easy integration
- **Documentation**: Include comprehensive PURPOSE and HUMAN LOGIC sections

#### **🎯 Tool vs Module vs Step Decision Matrix**

| Characteristic | Module | Tool | Step |
|---------------|--------|------|------|
| **Scope** | Complete functional area | Reusable component | Single operation |
| **Usage** | Imported once per area | Called repeatedly | Called within module |
| **Dependencies** | May use tools | Independent/minimal | Part of larger workflow |
| **Examples** | `m_Card_Processing` | `cWeChat_Screenshot_Finder` | `detect_width()` |
| **Human Logic** | "What does this do?" | "How do I do this?" | "What's this step?" |

#### **🚀 Benefits of Tool-Based Architecture**
- **Human Comprehension**: Clear mental model of code organization
- **Maintainability**: Single source of truth for repeated functionality  
- **Performance**: Centralized caching and optimization opportunities
- **Testing**: Isolated tool testing separate from complex module logic
- **Documentation**: Clear guidelines for consistent architecture decisions

#### **⚠️ Common Tool Candidates to Watch For**
- `get_latest_screenshot()` - Screenshot finding (✅ Now implemented as tool)
- `validate_*()` functions called across modules
- `find_*()` functions used by multiple detectors
- `cache_*()` functions for performance optimization
- `format_*()` functions for consistent output formatting

**This structural logic ensures code remains intuitive to human developers and maintains clear separation of concerns.**

### 11.3 KISS Principle Implementation

**Core Philosophy**: All development follows KISS principles established in Section 0.

**Before Writing Any Code**:
1. ✅ Can existing code handle this? → Use it
2. ✅ Can a simple function work? → Write it inline  
3. ✅ Is a new class really needed? → Probably not
4. ✅ User approved complexity? → Only then proceed

**Prohibited Without User Approval**:
- Creating new `.py` files or modules
- Creating new classes (even within existing files)
- Adding complex abstractions or design patterns

### 11.4 Mandatory Requirements

**Every feature MUST have diagnostic interface** for human verification

**Standard Components**:
3. **Visual verification** (MANDATORY for all computer vision features)
4. **Error handling** with detailed diagnostics
5. **Input/Output Contracts** (MANDATORY for all processing steps)
6. **Class Section Headers with I/O Info** (MANDATORY for all major classes and modules)

**Input/Output Contract Requirements for ALL Processing Steps**:
- **Mandatory Documentation**: Every function/method must define clear input/output contracts
- **Standardized Format**: Use INPUT CONTRACT and OUTPUT CONTRACT sections
- **Complete Specifications**: Include data types, constraints, expected formats, and side effects
- **Failure Modes**: Document all possible failure conditions and return values
- **Dependencies**: List required prerequisites and upstream processing results

**Contract Format Template**:
```python
def process_step(self, image_path: str, coord_context = None) -> ResultType:
    """
    Brief description of processing step
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
    - coord_context: Optional[WeChatCoordinateContext] - Context for integration
    - return_context: bool - Whether to return coordinate context with results
    
    📌 OUTPUT CONTRACT:
    Standard Mode:
    - Success: SpecificType - Detailed description of success return value
    - Failure: None or Empty - Description of failure conditions
    
    Side Effects:
    - File generation, context updates, validation requirements
    - Dependencies on other processing steps
    """
```

**Class Section Header Requirements for ALL Modules**:
- **MANDATORY**: All major classes must have standardized section headers
- **INPUT/OUTPUT REQUIRED**: Every class header MUST specify input/output information
- **Format**: Use ASCII art box style with descriptive title in ALL CAPS
- **Placement**: Immediately before class definition, after any preceding code blocks
- **Consistency**: Follow the established pattern from existing modules
- **Purpose**: Visual organization and immediate input/output clarity
- **Validation**: Claude must verify input/output info matches class functionality

**Class Section Header Template**:
Each class do one thing. 
```python
# =============================================================================
# DESCRIPTIVE CLASS PURPOSE IN ALL CAPS
# INPUT: file_types, data_types, parameters
# OUTPUT: file_types, data_types, results
# =============================================================================

It could be treated a phase in the modual. Within the class, it could have multiple steps as methods.

# =============================================================================
# PHASE 1: LEFT BOUNDARY DETECTOR
# INPUT: file_types, data_types, parameters
# OUTPUT: file_types, data_types, results
# =============================================================================

It could be a tool that is called by multiple phases.
# =============================================================================
# Tool 1: Get last screenshot
# INPUT: file_types, data_types, parameters
# OUTPUT: file_types, data_types, results
# =============================================================================


class YourClassName:
    """
    Class documentation with full CLAUDE.md documentation structure
    """
```

**MANDATORY INPUT/OUTPUT FORMAT**:
- **INPUT**: Concise summary of what data/files the class processes
- **OUTPUT**: Concise summary of what data/files the class produces  
- **Format**: Keep each line under 80 characters for readability
- **Consistency**: Use standardized terminology across all classes

**Examples of Proper Section Headers**:
```python
# =============================================================================
# WECHAT COORDINATE CONTEXT SYSTEM
# INPUT: WeChat screenshots, coordinate data, window dimensions
# OUTPUT: coordinate context objects, calibrated positions
# =============================================================================


```

**📚 Complete Example - Header + Docstring Integration**:
```python
# =============================================================================
# PHASE 6: TIME BOX DETECTOR  
# INPUT: WeChat screenshots (PNG/JPG), detection parameters, ROI bounds
# OUTPUT: avatar coordinates, confidence scores, debug visualizations
# =============================================================================

class AvatarDetector:
    """
    Detects circular avatar images in WeChat contact cards using OpenCV.
    
    📋 PURPOSE:
    This detector implements PHASE 3 of the WeChat card processing pipeline.
    Locates and extracts avatar positions to enable precise click targeting.
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
    - roi_bounds: Tuple[int,int,int,int] - (x,y,w,h) region of interest
    - min_radius: int - Minimum avatar radius in pixels (default: 15)
    - max_radius: int - Maximum avatar radius in pixels (default: 35)
    
    📌 OUTPUT CONTRACT:
    - Success: List[Dict] - [{"x": int, "y": int, "radius": int, "confidence": float}]
    - Failure: [] - Empty list if no avatars detected
    - Side Effects: Generates debug_avatar_detection_TIMESTAMP.png if debug enabled
    
    🔧 ALGORITHM:
    1. Load image and extract specified ROI region
    2. Apply Gaussian blur and convert to grayscale
    3. Use HoughCircles detection with adaptive parameters
    4. Filter results by radius constraints and confidence thresholds
    5. Return sorted results by confidence score
    """

    # Step 1: Get validated card and avatar data
        cards, card_detection_info = self.card_boundary_detector.detect_cards(image_path)
        if not cards:
            print("❌ No cards available for time detection")
            return [], {}
            
    # Step 2Load image for processing
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Failed to load image: {image_path}")
        return [], {}
            
    # Step 3: Get panel right boundary from width detection
    width_boundaries = card_detection_info.get("width_boundaries")
    if not width_boundaries:
        print("❌ No width boundaries available for time detection")
        return [], {}
```

**🔍 Header-Docstring Relationship**:
- **Header INPUT** summarizes data types → **Docstring INPUT CONTRACT** provides parameter details
- **Header OUTPUT** summarizes result types → **Docstring OUTPUT CONTRACT** specifies exact formats  
- **Header** gives immediate overview → **Docstring** provides implementation details
- Both must be consistent and maintained together

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

### 11.5 Development Workflow

**Pre-Development**:
1. Apply KISS checklist (Section 11.3)
2. Review [`process.md`](process.md) and relevant documentation
3. Verify WeChat setup and coordinate calibration
4. Check existing diagnostic tools for similar patterns

**Implementation**:
1. **Architecture Decision**: Default to inline implementation in existing files
2. **Input/Output Contracts**: Define clear contracts with parameters and failure modes
3. **Feature Implementation**: Implement with comprehensive error handling
4. **Diagnostic Interface**: Create endpoint in `step_diagnostic_server.py` and frontend button
5. **Visual Verification**: Implement overlay generation for computer vision features
6. **Testing**: Test error handling, edge cases, and visual accuracy

**Post-Development**:
1. **MANDATORY**: Update [`process.md`](process.md) with workflow changes
2. Move temporary files to `TestRun/` directory
3. Verify end-to-end diagnostic functionality
4. Update relevant documentation files
5. **🚨 MANDATORY VALIDATION**: Verify documentation accuracy and consistency

### 11.6 Variable Naming Standards

**🎯 CRITICAL RULE**: Use descriptive variable names that indicate file type and purpose.

**File Path Variable Naming Conventions**:

**✅ CORRECT Examples**:
```python
# Full file paths
filepath_screenshot = os.path.join(output_dir, "20250905_203603_WeChat.png")
filepath_coordinates = os.path.join(output_dir, "wechat_window_coords.json") 
filepath_config = os.path.join(config_dir, "settings.json")
filepath_log = os.path.join(log_dir, "debug.log")

# Just filenames (basenames)
filename_screenshot = "20250905_203603_WeChat.png"
filename_coordinates = "wechat_window_coords.json"
filename_temp_screenshot = "20250905_203603_OCR_fullscreen.png"
```

**❌ INCORRECT Examples**:
```python
filepath = os.path.join(output_dir, filename)  # Too generic!
path = "some/file.png"                         # Ambiguous!
file = "data.json"                            # No context!
filename = "screenshot.png"                    # Generic - what type of file?
custom_filename = "test.png"                  # Unclear purpose
temp_filename = "temp.png"                    # No file type indication
```

**Naming Rules**:
1. **Use prefix to indicate path level**:
   - **`filepath_`** prefix for full paths: `/path/to/file.png`
   - **`filename_`** prefix for basenames: `file.png`

2. **Always use descriptive suffixes** indicating file type:
   - `filepath_screenshot` / `filename_screenshot` for PNG/JPG image files
   - `filepath_coordinates` / `filename_coordinates` for JSON coordinate files
   - `filepath_config` / `filename_config` for configuration files
   - `filepath_log` / `filename_log` for log files
   - `filepath_temp_screenshot` / `filename_temp_screenshot` for temporary files

3. **Never use generic names** without context:
   - ❌ `filepath`, `path`, `file` 
   - ❌ `filename`, `custom_filename`, `temp_filename` (generic - no file type)
   - ❌ `f`, `p`, `fn` (abbreviations)

3. **Consistency across methods**: Similar variables in different methods should follow the same naming pattern

4. **Method parameter clarity**: Parameters should be self-documenting
   ```python
   # ✅ GOOD: Clear distinction between basename and full path
   def capture_screenshot(self, filename_screenshot: str = None) -> Optional[str]:
       if not filename_screenshot:
           filename_screenshot = f"{timestamp}_WeChat.png"
       filepath_screenshot = os.path.join(self.output_dir, filename_screenshot)
       return filepath_screenshot
   
   # ❌ BAD: Generic, confusing variable names
   def capture_screenshot(self, filename: str = None) -> Optional[str]:
       if not filename:
           filename = f"{timestamp}_WeChat.png" 
       path = os.path.join(self.output_dir, filename)
       return path
   ```

**Benefits**:
- **Eliminates Path Confusion**: Clear distinction between basename (`filename_`) and full path (`filepath_`)
- **Eliminates Type Confusion**: Clear distinction between different file types (_screenshot, _coordinates, etc.)
- **Improves Maintainability**: Self-documenting variable names that indicate both path level and file type
- **Reduces Bugs**: Less chance of passing wrong file type or path level to methods
- **Consistent Pattern**: All functions follow same parameter naming conventions
- **Enforces Standards**: Consistent patterns across the codebase

**Validation Requirement**: All filepath variables must be validated during documentation accuracy checks (Section 10.0).

### 5.1 Card Processing Module Filename Conventions

The Card Processing module (`modules/m_Card_Processing.py`) generates debug visualization files that follow a strict naming convention for easy identification and organization.

**Naming Pattern**: `{timestamp}_{section#}_{description}.png`

**Section Mapping**:
- **Section 1**: `01_` - SimpleWidthDetector (Width Detection)
- **Section 2**: `02_` - RightBoundaryDetector (High-Contrast Preprocessing) 
- **Section 3**: `03_` - CardAvatarDetector (Avatar Detection)
- **Section 4**: `04_` - CardBoundaryDetector (Card Boundaries)
- **Section 5**: `05_` - ContactNameBoundaryDetector (Contact Names)
  - **Simple**: `05_ContactName_Boundaries_Xnames_Ycards.png` - Basic overlay visualization
  - **Comprehensive**: `05_Debug_ContactNameDetection_Xsuccess_Yfailed.png` - Multi-panel debug output (NEW)
- **Section 6**: `06_` - TimeBoxDetector (Time Boxes)
  - **Comprehensive**: `06_Debug_TimeDetection_Xsuccess_Yfailed.png` - Multi-panel debug output

**File Examples**:
```
20250905_203603_01_EnhancedDualBoundary_365px.png
20250905_203603_01_SimpleWidth_365px.png
20250905_203603_02_photoshop_levels_gamma.png
20250905_203603_03_advanced_avatar_detection.png
20250905_203603_04_Enhanced_Card_Avatar_Boundaries_9cards.png
20250905_203603_05_ContactName_Boundaries_2names_9cards.png        # Simple overlay
20250905_203603_05_Debug_ContactNameDetection_2success_7failed.png # Comprehensive debug (NEW)
20250905_203603_06_Complete_Analysis_7times_9cards.png
20250905_203603_06_Debug_TimeDetection_7success_2failed.png        # Comprehensive debug
```

### 5.2 Comprehensive Debug Visualization System (NEW)

**Enhanced Visual Debugging**: Section 5 (Contact Name Detection) now includes comprehensive debug visualization matching the quality of time detection diagnostic output.

**Features**:
- **Multi-Panel Layout**: Main overview + individual ROI analysis + binary processing + statistical charts
- **Success/Failure Indicators**: Color-coded annotations showing detection results per card
- **Processing Pipeline**: Visual representation of white text detection, morphological operations, and filtering steps
- **Algorithm Parameters**: Real-time display of thresholds, size constraints, and confidence scores
- **ROI Analysis**: Detailed view of search regions, binary masks, and contour detection per card

**API Endpoints**:
- **Standard**: `/api/test-contact-name-boundary-detector` - Enhanced to use comprehensive debug by default
- **Dedicated**: `/api/test-contact-name-comprehensive-debug` - Comprehensive debug visualization only

**Generated Files**: `YYYYMMDD_HHMMSS_05_Debug_ContactNameDetection_Xsuccess_Yfailed.png`

**Implementation Notes**:
- All visualization methods use consistent `{timestamp}_0{section}_` prefix
- Debug files are saved to `pic/screenshots/` directory
- Timestamp format: `YYYYMMDD_HHMMSS`
- Additional metadata (dimensions, counts) included in filename for quick reference

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

## 9. Security Alert

🚨 **CRITICAL VULNERABILITIES** - [Full Security Analysis](docs/SECURITY.md)

**Immediate Action Required**:
1. **SQL Injection Risk** - Database uses string interpolation instead of parameterized queries
2. **Input Validation Missing** - No sanitization of AI-processed messages  
3. **Credential Exposure** - API keys stored in plain text environment variables
4. **Data Encryption Missing** - Screenshots and database unencrypted

**Current Status**: ⚠️ **High Risk** - Production use not recommended until fixes applied

**Remediation**: See [Security Guide](docs/SECURITY.md) for detailed fixes and implementation roadmap

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

### Screenshot Processing Protocol (NEW)
🚨 **MANDATORY RULE**: All future screenshot and image processing development must use `modules/m_screenshot_processor.py`
- **NEVER create new screenshot capture modules** - extend existing consolidated module
- **NEVER import deleted modules** (`m_ScreenShot_WeChatWindow`, old `screenshot_processor`)
- **ALWAYS use** functions with 'f' prefix: `fcapture_screenshot()`, `fcapture_messages_screenshot()`
- **SINGLE SOURCE OF TRUTH**: One module, one approach, consistent architecture
- **Cross-platform compatibility**: Handles macOS Quartz, OCR fallback, window detection

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




## 10. Getting Started

### 📋 Quick Reference

**Essential Commands**:
```bash
# Start diagnostic interface
python step_diagnostic_server.py    # → http://localhost:5001

# Run the bot
python app.py

# Test components
python capture/deal_chatbox.py       # Test OCR
python modules/m_screenshot_processor.py  # Test screenshot processing
```

**Key Files**:
- `CLAUDE.md` - This development guide
- `process.md` - Definitive workflow guide  
- `Constants.py` - GUI coordinates (system-specific)
- `step_diagnostic_server.py` - Web diagnostic console

**Critical Directories**:
- `modules/` - Core processing components
- `docs/` - Detailed documentation
- `pic/screenshots/` - Image storage and debug output

**Before You Start**:
1. ✅ WeChat desktop running and visible
2. ✅ Check health indicators (Section 1)  
3. ✅ Read KISS principles (Section 0)
4. ✅ Review workflow (process.md)

---

### 10.1 For New Developers
1. **Read**: [`docs/SETUP.md`](docs/SETUP.md) for installation and configuration
2. **Understand**: [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for system design
3. **Review**: [`process.md`](process.md) for definitive workflow guide

### 10.2 For System Operations
1. **Security**: [`docs/SECURITY.md`](docs/SECURITY.md) for vulnerability analysis and fixes
2. **Diagnostics**: [`docs/DIAGNOSTICS.md`](docs/DIAGNOSTICS.md) for troubleshooting tools
3. **Maintenance**: [`docs/MAINTENANCE.md`](docs/MAINTENANCE.md) for operational procedures

### 10.3 Quick Commands
```bash
# Development diagnostics
python step_diagnostic_server.py     # WDC (Web Diagnostic Console)

# Component testing
python capture/deal_chatbox.py       # Test OCR
python capture/monitor_new_message.py # Test detection

# Module testing (NEW)
python modules/image_utils.py        # Test image utility functions
python modules/visualization_engine.py # Test visualization engine
python modules/m_screenshot_processor.py # Test screenshot processing

# Database inspection
sqlite3 history.db "SELECT * FROM conversations LIMIT 5;"
```

### 10.4 Modular Development Patterns (NEW)

**Using Detector Classes** (Recommended approach):
```python
from modules import m_Card_Processing

# Initialize detectors
boundary_coordinator = m_Card_Processing.BoundaryCoordinator()
avatar_detector = m_Card_Processing.CardAvatarDetector()

# Use as before - full backward compatibility
width_result = boundary_coordinator.detect_width("screenshot.png")
avatars, info = avatar_detector.detect_avatars("screenshot.png")
```

### 10.5 Detector Class Documentation Standards

**MANDATORY**: Every detector class in `m_Card_Processing.py` must follow this documentation standard:

**NOTE**: The processing pipeline uses **PHASES** (major processing stages) containing **algorithm steps** (individual operations within each phase). This creates clear hierarchical organization.

#### Class Documentation Template:
```python
class DetectorClassName:
    """
    Brief one-line description of what this detector does.
    
    📋 PURPOSE:
    Detailed explanation of the detector's role in the processing pipeline.
    This detector implements PHASE X of the 6-phase WeChat card processing pipeline.
    What specific problem does it solve? What does it detect or analyze?
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
    - param2: Type - Description of parameter and constraints
    - param3: Optional[Type] - Optional parameters with defaults
    
    📌 OUTPUT CONTRACT:
    - Success: ReturnType - Detailed description of success return value
    - Failure: None or ErrorType - Description of failure conditions
    - Example: Tuple[int, int, int] - (left_boundary, right_boundary, width)
    
    🔧 ALGORITHM:
    1. Load and validate input data
    2. Apply preprocessing transformations  
    3. Execute core detection/analysis logic
    4. Post-process and validate results
    5. Generate output with confidence scoring
    
    📊 KEY PARAMETERS:
    - PARAM_NAME = value  # What this parameter controls
    - THRESHOLD = 0.5     # Why this threshold was chosen
    
    🎨 VISUAL OUTPUT:
    - Debug mode generates: filename_pattern.png
    - Visualization shows: what elements are highlighted
    - Color coding: green=detected, red=failed, etc.
    
    🔍 DEBUG VARIABLES:
    - Key variables used for visualization:
      • variable_name: Type - Description of what it represents
      • confidence_score: float - Used to show detection quality
      • profile_array: np.array - Plotted as intensity graph
    - Debug triggers: When debug_mode=True or specific thresholds exceeded
    - Output format: Images saved to pic/screenshots/ with timestamp
    
    ⚙️ DEPENDENCIES:
    - Required: opencv-python, numpy
    - Optional: visualization_engine (for debug output)
    - Integrates with: Other detector classes if applicable
    """
```

#### Method Documentation Requirements:
```python
def detect_method(self, image_path: str, other_params) -> ReturnType:
    """
    Brief description of what this method does.
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to screenshot file
    - other_params: Type - Description and constraints
    
    📌 OUTPUT CONTRACT:
    - Success: Type - What is returned on success
    - Failure: None - What happens on failure
    
    Side Effects:
    - File generation, context updates, etc.
    """
```

#### Example Implementation:
```python
class LeftBoundaryDetector:
    """
    Detects the left boundary of WeChat conversation area using vertical edge detection.
    
    📋 PURPOSE:
    Identifies the visual boundary between WeChat's sidebar and the main conversation
    area by finding the strongest vertical edge in the left portion of the screen.
    This detector implements PHASE 1 of the 6-phase WeChat card processing pipeline.
    This establishes the left coordinate for all subsequent message card processing.
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
    - debug_mode: bool - Enable visualization output (default: False)
    
    📌 OUTPUT CONTRACT:
    - Success: int - X-coordinate of left boundary in pixels
    - Failure: None - Returns None if detection fails
    - Range: Typically 50-150px from left edge
    
    🔧 ALGORITHM:
    1. Load image and focus on left 65% (conversation area)
    2. Apply Sobel edge detection to find vertical edges
    3. Create 1D intensity profile by averaging gradients
    4. Find strongest edge peak using Gaussian smoothing
    5. Apply 8px offset for actual boundary position
    
    📊 KEY PARAMETERS:
    - CONVERSATION_WIDTH_RATIO = 0.65  # Search left 65% of screen
    - SIDEBAR_OFFSET = 8               # Pixels offset from edge to boundary
    - EDGE_THRESHOLD_LOW = 30          # Minimum edge strength
    
    🎨 VISUAL OUTPUT:
    - Debug file: YYYYMMDD_HHMMSS_01_LeftBoundary_XXXpx.png
    - Yellow line: Detected edge position
    - Green line: Actual boundary (with offset)
    - Text overlay: Coordinates and confidence score
    
    🔍 DEBUG VARIABLES:
    - Key variables used for visualization:
      • detected_edge: int - X-coordinate where vertical edge detected (yellow line)
      • left_boundary: int - Final boundary after offset applied (green line)
      • confidence: float (0-1) - Peak strength vs noise ratio (text overlay)
      • profile: np.array - 1D gradient intensity profile (could be plotted)
    - Debug triggers: When debug_mode=True in constructor
    - Output format: PNG saved to pic/screenshots/ with timestamp prefix
    
    ⚙️ DEPENDENCIES:
    - Required: opencv-python, numpy
    - Optional: image_utils.find_vertical_edge_x (modular import)
    - Integrates with: BoundaryCoordinator for complete width detection
    """
```

#### Documentation Validation Checklist:
- [ ] Class has complete docstring with all sections
- [ ] PURPOSE clearly explains the detector's role
- [ ] INPUT CONTRACT lists all parameters with types
- [ ] OUTPUT CONTRACT specifies return values and failure modes
- [ ] ALGORITHM provides step-by-step explanation
- [ ] KEY PARAMETERS documents important constants
- [ ] VISUAL OUTPUT describes debug visualizations
- [ ] DEBUG VARIABLES lists key variables used in visualization
- [ ] DEPENDENCIES lists required and optional imports

#### Debug Documentation Best Practices:
1. **List ALL variables** passed to debug visualization methods
2. **Explain visual mapping**: Which variable creates which visual element
3. **Note unused variables**: If a variable is passed but not visualized (e.g., profile array)
4. **Describe triggers**: When debug mode activates (constructor flag, thresholds, errors)
5. **Include examples**: Show actual values from test runs when possible

**Using Individual Modules** (Advanced/custom development):
```python
from modules import m_screenshot_processor, visualization_engine, image_utils

# Screenshot processing - ALL screenshot operations must use this module
screenshot_path, analysis = m_screenshot_processor.fcapture_and_process_screenshot()
basic_screenshot = m_screenshot_processor.fcapture_screenshot()
message_screenshot = m_screenshot_processor.fcapture_messages_screenshot()

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
from modules import m_Card_Processing, m_screenshot_processor, visualization_engine

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

## 12. Documentation Maintenance Requirements (General)

### 12.1 🚨 CRITICAL: Documentation Validation Protocol

**MANDATORY RULE**: Claude must verify documentation accuracy and consistency after EVERY code change.

**Documentation Validation Checklist** (MUST complete after every development task):
- [ ] **Comments Accuracy**: All code comments match actual implementation
- [ ] **CLAUDE.md Consistency**: File paths, function names, and examples are correct
- [ ] **Import Statements**: All import examples in documentation work with current codebase
- [ ] **Function Signatures**: All documented functions exist with correct parameters
- [ ] **File Structure**: Directory listings match actual project structure
- [ ] **Module References**: No references to deleted or renamed modules
- [ ] **API Endpoints**: All diagnostic endpoints mentioned actually exist
- [ ] **Dependencies**: All documented libraries and requirements are current

**Validation Process**:
1. **Before Code Changes**: Note which documentation sections will be affected
2. **During Development**: Update comments in code as functions are modified
3. **After Code Changes**: Review and update ALL affected documentation sections
4. **Final Check**: Verify all examples, imports, and references work correctly

**Documentation Impact Assessment** - Update these when making changes:
- **File/Module Changes** → Update CLAUDE.md project structure and import examples
- **Function Changes** → Update CLAUDE.md function listings and signatures
- **Architecture Changes** → Update CLAUDE.md, process.md, and relevant docs/ files
- **New Features** → Update CLAUDE.md, add examples, update workflow documentation
- **Workflow Changes** → Update process.md with step-by-step accuracy
- **Performance Changes** → Update benchmarks and timing information
- **API Changes** → Update diagnostic endpoint documentation

### 12.2 CRITICAL: Process Documentation Updates

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


---

**Note**: This main `CLAUDE.md` provides overview and navigation. Detailed information is in the specialized documentation files in the [`docs/`](docs/) directory.


### 🧪 MANDATORY MANUAL CODE TESTING PATTERN

**CRITICAL RULE**: ALL modules MUST include a simplified Manual Code Testing section.

**SIMPLIFIED SMOKE TEST REQUIREMENTS**:
- **Purpose**: Simply instantiate all classes/call main functions to verify imports work
- **Pattern**: All modules MUST include `if __name__ == "__main__":` section
- **Performance**: Must complete quickly with clear pass/fail indication
- **Simplicity**: Just instantiate classes in order - modules already have their own debug processes

**SIMPLIFIED SMOKE TEST TEMPLATE**:
```python
# =============================================================================
# MANUAL CODE TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Code Testing - [MODULE_DISPLAY_NAME]")
    print("=" * 60)
    print("🔍 [DEBUG] Smoke test ENTRY")
    
    try:
        # Simply instantiate each class in order
        print("   🔧 Testing Class1...")
        obj1 = Class1()
        print("   ✅ Class1 instantiated successfully")
        
        print("   🔧 Testing Class2...")  
        obj2 = Class2()
        print("   ✅ Class2 instantiated successfully")
        
        print("🏁 [DEBUG] Smoke test PASSED")
        
    except Exception as e:
        print(f"   ❌ [ERROR] Smoke test FAILED: {str(e)}")
        print("🏁 [DEBUG] Smoke test FAILED")
```

**EXECUTION VERIFICATION**:
Every smoke test must be verifiable by running:
```bash
python modules/module_name.py
```
And produce clear console output indicating success or failure.