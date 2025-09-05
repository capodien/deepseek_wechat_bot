# WeChat Bot Process Documentation

## Overview

This WeChat desktop automation bot uses computer vision (OCR), AI integration (DeepSeek), and GUI automation to automatically respond to WeChat messages. The system monitors WeChat windows, detects new messages, extracts text content, generates intelligent responses, and types them back into WeChat.

## System Architecture

### Core Components

1. **Main Application** (`app.py`) - Core orchestration and main execution loop
2. **Message Detection** (`capture/monitor_new_message.py`) - Detects new message notifications
3. **OCR Processing** (`capture/deal_chatbox.py`) - Extracts text from screenshots using EasyOCR
4. **Contact Recognition** (`capture/get_name_free.py`) - Identifies sender names
5. **AI Integration** (`deepseek/deepseekai.py`) - Generates responses using DeepSeek API
6. **Database Layer** (`db/db.py`) - Stores conversation history and messages
7. **WDC (Web Diagnostic Console)** (`step_diagnostic_server.py`) - Real-time monitoring web interface

### Modular Components (NEW ARCHITECTURE)

8. **Card Processing Orchestrator** (`modules/m_Card_Processing.py`) - Main detector classes and coordination
   - `SimpleWidthDetector` - Message card width boundary detection
   - `CardAvatarDetector` - Avatar position detection within cards
   - `CardBoundaryDetector` - Individual card boundary detection
   - `ContactNameBoundaryDetector` - Name region detection
   - `TimeBoxDetector` - Timestamp detection
   - `RightBoundaryDetector` - Right boundary detection with preprocessing

9. **Screenshot Processor** (`modules/screenshot_processor.py`) - Screenshot I/O operations and processing
   - `capture_and_process_screenshot()` - Live capture with comprehensive analysis
   - `process_screenshot_file()` - Process existing screenshot files
   - `get_live_card_analysis()` - Real-time analysis with optional visualizations

10. **Visualization Engine** (`modules/visualization_engine.py`) - Centralized visualization utilities
    - `VisualizationEngine` class for consistent visual overlays
    - Heatmap generation and composite visualizations
    - Standardized color schemes and debug outputs

11. **Image Utilities** (`modules/image_utils.py`) - Shared image processing functions
    - `find_vertical_edge_x()` - Vertical edge detection with confidence scoring
    - `apply_level_adjustment()` - Photoshop-style image preprocessing
    - Common edge detection and morphological operations

### Key Libraries

- **pyautogui**: Cross-platform GUI automation (clicking, typing, screenshots)
- **easyocr**: Optical Character Recognition for Chinese/English text
- **opencv-python**: Image processing and computer vision
- **openai**: DeepSeek API integration (OpenAI-compatible interface)
- **flask/socket.io**: Web diagnostic panel with real-time updates
- **sqlite3**: Local database storage
- **pyperclip**: Clipboard operations for text input

## Main Process Flow

### Initialization Phase

```
1. Load Configuration
   ├── Read names.txt (monitored contacts)
   ├── Load environment variables (.env)
   ├── Initialize SAFE_MODE and WORK_MODE settings
   └── Setup logging and diagnostic systems

2. System Setup
   ├── Start diagnostic web server (port 5001)
   ├── Initialize OCR readers (Chinese/English, GPU acceleration)
   ├── Setup keyboard listeners (ESC key monitoring)
   ├── Connect to database systems
   └── Setup cross-platform GUI automation

3. Pre-execution Tasks
   ├── Clean old screenshots (>200 files → cleanup)
   ├── Add monitored users to AI conversation history
   ├── Update diagnostic panel status
   └── Begin main monitoring loop
```

### Main Execution Loop

The bot operates in a continuous 7-step cycle:

#### Step 1: Screenshot Capture
```python
#Location: modules/m_ScreenShot_WeChatWindow.py
#Method: WeChatScreenshotCapture
#Technique: Cross-platform GUI automation with window detection
Process:
1. Take fullscreen screenshot of WeChat window
2. Save to pic/screenshots/ with timestamp
3. Update diagnostic panel with screenshot info
4. Record capture performance metrics
```

#### Step 2: New Message Detection
```python
# Location: capture/monitor_new_message.py
# Method: Selective avatar-based detection with OCR filtering
# Technique: Avatar detection + Username OCR + Red dot detection + Smart filtering
Process:

1. Primary: Selective red dot detection (optimized workflow)
Step 1: Take the screen shot

Step 2: Detect message card boundaries first (using the obvious visual separations)
Step 3: Within each detected card, find the avatar
│   Use TestRun/opencv_adaptive_detector.py
   │    1. detect_contact_cards() method processes the screenshot
         2. Usesn
         3. Filters results by size and aspect ratio
         4. Calculates click coordinates (center + 70px offset)
         5. Formats the results in the "• Card X: Center(x, y) → OpenCV adaptive thresholding and contour detectio Click(x, y)" format
         6. each card (coordinates) must be numbered so it can be referenced later. 

Step 2.5: Define OCR Zone Boundaries (ENHANCED)
│   Use modules/m_OCRZone_MessageCards.py with Real Boundary Detection
│   ├── Input: Latest WeChat screenshot (auto-detected from pic/screenshots/)
│   ├── Enhanced Boundary Detection (modules/m_CardBoundaryDetection.py):
│   │   1. Horizontal Sobel edge detection to find natural card breaks
│   │   2. Content-aware clustering to identify actual message boundaries
│   │   3. Variable-height card detection (replaces uniform 80px grid)
│   │   4. Adaptive thresholding with median + k*MAD for robustness
│   │   5. Minimum card height enforcement (40px) with boundary validation
│   ├── OCR Zone Calculation for each REAL card boundary:
│   │   1. Calculate Avatar Zone: Circular area around detected avatar (50x50px default)
│   │   2. Calculate Username Zone: Text area right of avatar (200x25px, offset +10px right, -5px up)
│   │   3. Calculate Timestamp Zone: Right-aligned time area (100x20px, offset +200px right, -10px up)
│   │   4. Calculate Message Preview Zone: Main content area (250x20px, offset +10px right, +15px down)
│   │   5. Apply adaptive sizing based on ACTUAL card dimensions
│   │   6. Add configurable padding to all zones (5px default)
│   │   7. Validate zone positions within real card boundaries
│   │   8. Generate visual overlay with color-coded zones and boundary rectangles
│   ├── Output: Enhanced message cards with PRECISE OCR zone definitions aligned to actual content
│   ├── Performance: <50ms boundary detection + <100ms zone definition = <150ms total
│   ├── Detection Method: "real_boundary_detection" (vs legacy "1d_projection")
│   └── Visual validation: Overlay images showing real card boundaries and accurate zone alignment

3. Step 3: Calculate all other regions relative to the card boundaries
   1. Time
   2. Name
   3. Message previous box

   
   │
   ├── Step 2: Early filtering loop
   │   relevant_contacts = []
   │   for avatar in avatars:
   │       # Extract small username region (relative to avatar)
   │         │       
   │       # Quick OCR on small region (500ms vs 5000ms)
   │    
   │       # Early filter: Skip if not monitored
   │       Skip if username not in in the list names.txt
   │           continue  # SKIP - saves 8+ seconds per contact!
   │       
   │       # Only relevant contacts reach here
   │          if username not in in the list names.txt
               return the card number. 
   │
   └── Step 3: Process only relevant contacts
       for contact in relevant_contacts:
           # Check red dot status
           has_notification = self.check_red_dot(contact['card_region'])
           
           if has_notification:
               # Return click coordinates at center of avatar
               return True, contact['avatar']['avatar_center'], contact['username']

2. Fallback: Text change detection (detect_new_message_by_text_change)
   ├── Compare current screenshot with previous
   ├── Detect text differences in message area
   └── Return coordinates if change detected

3. Output detection results:
   ├── Return detection status (True/False)
   ├── Provide click coordinates if message found (avatar center)
   ├── Report contact username for matched contact
   └── Report detection method used ("selective_avatar" or "text_change")
```

#### Step 3: Message Click Action
```python
Location: app.py (main execution loop)
#Method: pyautogui.click + platform-specific movement
#Technique: Cross-platform GUI automation with natural movement simulation
Process:
1. Validate detection results from Step 2
2. Execute click at detected coordinates
   ├── Use pyautogui.click() for cross-platform compatibility
   ├── Apply random movement duration (0.2-0.5s) for natural behavior
   └── Platform-specific coordinate adjustment if needed
3. Wait for UI response (message window focus)
4. Proceed to contact identification
```

#### Step 4: Contact Recognition
```python
# Location: capture/get_name_free.py -> get_friend_name()
Process:
1. Take new screenshot after clicking message
2. Extract contact name using two methods:
   a) Search button detection method:
      ├── Look for WeChat search button element
      ├── Extract name from nearby text regions
      └── Higher accuracy but may fail on UI changes
   
   b) Fallback simplified method:
      ├── Use coordinate-based text extraction
      ├── More robust but less precise
      └── Always available as backup

3. Validate contact against monitoring list (names.txt)
4. Skip processing if contact not in monitored list
```

#### Step 5: OCR Text Extraction
```python
# Location: capture/deal_chatbox.py -> get_chat_messages()
Process:
1. Capture message area screenshot
2. Detect theme mode (light/dark) using brightness analysis
3. Apply theme-specific preprocessing:
   ├── Color space conversion (BGR/HSV)
   ├── Brightness adjustment
   ├── Contrast enhancement
   └── Noise reduction

4. EasyOCR Processing:
   ├── Load Chinese/English recognition models
   ├── Extract text regions with confidence scores
   ├── Classify as sent/received messages
   └── Filter low-confidence results

5. Post-processing:
   ├── Clean text (remove artifacts, fix spacing)
   ├── Extract latest message from conversation
   ├── Handle multi-line messages
   └── Return structured message data

Performance Analysis:
├── Image loading: ~20ms (0.2-0.4%)
├── Theme detection: ~0.2ms
├── Region detection: ~13-15ms
└── OCR processing: ~5000-9000ms (99.5%+ of total time)
```

#### Step 6: AI Response Generation
```python
# Location: deepseek/deepseekai.py -> reply()
Process:
1. Message Context Preparation:
   ├── Retrieve last 4 messages from conversation history
   ├── Add system prompt for WeChat chat context
   ├── Include error correction and tone instructions
   └── Format for DeepSeek API

2. API Request (Streaming):
   ├── Model: DeepSeek Chat
   ├── Temperature: 0.5, Top-p: 0.7
   ├── Max tokens: 384
   ├── Stream: True (real-time character output)
   └── Include usage statistics: False

3. Real-time Response Processing:
   ├── Receive streaming response character-by-character
   ├── Copy each character to clipboard
   ├── Paste immediately into WeChat input field
   └── Create typing animation effect

4. Response Finalization:
   ├── Clean response text (remove artifacts, tags)
   ├── Store in conversation history
   ├── Respect safe_mode setting for Enter key
   └── Generate performance report

Performance Breakdown:
├── API first response: 1100-1400ms (40-48%)
├── Streaming receive: 1300-2100ms (47-61%)
├── Content cleaning: <1ms (0.0%)
└── Total API time: 2700-3400ms
```

#### Step 7: Message Input & Transmission
```python
# Location: app.py -> send_reply() + deepseekai.py -> reply()
Process:
1. Dual Input Method:
   a) Real-time Streaming (during AI generation):
      ├── Character-by-character clipboard paste
      ├── Platform-specific hotkeys (Cmd+V/Ctrl+V)
      └── Creates natural typing animation

   b) Final Reply (via send_reply function):
      ├── Copy complete message to clipboard
      ├── Select all text in input field (Cmd+A/Ctrl+A)
      ├── Paste complete message
      └── Optional Enter key press based on mode

2. Safety Mode Handling:
   ├── SAFE_MODE=True: Type message, wait for manual Enter
   ├── SAFE_MODE=False: Type message and auto-press Enter
   └── Display appropriate user prompts

3. Performance Tracking:
   ├── Input timing measurement
   ├── Safe mode status logging
   └── Success/failure validation
```

## Operation Modes

### WORK_MODE Options

#### 1. CHAT Mode (Default)
```python
WORK_MODE = "chat"
- Uses AI (DeepSeek) to generate intelligent responses
- Maintains conversation context and history
- Provides personalized, contextual replies
- Higher processing time due to AI inference
```

#### 2. FORWARD Mode
```python
WORK_MODE = "forward"  
- Simple message forwarding with prefix
- No AI processing required
- Adds "[自动转发]" prefix to original messages
- Minimal processing time and resources
```

### SAFE_MODE Options

#### Safe Mode (SAFE_MODE = True)
```python
Behavior:
- Types AI response into WeChat input field
- DOES NOT press Enter automatically  
- Displays: "🔒 安全模式 - 消息已输入但未发送"
- Requires manual Enter key press to send
- Allows review and editing before sending
```

#### Auto Mode (SAFE_MODE = False)  
```python
Behavior:
- Types AI response into WeChat input field
- Automatically presses Enter to send message
- No manual intervention required
- Fully automated response system
```

## Data Management

### Database Systems

#### 1. Conversation History (`history.db`)
```sql
Table: conversations
- user_id: Contact identifier
- role: "user" or "assistant" 
- content: Message text content
- timestamp: Message timestamp
- Used for AI context and conversation continuity
```

#### 2. Message Logs (`messages.db`)
```sql  
Table: messages
- contact: Sender name
- message: Original message content  
- reply: Generated AI response
- timestamp: Processing timestamp
- Used for debugging and audit trails
```

### File Management

#### Screenshot Organization
```
pic/
├── screenshots/     # Main WeChat window captures
├── message/        # Message area specific captures  
└── chatname/       # Contact name region captures

Auto-cleanup Policy:
- Triggers when >200 files detected
- Removes files older than 3-7 days
- Keeps most recent 100 files
- Maintains ~60-70MB storage usage
```

#### Configuration Files
```
names.txt          # Monitored contact list (one per line)
.env              # Environment variables (DEEPSEEK_API_KEY)
Constants.py      # GUI coordinate constants
CLAUDE.md        # Development documentation
```

## Diagnostic & Monitoring

### Real-time Web Dashboard (Port 5001)

#### Features
- **Live Status Monitoring**: Bot running state, current contact, message counts
- **Process Flow Tracking**: 7-step process visualization with real-time status
- **Performance Metrics**: OCR timing, API response time, total processing time
- **Screenshot Debugging**: Live view of captured screenshots for each step
- **Control Interface**: Start/Stop/Restart bot remotely
- **Error Logging**: Real-time error tracking and debugging information

#### WebSocket Integration  
- **Real-time Updates**: Status changes broadcast immediately
- **Process Tracking**: Step-by-step progress monitoring
- **Screenshot Updates**: Live image feeds for debugging
- **Log Streaming**: Continuous error and info log display

### Performance Monitoring

#### Timing Analysis
```python
Typical Processing Times:
├── Screenshot Capture: 20-50ms
├── Message Detection: 100-500ms  
├── Contact Recognition: 100-300ms
├── OCR Processing: 5000-9000ms (largest bottleneck)
├── AI Generation: 2700-3400ms
└── Message Input: 100-300ms

Total Cycle Time: 8-13 seconds per message
```

#### Resource Usage
- **Memory**: ~100-500MB (depends on OCR models)
- **Storage**: 60-70MB for screenshots (auto-managed)
- **Network**: Minimal (only DeepSeek API calls)
- **CPU**: High during OCR processing, low otherwise

## Error Handling & Recovery

### Graceful Degradation
1. **OCR Failures**: Retry with different preprocessing
2. **API Timeouts**: Skip response generation, continue monitoring  
3. **GUI Automation Errors**: Retry with random delays
4. **Contact Recognition Failures**: Use fallback simplified method
5. **Screenshot Failures**: Continue with next monitoring cycle

### Safety Mechanisms
- **ESC Key Emergency Stop**: Immediate bot termination
- **Safe Mode**: Manual message review before sending
- **Contact Filtering**: Only respond to monitored contacts
- **Rate Limiting**: Built-in delays prevent spam detection
- **Error Logging**: Comprehensive error tracking and reporting

### Recovery Strategies  
- **Auto-retry**: Failed operations retry automatically
- **Fallback Methods**: Multiple approaches for critical functions
- **State Recovery**: Resume from last known good state
- **Clean Shutdown**: Proper resource cleanup on exit

## Security & Privacy

### Data Protection
- **Local Processing**: All OCR and processing done locally
- **API Encryption**: HTTPS for DeepSeek API communication
- **No Cloud Storage**: Screenshots and logs stored locally only
- **Conversation Privacy**: Message history encrypted in local database

### Access Control
- **Environment Variables**: API keys stored securely
- **Contact Filtering**: Restricted to approved contact list
- **Local Network**: Diagnostic panel only accessible locally
- **Screenshot Cleanup**: Automatic deletion of old sensitive images

## Platform Support

### Cross-Platform Compatibility
- **macOS**: Full support with native automation
- **Windows**: Full support with platform-specific adaptations  
- **Linux**: Partial support (requires testing and configuration)

### Platform-Specific Features
```python
macOS:
- Command key shortcuts (Cmd+A, Cmd+V)
- Native screenshot APIs
- Optimized OCR coordinate detection

Windows:  
- Control key shortcuts (Ctrl+A, Ctrl+V)
- Windows-specific message detection algorithms
- Adapted GUI automation coordinates
```

## Performance Optimization

### Bottleneck Analysis
1. **OCR Processing (99.5% of time)**
   - GPU acceleration enabled by default
   - Image preprocessing optimization
   - Multi-language model efficiency

2. **AI API Calls (15-20% of total time)**
   - Streaming responses for faster perception
   - Optimized context window usage
   - Response caching opportunities

### Optimization Strategies
- **Parallel Processing**: Screenshot capture while processing previous message
- **Caching**: Repeated contact recognition results
- **Image Optimization**: Reduce screenshot file sizes
- **Model Efficiency**: Use lightweight OCR models when possible

## Development & Maintenance

### Code Organization
```
├── app.py                    # Main application orchestration
├── capture/                  # Computer vision and detection
│   ├── monitor_new_message.py    # Message notification detection  
│   ├── deal_chatbox.py           # OCR text extraction
│   ├── get_name_free.py          # Contact name recognition
│   └── text_change_monitor.py    # Text change detection
├── deepseek/                 # AI integration
│   └── deepseekai.py             # DeepSeek API interface
├── db/                       # Database layer
│   └── db.py                     # SQLite operations
├── step_diagnostic_server.py # WDC (Web Diagnostic Console) server
├── step_diagnostic.html      # WDC interface UI
└── Constants.py              # Configuration constants
```

### Testing & Debugging
- **Unit Tests**: Individual component testing
- **Integration Tests**: End-to-end workflow validation
- **Debug Scripts**: OCR accuracy testing, API validation
- **Performance Profiling**: Timing analysis and optimization
- **Visual Debugging**: Screenshot analysis tools

This documentation provides a complete technical overview of the WeChat bot's architecture, processes, and implementation details. The system combines computer vision, artificial intelligence, and automation to create an intelligent, responsive chat assistant.