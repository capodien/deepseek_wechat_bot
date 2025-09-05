# Setup Guide

Complete installation and environment setup for the WeChat automation bot.

## System Requirements

- **Python**: 3.8+ (recommended: 3.10+)
- **OS**: macOS 10.14+ or Windows 10+
- **GPU**: CUDA-compatible GPU (optional, for OCR acceleration)
- **Memory**: 4GB+ RAM (8GB+ recommended)
- **WeChat**: Desktop version 3.0+

## Installation

### 1. Clone and Setup Environment

```bash
# Clone repository
git clone <repository-url>
cd deepseek_wechat_bot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Critical Dependencies & Versions

```bash
pip install easyocr>=1.6.0          # Chinese/English OCR processing
pip install opencv-python>=4.5.0     # Image processing and detection
pip install pyautogui>=0.9.54        # Cross-platform GUI automation
pip install openai>=1.0.0            # DeepSeek API integration (OpenAI-compatible)
pip install Pillow>=8.0.0            # Image manipulation
pip install numpy>=1.21.0            # Numerical processing
pip install flask>=2.0.0             # Diagnostic web interface
```

### 3. Environment Configuration

```bash
# Copy environment template
cp .env.example .env

# Edit with your API key
DEEPSEEK_API_KEY=your_api_key_here
```

### 4. WeChat Desktop Setup

1. **Install WeChat Desktop**: Download and install WeChat desktop application
2. **Login**: Log into your WeChat account
3. **Window Positioning**: Position WeChat window consistently (coordinates in `Constants.py` are hardcoded)
4. **Contact List**: Add contacts to monitor in `names.txt` (one name per line)

### 5. Coordinate Calibration

**CRITICAL**: Screen coordinates must match your system:

```python
# Edit Constants.py to match your WeChat window
WECHAT_WINDOW = (x, y, width, height)  # Adjust for your screen resolution
```

Use diagnostic tools to verify coordinates:
```bash
python step_diagnostic_server.py
# Visit http://localhost:5001 to test coordinates
```

## Database Setup

```bash
# Initialize databases
python -c "from db.db import *; create_db(); create_messagesdb()"
```

## Verification

### Test Installation
```bash
# Test OCR functionality
python capture/deal_chatbox.py

# Test message recognition
python capture/monitor_new_message.py

# Test friend name extraction
python capture/get_name_free.py
```

### Launch Diagnostic Interface
```bash
# Diagnostics with visual verification
python step_diagnostic_server.py
# Visit http://localhost:5001
```

### Run the Bot
```bash
# Start the main application (WeChat must be open and visible)
python app.py
```

## Common Setup Issues

### GPU Setup for OCR
```bash
# Check GPU availability
python -c "import easyocr; print(easyocr.Reader(['ch_sim'], gpu=True))"

# If GPU fails, modify to CPU mode in capture/deal_chatbox.py:
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=False)
```

### Coordinate Calibration
- Use diagnostic tools to verify click positions
- Take screenshots and compare with expected regions
- Adjust `Constants.py` values incrementally
- Test each coordinate change immediately

### Permission Issues (macOS)
```bash
# Grant accessibility permissions to terminal/IDE
# System Preferences > Security & Privacy > Accessibility
# Add Terminal.app or your IDE to allowed applications
```

### API Connection Test
```bash
# Test DeepSeek API connection
curl -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models
```

## Performance Optimization

### Hardware Recommendations
- **SSD**: For faster screenshot I/O
- **GPU**: NVIDIA GPU with CUDA for OCR acceleration
- **RAM**: 8GB+ for smooth processing of multiple screenshots

### Software Optimization
```python
# Enable GPU acceleration for OCR
OCR_READER = easyocr.Reader(['ch_sim', 'en'], gpu=True)

# Optimize screenshot region targeting
roi = image[y1:y2, x1:x2]  # Extract specific regions instead of full screen
```

## Next Steps

After successful setup:
1. Review [Architecture Guide](ARCHITECTURE.md) to understand system components
2. Use [Diagnostic Tools](DIAGNOSTICS.md) for testing and development
3. Check [Security Considerations](SECURITY.md) for important security fixes needed
4. Follow [Best Practices](MAINTENANCE.md) for ongoing maintenance