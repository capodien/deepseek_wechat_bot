# TestRun Directory

This directory contains **temporary test files** created during development and debugging processes. These files are NOT part of the core bot functionality.

## Contents

### Test Scripts
- `test_*.py` - Various component testing scripts
- `debug_*.py` - Debugging utilities and analysis tools
- `standalone_*.py` - Alternative implementations for testing
- `cleanup_*.py` - Utility scripts for maintenance
- `find_*.py` - Discovery and detection testing tools

### Purpose
- **Development Testing**: Scripts for testing individual components
- **Debugging**: Tools for analyzing bot behavior and issues
- **Prototyping**: Experimental features and alternative approaches
- **Utilities**: Helper scripts for maintenance and analysis
- **Legacy Code**: Old implementations kept for reference

### Note
These files are temporary and used for development purposes only. The core bot functionality is implemented in the main directory structure:

**Core Bot Structure:**
- `app.py` - Main application
- `capture/` - Computer vision and detection
- `deepseek/` - AI integration  
- `db/` - Database layer
- `step_diagnostic_server.py` - Diagnostic panel
- `step_diagnostic.html` - Diagnostic interface

**Do not move production code into this TestRun directory.**