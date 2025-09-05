#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Log_Diagnostic_Console (LDC) Web Server
Lightweight server to auto-serve diagnostic logs and enable Python function triggers
"""

import os
import sys
import json
import glob
from pathlib import Path
from flask import Flask, jsonify, send_file, request
from datetime import datetime
import threading
import time

# Add current directory to path for imports
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# Import diagnostic modules
try:
    from diagnostic_logger import get_logger, log_detection, log_ocr, log_performance, log_error
    from TestRun.logger_integration_example import (
        example_avatar_detection_with_logging,
        example_ocr_with_logging, 
        example_message_detection_with_logging,
        example_click_action_with_logging
    )
    PYTHON_FUNCTIONS_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Warning: Some Python functions not available: {e}")
    PYTHON_FUNCTIONS_AVAILABLE = False

app = Flask(__name__)

# Configuration
LOG_DIR = Path("diagnostic_logs")
PORT = 5002
DEBUG = False


@app.route('/')
def index():
    """Serve the diagnostic viewer HTML"""
    return send_file('diagnostic_viewer.html')


@app.route('/api/status')
def status():
    """Server status and capabilities"""
    return jsonify({
        "server": "Log_Diagnostic_Console (LDC)",
        "version": "1.0",
        "timestamp": datetime.now().isoformat(),
        "log_directory": str(LOG_DIR),
        "python_functions": PYTHON_FUNCTIONS_AVAILABLE,
        "capabilities": {
            "auto_load": True,
            "function_triggers": PYTHON_FUNCTIONS_AVAILABLE,
            "log_monitoring": True
        }
    })


@app.route('/api/latest-log')
def latest_log():
    """Get the most recent log file data"""
    try:
        log_files = list(LOG_DIR.glob("diagnostic_*.json"))
        if not log_files:
            return jsonify({"error": "No log files found", "logs": []}), 404
        
        # Find the most recent file by modification time
        latest_file = max(log_files, key=lambda f: f.stat().st_mtime)
        
        with open(latest_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return jsonify({
            "filename": latest_file.name,
            "modified": datetime.fromtimestamp(latest_file.stat().st_mtime).isoformat(),
            "events": data,
            "count": len(data) if isinstance(data, list) else 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs')
def list_logs():
    """List all available log files"""
    try:
        log_files = list(LOG_DIR.glob("diagnostic_*.json"))
        logs_info = []
        
        for log_file in sorted(log_files, key=lambda f: f.stat().st_mtime, reverse=True):
            try:
                with open(log_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    event_count = len(data) if isinstance(data, list) else 1
                    
                logs_info.append({
                    "filename": log_file.name,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                    "size_bytes": log_file.stat().st_size,
                    "event_count": event_count
                })
            except Exception as e:
                logs_info.append({
                    "filename": log_file.name,
                    "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
                    "size_bytes": log_file.stat().st_size,
                    "error": str(e)
                })
                
        return jsonify({"logs": logs_info, "total": len(logs_info)})
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/log/<filename>')
def get_specific_log(filename):
    """Get a specific log file by name"""
    try:
        log_file = LOG_DIR / filename
        if not log_file.exists():
            return jsonify({"error": f"Log file {filename} not found"}), 404
            
        with open(log_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        return jsonify({
            "filename": filename,
            "modified": datetime.fromtimestamp(log_file.stat().st_mtime).isoformat(),
            "events": data,
            "count": len(data) if isinstance(data, list) else 1
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-avatar-detection', methods=['POST'])
def run_avatar_detection():
    """Trigger avatar detection function"""
    if not PYTHON_FUNCTIONS_AVAILABLE:
        return jsonify({"error": "Python functions not available"}), 503
        
    try:
        start_time = time.time()
        
        # Run the detection function
        result = example_avatar_detection_with_logging()
        
        duration = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "function": "avatar_detection",
            "result": {
                "detections_found": len(result) if result else 0,
                "execution_time_ms": duration
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_error("diagnostic_web_server", e, {"function": "run_avatar_detection"})
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-message-detection', methods=['POST'])
def run_message_detection():
    """Trigger message detection function"""
    if not PYTHON_FUNCTIONS_AVAILABLE:
        return jsonify({"error": "Python functions not available"}), 503
        
    try:
        start_time = time.time()
        
        result = example_message_detection_with_logging()
        
        duration = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "function": "message_detection", 
            "result": {
                "messages_found": len(result) if result else 0,
                "execution_time_ms": duration
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_error("diagnostic_web_server", e, {"function": "run_message_detection"})
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-ocr', methods=['POST'])
def run_ocr():
    """Trigger OCR processing function"""
    if not PYTHON_FUNCTIONS_AVAILABLE:
        return jsonify({"error": "Python functions not available"}), 503
        
    try:
        start_time = time.time()
        
        result = example_ocr_with_logging()
        
        duration = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "function": "ocr_processing",
            "result": {
                "text_extracted": result if isinstance(result, str) else str(result),
                "characters_count": len(result) if isinstance(result, str) else 0,
                "execution_time_ms": duration
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_error("diagnostic_web_server", e, {"function": "run_ocr"})
        return jsonify({"error": str(e)}), 500


@app.route('/api/run-click', methods=['POST'])
def run_click():
    """Trigger click action function"""
    if not PYTHON_FUNCTIONS_AVAILABLE:
        return jsonify({"error": "Python functions not available"}), 503
        
    try:
        start_time = time.time()
        
        result = example_click_action_with_logging()
        
        duration = (time.time() - start_time) * 1000
        
        return jsonify({
            "success": True,
            "function": "click_action",
            "result": {
                "click_successful": result,
                "execution_time_ms": duration
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        log_error("diagnostic_web_server", e, {"function": "run_click"})
        return jsonify({"error": str(e)}), 500


@app.route('/api/clear-logs', methods=['POST'])
def clear_logs():
    """Clear all log files (use with caution)"""
    try:
        log_files = list(LOG_DIR.glob("diagnostic_*.json"))
        deleted_count = 0
        
        for log_file in log_files:
            log_file.unlink()
            deleted_count += 1
            
        # Also clear subdirectories
        for subdir in ['screenshots', 'overlays']:
            subdir_path = LOG_DIR / subdir
            if subdir_path.exists():
                for file in subdir_path.glob("*"):
                    file.unlink()
        
        return jsonify({
            "success": True,
            "deleted_files": deleted_count,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404


@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500


def ensure_log_directory():
    """Ensure log directory exists"""
    LOG_DIR.mkdir(exist_ok=True)
    (LOG_DIR / "screenshots").mkdir(exist_ok=True)
    (LOG_DIR / "overlays").mkdir(exist_ok=True)


def print_startup_info():
    """Print startup information"""
    print("🌐 Log_Diagnostic_Console (LDC) Starting...")
    print("=" * 50)
    print(f"📂 Log Directory: {LOG_DIR}")
    print(f"🔗 Server URL: http://localhost:{PORT}")
    print(f"🐍 Python Functions: {'✅ Available' if PYTHON_FUNCTIONS_AVAILABLE else '❌ Limited'}")
    print("\n🚀 Available Endpoints:")
    print(f"   • http://localhost:{PORT}/ - LDC Viewer")
    print(f"   • http://localhost:{PORT}/api/status - Server Status")
    print(f"   • http://localhost:{PORT}/api/latest-log - Auto-load Latest")
    print(f"   • http://localhost:{PORT}/api/logs - List All Logs")
    
    if PYTHON_FUNCTIONS_AVAILABLE:
        print("\n🎮 Function Triggers:")
        print(f"   • POST /api/run-avatar-detection")
        print(f"   • POST /api/run-message-detection") 
        print(f"   • POST /api/run-ocr")
        print(f"   • POST /api/run-click")
    
    print("\n💡 Usage:")
    print("   1. Open http://localhost:5002 in your browser (LDC Interface)")
    print("   2. Logs will auto-load automatically")
    print("   3. Use buttons to trigger Python functions")
    print("   4. Drag-and-drop still works as fallback")
    print("\n🔄 Server running... (Ctrl+C to stop)")
    print("=" * 50)


if __name__ == "__main__":
    ensure_log_directory()
    print_startup_info()
    
    try:
        app.run(
            host='0.0.0.0', 
            port=PORT, 
            debug=DEBUG,
            threaded=True
        )
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
    except Exception as e:
        print(f"\n💥 Server error: {e}")
    finally:
        # Cleanup
        logger = get_logger()
        logger.close()
        print("🔄 Cleanup completed")