#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Coordinate System Diagnostic Server

This server provides web-based testing and validation for the universal 
coordinate system implementation in m_Card_Processing.py.

Features:
- Test coordinate context creation and population
- Validate coordinate accuracy and consistency
- Visualize coordinate overlays on screenshots
- Export coordinate data for debugging
"""

from flask import Flask, request, jsonify, render_template_string
from flask_cors import CORS
import os
import sys
import json
import cv2
import numpy as np
from datetime import datetime

# Add modules directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'modules'))

try:
    from modules import m_Card_Processing
    from modules.m_Card_Processing import WeChatCoordinateContext
    CARD_PROCESSING_AVAILABLE = True
except ImportError:
    try:
        import m_Card_Processing
        from m_Card_Processing import WeChatCoordinateContext
        CARD_PROCESSING_AVAILABLE = True
    except ImportError:
        print("❌ Card processing module not available")
        CARD_PROCESSING_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# HTML template for the diagnostic interface
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Universal Coordinate System - Diagnostic Interface</title>
    <style>
        body { font-family: Arial, sans-serif; margin: 20px; }
        .header { color: #333; border-bottom: 2px solid #007acc; padding-bottom: 10px; }
        .section { margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 5px; }
        .button { 
            background: #007acc; color: white; padding: 10px 20px; 
            border: none; border-radius: 5px; cursor: pointer; margin: 5px;
        }
        .button:hover { background: #005c99; }
        .result { 
            margin: 10px 0; padding: 10px; 
            border-left: 4px solid #007acc; background: #f8f9fa;
            font-family: monospace; white-space: pre-wrap;
        }
        .success { border-left-color: #28a745; }
        .error { border-left-color: #dc3545; }
        .warning { border-left-color: #ffc107; }
        .image-result { text-align: center; margin: 15px 0; }
        .image-result img { max-width: 100%; height: auto; border: 1px solid #ddd; }
        .coordinate-data { 
            max-height: 400px; overflow-y: auto; 
            background: #f1f1f1; padding: 10px; border-radius: 5px;
        }
    </style>
</head>
<body>
    <h1 class="header">🎯 Universal Coordinate System - Diagnostic Interface</h1>
    
    <div class="section">
        <h2>📊 System Status</h2>
        <div id="system-status">Loading...</div>
        <button class="button" onclick="checkSystemStatus()">Refresh Status</button>
    </div>
    
    <div class="section">
        <h2>🔧 Coordinate System Testing</h2>
        <button class="button" onclick="testCoordinateContext()">Test Coordinate Context Creation</button>
        <button class="button" onclick="testDetectorIntegration()">Test Detector Integration</button>
        <button class="button" onclick="testFullPipeline()">Test Full Processing Pipeline</button>
        <div id="coordinate-results"></div>
    </div>
    
    <div class="section">
        <h2>📈 Coordinate Visualization</h2>
        <button class="button" onclick="visualizeCoordinates()">Generate Coordinate Overlay</button>
        <button class="button" onclick="compareCoordinateSystems()">Compare Legacy vs Universal</button>
        <div id="visualization-results"></div>
    </div>
    
    <div class="section">
        <h2>🔍 Coordinate Validation</h2>
        <button class="button" onclick="validateCoordinates()">Validate All Coordinates</button>
        <button class="button" onclick="exportCoordinateData()">Export Coordinate Data</button>
        <div id="validation-results"></div>
    </div>

    <script>
        function checkSystemStatus() {
            fetch('/api/system-status')
                .then(response => response.json())
                .then(data => {
                    document.getElementById('system-status').innerHTML = 
                        '<div class="result ' + (data.status === 'ready' ? 'success' : 'error') + '">' +
                        JSON.stringify(data, null, 2) + '</div>';
                });
        }
        
        function testCoordinateContext() {
            document.getElementById('coordinate-results').innerHTML = '<div class="result">Testing coordinate context...</div>';
            fetch('/api/test-coordinate-context')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'error';
                    document.getElementById('coordinate-results').innerHTML = 
                        '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                });
        }
        
        function testDetectorIntegration() {
            document.getElementById('coordinate-results').innerHTML = '<div class="result">Testing detector integration...</div>';
            fetch('/api/test-detector-integration')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'error';
                    document.getElementById('coordinate-results').innerHTML = 
                        '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                });
        }
        
        function testFullPipeline() {
            document.getElementById('coordinate-results').innerHTML = '<div class="result">Testing full pipeline...</div>';
            fetch('/api/test-full-pipeline')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'error';
                    let html = '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                    
                    if (data.coordinate_overlay) {
                        html += '<div class="image-result">' +
                               '<h4>Coordinate Overlay:</h4>' +
                               '<img src="' + data.coordinate_overlay + '" alt="Coordinate Overlay">' +
                               '</div>';
                    }
                    
                    document.getElementById('coordinate-results').innerHTML = html;
                });
        }
        
        function visualizeCoordinates() {
            document.getElementById('visualization-results').innerHTML = '<div class="result">Generating visualization...</div>';
            fetch('/api/visualize-coordinates')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'error';
                    let html = '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                    
                    if (data.visualization_path) {
                        html += '<div class="image-result">' +
                               '<h4>Coordinate Visualization:</h4>' +
                               '<img src="' + data.visualization_path + '" alt="Coordinate Visualization">' +
                               '</div>';
                    }
                    
                    document.getElementById('visualization-results').innerHTML = html;
                });
        }
        
        function compareCoordinateSystems() {
            document.getElementById('visualization-results').innerHTML = '<div class="result">Comparing coordinate systems...</div>';
            fetch('/api/compare-coordinate-systems')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'warning';
                    document.getElementById('visualization-results').innerHTML = 
                        '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                });
        }
        
        function validateCoordinates() {
            document.getElementById('validation-results').innerHTML = '<div class="result">Validating coordinates...</div>';
            fetch('/api/validate-coordinates')
                .then(response => response.json())
                .then(data => {
                    const className = data.validation_passed ? 'success' : 'warning';
                    document.getElementById('validation-results').innerHTML = 
                        '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                });
        }
        
        function exportCoordinateData() {
            document.getElementById('validation-results').innerHTML = '<div class="result">Exporting coordinate data...</div>';
            fetch('/api/export-coordinate-data')
                .then(response => response.json())
                .then(data => {
                    const className = data.success ? 'success' : 'error';
                    let html = '<div class="result ' + className + '">' + JSON.stringify(data, null, 2) + '</div>';
                    
                    if (data.export_path) {
                        html += '<div class="coordinate-data">' +
                               '<h4>Exported Coordinate Data:</h4>' +
                               '<pre>' + JSON.stringify(data.coordinate_data, null, 2) + '</pre>' +
                               '</div>';
                    }
                    
                    document.getElementById('validation-results').innerHTML = html;
                });
        }
        
        // Initialize page
        checkSystemStatus();
    </script>
</body>
</html>
"""

@app.route('/')
def index():
    """Main diagnostic interface page"""
    return render_template_string(HTML_TEMPLATE)

@app.route('/api/system-status')
def system_status():
    """Check system status and available components"""
    try:
        status = {
            "status": "ready" if CARD_PROCESSING_AVAILABLE else "error",
            "timestamp": datetime.now().isoformat(),
            "components": {
                "card_processing_module": CARD_PROCESSING_AVAILABLE,
                "coordinate_context_class": False,
                "detector_classes": {
                    "SimpleWidthDetector": False,
                    "CardAvatarDetector": False, 
                    "CardBoundaryDetector": False,
                    "ContactNameBoundaryDetector": False,
                    "TimeBoxDetector": False
                }
            }
        }
        
        if CARD_PROCESSING_AVAILABLE:
            # Test coordinate context class
            try:
                context = WeChatCoordinateContext("/test/path.png", (800, 600))
                status["components"]["coordinate_context_class"] = True
            except:
                pass
            
            # Test detector classes
            try:
                detector = m_Card_Processing.SimpleWidthDetector()
                status["components"]["detector_classes"]["SimpleWidthDetector"] = True
            except:
                pass
                
            try:
                detector = m_Card_Processing.CardAvatarDetector()
                status["components"]["detector_classes"]["CardAvatarDetector"] = True
            except:
                pass
                
            try:
                detector = m_Card_Processing.CardBoundaryDetector()
                status["components"]["detector_classes"]["CardBoundaryDetector"] = True
            except:
                pass
                
            try:
                detector = m_Card_Processing.ContactNameBoundaryDetector()
                status["components"]["detector_classes"]["ContactNameBoundaryDetector"] = True
            except:
                pass
                
            try:
                detector = m_Card_Processing.TimeBoxDetector()
                status["components"]["detector_classes"]["TimeBoxDetector"] = True
            except:
                pass
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/test-coordinate-context')
def test_coordinate_context():
    """Test basic coordinate context creation and operations"""
    try:
        if not CARD_PROCESSING_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Card processing module not available"
            })
        
        # Test coordinate context creation
        context = WeChatCoordinateContext("/test/image.png", (800, 600))
        
        # Test adding global boundary
        context.add_global_boundary("conversation_area", [50, 0, 700, 600], "test", 0.95)
        
        # Test adding cards
        context.add_card(1, [100, 50, 600, 100], "test", 0.9)
        context.add_card(2, [100, 200, 600, 100], "test", 0.9)
        
        # Test adding components
        context.add_component(1, "avatar", [80, 70, 40, 40], "test", 0.98, False)
        context.add_component(1, "contact_name", [130, 60, 100, 20], "test", 0.85, True, "contact_name")
        context.add_component(1, "timestamp", [500, 60, 80, 20], "test", 0.8, True, "timestamp")
        
        # Test validation
        validation_result = context.validate_coordinates()
        
        # Test data extraction
        context_data = context.to_dict()
        ocr_regions = context.extract_all_regions("contact_names")
        
        return jsonify({
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "tests_passed": {
                "context_creation": True,
                "global_boundary_addition": len(context_data["global_boundaries"]) > 0,
                "card_addition": len(context_data["cards"]) == 2,
                "component_addition": len(context_data["cards"][0]["components"]) == 3,
                "coordinate_validation": len(validation_result["errors"]) == 0,
                "ocr_region_extraction": len(ocr_regions) > 0
            },
            "coordinate_data": context_data,
            "validation_result": validation_result,
            "ocr_regions_found": len(ocr_regions)
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/test-detector-integration')
def test_detector_integration():
    """Test detector methods with coordinate context integration"""
    try:
        if not CARD_PROCESSING_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Card processing module not available"
            })
        
        # Find a test image
        test_image = None
        pic_dir = os.path.join(os.path.dirname(__file__), "pic", "screenshots")
        if os.path.exists(pic_dir):
            for file in os.listdir(pic_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(pic_dir, file)
                    break
        
        if not test_image:
            return jsonify({
                "success": False,
                "error": "No test images found in pic/screenshots directory"
            })
        
        results = {
            "success": True,
            "timestamp": datetime.now().isoformat(),
            "test_image": os.path.basename(test_image),
            "detector_tests": {}
        }
        
        # Test SimpleWidthDetector with coordinate context
        try:
            detector = m_Card_Processing.SimpleWidthDetector()
            width_result, coord_context = detector.detect_width(test_image, return_context=True)
            
            results["detector_tests"]["SimpleWidthDetector"] = {
                "success": width_result is not None,
                "width_detected": width_result,
                "coordinate_context_created": coord_context is not None,
                "global_boundaries": len(coord_context.context["global_boundaries"]) if coord_context else 0
            }
        except Exception as e:
            results["detector_tests"]["SimpleWidthDetector"] = {
                "success": False,
                "error": str(e)
            }
        
        # Test CardAvatarDetector with coordinate context  
        try:
            detector = m_Card_Processing.CardAvatarDetector()
            (avatars, info), coord_context = detector.detect_avatars(test_image, return_context=True)
            
            results["detector_tests"]["CardAvatarDetector"] = {
                "success": len(avatars) > 0,
                "avatars_detected": len(avatars),
                "coordinate_context_created": coord_context is not None,
                "global_boundaries": len(coord_context.context["global_boundaries"]) if coord_context else 0
            }
        except Exception as e:
            results["detector_tests"]["CardAvatarDetector"] = {
                "success": False, 
                "error": str(e)
            }
        
        return jsonify(results)
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

@app.route('/api/test-full-pipeline')  
def test_full_pipeline():
    """Test the full processing pipeline with coordinate context"""
    try:
        if not CARD_PROCESSING_AVAILABLE:
            return jsonify({
                "success": False,
                "error": "Card processing module not available"
            })
        
        # Find a test image
        test_image = None
        pic_dir = os.path.join(os.path.dirname(__file__), "pic", "screenshots")
        if os.path.exists(pic_dir):
            for file in os.listdir(pic_dir):
                if file.endswith(('.png', '.jpg', '.jpeg')):
                    test_image = os.path.join(pic_dir, file)
                    break
        
        if not test_image:
            return jsonify({
                "success": False,
                "error": "No test images found in pic/screenshots directory"
            })
        
        # Run the full pipeline
        results = m_Card_Processing.process_screenshot_file(test_image)
        
        # Create coordinate overlay visualization
        overlay_path = None
        if results.get("coordinate_context"):
            try:
                overlay_path = create_coordinate_overlay_visualization(test_image, results["coordinate_context"])
            except Exception as e:
                print(f"Failed to create overlay: {e}")
        
        return jsonify({
            "success": results.get("processing_successful", False),
            "timestamp": datetime.now().isoformat(),
            "test_image": os.path.basename(test_image),
            "processing_results": {
                "width_detected": results.get("step1_width_detected"),
                "avatars_detected": results.get("step3_avatars_detected"),
                "cards_detected": results.get("step4_cards_detected"),
                "names_detected": results.get("step5_names_detected"),
                "times_detected": results.get("step6_times_detected")
            },
            "coordinate_system": {
                "context_created": results.get("coordinate_context") is not None,
                "validation_passed": not results.get("coordinate_validation", {}).get("errors", []),
                "total_ocr_regions": results.get("ocr_regions", {}).get("total_regions", 0)
            },
            "coordinate_overlay": f"/static/{os.path.basename(overlay_path)}" if overlay_path else None
        })
        
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        })

def create_coordinate_overlay_visualization(image_path: str, coord_context: dict) -> str:
    """Create a visualization showing all coordinates from the context"""
    try:
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError("Could not load image")
        
        overlay = img.copy()
        
        # Draw global boundaries in blue
        for boundary_name, boundary_data in coord_context.get("global_boundaries", {}).items():
            bbox = boundary_data["bbox"]
            x, y, w, h = bbox
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Blue
            cv2.putText(overlay, boundary_name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        
        # Draw cards and components
        colors = {
            "avatar": (0, 255, 0),      # Green
            "contact_name": (0, 255, 255),  # Yellow
            "timestamp": (255, 0, 255),  # Magenta
            "message_content": (128, 128, 255)  # Light blue
        }
        
        for card in coord_context.get("cards", []):
            if card is None:
                continue
                
            card_id = card["card_id"]
            card_bbox = card["card_region"]["bbox"]
            x, y, w, h = card_bbox
            
            # Draw card boundary in red
            cv2.rectangle(overlay, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red
            cv2.putText(overlay, f"Card {card_id}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Draw components
            for comp_type, comp_data in card.get("components", {}).items():
                comp_bbox = comp_data["bbox"]
                cx, cy, cw, ch = comp_bbox
                color = colors.get(comp_type, (128, 128, 128))
                
                cv2.rectangle(overlay, (cx, cy), (cx + cw, cy + ch), color, 1)
                cv2.putText(overlay, comp_type, (cx, cy + ch + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)
        
        # Add legend
        legend_y = 30
        cv2.putText(overlay, "Universal Coordinate System Overlay", (10, legend_y), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        legend_y += 25
        cv2.putText(overlay, "Blue=Global | Red=Cards | Green=Avatar | Yellow=Names | Magenta=Times", 
                   (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Save overlay
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        overlay_filename = f"{timestamp}_coordinate_overlay.png"
        
        # Ensure static directory exists
        static_dir = os.path.join(os.path.dirname(__file__), "static")
        os.makedirs(static_dir, exist_ok=True)
        
        overlay_path = os.path.join(static_dir, overlay_filename)
        cv2.imwrite(overlay_path, overlay)
        
        return overlay_path
        
    except Exception as e:
        print(f"Error creating coordinate overlay: {e}")
        return None

@app.route('/static/<filename>')
def serve_static(filename):
    """Serve static files (images)"""
    static_dir = os.path.join(os.path.dirname(__file__), "static")
    return app.send_static_file(filename)

# Additional API endpoints for completeness
@app.route('/api/visualize-coordinates')
def visualize_coordinates():
    return jsonify({"success": False, "error": "Not implemented yet"})

@app.route('/api/compare-coordinate-systems') 
def compare_coordinate_systems():
    return jsonify({"success": False, "error": "Not implemented yet"})

@app.route('/api/validate-coordinates')
def validate_coordinates():
    return jsonify({"success": False, "error": "Not implemented yet"})

@app.route('/api/export-coordinate-data')
def export_coordinate_data():
    return jsonify({"success": False, "error": "Not implemented yet"})

if __name__ == '__main__':
    print("🎯 Starting Universal Coordinate System Diagnostic Server")
    print("📍 Access the diagnostic interface at: http://localhost:5002")
    print("📊 Available endpoints:")
    print("   GET  /                           - Main diagnostic interface")
    print("   GET  /api/system-status          - Check system status")
    print("   GET  /api/test-coordinate-context - Test coordinate context")
    print("   GET  /api/test-detector-integration - Test detector integration") 
    print("   GET  /api/test-full-pipeline     - Test full processing pipeline")
    print("")
    
    app.run(debug=True, host='0.0.0.0', port=5002)