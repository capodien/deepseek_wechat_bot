#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Levels Adjustment Testing Server
Provides API endpoints for the HTML testing tool
"""

from flask import Flask, request, jsonify, send_from_directory, send_file
import cv2
import numpy as np
import os
import time
from datetime import datetime
import sys

# Add current directory to path for imports
sys.path.append('.')

app = Flask(__name__)

# Configuration
TEST_IMAGE_PATH = "pic/Sample Screen Shoot.png"
OUTPUT_DIR = "pic/screenshots"
STATIC_DIR = "pic/screenshots"

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

class LevelsAdjustmentTester:
    """Test different level adjustment parameters"""
    
    def __init__(self):
        self.test_image_path = TEST_IMAGE_PATH
        
    def apply_levels_adjustment(self, gray, in_black, in_white, gamma):
        """Apply Photoshop-style levels adjustment"""
        # Step 1: Normalize to 0-1 range and clip
        arr = np.clip((gray - in_black) / (in_white - in_black), 0, 1)
        
        # Step 2: Apply gamma correction and scale to 0-255
        arr = (arr ** (1/gamma)) * 255
        
        # Convert to uint8
        return arr.astype(np.uint8)
    
    def test_levels(self, in_black, in_white, gamma):
        """Test level adjustment with given parameters"""
        start_time = time.time()
        
        # Load image
        img = cv2.imread(self.test_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {self.test_image_path}")
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply levels adjustment
        adjusted = self.apply_levels_adjustment(gray, in_black, in_white, gamma)
        
        # Save images with timestamps
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        original_filename = f"{timestamp}_original_grayscale.png"
        adjusted_filename = f"{timestamp}_levels_{in_black}_{in_white}_{gamma:.2f}.png"
        
        original_path = os.path.join(OUTPUT_DIR, original_filename)
        adjusted_path = os.path.join(OUTPUT_DIR, adjusted_filename)
        
        cv2.imwrite(original_path, gray)
        cv2.imwrite(adjusted_path, adjusted)
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'processing_time': processing_time,
            'original_path': f'/images/{original_filename}',
            'adjusted_path': f'/images/{adjusted_filename}',
            'parameters': {
                'input_black': in_black,
                'input_white': in_white,
                'gamma': gamma
            }
        }
    
    def detect_right_boundary(self, in_black, in_white, gamma, edge_threshold=15, smoothing_size=9, search_start=40):
        """Test right boundary detection with level adjustment"""
        start_time = time.time()
        
        # Load and process image
        img = cv2.imread(self.test_image_path)
        if img is None:
            raise FileNotFoundError(f"Could not load image: {self.test_image_path}")
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_height, img_width = gray.shape
        
        # Apply levels adjustment
        adjusted = self.apply_levels_adjustment(gray, in_black, in_white, gamma)
        
        # Apply edge detection (Sobel)
        sobel_x = cv2.Sobel(adjusted, cv2.CV_64F, dx=1, dy=0, ksize=5)
        
        # Create horizontal projection
        edge_projection = np.sum(np.abs(sobel_x), axis=0)
        
        # Apply smoothing
        from scipy.ndimage import uniform_filter1d
        smoothed_projection = uniform_filter1d(edge_projection, size=smoothing_size, mode='nearest')
        
        # Calculate threshold
        max_strength = np.max(smoothed_projection)
        threshold = max_strength * (edge_threshold / 100.0)
        
        # Find boundary candidates
        search_start_px = max(500, int(img_width * (search_start / 100.0)))
        search_end_px = min(len(smoothed_projection) - 10, int(img_width * 0.95))
        
        boundary_candidates = []
        for x in range(search_start_px, search_end_px):
            if smoothed_projection[x] > threshold:
                # Check for local maximum
                left_neighbor = smoothed_projection[x-1] if x > 0 else 0
                right_neighbor = smoothed_projection[x+1] if x < len(smoothed_projection) - 1 else 0
                
                if smoothed_projection[x] >= left_neighbor and smoothed_projection[x] >= right_neighbor:
                    boundary_candidates.append((x, smoothed_projection[x]))
        
        # Select rightmost boundary
        right_boundary = None
        if boundary_candidates:
            boundary_candidates.sort(key=lambda b: b[0], reverse=True)
            
            # Prefer boundaries > 800px
            for x, strength in boundary_candidates:
                if x >= 800:
                    right_boundary = x
                    break
            
            if right_boundary is None:
                right_boundary = boundary_candidates[0][0]
        else:
            right_boundary = int(img_width * 0.8)  # Fallback
        
        processing_time = int((time.time() - start_time) * 1000)
        
        return {
            'success': True,
            'processing_time': processing_time,
            'right_boundary': right_boundary,
            'candidates_found': len(boundary_candidates),
            'max_edge_strength': int(max_strength),
            'threshold_used': int(threshold),
            'search_range': f"{search_start_px}-{search_end_px}px"
        }

# Initialize tester
tester = LevelsAdjustmentTester()

@app.route('/')
def index():
    """Serve the HTML tool"""
    return send_file('levels_adjustment_tool.html')

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve images from screenshots directory"""
    return send_from_directory(STATIC_DIR, filename)

@app.route('/original-image')
def serve_original_image():
    """Serve the original Sample Screen Shoot.png"""
    return send_file(TEST_IMAGE_PATH)

@app.route('/api/test-levels', methods=['POST'])
def test_levels():
    """Test level adjustment parameters"""
    try:
        data = request.json
        in_black = data.get('input_black', 53)
        in_white = data.get('input_white', 107)
        gamma = data.get('gamma', 0.67)
        
        result = tester.test_levels(in_black, in_white, gamma)
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/test-boundary-detection', methods=['POST'])
def test_boundary_detection():
    """Test boundary detection with parameters"""
    try:
        data = request.json
        in_black = data.get('input_black', 53)
        in_white = data.get('input_white', 107)
        gamma = data.get('gamma', 0.67)
        edge_threshold = data.get('edge_threshold', 15)
        smoothing_size = data.get('smoothing_size', 9)
        search_start = data.get('search_start', 40)
        
        result = tester.detect_right_boundary(
            in_black, in_white, gamma, 
            edge_threshold, smoothing_size, search_start
        )
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/api/run-full-pipeline', methods=['POST'])
def run_full_pipeline():
    """Run complete width detection pipeline"""
    try:
        data = request.json
        in_black = data.get('input_black', 53)
        in_white = data.get('input_white', 107)
        gamma = data.get('gamma', 0.67)
        
        # Import the actual detector
        from modules.m_Card_Processing import SimpleWidthDetector
        
        # Update detector parameters
        detector = SimpleWidthDetector()
        detector.right_detector.INPUT_BLACK_POINT = in_black
        detector.right_detector.INPUT_WHITE_POINT = in_white
        detector.right_detector.GAMMA = gamma
        
        # Run detection
        start_time = time.time()
        result = detector.detect_width(TEST_IMAGE_PATH)
        processing_time = int((time.time() - start_time) * 1000)
        
        if result:
            left, right, width = result
            return jsonify({
                'success': True,
                'processing_time': processing_time,
                'left': left,
                'right': right,
                'width': width,
                'parameters': {
                    'input_black': in_black,
                    'input_white': in_white,
                    'gamma': gamma
                }
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Width detection failed'
            })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

if __name__ == '__main__':
    print("🎨 Starting Levels Adjustment Testing Server...")
    print(f"📸 Test image: {TEST_IMAGE_PATH}")
    print(f"📁 Output directory: {OUTPUT_DIR}")
    print(f"🌐 Open your browser to: http://localhost:5003")
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5003)