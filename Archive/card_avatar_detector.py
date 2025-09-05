#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Advanced Avatar Detection Module
Uses gradient projection and geometric filtering for robust WeChat avatar detection
Based on advanced computer vision techniques with WeChat-specific optimizations
"""

import cv2 as cv
import numpy as np
import os
import sys
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Import width detector for dynamic boundary detection
try:
    from modules.Card_Width_Detector import SimpleWidthDetector
except ImportError:
    # Fallback import path
    from Card_Width_Detector import SimpleWidthDetector

class CardAvatarDetector:
    """
    Advanced avatar detector using gradient projection and geometric filtering
    Specifically optimized for WeChat message card layouts
    """
    
    def __init__(self):
        # Initialize width detector for dynamic boundary detection
        self.width_detector = SimpleWidthDetector()
        
        # Avatar size constraints (adjustable based on DPI)
        self.MIN_AVATAR_SIZE = 25       # Minimum avatar dimension
        self.MAX_AVATAR_SIZE = 140      # Maximum avatar dimension
        self.MIN_ASPECT_RATIO = 0.85    # Nearly square avatars
        self.MAX_ASPECT_RATIO = 1.15    # Nearly square avatars
        
        # WeChat-specific panel parameters (fallback values if width detection fails)
        self.EXPECTED_CONVERSATION_AREA_RATIO = 0.65  # Conversation area is ~65% of screen width
        self.AVATAR_COLUMN_WIDTH = 120  # Avatar column is ~120px wide (fallback)
        self.AVATAR_COLUMN_START = 60   # Avatar column starts around x=60px (fallback)
        
        # Edge detection parameters
        self.BILATERAL_D = 5            # Bilateral filter diameter
        self.BILATERAL_SIGMA_COLOR = 35 # Color sigma for bilateral filter
        self.BILATERAL_SIGMA_SPACE = 35 # Space sigma for bilateral filter
        self.CANNY_LOW = 50            # Lower threshold for Canny
        self.CANNY_HIGH = 120          # Upper threshold for Canny
        
        # Morphology parameters
        self.DILATE_KERNEL_SIZE = (3, 3)  # Dilation kernel
        self.DILATE_ITERATIONS = 1        # Dilation iterations
        
        # NMS parameters
        self.NMS_IOU_THRESHOLD = 0.2    # IoU threshold for non-maximum suppression
        
        # Solidity threshold (filled vs hollow shapes)
        self.MIN_SOLIDITY = 0.6         # Minimum solidity for avatar shapes
        
    def find_panel_right_edge(self, gray: np.ndarray, xL: int = 0, xR: Optional[int] = None, 
                              y0: int = 0, y1: Optional[int] = None) -> int:
        """
        1-D x-gradient projection to locate the right border of the list panel.
        Uses gradient analysis to find the strongest vertical edge (panel boundary)
        """
        H, W = gray.shape
        xR = W if xR is None else xR
        y1 = H if y1 is None else y1
        
        # Extract the search band
        band = gray[y0:y1, xL:xR]
        
        # Compute horizontal gradient (vertical edges)
        diff = np.abs(band[:, 1:].astype(np.int16) - band[:, :-1].astype(np.int16))
        
        # Create 1D profile by averaging across height
        prof = diff.mean(axis=0).astype(np.float32)
        
        # Smooth the profile to reduce noise
        prof = cv.GaussianBlur(prof.reshape(1, -1), (1, 7), 0).ravel()
        
        # Find the strongest vertical edge
        xr_local = int(np.argmax(prof)) + 1
        
        return xL + xr_local

    def nms_boxes(self, boxes: List[List[int]], iou_thresh: float = 0.3) -> List[List[int]]:
        """
        Non-maximum suppression on axis-aligned boxes [x,y,w,h].
        Eliminates overlapping detections to keep only the best ones.
        """
        if not boxes:
            return []
        
        b = np.array(boxes, dtype=np.float32)
        x1, y1 = b[:, 0], b[:, 1]
        x2, y2 = b[:, 0] + b[:, 2], b[:, 1] + b[:, 3]
        scores = b[:, 2] * b[:, 3]  # Use area as score
        
        # Sort by score (area) in descending order
        idxs = scores.argsort()[::-1]
        keep = []
        
        while idxs.size > 0:
            i = idxs[0]
            keep.append(i)
            
            # Compute IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[idxs[1:]])
            yy1 = np.maximum(y1[i], y1[idxs[1:]])
            xx2 = np.minimum(x2[i], x2[idxs[1:]])
            yy2 = np.minimum(y2[i], y2[idxs[1:]])
            
            w = np.maximum(0, xx2 - xx1)
            h = np.maximum(0, yy2 - yy1)
            inter = w * h
            
            iou = inter / ((x2[i] - x1[i]) * (y2[i] - y1[i]) + 
                          (x2[idxs[1:]] - x1[idxs[1:]]) * (y2[idxs[1:]] - y1[idxs[1:]]) - 
                          inter + 1e-6)
            
            # Keep boxes with IoU below threshold
            idxs = idxs[1:][iou < iou_thresh]
        
        return [boxes[i] for i in keep]

    def detect_avatars(self, image_path: str) -> Tuple[List[Dict], Dict]:
        """
        Detect avatars using gradient projection and geometric filtering.
        
        Returns:
            Tuple of (avatar_results, detection_info)
            - avatar_results: List of detected avatars with bbox and center
            - detection_info: Diagnostic information about the detection process
        """
        # Load image
        img = cv.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
        
        H, W = img.shape[:2]
        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        print(f"🎯 Advanced Avatar Detection: {os.path.basename(image_path)}")
        print(f"📐 Image size: {W}x{H}")
        
        # Step 1: Use dynamic width detection to determine card boundaries
        print(f"  🔍 Detecting dynamic card boundaries...")
        width_result = self.width_detector.detect_width(image_path)
        
        if width_result is not None:
            left_boundary, right_boundary, detected_width = width_result
            # Use dynamic left boundary + fixed avatar column width (avatars are only in leftmost ~120px)
            x0 = left_boundary
            wR = self.AVATAR_COLUMN_WIDTH  # Keep avatar column width, only left boundary is dynamic
            avatar_search_area = (x0, 0, wR, H)
            boundary_source = "dynamic"
            print(f"  ✅ Using dynamic left boundary: {left_boundary}px, avatar column: {wR}px wide (search area: {x0}-{x0+wR}px)")
        else:
            # Fallback to hardcoded values
            conversation_width = int(W * self.EXPECTED_CONVERSATION_AREA_RATIO)
            x0 = self.AVATAR_COLUMN_START
            wR = self.AVATAR_COLUMN_WIDTH
            avatar_search_area = (x0, 0, wR, H)
            boundary_source = "fallback"
            print(f"  ⚠️ Width detection failed, using fallback boundaries: x={x0}, width={wR}px")
        
        # Extract avatar search region from the detected/fallback boundaries
        y0, hR = 0, H
        panel = img[y0:y0+hR, x0:x0+wR]
        gray_p = cv.cvtColor(panel, cv.COLOR_BGR2GRAY)
        
        print(f"  📊 Avatar search area: x={x0}-{x0+wR}px ({boundary_source} left boundary, fixed avatar column width)")
        
        # Step 2: Edge detection with morphology to reveal rounded-square thumbnails
        blur = cv.bilateralFilter(gray_p, self.BILATERAL_D, 
                                  self.BILATERAL_SIGMA_COLOR, 
                                  self.BILATERAL_SIGMA_SPACE)
        
        edges = cv.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)
        edges = cv.dilate(edges, cv.getStructuringElement(cv.MORPH_RECT, self.DILATE_KERNEL_SIZE), 
                         iterations=self.DILATE_ITERATIONS)
        
        # Step 3: Find contours and apply geometric filters
        cnts, _ = cv.findContours(edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        print(f"  🔍 Found {len(cnts)} contours")
        
        candidates = []
        for c in cnts:
            x, y, w, h = cv.boundingRect(c)
            area = w * h
            
            # Size filter
            if area < self.MIN_AVATAR_SIZE**2 or area > self.MAX_AVATAR_SIZE**2:
                continue
            
            # Aspect ratio filter (nearly square)
            ar = w / float(h)
            if ar < self.MIN_ASPECT_RATIO or ar > self.MAX_ASPECT_RATIO:
                continue
            
            # Position filter (avatars should be within the avatar search area)
            # Since we're already searching in the avatar column, all detections are valid positions
            # Additional filter: ensure avatars are not at the very edge
            if x < 5 or x > wR - 10:
                continue
            
            # Solidity filter (prefer filled thumbnails vs thin edges)
            hull = cv.convexHull(c)
            hull_area = cv.contourArea(hull)
            if hull_area > 0:
                solidity = cv.contourArea(c) / hull_area
                if solidity < self.MIN_SOLIDITY:
                    continue
            
            candidates.append([x, y, w, h])
        
        print(f"  ✅ Filtered to {len(candidates)} avatar candidates")
        
        # Step 4: Apply non-maximum suppression
        boxes = self.nms_boxes(candidates, iou_thresh=self.NMS_IOU_THRESHOLD)
        print(f"  🎯 After NMS: {len(boxes)} final avatars")
        
        # Step 5: Sort by y-coordinate (top to bottom) and convert to results format
        boxes = sorted(boxes, key=lambda b: (b[1], b[0]))
        
        results = []
        for i, (x, y, w, h) in enumerate(boxes):
            # Convert back to full-image coordinates
            X, Y = x + x0, y + y0
            cx, cy = X + w // 2, Y + h // 2
            
            result = {
                "bbox": [int(X), int(Y), int(w), int(h)],
                "center": [int(cx), int(cy)],
                "avatar_id": i + 1,
                "area": int(w * h),
                "aspect_ratio": round(w / float(h), 3),
                "position_in_panel": round(x / float(wR), 3)
            }
            results.append(result)
            
            print(f"    👤 Avatar {i+1}: {w}×{h}px at ({X}, {Y}), center=({cx}, {cy})")
        
        # Detection info for diagnostics
        detection_info = {
            "avatar_search_area": avatar_search_area,
            "boundary_source": boundary_source,
            "total_contours": len(cnts),
            "candidates_after_filtering": len(candidates),
            "final_avatars": len(boxes),
            "processing_steps": [
                f"Dynamic width detection ({boundary_source} boundaries)",
                "Avatar region definition",
                "Bilateral filtering",
                "Canny edge detection", 
                "Morphological dilation",
                "Contour analysis",
                "Geometric filtering",
                "Non-maximum suppression"
            ]
        }
        
        # Add width detection specific info if dynamic boundaries were used
        if width_result is not None:
            detection_info["width_detection"] = {
                "left_boundary": width_result[0],
                "right_boundary": width_result[1],
                "detected_width": width_result[2]
            }
        else:
            detection_info["width_detection"] = {
                "failed": True,
                "fallback_start": self.AVATAR_COLUMN_START,
                "fallback_width": self.AVATAR_COLUMN_WIDTH
            }
        
        return results, detection_info

    def create_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive visualization showing the detection process.
        Includes panel boundary, contours, filtered candidates, and final results.
        """
        # Load original image
        img = cv.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating advanced avatar detection visualization...")
        
        # Get detection results
        avatars, info = self.detect_avatars(image_path)
        
        # Create visualization overlay
        result = img.copy()
        H, W = result.shape[:2]
        
        # Draw avatar search area (rectangle)
        search_area = info.get("avatar_search_area", (60, 0, 120, H))
        x0, y0, wR, hR = search_area
        cv.rectangle(result, (x0, y0), (x0 + wR, y0 + hR), (255, 0, 0), 2)
        cv.putText(result, f"Avatar Area: {x0}-{x0+wR}px", (x0 + 5, 30), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw avatar detections
        for i, avatar in enumerate(avatars):
            x, y, w, h = avatar["bbox"]
            cx, cy = avatar["center"]
            
            # Draw bounding box (green)
            cv.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point (red circle)
            cv.circle(result, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw avatar ID
            cv.putText(result, f"A{avatar['avatar_id']}", (x, y - 5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center coordinates
            cv.putText(result, f"({cx},{cy})", (cx + 5, cy - 5), 
                      cv.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Add legend and stats
        legend_y = 60
        cv.putText(result, "Advanced Avatar Detection Results:", (10, legend_y), 
                  cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv.putText(result, f"Blue=Avatar Search Area  Green=Avatar Box  Red=Center", 
                  (10, legend_y + 25), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection statistics
        stats_y = legend_y + 50
        cv.putText(result, f"Detected: {len(avatars)} avatars", 
                  (10, stats_y), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(result, f"Search area: {x0}-{x0+wR}px (width={wR}px)", 
                  (10, stats_y + 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv.putText(result, f"Contours: {info.get('total_contours', 0)} -> "
                          f"Candidates: {info.get('candidates_after_filtering', 0)} -> "
                          f"Final: {info.get('final_avatars', 0)}", 
                  (10, stats_y + 40), cv.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_advanced_avatar_detection.png"
        
        # Save visualization
        cv.imwrite(output_path, result)
        print(f"✅ Advanced avatar detection visualization saved: {output_path}")
        
        return os.path.basename(output_path)

    def get_avatar_positions(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Get avatar center positions for integration with boundary detection.
        Returns list of (x, y) center coordinates.
        """
        avatars, _ = self.detect_avatars(image_path)
        return [tuple(avatar["center"]) for avatar in avatars]

if __name__ == "__main__":
    # Test the detector with dynamic width integration
    detector = CardAvatarDetector()
    
    # Test on available screenshots (updated to use existing files)
    test_images = [
        "pic/screenshots/20250904_235942_WeChat.png",
        "pic/screenshots/SimpleWidth_750px_20250905_000857.png"
    ]
    
    print("🎯 Testing CardAvatarDetector with Dynamic Width Integration")
    print("="*70)
    
    for img_path in test_images:
        if os.path.exists(img_path):
            print(f"\n{'='*60}")
            print(f"Testing: {img_path}")
            print('='*60)
            
            import time
            start_time = time.time()
            avatars, info = detector.detect_avatars(img_path)
            processing_time = (time.time() - start_time) * 1000
            
            print(f"⚡ Processing time: {processing_time:.1f}ms")
            print(f"📊 Results: {len(avatars)} avatars detected")
            print(f"🔧 Boundary source: {info.get('boundary_source', 'unknown')}")
            
            # Show width detection details
            width_info = info.get('width_detection', {})
            if 'failed' not in width_info:
                print(f"📏 Dynamic boundaries: left={width_info.get('left_boundary')}, right={width_info.get('right_boundary')}, width={width_info.get('detected_width')}px")
            else:
                print(f"⚠️ Used fallback boundaries: start={width_info.get('fallback_start')}, width={width_info.get('fallback_width')}px")
            
            # Create visualization
            viz_file = detector.create_visualization(img_path)
            print(f"🎨 Visualization: {viz_file}")
        else:
            print(f"⚠️ Test image not found: {img_path}")
    
    print(f"\n✅ Dynamic width integration testing complete!")