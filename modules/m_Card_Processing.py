#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Consolidated Card Processing Module (m_Card_Processing.py)

This module consolidates six card processing functionalities with a universal coordinate system:

## Core Detector Classes:
1. BoundaryCoordinator - Detects message card width boundaries
2. RightBoundaryDetector - Enhanced right boundary detection
3. CardAvatarDetector - Detects avatar positions within cards  
4. CardBoundaryDetector - Detects individual card boundaries
5. ContactNameBoundaryDetector - Detects contact name boundaries
6. TimeBoxDetector - Detects timestamp boundaries

## Universal Coordinate System:

### WeChatCoordinateContext Class:
Unified coordinate management system that provides:
- Standardized coordinate storage for all processing phases
- OCR-ready region extraction with confidence scoring
- Coordinate validation against image boundaries
- Legacy format conversion for backward compatibility

### Usage Patterns:

#### Basic Usage (Legacy Compatible):
```python
# All existing code continues to work unchanged
detector = cBoundaryCoordinator()
result = detector.detect_width("screenshot.png")  # Returns (left, right, width)

avatars, info = cCardAvatarDetector().detect_avatars("screenshot.png")
cards, info = cCardBoundaryDetector().detect_cards("screenshot.png")
```

#### Universal Coordinate System Usage:
```python
# Option 1: Let detectors create coordinate context automatically
detector = cBoundaryCoordinator()
result, coord_context = detector.detect_width("screenshot.png", return_context=True)
# coord_context now contains structured coordinate data

# Option 2: Provide existing coordinate context for population
coord_context = cWeChatCoordinateContext("screenshot.png", (800, 600))
result = detector.detect_width("screenshot.png", coord_context=coord_context)
# coord_context is now populated with width boundaries

# Option 3: Full pipeline with comprehensive coordinate structure
results = process_screenshot_file("screenshot.png")
coord_context = results["coordinate_context"]  # Complete coordinate structure
ocr_regions = results["ocr_regions"]  # OCR-ready extraction regions
```

### Coordinate Context Structure:
```python
{
    "image_metadata": {
        "source_image": "screenshot.png",
        "dimensions": {"width": 800, "height": 600},
        "processing_timestamp": "20250905_143022"
    },
    "global_boundaries": {
        "conversation_area": {
            "bbox": [50, 0, 700, 600],  # [x, y, width, height]
            "source_step": "BoundaryCoordinator.detect_width",
            "confidence": 0.95
        }
    },
    "cards": [
        {
            "card_id": 1,
            "card_region": {"bbox": [100, 50, 600, 100], "confidence": 0.95},
            "components": {
                "avatar": {"bbox": [80, 70, 40, 40], "ocr_suitable": false},
                "contact_name": {"bbox": [130, 60, 100, 20], "ocr_suitable": true, "expected_content": "contact_name"},
                "timestamp": {"bbox": [500, 60, 80, 20], "ocr_suitable": true, "expected_content": "timestamp"}
            }
        }
    ],
    "ocr_extraction_regions": {
        "contact_names": [{"card_id": 1, "bbox": [130, 60, 100, 20], "confidence": 0.85}],
        "timestamps": [{"card_id": 1, "bbox": [500, 60, 80, 20], "confidence": 0.8}],
        "messages": [],
        "avatars": []
    }
}
```

### OCR Integration:
```python
# Extract OCR-ready regions by content type
contact_name_regions = coord_context.extract_all_regions("contact_names")
timestamp_regions = coord_context.extract_all_regions("timestamps")

# Batch OCR processing using OCRExtractionUtils
ocr_batch = cOCRExtractionUtils.create_ocr_batch_from_context(
    coord_context, "screenshot.png", ["contact_names", "timestamps"]
)
```

### Validation:
```python
# Validate all coordinates against image boundaries
validation_result = coord_context.validate_coordinates()
if validation_result["errors"]:
    print("Coordinate validation failed:", validation_result["errors"])
```

### Diagnostic Testing:
Use the coordinate diagnostic server for testing and validation:
```bash
python coordinate_diagnostic_server.py
# Visit http://localhost:5002 for web interface
```

Each detector class is implemented for modular usage with full backward compatibility.
"""

import cv2
import numpy as np
import os
import sys

# Fix matplotlib backend for threading compatibility on macOS
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend to prevent threading issues
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Tool class import (Human Structural Logic: Repeatedly-called functionality = Tool Class)
try:
    from m_screenshot_finder_tool import cWeChat_Screenshot_Finder
except ImportError:
    # Fallback for environments where tool class not available
    cWeChat_Screenshot_Finder = None

# Import consolidated modular components
try:
    from . import m_screenshot_processor
    from . import visualization_engine
    from . import image_utils
    SCREENSHOT_AVAILABLE = True
except ImportError:
    # Fallback imports for direct execution
    try:
        import modules.m_screenshot_processor as m_screenshot_processor
        import visualization_engine
        import image_utils
        SCREENSHOT_AVAILABLE = True
    except ImportError:
        print("⚠️  Modular components not available. Screenshot capture disabled.")
        SCREENSHOT_AVAILABLE = False


# =============================================================================
# WECHAT COORDINATE CONTEXT SYSTEM
# =============================================================================

class cWeChatCoordinateContext:
    """
    Unified coordinate management system for WeChat processing pipeline
    
    Provides standardized coordinate storage, OCR-ready region extraction,
    and consistent API across all 6 processing phases.
    """
    
    def __init__(self, image_path: str, image_dimensions: Tuple[int, int]):
        """
        Initialize coordinate context for a WeChat screenshot
        
        Args:
            image_path: Path to the screenshot being processed
            image_dimensions: (width, height) of the screenshot
        """
        self.context = {
            "image_metadata": {
                "source_image": os.path.basename(image_path),
                "full_path": image_path,
                "dimensions": {"width": image_dimensions[0], "height": image_dimensions[1]},
                "processing_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S")
            },
            "global_boundaries": {},
            "cards": [],
            "ocr_extraction_regions": {
                "contact_names": [],
                "timestamps": [],
                "messages": [],
                "avatars": []
            }
        }
    
    def add_global_boundary(self, boundary_type: str, bbox: List[int], source_step: str, confidence: float = 1.0):
        """Add global boundary coordinates (e.g., conversation area)"""
        self.context["global_boundaries"][boundary_type] = {
            "bbox": bbox,  # [x, y, w, h]
            "source_step": source_step,
            "confidence": confidence
        }
    
    def add_card(self, card_id: int, card_bbox: List[int], source_step: str, confidence: float = 1.0):
        """Add a new card with its bounding box"""
        card = {
            "card_id": card_id,
            "card_region": {
                "bbox": card_bbox,  # [x, y, w, h]
                "source_step": source_step,
                "confidence": confidence
            },
            "components": {}
        }
        
        # Ensure we have enough slots for this card
        while len(self.context["cards"]) < card_id:
            self.context["cards"].append(None)
        
        self.context["cards"][card_id - 1] = card
    
    def add_component(self, card_id: int, component_type: str, bbox: List[int], 
                     source_step: str, confidence: float = 1.0, ocr_suitable: bool = True,
                     expected_content: str = None, **kwargs):
        """Add a component to a specific card"""
        if card_id > len(self.context["cards"]) or self.context["cards"][card_id - 1] is None:
            raise ValueError(f"Card {card_id} does not exist. Add card first.")
        
        component_data = {
            "bbox": bbox,  # [x, y, w, h] 
            "source_step": source_step,
            "confidence": confidence,
            "ocr_suitable": ocr_suitable
        }
        
        if expected_content:
            component_data["expected_content"] = expected_content
            
        # Add any additional data passed via kwargs
        component_data.update(kwargs)
        
        self.context["cards"][card_id - 1]["components"][component_type] = component_data
        
        # Add to OCR extraction regions if suitable
        if ocr_suitable and expected_content:
            region_key = self._get_ocr_region_key(expected_content)
            if region_key:
                self.context["ocr_extraction_regions"][region_key].append({
                    "card_id": card_id,
                    "bbox": bbox,
                    "confidence": confidence
                })
    
    def _get_ocr_region_key(self, expected_content: str) -> Optional[str]:
        """Map expected content to OCR region key"""
        content_mapping = {
            "contact_name": "contact_names",
            "timestamp": "timestamps", 
            "message_text": "messages",
            "message_content": "messages"
        }
        return content_mapping.get(expected_content)
    
    def extract_ocr_region(self, component_type: str, card_id: int = None) -> Optional[Dict]:
        """
        Extract specific region for OCR processing
        
        Args:
            component_type: Type of component ('contact_name', 'timestamp', 'message_content')
            card_id: Specific card ID, or None for all cards
            
        Returns:
            Dictionary with region info or None if not found
        """
        if card_id is not None:
            if card_id > len(self.context["cards"]) or self.context["cards"][card_id - 1] is None:
                return None
                
            card = self.context["cards"][card_id - 1]
            if component_type in card["components"]:
                component = card["components"][component_type]
                if component.get("ocr_suitable", False):
                    return {
                        "card_id": card_id,
                        "bbox": component["bbox"],
                        "confidence": component["confidence"],
                        "expected_content": component.get("expected_content"),
                        "source_step": component["source_step"]
                    }
        
        return None
    
    def extract_all_regions(self, content_type: str) -> List[Dict]:
        """
        Extract all regions of a specific content type for batch OCR processing
        
        Args:
            content_type: 'contact_names', 'timestamps', 'messages', 'avatars'
            
        Returns:
            List of region dictionaries sorted by confidence (highest first)
        """
        if content_type not in self.context["ocr_extraction_regions"]:
            return []
        
        regions = self.context["ocr_extraction_regions"][content_type].copy()
        # Sort by confidence, highest first
        regions.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return regions
    
    def get_card_count(self) -> int:
        """Get number of cards in context"""
        return len([card for card in self.context["cards"] if card is not None])
    
    def validate_coordinates(self) -> Dict[str, List[str]]:
        """
        Validate all coordinates against image boundaries
        
        Returns:
            Dictionary with validation results and any errors found
        """
        errors = []
        warnings = []
        
        img_width = self.context["image_metadata"]["dimensions"]["width"]
        img_height = self.context["image_metadata"]["dimensions"]["height"]
        
        def validate_bbox(bbox: List[int], context_name: str) -> None:
            x, y, w, h = bbox
            if x < 0 or y < 0:
                errors.append(f"{context_name}: Negative coordinates ({x}, {y})")
            if x + w > img_width or y + h > img_height:
                errors.append(f"{context_name}: Coordinates exceed image bounds")
            if w <= 0 or h <= 0:
                errors.append(f"{context_name}: Invalid dimensions {w}×{h}")
        
        # Validate global boundaries
        for boundary_type, boundary_data in self.context["global_boundaries"].items():
            validate_bbox(boundary_data["bbox"], f"Global boundary {boundary_type}")
        
        # Validate card regions and components
        for card in self.context["cards"]:
            if card is None:
                continue
                
            card_id = card["card_id"]
            validate_bbox(card["card_region"]["bbox"], f"Card {card_id} region")
            
            for comp_type, comp_data in card["components"].items():
                validate_bbox(comp_data["bbox"], f"Card {card_id} {comp_type}")
        
        return {"errors": errors, "warnings": warnings}
    
    def to_dict(self) -> Dict:
        """Return the complete coordinate context as dictionary"""
        return self.context.copy()
    
    def to_legacy_format(self, card_id: int) -> Dict:
        """
        Convert to legacy coordinate format for backward compatibility
        
        Args:
            card_id: ID of card to convert
            
        Returns:
            Dictionary in legacy format
        """
        if card_id > len(self.context["cards"]) or self.context["cards"][card_id - 1] is None:
            return {}
        
        card = self.context["cards"][card_id - 1]
        card_bbox = card["card_region"]["bbox"]
        
        # Convert to legacy format used by existing code
        legacy_card = {
            "card_id": card_id,
            "bbox": card_bbox,
            "boundaries": {
                "card": {
                    "left": card_bbox[0],
                    "right": card_bbox[0] + card_bbox[2], 
                    "top": card_bbox[1],
                    "bottom": card_bbox[1] + card_bbox[3]
                }
            }
        }
        
        # Add avatar data if present
        if "avatar" in card["components"]:
            avatar_data = card["components"]["avatar"]
            avatar_bbox = avatar_data["bbox"]
            legacy_card["avatar"] = {
                "bbox": avatar_bbox,
                "center": avatar_data.get("center", [avatar_bbox[0] + avatar_bbox[2]//2, 
                                                   avatar_bbox[1] + avatar_bbox[3]//2])
            }
            legacy_card["boundaries"]["avatar"] = {
                "left": avatar_bbox[0],
                "right": avatar_bbox[0] + avatar_bbox[2],
                "top": avatar_bbox[1], 
                "bottom": avatar_bbox[1] + avatar_bbox[3]
            }
        
        # Add name boundary if present
        if "contact_name" in card["components"]:
            name_data = card["components"]["contact_name"]
            legacy_card["name_boundary"] = {
                "bbox": name_data["bbox"],
                "confidence": name_data["confidence"],
                "detection_method": name_data.get("detection_method", "unified_context")
            }
        
        # Add time box if present
        if "timestamp" in card["components"]:
            time_data = card["components"]["timestamp"] 
            legacy_card["time_box"] = {
                "bbox": time_data["bbox"],
                "density_score": time_data.get("density_score", time_data["confidence"]),
                "detection_method": time_data.get("detection_method", "unified_context")
            }
        
        return legacy_card


# =============================================================================
# COORDINATE CONVERSION UTILITIES
# =============================================================================

class cCoordinateConverter:
    """Utilities for converting between different coordinate formats"""
    
    @staticmethod
    def to_bbox_format(left: int, top: int, right: int, bottom: int) -> List[int]:
        """Convert left/top/right/bottom to [x, y, w, h] bbox format"""
        return [left, top, right - left, bottom - top]
    
    @staticmethod 
    def from_bbox_format(bbox: List[int]) -> Tuple[int, int, int, int]:
        """Convert [x, y, w, h] bbox to (left, top, right, bottom)"""
        x, y, w, h = bbox
        return (x, y, x + w, y + h)
    
    @staticmethod
    def bbox_to_center(bbox: List[int]) -> List[int]:
        """Calculate center point from bbox [x, y, w, h]"""
        x, y, w, h = bbox
        return [x + w // 2, y + h // 2]
    
    @staticmethod
    def center_to_bbox(center: List[int], width: int, height: int) -> List[int]:
        """Calculate bbox from center point and dimensions"""
        cx, cy = center
        x = cx - width // 2
        y = cy - height // 2
        return [x, y, width, height]
    
    @staticmethod
    def validate_bbox(bbox: List[int], image_width: int, image_height: int) -> bool:
        """Validate bbox is within image boundaries"""
        x, y, w, h = bbox
        return (x >= 0 and y >= 0 and w > 0 and h > 0 and 
                x + w <= image_width and y + h <= image_height)
    
    @staticmethod
    def clip_bbox_to_image(bbox: List[int], image_width: int, image_height: int) -> List[int]:
        """Clip bbox to stay within image boundaries"""
        x, y, w, h = bbox
        
        # Clip to image boundaries
        x = max(0, min(x, image_width - 1))
        y = max(0, min(y, image_height - 1))
        w = min(w, image_width - x)
        h = min(h, image_height - y)
        
        return [x, y, max(1, w), max(1, h)]
    
    @staticmethod
    def bbox_overlap(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate overlap ratio between two bboxes (0.0 to 1.0)"""
        x1, y1, w1, h1 = bbox1
        x2, y2, w2, h2 = bbox2
        
        # Calculate intersection
        left = max(x1, x2)
        top = max(y1, y2)
        right = min(x1 + w1, x2 + w2)
        bottom = min(y1 + h1, y2 + h2)
        
        if left >= right or top >= bottom:
            return 0.0
        
        intersection_area = (right - left) * (bottom - top)
        bbox1_area = w1 * h1
        bbox2_area = w2 * h2
        union_area = bbox1_area + bbox2_area - intersection_area
        
        return intersection_area / union_area if union_area > 0 else 0.0
    
    @staticmethod
    def bbox_distance(bbox1: List[int], bbox2: List[int]) -> float:
        """Calculate distance between bbox centers"""
        center1 = cCoordinateConverter.bbox_to_center(bbox1)
        center2 = cCoordinateConverter.bbox_to_center(bbox2)
        
        dx = center2[0] - center1[0]
        dy = center2[1] - center1[1]
        
        return (dx * dx + dy * dy) ** 0.5
    
    @staticmethod
    def legacy_avatar_to_unified(avatar_dict: Dict) -> Dict:
        """Convert legacy avatar format to unified coordinate format"""
        bbox = avatar_dict.get("bbox", [0, 0, 0, 0])
        center = avatar_dict.get("center", cCoordinateConverter.bbox_to_center(bbox))
        
        return {
            "bbox": bbox,
            "center": center,
            "avatar_id": avatar_dict.get("avatar_id", 1),
            "area": bbox[2] * bbox[3] if len(bbox) == 4 else 0,
            "aspect_ratio": bbox[2] / bbox[3] if len(bbox) == 4 and bbox[3] > 0 else 1.0,
            "source_step": "STEP_3",
            "confidence": 0.95,
            "ocr_suitable": False,
            "expected_content": None
        }
    
    @staticmethod
    def legacy_card_to_unified(card_dict: Dict) -> Tuple[List[int], Dict]:
        """
        Convert legacy card format to unified coordinate format
        
        Returns:
            Tuple of (card_bbox, components_dict)
        """
        # Extract card bbox
        card_bbox = card_dict.get("bbox", [0, 0, 0, 0])
        
        components = {}
        
        # Convert avatar component
        if "avatar" in card_dict:
            avatar_data = cCoordinateConverter.legacy_avatar_to_unified(card_dict["avatar"])
            components["avatar"] = avatar_data
        
        # Convert name boundary component
        if "name_boundary" in card_dict:
            name_data = card_dict["name_boundary"]
            components["contact_name"] = {
                "bbox": name_data.get("bbox", [0, 0, 0, 0]),
                "source_step": "STEP_5",
                "confidence": name_data.get("confidence", 0.8),
                "ocr_suitable": True,
                "expected_content": "contact_name",
                "detection_method": name_data.get("detection_method", "legacy")
            }
        
        # Convert time box component
        if "time_box" in card_dict:
            time_data = card_dict["time_box"]
            components["timestamp"] = {
                "bbox": time_data.get("bbox", [0, 0, 0, 0]),
                "source_phase": "PHASE_6", 
                "confidence": time_data.get("density_score", 0.8),
                "ocr_suitable": True,
                "expected_content": "timestamp",
                "density_score": time_data.get("density_score", 0.8),
                "detection_method": time_data.get("detection_method", "legacy")
            }
        
        return card_bbox, components


# =============================================================================
# OCR EXTRACTION UTILITIES
# =============================================================================

class cOCRExtractionUtils:
    """Utilities for extracting regions from coordinate context for OCR processing"""
    
    @staticmethod
    def extract_region_from_image(image_path: str, bbox: List[int]) -> Optional[np.ndarray]:
        """
        Extract a specific region from image for OCR processing
        
        Args:
            image_path: Path to source image
            bbox: [x, y, w, h] region to extract
            
        Returns:
            Extracted region as numpy array or None if failed
        """
        try:
            import cv2
            img = cv2.imread(image_path)
            if img is None:
                return None
            
            x, y, w, h = bbox
            
            # Validate and clip bbox to image boundaries
            img_height, img_width = img.shape[:2]
            clipped_bbox = cCoordinateConverter.clip_bbox_to_image(bbox, img_width, img_height)
            x, y, w, h = clipped_bbox
            
            # Extract region
            region = img[y:y+h, x:x+w]
            return region
            
        except Exception as e:
            print(f"❌ Failed to extract region {bbox}: {e}")
            return None
    
    @staticmethod
    def batch_extract_regions(image_path: str, regions: List[Dict], min_confidence: float = 0.5) -> List[Dict]:
        """
        Extract multiple regions from image for batch OCR processing
        
        Args:
            image_path: Path to source image
            regions: List of region dictionaries with bbox and confidence
            min_confidence: Minimum confidence threshold for extraction
            
        Returns:
            List of dictionaries with extracted regions and metadata
        """
        results = []
        
        for region in regions:
            if region.get('confidence', 0.0) < min_confidence:
                continue
                
            bbox = region.get('bbox')
            if not bbox:
                continue
            
            extracted_region = cOCRExtractionUtils.extract_region_from_image(image_path, bbox)
            if extracted_region is not None:
                results.append({
                    'card_id': region.get('card_id'),
                    'bbox': bbox,
                    'confidence': region.get('confidence', 0.0),
                    'expected_content': region.get('expected_content'),
                    'image_region': extracted_region,
                    'region_size': (bbox[2], bbox[3])  # w, h
                })
        
        return results
    
    @staticmethod
    def save_extracted_regions(regions: List[Dict], output_dir: str = "ocr_regions") -> List[str]:
        """
        Save extracted regions as individual image files for OCR processing
        
        Args:
            regions: List of region dictionaries from batch_extract_regions
            output_dir: Directory to save region images
            
        Returns:
            List of saved file paths
        """
        import cv2
        import os
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        saved_paths = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for i, region in enumerate(regions):
            card_id = region.get('card_id', i+1)
            content_type = region.get('expected_content', 'unknown')
            
            filename = f"{timestamp}_card{card_id}_{content_type}_{i+1}.png"
            filepath = os.path.join(output_dir, filename)
            
            try:
                cv2.imwrite(filepath, region['image_region'])
                saved_paths.append(filepath)
                print(f"💾 Saved OCR region: {filename}")
            except Exception as e:
                print(f"❌ Failed to save region {filename}: {e}")
        
        return saved_paths
    
    @staticmethod
    def create_ocr_batch_from_context(coord_context: cWeChatCoordinateContext, 
                                    image_path: str,
                                    content_types: List[str] = None,
                                    min_confidence: float = 0.5) -> Dict[str, List[Dict]]:
        """
        Create OCR-ready batches from coordinate context
        
        Args:
            coord_context: ccWeChatCoordinateContext instance
            image_path: Path to source image
            content_types: List of content types to extract ('contact_names', 'timestamps', 'messages')
            min_confidence: Minimum confidence threshold
            
        Returns:
            Dictionary with extracted regions by content type
        """
        if content_types is None:
            content_types = ['contact_names', 'timestamps', 'messages']
        
        ocr_batches = {}
        
        for content_type in content_types:
            regions = coord_context.extract_all_regions(content_type)
            extracted = cOCRExtractionUtils.batch_extract_regions(
                image_path, regions, min_confidence
            )
            ocr_batches[content_type] = extracted
            print(f"📋 Prepared {len(extracted)} {content_type} regions for OCR")
        
        return ocr_batches


# =============================================================================
# PHASE 1: LEFT BOUNDARY DETECTOR
# =============================================================================

class cLeftBoundaryDetector:
    """
    Detects the left boundary of WeChat conversation area using vertical edge detection.
    
    📋 PURPOSE:
    Identifies the visual boundary between WeChat's sidebar and the main conversation
    area by finding the strongest vertical edge in the left portion of the screen.
    This establishes the left coordinate for all subsequent message card processing.
    
    📌 INPUT CONTRACT:
    - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
    - debug_mode: bool - Enable visualization output (default: False)
    
    📌 OUTPUT CONTRACT:
    - Success: int - X-coordinate of left boundary in pixels
    - Failure: None - Returns None if detection fails
    - Range: Typically 50-150px from left edge
    
    🔧 ALGORITHM:
    1. Load image and focus on left 15% (to find avatar column start)
    2. Apply Sobel edge detection to find vertical edges
    3. Create 1D intensity profile by averaging gradients
    4. Find strongest edge peak using Gaussian smoothing
    5. Apply 10px LEFT offset for actual boundary (10px before avatar start)
    
    📊 KEY PARAMETERS:
    - CONVERSATION_WIDTH_RATIO = 0.15  # Search left 15% to find avatar left edge
    - SIDEBAR_OFFSET = 10              # Pixels offset LEFT from edge (10px before avatar)
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
    - Optional: image_utils.ffind_vertical_edge_x (modular import)
    - Integrates with: BoundaryCoordinator for complete width detection
    """
    
    def __init__(self, debug_mode: bool = False):
        # Debug mode control for visualization generation
        self.debug_mode = debug_mode

        # Edge detection parameters
        self.CONVERSATION_WIDTH_RATIO = 0.15  # Focus on left 15% to find avatar column start
        self.EDGE_THRESHOLD_LOW = 30
        self.EDGE_THRESHOLD_HIGH = 100
        self.SIDEBAR_OFFSET = 10  # Offset LEFT from detected edge (10px before avatar start)
        
    def detect_left_boundary(self, image_path: str) -> Optional[int]:
        """
        Detect left boundary of WeChat conversation area
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
        
        📌 OUTPUT CONTRACT:
        - Success: int - Left boundary x-coordinate in pixels
        - Failure: None - Detection failedLet
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True
        """
        try:
            # Load and validate image
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Could not load image: {image_path}")
                return None
                
            print(f"🎯 1. Left Boundary Detection: {os.path.basename(image_path)}")
            img_height, img_width = img.shape[:2]
            print(f"📐 Image dimensions: {img_width}×{img_height}")
            
            # Use modular image utils if available, otherwise fallback
            if SCREENSHOT_AVAILABLE:
                x_pos, confidence, profile = image_utils.ffind_vertical_edge_x(
                    img, x0=0, x1=int(img_width * self.CONVERSATION_WIDTH_RATIO), rightmost=False
                )
            else:
                x_pos, confidence, profile = self._find_vertical_edge_fallback(
                    img, x0=0, x1=int(img_width * self.CONVERSATION_WIDTH_RATIO), rightmost=False
                )
            
            # Apply sidebar offset for actual boundary
            left_boundary = max(0, x_pos - self.SIDEBAR_OFFSET)
            
            print(f"🔍 Edge detected at: {x_pos}px (confidence: {confidence:.3f})")
            print(f"✅ Left boundary: {left_boundary}px (with {self.SIDEBAR_OFFSET}px offset)")
            
            # Generate debug visualization if enabled
            if self.debug_mode:
                self._generate_debug_visualization(img, x_pos, left_boundary, confidence, profile)
            
            return left_boundary
            
        except Exception as e:
            print(f"❌ Left boundary detection error: {e}")
            return None
    
    def _find_vertical_edge_fallback(self, img, x0=0, x1=None, y0=0, y1=None, rightmost=True):
        """
        Legacy fallback implementation of find_vertical_edge_x
        Return the x (in original image coords) of the dominant vertical edge inside ROI.
        """
        if img.ndim == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        H, W = gray.shape
        x1 = W if x1 is None else x1
        y1 = H if y1 is None else y1
        roi = gray[y0:y1, x0:x1]

        # Optional denoise/normalize for dark UIs
        roi = cv2.bilateralFilter(roi, d=5, sigmaColor=25, sigmaSpace=25)

        # Compute horizontal gradient (vertical edges) → 1D profile
        # (Sobel is a bit smoother than simple diff)
        sobelx = cv2.Sobel(roi, cv2.CV_32F, 1, 0, ksize=3)
        prof = np.mean(np.abs(sobelx), axis=0)  # average over rows → shape (roiW,)

        # Smooth & pick peak
        prof = cv2.GaussianBlur(prof.reshape(1,-1), (1,7), 0).ravel()
        if rightmost:
            idx = int(np.argmax(prof[::-1]))          # strongest from right
            xr  = (x1 - 1) - idx
        else:
            xr  = int(np.argmax(prof)) + x0

        # Confidence (0–1): peak vs neighborhood
        peak = prof[(xr - x0)]
        med  = float(np.median(prof))
        mad  = float(np.median(np.abs(prof - med)) + 1e-6)
        conf = max(0.0, min(1.0, (peak - med) / (6*mad)))  # rough score

        return xr, conf, prof
    
    def _generate_debug_visualization(self, img, detected_edge, left_boundary, confidence, profile):
        """Generate debug visualization showing left boundary detection"""
        try:
            from datetime import datetime
            
            # Create visualization overlay
            result = img.copy()
            img_height, img_width = img.shape[:2]
            
            # Draw detected edge line (yellow)
            cv2.line(result, (detected_edge, 0), (detected_edge, img_height), (0, 255, 255), 2)
            
            # Draw actual left boundary (green)  
            cv2.line(result, (left_boundary, 0), (left_boundary, img_height), (0, 255, 0), 3)
            
            # Add text annotations
            cv2.putText(result, f"Edge: {detected_edge}px", (detected_edge + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            cv2.putText(result, f"Boundary: {left_boundary}px", (left_boundary + 10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(result, f"Confidence: {confidence:.3f}", (10, img_height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Save debug visualization
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_01_LeftBoundary_{left_boundary}px.png"
            
            os.makedirs("pic/screenshots", exist_ok=True)
            cv2.imwrite(output_path, result)
            print(f"🎨 Left boundary debug visualization: {output_path}")
            
        except Exception as e:
            print(f"⚠️ Could not generate debug visualization: {e}")


# Legacy compatibility - keep find_vertical_edge_x function for backward compatibility
if SCREENSHOT_AVAILABLE:
    find_vertical_edge_x = image_utils.ffind_vertical_edge_x
else:
    def find_vertical_edge_x(img, x0=0, x1=None, y0=0, y1=None, rightmost=True):
        """Legacy compatibility function - use LeftBoundaryDetector class instead"""
        detector = cLeftBoundaryDetector()
        return detector._find_vertical_edge_fallback(img, x0, x1, y0, y1, rightmost)


# =============================================================================
# PHASE 2: RIGHT BOUNDARY DETECTOR
# =============================================================================

class cRightBoundaryDetector:
    """
    Detects the right boundary of WeChat conversation area using horizontal pixel difference analysis.
    
    📋 PURPOSE:
    Identifies the visual boundary that marks the right edge of the WeChat conversation
    content area by analyzing horizontal pixel differences in high-contrast processed images.
    Uses Photoshop-style level adjustments and red region detection to find strong
    vertical transitions that indicate content boundaries.
    
    📌 INPUT CONTRACT:
    - img: np.ndarray - WeChat screenshot image (BGR or grayscale)
    - img_width: int - Image width in pixels for boundary calculations
    - preprocessed_image_path: Optional[str] - Path for preprocessed image (unused)
    - debug_mode: bool - Enable visualization output (default: False)
    
    📌 OUTPUT CONTRACT:
    - Success: int - X-coordinate of right boundary in pixels
    - Failure: int - Fallback boundary (80% of image width)
    - Range: Typically 300-800px from left edge (conversation content area)
    
    🔧 ALGORITHM:
    1. Apply Photoshop-style levels adjustment (gamma correction)
    2. Calculate horizontal pixel differences using np.diff()
    3. Identify red regions (transitions > 100 intensity)
    4. Search for strongest vertical boundary in center region (360-1368px)
    5. Rank boundary candidates by red pixel intensity
    6. Return strongest boundary or geometric fallback
    
    📊 KEY PARAMETERS:
    - INPUT_BLACK_POINT = 32       # Photoshop levels input black
    - INPUT_WHITE_POINT = 107      # Photoshop levels input white  
    - GAMMA = 0.67                 # Gamma correction value
    - EDGE_THRESHOLD = 0.10        # 10% threshold for preprocessing
    - PREPROCESSED_THRESHOLD = 0.005  # 0.5% threshold for transitions
    - MIN_BOUNDARY_PX = 200        # Minimum boundary position
    
    🎨 VISUAL OUTPUT:
    - Preprocessing: YYYYMMDD_HHMMSS_02_photoshop_levels_gamma.png
    - Heatmap: YYYYMMDD_HHMMSS_03_horizontal_differences.png
    - With boundary: YYYYMMDD_HHMMSS_04_horizontal_differences_with_boundary.png
    - Dual-plot visualization showing heatmap + profile with red boundary line
    
    🔍 DEBUG VARIABLES:
    - Key variables used for visualization:
      • scaled: np.ndarray - Level-adjusted grayscale image (preprocessing output)
      • diff_x: np.ndarray - Horizontal pixel differences matrix (heatmap data)
      • red_regions: np.ndarray - Boolean mask of strong transitions >100
      • boundary_candidates: List[Tuple[int, int]] - (x_position, intensity) pairs
      • strongest_x: int - Final detected boundary x-coordinate (red line)
      • profile: np.ndarray - Mean differences per column (profile plot)
    - Debug triggers: When debug_mode=True in constructor
    - Output format: PNG heatmaps saved to pic/screenshots/ with timestamp
    
    ⚙️ DEPENDENCIES:
    - Required: opencv-python, numpy, matplotlib
    - Optional: None (self-contained implementation)
    - Integrates with: BoundaryCoordinator for complete width detection
    """
    
    def __init__(self, debug_mode: bool = False):
        # Debug mode control for visualization generation
        self.debug_mode = debug_mode
        
        # Optimized Photoshop levels parameters for high contrast preprocessing
        self.INPUT_BLACK_POINT = 32     # Optimized input black point (27-36 range) for distinctive edges
        self.INPUT_WHITE_POINT = 107    # Photoshop input white point
        self.GAMMA = 0.67               # Photoshop gamma value
        
        # Detection parameters
        self.EDGE_THRESHOLD = 0.10      # 10% threshold for internal preprocessing
        self.PREPROCESSED_THRESHOLD = 0.005  # 0.5% threshold for white-to-black transitions
        self.SMOOTHING_SIZE = 5         # Smoothing kernel size (reduced for sharper transitions)
        self.MIN_BOUNDARY_PX = 200      # Minimum boundary position for message content (lowered to detect left boundaries)
        
    def _apply_level_adjustment(self, gray: np.ndarray) -> np.ndarray:
        """
        Apply Photoshop-style levels adjustment using your exact method with gamma correction
        Creates high contrast white cards on black background for clear boundary detection
        """
        print(f"  🎨 Applying Photoshop-style levels adjustment with gamma correction...")
        
        # Apply your exact method with gamma correction
        in_black, in_white = self.INPUT_BLACK_POINT, self.INPUT_WHITE_POINT
        gamma = self.GAMMA
        
        # Normalize to 0-1 range and clip
        arr = np.clip((gray - in_black) / (in_white - in_black), 0, 1)
        
        # Apply gamma correction and scale to 0-255
        arr = (arr ** (1/gamma)) * 255
        
        # Convert to uint8
        scaled = arr.astype(np.uint8)
        
        # Save the result
        self._save_preprocessing_image(scaled, "02_photoshop_levels_gamma.png")
        
        print(f"  🎨 Photoshop levels with gamma applied:")
        print(f"    - Input black: {in_black}")
        print(f"    - Input white: {in_white}")
        print(f"    - Gamma: {gamma}")
        print(f"    - Normalize: arr = clip((pixel - {in_black}) / ({in_white} - {in_black}), 0, 1)")
        print(f"    - Gamma correction: arr = (arr ** (1/{gamma})) * 255")
        print(f"  📸 Image saved: 02_photoshop_levels_gamma.png")
        
        return scaled
    
    def _save_preprocessing_image(self, img: np.ndarray, filename: str):
        """Save preprocessing phase images for visualization"""
        import os
        from datetime import datetime
        
        try:
            # Create screenshot folder if it doesn't exist
            screenshot_dir = "pic/screenshots"
            if not os.path.exists(screenshot_dir):
                os.makedirs(screenshot_dir)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = os.path.join(screenshot_dir, f"{timestamp}_{filename}")
            
            # Save grayscale image
            import cv2
            cv2.imwrite(filepath, img)
            
        except Exception as e:
            print(f"  ⚠️ Could not save preprocessing image: {e}")
    
    def _generate_horizontal_differences_heatmap(self, diff_x: np.ndarray, detected_boundary: int = None, filename_suffix: str = "horizontal_differences") -> str:
        """
        Generate horizontal pixel differences heatmap visualization similar to your reference image
        
        Args:
            diff_x: Horizontal pixel differences array from np.diff()
            detected_boundary: X-axis position of detected boundary (optional, for overlay)
            filename_suffix: Suffix for the output filename
            
        Returns:
            Path to saved heatmap image
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from datetime import datetime
            
            print(f"  🎨 Generating horizontal differences heatmap with profile plot...")
            
            # Create figure with precise GridSpec layout for identical physical x-axis lengths
            fig = plt.figure(figsize=(16, 12))
            
            # Create grid: plotting area (90%) + colorbar space (10%)
            # This ensures colorbar doesn't affect plot widths
            gs = gridspec.GridSpec(2, 2, 
                                 width_ratios=[10, 1],  # 10:1 ratio for plot:colorbar  
                                 height_ratios=[3, 1],   # 3:1 ratio for heatmap:profile
                                 left=0.08, right=0.95, top=0.95, bottom=0.08,
                                 wspace=0.05, hspace=0.3)
            
            # Both plots use the same column (0) for identical physical width
            ax_heatmap = fig.add_subplot(gs[0, 0])  # Top plot, column 0
            ax_profile = fig.add_subplot(gs[1, 0])   # Bottom plot, column 0
            cax = fig.add_subplot(gs[0, 1])         # Colorbar in column 1
            
            # HEATMAP (Top Plot): Use RdBu_r colormap with explicit extent for coordinate alignment
            height, width = diff_x.shape
            
            # Define coordinate system that matches the actual image pixels (0 to width-1)
            x_extent = [0, width - 1]  # Use actual image pixel coordinates  
            y_extent = [height, 0]     # Image coordinates (top=0, bottom=height)
            
            im = ax_heatmap.imshow(diff_x, cmap='RdBu_r', aspect='auto', vmin=-200, vmax=200,
                                  extent=[x_extent[0], x_extent[1], y_extent[0], y_extent[1]])
            
            # Add colorbar in dedicated space to avoid affecting plot widths
            cbar = plt.colorbar(im, cax=cax)
            cbar.set_label('Pixel Difference Value', rotation=270, labelpad=20, fontsize=12)
            
            # Set title and labels for heatmap
            ax_heatmap.set_title('Horizontal Pixel Differences (Red Lines = Target Boundaries)', fontsize=16, fontweight='bold', pad=20)
            ax_heatmap.set_xlabel('X Position (pixels)', fontsize=12)
            ax_heatmap.set_ylabel('Y Position (pixels)', fontsize=12)
            
            # Add red vertical line at detected boundary position in heatmap
            if detected_boundary is not None:
                ax_heatmap.axvline(x=detected_boundary, color='red', linewidth=3, alpha=0.9, 
                                  label=f'Detected Boundary: {detected_boundary}px', linestyle='-')
                
                # Add red circle markers at top and bottom for visibility
                ax_heatmap.scatter([detected_boundary, detected_boundary], [50, diff_x.shape[0]-50], 
                                  color='red', s=120, marker='o', zorder=5, edgecolor='white', linewidth=2)
                
                ax_heatmap.legend(loc='upper right', fontsize=12, framealpha=0.9)
            
            # PROFILE PLOT (Bottom): Line plot showing pixel difference profile vs x-position
            print(f"  📊 Creating pixel difference profile plot with exact coordinate projection...")
            
            # Calculate average pixel differences across Y-axis for each X position
            profile_data = np.mean(diff_x, axis=0)
            
            # Use the same coordinate system as the heatmap for perfect alignment
            num_points = len(profile_data)
            
            # Create x-coordinates that match heatmap exactly (0 to width-1)
            actual_x_positions = np.linspace(0, width - 1, num_points)
            
            # Create line plot using real pixel coordinates (not indices)
            ax_profile.plot(actual_x_positions, profile_data, 'b-', linewidth=2, label='Average Pixel Difference')
            ax_profile.set_xlabel('X Position (pixels)', fontsize=12)
            ax_profile.set_ylabel('Pixel Difference Value', fontsize=12)
            ax_profile.set_title('Horizontal Pixel Difference Profile', fontsize=14, fontweight='bold')
            ax_profile.grid(True, alpha=0.3)
            
            # Add zero line for reference
            ax_profile.axhline(y=0, color='black', linestyle='-', alpha=0.5, linewidth=1)
            
            # Mark positive (red) and negative (blue) regions with background colors
            ax_profile.axhspan(0, np.max(profile_data), alpha=0.1, color='red', label='Positive (Red) Regions')
            ax_profile.axhspan(np.min(profile_data), 0, alpha=0.1, color='blue', label='Negative (Blue) Regions')
            
            # Add boundary markers to line plot using real pixel coordinates
            if detected_boundary is not None:
                # Find the corresponding profile value for the detected boundary position
                # Map detected_boundary (pixel coordinate) to profile_data index
                boundary_index = int(detected_boundary * num_points / (width - 1))
                boundary_index = max(0, min(boundary_index, len(profile_data) - 1))  # Clamp to valid range
                boundary_value = profile_data[boundary_index]
                
                # Use actual pixel coordinate (not index) for plotting
                ax_profile.axvline(x=detected_boundary, color='red', linewidth=3, alpha=0.9,
                                  linestyle='-', label=f'Detected Boundary: {detected_boundary}px')
                
                # Add marker dot at the boundary position using real coordinates
                ax_profile.scatter([detected_boundary], [boundary_value], color='red', s=100, 
                                  marker='o', zorder=5, edgecolor='white', linewidth=2)
                
                print(f"  📍 Red boundary line marked at x={detected_boundary}px (profile value: {boundary_value:.1f})")
            
            ax_profile.legend(loc='upper left', fontsize=10, framealpha=0.9)
            
            # Force exact x-axis alignment by setting identical limits AND tick positions
            ax_heatmap.set_xlim(0, width - 1)
            ax_profile.set_xlim(0, width - 1)
            
            # Generate identical tick positions for both plots
            # Use major ticks every 200 pixels for clean alignment
            major_ticks = np.arange(0, width, 200)
            if major_ticks[-1] < width - 1:
                major_ticks = np.append(major_ticks, width - 1)
            
            # Apply identical tick positions to both plots
            ax_heatmap.set_xticks(major_ticks)
            ax_profile.set_xticks(major_ticks)
            
            # Set identical tick labels (optional: format as integers)
            tick_labels = [f'{int(tick)}' for tick in major_ticks]
            ax_heatmap.set_xticklabels(tick_labels)
            ax_profile.set_xticklabels(tick_labels)
            
            # Verify perfect alignment achieved
            print(f"  📊 Perfect x-axis alignment achieved:")
            print(f"      - Heatmap extent: {ax_heatmap.get_xlim()}")
            print(f"      - Profile extent: {ax_profile.get_xlim()}")
            print(f"      - Coordinate range: 0 to {width-1} pixels")
            print(f"      - Tick positions: {major_ticks.tolist()}")
            print(f"      - Alignment status: {'✅ PERFECT' if ax_heatmap.get_xlim() == ax_profile.get_xlim() else '⚠️ MISALIGNED'}")
            print(f"      - Physical width alignment: ✅ IDENTICAL (GridSpec ensures same column width)")
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{timestamp}_{filename_suffix}.png"
            
            # Save to screenshots directory
            screenshot_dir = "pic/screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            filepath = os.path.join(screenshot_dir, filename)
            
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()  # Clean up memory
            
            print(f"  🎨 Enhanced dual-plot visualization saved: {filename}")
            print(f"      - Top: Horizontal differences heatmap")
            print(f"      - Bottom: Pixel difference profile plot")
            return filepath
            
        except Exception as e:
            print(f"  ⚠️ Could not generate horizontal differences heatmap: {e}")
            return ""
    
    def detect_right_boundary(self, img: np.ndarray = None, img_width: int = None, 
                              preprocessed_image_path: str = None, coord_context: 'cWeChatCoordinateContext' = None) -> int:
        """
        Simplified right boundary detection using horizontal pixel difference visualization
        
        Based on the insight that horizontal pixel differences directly show boundaries as blue regions,
        this method:
        1. Creates horizontal pixel differences (like your visualization)
        2. Detects blue regions (strong negative transitions < -100)
        3. Finds rightmost boundary from visual pattern
        
        Args:
            img: Original image (optional, used for fallback preprocessing)
            img_width: Width of the original image (optional)
            preprocessed_image_path: Path to preprocessed level-adjusted image
            coord_context: Optional cWeChatCoordinateContext to populate with results
            
        Returns:
            int: boundary_position_px
        """
        print(f"  🎯 2. Simplified Visual Pattern Boundary Detection")
        
        # Load and prepare image with high-contrast preprocessing
        if preprocessed_image_path and os.path.exists(preprocessed_image_path):
            print(f"  📸 Loading preprocessed image: {os.path.basename(preprocessed_image_path)}")
            adjusted = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
        else:
            if img is None:
                raise ValueError("Either preprocessed_image_path or img must be provided")
            print(f"  🎨 Using original image with high-contrast preprocessing")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
            # Apply high-contrast preprocessing for better boundary detection (like reference image)
            adjusted = self._apply_level_adjustment(gray)
            
        img_width = img_width or adjusted.shape[1]
        
        # Create horizontal pixel differences (like your visualization)
        print(f"  📊 Creating horizontal pixel difference pattern")
        diff_x = np.diff(adjusted.astype(np.int16), axis=1)
        
        # Generate initial horizontal differences heatmap visualization (debug mode only)
        if self.debug_mode:
            self._generate_horizontal_differences_heatmap(diff_x, None, "03_horizontal_differences")
        
        # Detect red regions (strong positive transitions) - these are the boundaries we want
        red_threshold = 100  # Red regions in your visualization (high positive values)
        red_regions = diff_x > red_threshold
        
        print(f"  🔴 Detecting red regions (transitions > {red_threshold})")
        red_pixel_count = np.sum(red_regions)
        print(f"  📍 Found {red_pixel_count} red pixels representing strong transitions")
        
        # Find FIRST (leftmost) boundary from red regions
        # Count red pixels per column (vertical projection)
        red_column_intensity = np.sum(red_regions, axis=0)
        
        # Search in reasonable boundary range (adjusted to capture left boundaries around 400px)
        search_start = int(img_width * 0.25)  # Start earlier to capture boundaries around 400px
        search_end = int(img_width * 0.95)    # Skip right edge artifacts
        
        print(f"  🔍 Searching for FIRST red region from left in {search_start}-{search_end}px")
        
        # Find columns with significant red intensity (boundary regions)
        min_red_intensity = 2  # Minimum red pixels per column to be considered a boundary
        boundary_candidates = []
        
        for x in range(search_start, min(search_end, len(red_column_intensity))):
            if red_column_intensity[x] >= min_red_intensity:
                boundary_candidates.append((x, red_column_intensity[x]))
        
        print(f"  🎯 Found {len(boundary_candidates)} boundary candidates with red regions")
        
        # Enhanced red line detection reporting - show x-axis positions
        if boundary_candidates:
            print(f"  📍 Red Line X-Axis Position Analysis:")
            # Show top 10 strongest red regions with their x-axis positions
            top_candidates = sorted(boundary_candidates, key=lambda b: b[1], reverse=True)[:10]
            for i, (x, intensity) in enumerate(top_candidates, 1):
                print(f"      {i:2d}. x-axis position: {x:4d}px, red intensity: {intensity:3d}")
            
            # Show leftmost red regions specifically (these are our targets)
            leftmost_candidates = sorted(boundary_candidates, key=lambda b: b[0])[:5]
            print(f"  📍 LEFTMOST Red Line Positions (Target Boundaries):")
            for i, (x, intensity) in enumerate(leftmost_candidates, 1):
                print(f"      {i}. x={x:4d}px (red intensity: {intensity:3d})")
        
        # Select STRONGEST red boundary (highest intensity)
        if boundary_candidates:
            # Sort by intensity (strongest first)
            boundary_candidates.sort(key=lambda b: b[1], reverse=True)
            
            # Select the strongest boundary above minimum threshold
            for x, intensity in boundary_candidates:
                if x >= self.MIN_BOUNDARY_PX:
                    print(f"  ✅ RED LINE DETECTED ON X-AXIS (STRONGEST BOUNDARY):")
                    print(f"      🎯 X-axis position: {x}px")
                    print(f"      🔴 Red intensity: {intensity} pixels")
                    print(f"      📏 Distance from left edge: {x}px")
                    print(f"      📊 Strongest valid boundary: ✅ YES")
                    # Generate final heatmap with detected boundary marked (debug mode only)
                    if self.debug_mode:
                        self._generate_horizontal_differences_heatmap(diff_x, x, "04_horizontal_differences_with_boundary")
                    return x
            
            # Fallback to best available
            x, intensity = boundary_candidates[0]
            print(f"  ⚠️ RED LINE FALLBACK DETECTION:")
            print(f"      🎯 X-axis position: {x}px")
            print(f"      🔴 Red intensity: {intensity} pixels")
            print(f"      📏 Distance from left edge: {x}px")
            print(f"      📊 First valid boundary: ⚠️ BELOW MINIMUM")
            # Generate final heatmap with detected boundary marked (debug mode only)
            if self.debug_mode:
                self._generate_horizontal_differences_heatmap(diff_x, x, "04_horizontal_differences_with_boundary")
            return x
        
        # Final fallback
        fallback = int(img_width * 0.8)
        print(f"  ❌ NO RED LINES DETECTED - GEOMETRIC FALLBACK:")
        print(f"      🎯 X-axis position: {fallback}px")
        print(f"      📏 Distance from left edge: {fallback}px")
        print(f"      📊 Method: Geometric estimation (80% of width)")
        # Generate final heatmap with fallback boundary marked (debug mode only)
        if self.debug_mode:
            self._generate_horizontal_differences_heatmap(diff_x, fallback, "04_horizontal_differences_with_boundary")
        return fallback


class cBoundaryCoordinator:
    """Boundary coordination class that combines left and right boundary detection for complete conversation area analysis"""
    
    def __init__(self, debug_mode: bool = False):
        # Debug mode control for visualization generation
        self.debug_mode = debug_mode
        
        # Initialize both boundary detectors
        self.left_detector = cLeftBoundaryDetector(debug_mode=debug_mode)
        self.right_detector = cRightBoundaryDetector(debug_mode=debug_mode)
        
        # Dual-boundary coordination storage for visualization integration
        self._boundary_markers = {
            'left': {'position': None, 'confidence': None, 'method': None},
            'right': {'position': None, 'confidence': None, 'method': None}
        }
        
    def detect_width(self, image_path: str, preprocessed_image_path: str = None, 
                     coord_context: 'cWeChatCoordinateContext' = None, 
                     return_context: bool = False) -> Optional[Tuple[int, int, int]]:
        """
        Enhanced width detection with dual-boundary coordination and blue line visualization integration
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, min 800x600px)
        - preprocessed_image_path: Optional[str] - Pre-processed level-adjusted image path
        - coord_context: Optional[WeChatCoordinateContext] - Coordinate context to populate
        - return_context: bool - Whether to return coordinate context with results
        
        📌 OUTPUT CONTRACT:
        Standard Mode (return_context=False):
        - Success: Tuple[int, int, int] - (left_boundary, right_boundary, detected_width)
        - Failure: None
        
        Context Mode (return_context=True): 
        - Success: (Tuple[int, int, int], cWeChatCoordinateContext) 
        - Failure: (None, cWeChatCoordinateContext)
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True
        - Updates coordinate context with conversation_area boundaries
        - Populates boundary markers for downstream processing
        
        Implements synchronized left and right boundary detection with confidence scoring
        and visual marker data compatible with horizontal pixel difference analysis
        """
        print(f"🎯 1. Enhanced Dual-Boundary Width Detection: {os.path.basename(image_path)}")
        
        # Phase 1: Image Loading and Preparation
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        print(f"📐 Image dimensions: {img.shape[1]}×{img.shape[0]}")
        img_height, img_width = img.shape[:2]
        
        # Phase 2: Left Boundary Detection using LeftBoundaryDetector
        print(f"🔍 Phase 2: Left Boundary Detection")
        left_boundary = self.left_detector.detect_left_boundary(image_path)
        left_confidence = 0.8  # Default confidence, can be enhanced later
        
        # Store left boundary marker data
        self._boundary_markers['left'] = {
            'position': left_boundary,
            'confidence': left_confidence,
            'method': 'edge_based_sidebar'
        }
        
        # Phase 3: Right Boundary Detection (Enhanced Method)
        print(f"🔍 Phase 3: Right Boundary Detection")
        right_boundary = self.right_detector.detect_right_boundary(
            img=img,
            img_width=img_width,
            preprocessed_image_path=preprocessed_image_path
        )
        
        # Extract right boundary confidence and method from detector
        right_confidence = 0.5  # Default fallback
        right_method = 'unknown'
        if hasattr(self.right_detector, '_boundary_visualization_data'):
            viz_data = self.right_detector._boundary_visualization_data
            right_confidence = viz_data.get('confidence', 0.5)
            right_method = viz_data.get('method', 'unknown')
        
        # Store right boundary marker data
        self._boundary_markers['right'] = {
            'position': right_boundary,
            'confidence': right_confidence,
            'method': right_method
        }
        
        # Phase 4: Dual-Boundary Validation and Coordination
        if left_boundary is None or right_boundary is None:
            print("❌ Dual-boundary detection failed")
            return None
        
        width = right_boundary - left_boundary
        
        # Phase 5: Boundary Relationship Analysis
        self._analyze_boundary_relationships(left_boundary, right_boundary, width, img_width)
        
        # Phase 6: Enhanced Visual Output with Blue Line Integration (only in debug mode)
        if self.debug_mode:
            self._create_enhanced_visual_result(img, image_path)
        
        # Phase 7: Coordinate Context Integration
        results = (left_boundary, right_boundary, width)
        
        # Create or initialize coordinate context if needed
        if return_context and coord_context is None:
            img_height, img_width = img.shape[:2]
            coord_context = cWeChatCoordinateContext(image_path, (img_width, img_height))
        
        # Populate coordinate context if provided
        if coord_context is not None:
            conversation_bbox = [left_boundary, 0, width, img.shape[0]]
            coord_context.add_global_boundary(
                "conversation_area", 
                conversation_bbox, 
                "BoundaryCoordinator.detect_width", 
                min(self._boundary_markers['left']['confidence'], self._boundary_markers['right']['confidence'])
            )
        
        # Return based on requested format
        if return_context:
            return results, coord_context
        else:
            return results
    
    def _detect_left_boundary_with_confidence(self, conversation_area: np.ndarray) -> tuple:
        """Enhanced left boundary detection with confidence scoring"""
        # Use existing edge-based detection but add confidence calculation
        left_boundary = self._find_left_boundary_edge_based(conversation_area)
        
        if left_boundary is not None:
            # Calculate confidence based on edge strength and position consistency
            confidence = 0.8  # High confidence for sidebar edge detection
            print(f"  ✅ Left boundary: {left_boundary}px, confidence: {confidence:.3f}")
        else:
            confidence = 0.0
            print(f"  ❌ Left boundary detection failed")
        
        return left_boundary, confidence
    
    def _analyze_boundary_relationships(self, left: int, right: int, width: int, img_width: int):
        """Analyze relationships between detected boundaries for validation"""
        print(f"🔍 Phase 4: Boundary Relationship Analysis")
        
        # Calculate metrics
        left_ratio = left / img_width if img_width > 0 else 0
        right_ratio = right / img_width if img_width > 0 else 0
        width_ratio = width / img_width if img_width > 0 else 0
        
        # Relationship validation
        relationship_quality = "excellent"
        if width < 200:
            relationship_quality = "suspicious_narrow"
        elif width > img_width * 0.9:
            relationship_quality = "suspicious_wide"
        elif left < -50:
            relationship_quality = "left_overflow"
        elif right > img_width - 20:
            relationship_quality = "right_overflow"
        
        print(f"  📊 Boundary Metrics:")
        print(f"    Left position: {left}px ({left_ratio:.2%} of image)")
        print(f"    Right position: {right}px ({right_ratio:.2%} of image)")
        print(f"    Detected width: {width}px ({width_ratio:.2%} of image)")
        print(f"    Relationship quality: {relationship_quality}")
        
        # Store for visualization
        self._boundary_markers['relationship'] = {
            'width': width,
            'quality': relationship_quality,
            'ratios': {'left': left_ratio, 'right': right_ratio, 'width': width_ratio}
        }
    
    def _create_enhanced_visual_result(self, img: np.ndarray, image_path: str):
        """Create enhanced professional visual result with clean boundary markers"""
        left_data = self._boundary_markers['left']
        right_data = self._boundary_markers['right']
        
        if left_data['position'] is None or right_data['position'] is None:
            return
        
        # Create enhanced visualization with clean background
        result_img = img.copy()
        img_height, img_width = img.shape[:2]
        
        # Apply subtle overlay for better contrast
        overlay = result_img.copy()
        cv2.rectangle(overlay, (0, 0), (img_width, img_height), (245, 245, 245), -1)
        result_img = cv2.addWeighted(result_img, 0.7, overlay, 0.3, 0)
        
        left_pos = left_data['position']
        right_pos = right_data['position']
        width = right_pos - left_pos
        
        # Professional blue boundary lines with shadow effect
        line_color = (220, 120, 50)  # Professional blue
        shadow_color = (180, 100, 40)  # Darker shadow
        line_thickness = 4
        
        # Shadow lines (offset by 2px)
        cv2.line(result_img, (left_pos + 2, 2), (left_pos + 2, img_height), shadow_color, line_thickness)
        cv2.line(result_img, (right_pos + 2, 2), (right_pos + 2, img_height), shadow_color, line_thickness)
        
        # Main boundary lines
        cv2.line(result_img, (left_pos, 0), (left_pos, img_height), line_color, line_thickness)
        cv2.line(result_img, (right_pos, 0), (right_pos, img_height), line_color, line_thickness)
        
        # Professional info panel with rounded corners effect
        panel_width = max(350, int(width * 0.8))
        panel_height = 140
        panel_x = left_pos + (width - panel_width) // 2
        panel_y = 30
        
        # Panel shadow
        cv2.rectangle(result_img, (panel_x + 3, panel_y + 3), 
                     (panel_x + panel_width + 3, panel_y + panel_height + 3), 
                     (200, 200, 200), -1)
        
        # Main panel with gradient-like effect
        cv2.rectangle(result_img, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (250, 250, 250), -1)
        cv2.rectangle(result_img, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height), 
                     (100, 100, 100), 2)
        
        # Professional typography with better positioning
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_color = (50, 50, 50)  # Dark gray
        accent_color = (40, 80, 160)  # Professional blue
        
        # Title with larger font
        title_y = panel_y + 35
        cv2.putText(result_img, f"BOUNDARY DETECTION", 
                   (panel_x + 15, title_y), font, 0.9, text_color, 2)
        
        # Width measurement (prominent)
        width_y = title_y + 35
        cv2.putText(result_img, f"Detected Width: {width}px", 
                   (panel_x + 15, width_y), font, 0.8, accent_color, 2)
        
        # Boundary details
        details_y = width_y + 25
        cv2.putText(result_img, f"Left: {left_pos}px (conf: {left_data['confidence']:.2f})", 
                   (panel_x + 15, details_y), font, 0.6, text_color, 1)
        
        cv2.putText(result_img, f"Right: {right_pos}px (conf: {right_data['confidence']:.2f})", 
                   (panel_x + 15, details_y + 20), font, 0.6, text_color, 1)
        
        # Method info (smaller text)
        method_y = details_y + 45
        method_text = f"Methods: {left_data.get('method', 'auto')} + {right_data.get('method', 'auto')}"
        cv2.putText(result_img, method_text, 
                   (panel_x + 15, method_y), font, 0.5, (120, 120, 120), 1)
        
        # Professional circular markers at boundaries
        marker_size = 10
        marker_color = (220, 120, 50)  # Matching line color
        marker_border = (180, 100, 40)  # Darker border
        
        # Top markers with shadow effect
        cv2.circle(result_img, (left_pos + 1, 11), marker_size, marker_border, -1)
        cv2.circle(result_img, (right_pos + 1, 11), marker_size, marker_border, -1)
        cv2.circle(result_img, (left_pos, 10), marker_size, marker_color, -1)
        cv2.circle(result_img, (right_pos, 10), marker_size, marker_color, -1)
        
        # Bottom markers with shadow effect
        cv2.circle(result_img, (left_pos + 1, img_height - 9), marker_size, marker_border, -1)
        cv2.circle(result_img, (right_pos + 1, img_height - 9), marker_size, marker_border, -1)
        cv2.circle(result_img, (left_pos, img_height - 10), marker_size, marker_color, -1)
        cv2.circle(result_img, (right_pos, img_height - 10), marker_size, marker_color, -1)
        
        # Add subtle measurement arrows
        arrow_y = panel_y + panel_height + 20
        cv2.arrowedLine(result_img, (left_pos, arrow_y), (right_pos, arrow_y), accent_color, 2, tipLength=0.02)
        cv2.arrowedLine(result_img, (right_pos, arrow_y), (left_pos, arrow_y), accent_color, 2, tipLength=0.02)
        
        # Width label below arrows
        label_x = left_pos + width // 2 - 30
        cv2.putText(result_img, f"{width}px", (label_x, arrow_y + 25), font, 0.7, accent_color, 2)
        
        # Save enhanced result
        screenshot_dir = "pic/screenshots"
        os.makedirs(screenshot_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"{timestamp}_01_EnhancedDualBoundary_{width}px.png"
        output_path = os.path.join(screenshot_dir, output_filename)
        
        cv2.imwrite(output_path, result_img)
        print(f"  🎨 Enhanced dual-boundary visualization: {output_path}")
    
    def get_boundary_markers(self) -> dict:
        """Get boundary marker data for integration with blue line visualizations"""
        return self._boundary_markers
    
    def _save_visual_result(self, img: np.ndarray, left_boundary: int, right_boundary: int, width: int, image_path: str):
        """Save visual result showing detected boundaries to screenshots folder"""
        try:
            # Create output image with boundary overlays
            result_img = img.copy()
            img_height, img_width = img.shape[:2]
            
            # Draw left boundary line (green)
            cv2.line(result_img, (left_boundary, 0), (left_boundary, img_height), (0, 255, 0), 3)
            
            # Draw right boundary line (red)  
            cv2.line(result_img, (right_boundary, 0), (right_boundary, img_height), (0, 0, 255), 3)
            
            # Draw width measurement box at top
            cv2.rectangle(result_img, (left_boundary, 20), (right_boundary, 80), (255, 255, 255), -1)
            cv2.rectangle(result_img, (left_boundary, 20), (right_boundary, 80), (0, 0, 0), 2)
            
            # Add text showing measurements
            font = cv2.FONT_HERSHEY_SIMPLEX
            text = f"Width: {width}px"
            cv2.putText(result_img, text, (left_boundary + 10, 45), font, 0.8, (0, 0, 0), 2)
            
            left_text = f"L:{left_boundary}"
            cv2.putText(result_img, left_text, (left_boundary + 10, 65), font, 0.6, (0, 150, 0), 2)
            
            right_text = f"R:{right_boundary}"
            cv2.putText(result_img, right_text, (right_boundary - 80, 65), font, 0.6, (0, 0, 150), 2)
            
            # Create screenshot folder if it doesn't exist
            screenshot_dir = "pic/screenshots"
            os.makedirs(screenshot_dir, exist_ok=True)
            
            # Generate output filename with timestamp and measurements
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"{timestamp}_SimpleWidth_{width}px.png"
            output_path = os.path.join(screenshot_dir, output_filename)
            
            # Save the result
            cv2.imwrite(output_path, result_img)
            print(f"  📸 Visual result saved: {output_path}")
            
        except Exception as e:
            print(f"  ⚠️ Could not save visual result: {e}")
    
    def _find_left_boundary_edge_based(self, conversation_area: np.ndarray) -> Optional[int]:
        """Find the left boundary using sophisticated vertical edge detection"""
        print(f"  🕸️ Edge-based left boundary detection - looking for sidebar boundary")
        
        try:
            # Get the complete edge profile to analyze all peaks
            left_x, confidence, profile = find_vertical_edge_x(
                img=conversation_area,
                x0=0,              # Start from left edge
                x1=100,            # Only search first 100px (sidebar region)
                y0=0,              # Full height
                y1=None,           # Full height
                rightmost=False    # Find leftmost edge (sidebar boundary)
            )
            
            print(f"  📊 Strongest edge: x={left_x}px, confidence={confidence:.3f}")
            
            # Find ALL significant peaks in the profile, not just the strongest
            profile_peaks = []
            mean_profile = np.mean(profile)
            threshold = mean_profile * 1.2  # Lower threshold to catch weaker edges
            
            for i in range(1, len(profile) - 1):
                if (profile[i] > profile[i-1] and profile[i] > profile[i+1] and 
                    profile[i] > threshold):
                    # Calculate local confidence for this peak
                    local_region = profile[max(0, i-5):min(len(profile), i+6)]
                    local_med = np.median(local_region)
                    local_mad = np.median(np.abs(local_region - local_med)) + 1e-6
                    local_conf = (profile[i] - local_med) / (6 * local_mad)
                    
                    profile_peaks.append((i, profile[i], local_conf))
            
            # Sort peaks by position (left to right)
            profile_peaks.sort(key=lambda x: x[0])
            
            print(f"  📈 Found {len(profile_peaks)} significant edges:")
            for i, (pos, strength, conf) in enumerate(profile_peaks):
                print(f"    Edge {i+1}: x={pos}px, strength={strength:.0f}, conf={conf:.3f}")
            
            # Strategy: Use strongest edge and subtract 8px for actual sidebar boundary
            # Strongest edge is likely in message content, so subtract 8px to get true boundary
            if left_x is not None:
                # Apply 8px offset to get actual sidebar boundary
                actual_boundary = left_x - 8
                print(f"  ✅ Strongest edge at {left_x}px, applying -8px offset")
                print(f"  ✅ Actual sidebar boundary: {actual_boundary}px")
                return actual_boundary
            
            # Fallback: Use the original strongest edge if no reasonable alternatives
            if left_x >= 20:
                print(f"  ⚠️ Using strongest edge as fallback: {left_x}px")
                return left_x
            else:
                print(f"  ❌ No suitable edge found, using default")
                return 60  # Default fallback
        
        except Exception as e:
            print(f"  ❌ Edge detection failed: {e}")
            return 60  # Default fallback
    
    
    def get_latest_screenshot(self, screenshot_dir: str = "pic/screenshots") -> Optional[str]:
        """
        🔄 BACKWARD COMPATIBILITY WRAPPER
        
        Find the latest WeChat screenshot in the specified directory
        Uses tool class implementation for consistency (Human Structural Logic: Tool Class pattern)
        
        Args:
            screenshot_dir: Directory to search for screenshots
            
        Returns:
            Path to the latest screenshot file, or None if not found
        """
        # Use tool class if available (Human Structural Logic: Repeatedly-called functionality = Tool Class)
        if cWeChat_Screenshot_Finder is not None:
            try:
                finder = cWeChat_Screenshot_Finder()
                return finder.get_latest_screenshot(screenshot_dir)
            except Exception as e:
                print(f"⚠️ Tool class failed, using fallback: {e}")
        
        # Fallback implementation for environments where tool class not available
        if not os.path.exists(screenshot_dir):
            print(f"⚠️ Screenshot directory not found: {screenshot_dir}")
            return None
        
        # Look for files matching the WeChat screenshot pattern: YYYYMMDD_HHMMSS_WeChat.png
        screenshot_files = []
        for filename in os.listdir(screenshot_dir):
            if filename.endswith('_WeChat.png') and len(filename) >= 20:  # Minimum length check
                try:
                    # Verify the timestamp format
                    timestamp_part = filename.split('_WeChat.png')[0]
                    if len(timestamp_part) == 15 and timestamp_part.replace('_', '').isdigit():
                        screenshot_files.append(filename)
                except:
                    continue
        
        if not screenshot_files:
            print(f"⚠️ No WeChat screenshots found in {screenshot_dir}")
            return None
        
        # Sort by filename (which sorts by timestamp due to YYYYMMDD_HHMMSS format)
        latest_screenshot = sorted(screenshot_files)[-1]
        latest_path = os.path.join(screenshot_dir, latest_screenshot)
        
        print(f"📸 Using latest screenshot: {latest_screenshot}")
        return latest_path
    
    def create_width_visualization(self, image_path: str, output_path: str = None) -> Optional[str]:
        """
        Create simple visualization showing ONLY the detected width boundaries
        No avatars, no zones, just two vertical lines and width value
        """
        # Detect width
        detection_result = self.detect_width(image_path)
        if detection_result is None:
            return None
        
        left_boundary, right_boundary, width = detection_result
        
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        result = img.copy()
        img_height = result.shape[0]
        
        # Draw simple vertical lines for boundaries
        cv2.line(result, (left_boundary, 0), (left_boundary, img_height), (0, 255, 0), 2)  # Green left line
        cv2.line(result, (right_boundary, 0), (right_boundary, img_height), (0, 255, 0), 2)  # Green right line
        
        # Add width text at the top
        width_text = f"Width: {width}px"
        cv2.putText(result, width_text, (left_boundary, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
        
        # Add boundary labels
        cv2.putText(result, f"L:{left_boundary}", (left_boundary + 5, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(result, f"R:{right_boundary}", (right_boundary - 80, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_01_SimpleWidth_{width}px.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Simple width visualization saved: {output_path}")
        
        return os.path.basename(output_path)


# =============================================================================
# PHASE 3: CARD AVATAR DETECTOR
# =============================================================================

class cCardAvatarDetector:
    """
    Advanced avatar detector using gradient projection and geometric filtering
    Specifically optimized for WeChat message card layouts
    """
    
    def __init__(self, debug_mode: bool = False):
        # Initialize width detector for dynamic boundary detection
        self.width_detector = cBoundaryCoordinator(debug_mode=debug_mode)
        
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
        prof = cv2.GaussianBlur(prof.reshape(1, -1), (1, 7), 0).ravel()
        
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

    def detect_avatars(self, image_path: str, coord_context: 'cWeChatCoordinateContext' = None, 
                       return_context: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect avatars using gradient projection and geometric filtering.
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, conversation area visible)
        - coord_context: Optional[WeChatCoordinateContext] - Coordinate context for integration
        - return_context: bool - Whether to return coordinate context with results
        
        📌 OUTPUT CONTRACT:
        Standard Mode (return_context=False):
        - Success: Tuple[List[Dict], Dict] - (avatar_results, detection_info)
        - avatar_results: List of {"bbox": [x,y,w,h], "center": [cx,cy], "area": int, "avatar_id": int}
        - detection_info: {"total_avatars": int, "processing_time": float, "method": str}
        
        Context Mode (return_context=True):
        - Success: ((avatar_results, detection_info), cWeChatCoordinateContext)
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True
        - Updates coordinate context with avatar component data
        - Validates avatar positions against conversation area boundaries
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
        
        H, W = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        print(f"🎯 3. Advanced Avatar Detection: {os.path.basename(image_path)}")
        print(f"📐 Image size: {W}x{H}")
        
        # Use dynamic width detection to determine card boundaries
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
        
        # Add bounds checking to prevent empty panel
        if x0 >= W or x0 < 0 or wR <= 0 or x0 + wR > W:
            print(f"  ❌ Invalid avatar search bounds: x0={x0}, wR={wR}, image_width={W}")
            return [], {'total_contours': 0, 'candidates_after_filtering': 0, 'final_avatars': 0, 'avatar_search_area': None}
        
        panel = img[y0:y0+hR, x0:x0+wR]
        
        # Verify panel is not empty
        if panel.size == 0:
            print(f"  ❌ Empty panel extracted: bounds={x0}:{x0+wR}, {y0}:{y0+hR}")
            return [], {'total_contours': 0, 'candidates_after_filtering': 0, 'final_avatars': 0, 'avatar_search_area': None}
        
        gray_p = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)
        
        print(f"  📊 Avatar search area: x={x0}-{x0+wR}px ({boundary_source} left boundary, fixed avatar column width)")
        
        # Edge detection with morphology to reveal rounded-square thumbnails
        blur = cv2.bilateralFilter(gray_p, self.BILATERAL_D, 
                                  self.BILATERAL_SIGMA_COLOR, 
                                  self.BILATERAL_SIGMA_SPACE)
        
        edges = cv2.Canny(blur, self.CANNY_LOW, self.CANNY_HIGH)
        edges = cv2.dilate(edges, cv2.getStructuringElement(cv2.MORPH_RECT, self.DILATE_KERNEL_SIZE), 
                         iterations=self.DILATE_ITERATIONS)
        
        # Find contours and apply geometric filters
        cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  🔍 Found {len(cnts)} contours")
        
        candidates = []
        for c in cnts:
            x, y, w, h = cv2.boundingRect(c)
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
            hull = cv2.convexHull(c)
            hull_area = cv2.contourArea(hull)
            if hull_area > 0:
                solidity = cv2.contourArea(c) / hull_area
                if solidity < self.MIN_SOLIDITY:
                    continue
            
            candidates.append([x, y, w, h])
        
        print(f"  ✅ Filtered to {len(candidates)} avatar candidates")
        
        # Apply non-maximum suppression
        boxes = self.nms_boxes(candidates, iou_thresh=self.NMS_IOU_THRESHOLD)
        print(f"  🎯 After NMS: {len(boxes)} final avatars")
        
        # Sort by y-coordinate (top to bottom) and convert to results format
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
        
        # Coordinate Context Integration
        results_tuple = (results, detection_info)
        
        # Create or initialize coordinate context if needed
        if return_context and coord_context is None:
            H, W = img.shape[:2]
            coord_context = cWeChatCoordinateContext(image_path, (W, H))
        
        # Populate coordinate context with avatar data if provided
        if coord_context is not None and results:
            for i, avatar in enumerate(results, 1):
                # Note: Avatars are not assigned to specific cards at this stage
                # They will be associated with cards in the CardBoundaryDetector step
                avatar_bbox = avatar["bbox"]
                avatar_center = avatar["center"]
                
                # Add as a temporary global avatar marker
                coord_context.add_global_boundary(
                    f"avatar_{i}",
                    avatar_bbox,
                    "CardAvatarDetector.detect_avatars",
                    avatar.get("confidence", 0.95)
                )
        
        # Return based on requested format
        if return_context:
            return results_tuple, coord_context
        else:
            return results_tuple

    def create_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive visualization showing the detection process.
        Includes panel boundary, contours, filtered candidates, and final results.
        """
        # Load original image
        img = cv2.imread(image_path)
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
        cv2.rectangle(result, (x0, y0), (x0 + wR, y0 + hR), (255, 0, 0), 2)
        cv2.putText(result, f"Avatar Area: {x0}-{x0+wR}px", (x0 + 5, 30), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Draw avatar detections
        for i, avatar in enumerate(avatars):
            x, y, w, h = avatar["bbox"]
            cx, cy = avatar["center"]
            
            # Draw bounding box (green)
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw center point (red circle)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
            
            # Draw avatar ID
            cv2.putText(result, f"A{avatar['avatar_id']}", (x, y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Draw center coordinates
            cv2.putText(result, f"({cx},{cy})", (cx + 5, cy - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 255), 1)
        
        # Add legend and stats
        legend_y = 60
        cv2.putText(result, "3. Advanced Avatar Detection Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Avatar Search Area  Green=Avatar Box  Red=Center", 
                  (10, legend_y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Detection statistics
        stats_y = legend_y + 50
        cv2.putText(result, f"Detected: {len(avatars)} avatars", 
                  (10, stats_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Search area: {x0}-{x0+wR}px (width={wR}px)", 
                  (10, stats_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Contours: {info.get('total_contours', 0)} -> "
                          f"Candidates: {info.get('candidates_after_filtering', 0)} -> "
                          f"Final: {info.get('final_avatars', 0)}", 
                  (10, stats_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_03_advanced_avatar_detection.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Advanced avatar detection visualization saved: {output_path}")
        
        return os.path.basename(output_path)

    def get_avatar_positions(self, image_path: str) -> List[Tuple[int, int]]:
        """
        Get avatar center positions for integration with boundary detection.
        Returns list of (x, y) center coordinates.
        """
        avatars, _ = self.detect_avatars(image_path)
        return [tuple(avatar["center"]) for avatar in avatars]


# =============================================================================
# PHASE 4: CARD BOUNDARY DETECTOR - Enhanced_Card_Avatar_Boundaries 
# =============================================================================

class cCardBoundaryDetector:
    """
    Card Boundary Detector using avatar-centric approach
    Uses avatar positions to determine message card boundaries
    """
    
    def __init__(self, debug_mode: bool = False):
        self.avatar_detector = cCardAvatarDetector(debug_mode=debug_mode)
        self.width_detector = cBoundaryCoordinator(debug_mode=debug_mode)
        
        # Card boundary parameters
        self.MIN_CARD_HEIGHT = 60       # Minimum height for a card
        self.MAX_CARD_HEIGHT = 300      # Maximum height for a card
        self.CARD_PADDING = 10          # Padding around card boundaries
        
        # Coordinate validation parameters
        self.VALIDATION_TOLERANCE = 5   # Pixels tolerance for boundary validation
        self.ENABLE_VALIDATION = True   # Enable/disable coordinate validation
        
    def detect_cards(self, image_path: str, coord_context: 'cWeChatCoordinateContext' = None, 
                     return_context: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect individual message card boundaries using avatar positions
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, with visible message cards)
        - coord_context: Optional[WeChatCoordinateContext] - Coordinate context for integration
        - return_context: bool - Whether to return coordinate context with results
        
        📌 OUTPUT CONTRACT:
        Standard Mode (return_context=False):
        - Success: Tuple[List[Dict], Dict] - (card_results, detection_info)
        - card_results: List of {"id": int, "region": {"bbox": [x,y,w,h], "confidence": float}}
        - detection_info: {"total_cards": int, "processing_time": float, "method": str}
        
        Context Mode (return_context=True):
        - Success: ((card_results, detection_info), cWeChatCoordinateContext)
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True
        - Updates coordinate context with card region data
        - Dependencies: Requires BoundaryCoordinator and CardAvatarDetector results
        """
        print(f"🎯 4. Card Boundary Detection: {os.path.basename(image_path)}")
        
        # Get width boundaries
        width_result = self.width_detector.detect_width(image_path)
        if width_result is None:
            print("❌ Width detection failed")
            return [], {}
        
        left_boundary, right_boundary, width = width_result
        print(f"  📏 Card width: {width}px (left: {left_boundary}, right: {right_boundary})")
        
        # Get complete avatar data (not just positions)
        avatars, avatar_info = self.avatar_detector.detect_avatars(image_path)
        if not avatars:
            print("❌ No avatars detected")
            return [], {}
        
        print(f"  👥 Found {len(avatars)} avatars with complete boundary data")
        
        # Calculate card boundaries using midpoint approach with complete avatar data
        cards = []
        for i, avatar in enumerate(avatars):
            avatar_x, avatar_y = avatar["center"]  # Get center from complete avatar data
            
            # Calculate vertical boundaries (midpoint between adjacent avatars)
            if i == 0:
                # First card: start from top
                top = max(0, avatar_y - 40)
            else:
                # Midpoint between current and previous avatar
                prev_avatar_y = avatars[i-1]["center"][1]
                top = (prev_avatar_y + avatar_y) // 2
            
            if i == len(avatars) - 1:
                # Last card: extend to reasonable bottom
                bottom = avatar_y + 40
            else:
                # Midpoint between current and next avatar
                next_avatar_y = avatars[i+1]["center"][1]
                bottom = (avatar_y + next_avatar_y) // 2
            
            # Create card boundary with complete avatar data
            avatar_bbox = avatar["bbox"]  # [x, y, w, h]
            card = {
                "card_id": i + 1,
                "bbox": [left_boundary, top, width, bottom - top],
                "avatar": {
                    "bbox": avatar_bbox,
                    "center": avatar["center"],
                    "avatar_id": avatar["avatar_id"],
                    "area": avatar["area"],
                    "aspect_ratio": avatar["aspect_ratio"],
                    "position_in_panel": avatar.get("position_in_panel", 0.0)
                },
                "boundaries": {
                    "card": {
                        "left": left_boundary,
                        "right": right_boundary, 
                        "top": top,
                        "bottom": bottom
                    },
                    "avatar": {
                        "left": avatar_bbox[0],
                        "right": avatar_bbox[0] + avatar_bbox[2],
                        "top": avatar_bbox[1], 
                        "bottom": avatar_bbox[1] + avatar_bbox[3]
                    }
                }
            }
            cards.append(card)
            
            # Enhanced card information display
            avatar_dims = f"{avatar_bbox[2]}×{avatar_bbox[3]}px"
            print(f"    📄 Card {i+1}: {width}×{bottom-top}px at ({left_boundary}, {top}) | Avatar: {avatar_dims} at ({avatar_bbox[0]}, {avatar_bbox[1]})")
        
        detection_info = {
            "total_cards": len(cards),
            "card_width": width,
            "width_boundaries": (left_boundary, right_boundary),
            "avatar_count": len(avatars),
            "avatar_detection_info": avatar_info,
            "enhanced_data": True  # Flag indicating this includes complete avatar boundaries
        }
        
        # Validate coordinates if enabled
        if self.ENABLE_VALIDATION:
            validated_cards, validation_report = self._validate_card_avatar_coordinates(cards)
            detection_info["coordinate_validation"] = validation_report
            if validation_report["validation_passed"]:
                print(f"  ✅ Coordinate validation passed: All {len(cards)} cards have avatars within boundaries")
            else:
                print(f"  ⚠️  Coordinate validation warnings: {validation_report['warnings_count']} issues found")
                for warning in validation_report["warnings"][:3]:  # Show first 3 warnings
                    print(f"    ⚠️  {warning}")
                if len(validation_report["warnings"]) > 3:
                    print(f"    ... and {len(validation_report['warnings']) - 3} more warnings")
            cards = validated_cards

        # Coordinate Context Integration
        results_tuple = (cards, detection_info)
        
        # Create or initialize coordinate context if needed
        if return_context and coord_context is None:
            img = cv2.imread(image_path)
            if img is not None:
                H, W = img.shape[:2]
                coord_context = cWeChatCoordinateContext(image_path, (W, H))
        
        # Populate coordinate context with card data if provided
        if coord_context is not None and cards:
            for card in cards:
                card_id = card["card_id"]
                card_bbox = card["bbox"]
                
                # Add card to coordinate context
                coord_context.add_card(card_id, card_bbox, "CardBoundaryDetector.detect_cards", 0.95)
                
                # Add avatar component if available
                if "avatar_data" in card and card["avatar_data"]:
                    avatar_bbox = card["avatar_data"]["bbox"]
                    avatar_center = card["avatar_data"]["center"]
                    
                    coord_context.add_component(
                        card_id, "avatar", avatar_bbox, "CardBoundaryDetector.detect_cards", 0.98,
                        ocr_suitable=False, center=avatar_center, 
                        expected_content="avatar_image"
                    )
        
        # Return based on requested format
        if return_context:
            return results_tuple, coord_context
        else:
            return results_tuple

    def _validate_card_avatar_coordinates(self, cards: List[Dict]) -> Tuple[List[Dict], Dict]:
        """
        Validate that avatars are positioned within their corresponding card boundaries
        
        Args:
            cards: List of card dictionaries with avatar data
            
        Returns:
            Tuple of (validated_cards, validation_report)
        """
        validated_cards = []
        warnings = []
        corrections_made = 0
        
        for i, card in enumerate(cards):
            card_bbox = card["bbox"]  # [x, y, w, h]
            avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
            
            # Card boundaries
            card_left = card_bbox[0]
            card_right = card_bbox[0] + card_bbox[2]
            card_top = card_bbox[1] 
            card_bottom = card_bbox[1] + card_bbox[3]
            
            # Avatar boundaries
            avatar_left = avatar_bbox[0]
            avatar_right = avatar_bbox[0] + avatar_bbox[2]
            avatar_top = avatar_bbox[1]
            avatar_bottom = avatar_bbox[1] + avatar_bbox[3]
            
            # Validation checks with tolerance
            violations = []
            corrected_avatar = list(avatar_bbox)  # Copy for potential corrections
            
            # Check left boundary
            if avatar_left < card_left - self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar left ({avatar_left}) extends beyond card left ({card_left})")
                corrected_avatar[0] = card_left  # Correct x position
                
            # Check right boundary
            if avatar_right > card_right + self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar right ({avatar_right}) extends beyond card right ({card_right})")
                # Keep avatar width, adjust position if needed
                if corrected_avatar[0] + corrected_avatar[2] > card_right:
                    corrected_avatar[0] = card_right - corrected_avatar[2]
                    
            # Check top boundary
            if avatar_top < card_top - self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar top ({avatar_top}) extends beyond card top ({card_top})")
                corrected_avatar[1] = card_top  # Correct y position
                
            # Check bottom boundary  
            if avatar_bottom > card_bottom + self.VALIDATION_TOLERANCE:
                violations.append(f"Avatar bottom ({avatar_bottom}) extends beyond card bottom ({card_bottom})")
                # Keep avatar height, adjust position if needed
                if corrected_avatar[1] + corrected_avatar[3] > card_bottom:
                    corrected_avatar[1] = card_bottom - corrected_avatar[3]
            
            # Create validated card
            validated_card = card.copy()
            
            if violations:
                # Apply corrections and warn
                warning_msg = f"Card {card['card_id']}: {'; '.join(violations)}"
                warnings.append(warning_msg)
                
                # Update avatar bbox and recalculate center
                validated_card["avatar"]["bbox"] = corrected_avatar
                new_center = [
                    corrected_avatar[0] + corrected_avatar[2] // 2,
                    corrected_avatar[1] + corrected_avatar[3] // 2
                ]
                validated_card["avatar"]["center"] = new_center
                
                # Update boundaries dict
                validated_card["boundaries"]["avatar"] = {
                    "left": corrected_avatar[0],
                    "right": corrected_avatar[0] + corrected_avatar[2],
                    "top": corrected_avatar[1], 
                    "bottom": corrected_avatar[1] + corrected_avatar[3]
                }
                
                corrections_made += 1
            
            validated_cards.append(validated_card)
        
        # Generate validation report
        validation_report = {
            "validation_passed": len(warnings) == 0,
            "total_cards_checked": len(cards),
            "violations_found": len(warnings),
            "corrections_made": corrections_made,
            "warnings_count": len(warnings),
            "warnings": warnings,
            "tolerance_used": self.VALIDATION_TOLERANCE
        }
        
        return validated_cards, validation_report

    def create_card_visualization(self, image_path: str, output_path: str = None) -> str:
        """Create visualization showing detected card boundaries"""
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating card boundary visualization...")
        
        # Get detection results
        cards, detection_info = self.detect_cards(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Draw enhanced card boundaries with avatar boundaries
        for card in cards:
            # Card boundary data
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center across card width
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)  # Yellow line
            
            # Draw card ID and dimensions
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            cv2.putText(result, f"Card: {card_w}×{card_h}px", (card_x + 10, card_y + card_h - 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)
            
            # Draw avatar ID and dimensions
            cv2.putText(result, f"A{avatar_data['avatar_id']}", (avatar_x, avatar_y - 5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            cv2.putText(result, f"Avatar: {avatar_w}×{avatar_h}px", (card_x + 10, card_y + card_h - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        # Add enhanced legend and stats with validation info
        legend_y = 30
        cv2.putText(result, "4. Enhanced Card & Avatar Boundary Detection:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add validation status
        validation_info = detection_info.get("coordinate_validation", {})
        if validation_info:
            validation_status = "✅ VALIDATED" if validation_info.get("validation_passed", False) else "⚠️ CORRECTED"
            cv2.putText(result, f"Total Cards: {len(cards)} | {validation_status} | Complete Dataset",
                       (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            if validation_info.get("corrections_made", 0) > 0:
                cv2.putText(result, f"Corrections Applied: {validation_info['corrections_made']} cards adjusted",
                           (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
        else:
            cv2.putText(result, f"Total Cards: {len(cards)} | Complete Coordinate Dataset",
                       (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_04_Enhanced_Card_Avatar_Boundaries_{len(cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Enhanced card & avatar boundary visualization saved: {output_path}")
        
        return os.path.basename(output_path)


# =============================================================================
# PHASE 5: CONTACT NAME BOUNDARY DETECTOR  
# =============================================================================

class cContactNameBoundaryDetector:
    """
    Contact Name Boundary Detector for visual detection of name regions
    Detects white text boundaries above avatar center lines without OCR
    """
    
    def __init__(self, debug_mode: bool = False):
        self.card_boundary_detector = cCardBoundaryDetector(debug_mode=debug_mode)
        
        # White text detection parameters (optimized for light gray text)
        self.WHITE_THRESHOLD_MIN = 155      # Minimum brightness for white/light gray text
        self.WHITE_THRESHOLD_MAX = 255      # Maximum brightness
        
        # Name size constraints (optimized for left-side contact names)
        self.MIN_NAME_WIDTH = 20            # Minimum name width in pixels (increased for contact names)
        self.MAX_NAME_WIDTH = 180           # Maximum name width in pixels (reduced, names are shorter on left)
        self.MIN_NAME_HEIGHT = 10           # Minimum name height in pixels (increased for better text)
        self.MAX_NAME_HEIGHT = 30           # Maximum name height in pixels (increased for contact names)
        
        # Morphological operation parameters (optimized for contact names)
        self.MORPH_KERNEL_SIZE = (3, 2)     # Kernel for connecting text pixels (wider for names)
        self.MORPH_ITERATIONS = 2           # Number of morphological iterations (increased for better connectivity)
        
        # Search region parameters (optimized for right-side search)
        self.SEARCH_MARGIN_LEFT = 5         # Margin from avatar right edge
        self.SEARCH_MARGIN_RIGHT = 10       # Margin from card right edge  
        self.SEARCH_MARGIN_TOP = 12         # Margin from card top (optimized)
        
        # Detection confidence parameters (optimized for contact names)
        self.MIN_WHITE_PIXEL_RATIO = 0.12   # Minimum ratio of white pixels in region (balanced for names)
        
    def detect_name_boundaries(self, image_path: str, cards_with_times: List[Dict] = None, 
                             debug_mode: bool = False, coord_context: 'cWeChatCoordinateContext' = None,
                             return_context: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect contact name boundaries using visual detection only, avoiding time conflict areas
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, with visible contact names)
        - cards_with_times: Optional[List[Dict]] - Pre-detected cards with time box data
        - debug_mode: bool - Enable comprehensive debug visualization collection
        - coord_context: Optional[WeChatCoordinateContext] - Coordinate context for integration
        - return_context: bool - Whether to return coordinate context with results
        
        📌 OUTPUT CONTRACT:
        Standard Mode (return_context=False):
        - Success: Tuple[List[Dict], Dict] - (enhanced_cards_with_names, detection_info)
        - enhanced_cards_with_names: List of cards with added contact_name components
        - detection_info: {"total_names": int, "success_rate": float, "method": "white_text_visual"}
        
        Context Mode (return_context=True):
        - Success: ((enhanced_cards_with_names, detection_info), cWeChatCoordinateContext)
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True or self.debug_mode=True
        - Updates coordinate context with contact_name component data
        - Dependencies: Requires card boundary and avatar detection results
        """
        print(f"🎯 5. Contact Name Boundary Detection: {os.path.basename(image_path)}")
        
        # Get card data (use cards_with_times if provided, otherwise detect fresh)
        if cards_with_times:
            cards = cards_with_times
            card_detection_info = {"reused_cards_with_times": True}
            print(f"  📄 Using {len(cards)} cards with time box data")
        else:
            cards, card_detection_info = self.card_boundary_detector.detect_cards(image_path)
            if not cards:
                print("❌ No cards available for name detection")
                return [], {}
            
        # Load image for processing
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
            
        print(f"  📄 Processing {len(cards)} cards for name boundary detection")
        
        # Initialize comprehensive debug data collection
        debug_data = {
            "original_image": img.copy() if debug_mode else None,
            "image_path": image_path,
            "processing_steps": [],
            "roi_images": {},
            "binary_masks": {},
            "contour_data": {},
            "search_regions": {},
            "white_text_regions": {},
            "filtered_boundaries": {},
            "confidence_scores": {},
            "processing_time": {},
            "success_cards": [],
            "failed_cards": [],
            "detection_results": {},
            "filtering_results": {},
            "confidence_breakdown": {},
            "morphology_data": {},
            "statistical_analysis": {},
            "algorithm_parameters": {
                "WHITE_THRESHOLD_MIN": self.WHITE_THRESHOLD_MIN,
                "WHITE_THRESHOLD_MAX": self.WHITE_THRESHOLD_MAX,
                "MIN_NAME_WIDTH": self.MIN_NAME_WIDTH,
                "MAX_NAME_WIDTH": self.MAX_NAME_WIDTH,
                "MIN_NAME_HEIGHT": self.MIN_NAME_HEIGHT,
                "MAX_NAME_HEIGHT": self.MAX_NAME_HEIGHT,
                "MORPH_KERNEL_SIZE": self.MORPH_KERNEL_SIZE,
                "MORPH_ITERATIONS": self.MORPH_ITERATIONS,
                "MIN_WHITE_PIXEL_RATIO": self.MIN_WHITE_PIXEL_RATIO,
                "SEARCH_MARGIN_LEFT": self.SEARCH_MARGIN_LEFT,
                "SEARCH_MARGIN_RIGHT": self.SEARCH_MARGIN_RIGHT,
                "SEARCH_MARGIN_TOP": self.SEARCH_MARGIN_TOP
            }
        } if debug_mode else {}
        
        # Process each card for name boundaries
        enhanced_cards = []
        total_names_detected = 0
        
        for card_idx, card in enumerate(cards):
            enhanced_card = self._detect_name_boundary_for_card(img, card, debug_mode, debug_data, card_idx)
            enhanced_cards.append(enhanced_card)
            
            card_id = enhanced_card.get("card_id", card_idx)
            
            if enhanced_card.get("name_boundary"):
                total_names_detected += 1
                name_bbox = enhanced_card["name_boundary"]["bbox"]
                confidence = enhanced_card["name_boundary"]["confidence"]
                print(f"    📝 Card {card_id}: Name boundary {name_bbox[2]}×{name_bbox[3]}px at ({name_bbox[0]}, {name_bbox[1]}) | confidence={confidence:.3f}")
                
                # Track successful detection
                if debug_mode:
                    debug_data["success_cards"].append(card_id)
                    debug_data["detection_results"][card_id] = {
                        "status": "success",
                        "bbox": name_bbox,
                        "confidence": confidence,
                        "detection_method": enhanced_card["name_boundary"]["detection_method"],
                        "search_region": enhanced_card["name_boundary"]["search_region"]
                    }
            else:
                # Track failed detection
                if debug_mode:
                    debug_data["failed_cards"].append(card_id)
                    debug_data["detection_results"][card_id] = {
                        "status": "failed",
                        "reason": "No valid name boundary found",
                        "bbox": None,
                        "confidence": 0.0
                    }
        
        # Generate statistical analysis for debug data
        if debug_mode and debug_data:
            debug_data["statistical_analysis"] = self._generate_statistical_analysis(debug_data, enhanced_cards)
        
        # Generate detection summary with debug data
        detection_info = {
            "total_cards_processed": len(cards),
            "names_detected": total_names_detected,
            "detection_success_rate": total_names_detected / len(cards) if cards else 0,
            "card_detection_info": card_detection_info,
            "detection_method": "visual_boundary_only",
            "debug_data": debug_data if debug_mode else None
        }
        
        print(f"  ✅ Name boundary detection complete: {total_names_detected}/{len(cards)} cards have detected names")
        
        # Coordinate Context Integration
        results_tuple = (enhanced_cards, detection_info)
        
        # Create or initialize coordinate context if needed
        if return_context and coord_context is None:
            img = cv2.imread(image_path)
            if img is not None:
                H, W = img.shape[:2]
                coord_context = cWeChatCoordinateContext(image_path, (W, H))
        
        # Populate coordinate context with contact name data if provided
        if coord_context is not None and enhanced_cards:
            for card in enhanced_cards:
                if "name_boundary" in card and card["name_boundary"]:
                    card_id = card["card_id"]
                    name_data = card["name_boundary"]
                    name_bbox = name_data["bbox"]
                    
                    coord_context.add_component(
                        card_id, "contact_name", name_bbox, 
                        "ContactNameBoundaryDetector.detect_name_boundaries",
                        name_data.get("confidence", 0.8),
                        ocr_suitable=True, expected_content="contact_name"
                    )
        
        # Return based on requested format
        if return_context:
            return results_tuple, coord_context
        else:
            return results_tuple

    def _detect_name_boundary_for_card(self, img: np.ndarray, card: Dict, debug_mode: bool = False, 
                                      debug_data: Dict = None, card_idx: int = 0) -> Dict:
        """
        Detect name boundary for a single card using the detected name-time boundary
        
        Args:
            img: Original image as numpy array
            card: Card dictionary with boundary, time box, and avatar data
            debug_mode: Enable debug data collection
            debug_data: Debug data dictionary to populate
            card_idx: Card index for debug tracking
            
        Returns:
            Enhanced card dictionary with name boundary information
        """
        import time
        enhanced_card = card.copy()
        card_id = card.get("card_id", card_idx)
        
        if debug_mode and debug_data is not None:
            start_time = time.time()
            debug_data["processing_steps"].append(f"Processing card {card_id}")
        
        try:
            # Detect the left edge of grey timestamp text as boundary
            boundary_x = self._detect_grey_timestamp_left_edge(img, card["bbox"], card["avatar"])
            
            if boundary_x is not None:
                # Store boundary information in the card
                enhanced_card["name_time_boundary"] = {
                    "x": boundary_x,
                    "detection_method": "grey_timestamp_left_edge", 
                    "avatar_center_y": card["avatar"]["center"][1]
                }
                # Use the detected boundary to create search region LEFT of timestamp (where names are)
                search_region = self._create_search_region_left_of_timestamp(card, boundary_x)
                region_method = "timestamp_left_boundary"
                
                if debug_mode:
                    debug_data["processing_steps"].append(f"Card {card_id}: Timestamp left edge at x={boundary_x}")
            else:
                # Fallback to avatar right edge as boundary
                avatar_right = card["avatar"]["bbox"][0] + card["avatar"]["bbox"][2]
                enhanced_card["name_time_boundary"] = {
                    "x": avatar_right + 20,  # Small margin from avatar
                    "detection_method": "avatar_right_fallback",
                    "avatar_center_y": card["avatar"]["center"][1]
                }
                # Use avatar right edge as boundary for search region
                search_region = self._create_search_region_left_of_timestamp(card, avatar_right + 20)
                region_method = "avatar_right_fallback"
                
                if debug_mode:
                    debug_data["processing_steps"].append(f"Card {card_id}: No timestamp detected, using avatar right edge x={avatar_right + 20}")
            
            if search_region is None:
                if debug_mode:
                    debug_data["processing_steps"].append(f"Card {card_id}: No valid search region")
                return enhanced_card
            
            # Debug: Store search region
            if debug_mode:
                debug_data["search_regions"][card_id] = {
                    "bbox": search_region,
                    "method": region_method,
                    "card_bbox": card["bbox"],
                    "avatar_bbox": card["avatar"]["bbox"]
                }
                
                # Extract and store ROI image
                x, y, w, h = search_region
                roi_img = img[y:y+h, x:x+w].copy()
                debug_data["roi_images"][card_id] = roi_img
            
            # Detect white text regions in the search area
            white_text_regions = self._detect_white_text_regions(img, search_region, debug_mode, debug_data, card_id)
            
            if not white_text_regions:
                if debug_mode:
                    debug_data["processing_steps"].append(f"Card {card_id}: No white text regions found")
                return enhanced_card
            
            # Filter and extract name-sized boundaries
            name_boundary = self._filter_and_extract_boundaries(white_text_regions, search_region, debug_mode, debug_data, card_id)
            
            if name_boundary:
                confidence = self._calculate_boundary_confidence(img, name_boundary)
                enhanced_card["name_boundary"] = {
                    "bbox": name_boundary,  # [x, y, w, h]
                    "search_region": search_region,
                    "detection_method": "white_text_visual",
                    "confidence": confidence
                }
                
                        # Debug: Store detailed confidence breakdown
                if debug_mode:
                    debug_data["confidence_breakdown"][card_id] = self._calculate_detailed_confidence_breakdown(img, name_boundary)
                
                # Debug: Store final boundary and confidence
                if debug_mode:
                    debug_data["filtered_boundaries"][card_id] = name_boundary
                    debug_data["confidence_scores"][card_id] = confidence
        
        except Exception as e:
            print(f"    ⚠️ Card {card_id}: Name detection failed - {e}")
            if debug_mode:
                debug_data["processing_steps"].append(f"Card {card_id}: ERROR - {str(e)}")
        
        if debug_mode and debug_data is not None:
            end_time = time.time()
            debug_data["processing_time"][card_id] = end_time - start_time
        
        return enhanced_card
    
    def _create_search_region_with_boundary(self, card: Dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Create search region for name detection using the detected name-time boundary
        
        Args:
            card: Card dictionary with name_time_boundary data
            
        Returns:
            Search region as (x, y, w, h) or None if invalid
        """
        card_bbox = card["bbox"]  # [x, y, w, h]
        avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
        boundary_y = card["name_time_boundary"]["y"]
        
        card_x, card_y, card_w, card_h = card_bbox
        avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
        
        # Start search region below the detected boundary
        search_top = boundary_y + 2  # Small gap below boundary
        search_bottom = card_y + card_h - self.SEARCH_MARGIN_TOP  # End before card bottom
        
        # CORRECTED: Search RIGHT of avatar where names actually appear
        search_left = avatar_x + avatar_w + self.SEARCH_MARGIN_LEFT  # Start right of avatar
        search_right = card_x + card_w - self.SEARCH_MARGIN_RIGHT    # End before card edge
        
        # Validate search region dimensions
        if search_right <= search_left or search_bottom <= search_top:
            return None
            
        search_width = search_right - search_left
        search_height = search_bottom - search_top
        
        # Ensure minimum search area
        if search_width < self.MIN_NAME_WIDTH or search_height < self.MIN_NAME_HEIGHT:
            return None
        
        return (search_left, search_top, search_width, search_height)
    
    def _create_adaptive_search_region(self, card: Dict) -> Optional[Tuple[int, int, int, int]]:
        """
        Create adaptive search region based on card width, avatar position, and detected time boxes
        
        Args:
            card: Card dictionary with boundary data (may include time_box)
            
        Returns:
            Search region as (x, y, w, h) or None if invalid
        """
        card_bbox = card["bbox"]  # [x, y, w, h]
        avatar_center = card["avatar"]["center"]  # [x, y]
        avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
        
        card_x, card_y, card_w, card_h = card_bbox
        avatar_center_x, avatar_center_y = avatar_center
        avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
        
        # Search region above avatar where names are actually located
        search_top = card_y + self.SEARCH_MARGIN_TOP  # Start from card top with margin
        search_bottom = avatar_y - 5  # End above avatar with small gap
        
        # CORRECTED: Search RIGHT of avatar where names actually appear (original was right!)
        search_left = avatar_x + avatar_w + self.SEARCH_MARGIN_LEFT  # Start right of avatar
        search_right = card_x + card_w - self.SEARCH_MARGIN_RIGHT    # End before card edge
        
        # Adjust for detected time boxes to avoid conflicts
        if card.get("time_box"):
            time_bbox = card["time_box"]["bbox"]  # [x, y, w, h]
            time_x, time_y, time_w, time_h = time_bbox
            time_bottom = time_y + time_h
            
            # If time box extends into our search area, adjust search top
            if time_bottom > search_top:
                search_top = max(search_top, time_bottom + 2)  # Small gap below time box
        
        # Validate search region dimensions
        if search_right <= search_left or search_bottom <= search_top:
            return None
            
        search_width = search_right - search_left
        search_height = search_bottom - search_top
        
        # Ensure minimum search area
        if search_width < self.MIN_NAME_WIDTH or search_height < self.MIN_NAME_HEIGHT:
            return None
        
        return (search_left, search_top, search_width, search_height)
    
    def _detect_white_text_regions(self, img: np.ndarray, search_region: Tuple[int, int, int, int], 
                                 debug_mode: bool = False, debug_data: Dict = None, card_id: int = 0) -> List[Tuple[int, int, int, int]]:
        """
        Detect white text regions using enhanced projection analysis
        
        Uses horizontal projection analysis similar to time detection but optimized 
        for detecting white contact names below the boundary.
        
        Args:
            img: Original image
            search_region: Region to search as (x, y, w, h)
            debug_mode: Enable debug data collection
            debug_data: Debug data dictionary to populate
            card_id: Card ID for debug tracking
            
        Returns:
            List of detected text regions as (x, y, w, h) tuples
        """
        search_x, search_y, search_w, search_h = search_region
        
        # Extract search region from image
        roi = img[search_y:search_y + search_h, search_x:search_x + search_w]
        
        # Convert to grayscale for analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges (like time detection)
        filtered_roi = cv2.bilateralFilter(gray_roi, d=5, sigmaColor=35, sigmaSpace=35)
        
        # Enhanced white text detection using projection analysis
        # Calculate horizontal projection (row-wise intensity analysis)
        horizontal_proj = np.mean(filtered_roi, axis=1)
        
        # Find text regions using projection peaks (white text has higher intensity)
        # Smooth the projection to reduce noise
        if len(horizontal_proj) > 5:
            kernel_size = min(5, len(horizontal_proj) // 2)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed_proj = cv2.GaussianBlur(horizontal_proj.reshape(-1, 1), (1, kernel_size), 0).ravel()
        else:
            smoothed_proj = horizontal_proj
        
        # Find regions with high intensity (white text)
        # Use adaptive thresholding based on projection statistics
        proj_mean = np.mean(smoothed_proj)
        proj_std = np.std(smoothed_proj)
        text_threshold = proj_mean + (proj_std * 0.5)  # Regions significantly brighter than average
        
        # Find continuous regions above threshold
        text_regions_1d = []
        in_text_region = False
        region_start = 0
        
        for i, intensity in enumerate(smoothed_proj):
            if intensity >= text_threshold and not in_text_region:
                # Start of text region
                in_text_region = True
                region_start = i
            elif intensity < text_threshold and in_text_region:
                # End of text region
                in_text_region = False
                region_height = i - region_start
                if region_height >= self.MIN_NAME_HEIGHT:  # Minimum height for valid text
                    text_regions_1d.append((region_start, region_height))
        
        # Handle case where text region extends to end
        if in_text_region:
            region_height = len(smoothed_proj) - region_start
            if region_height >= self.MIN_NAME_HEIGHT:
                text_regions_1d.append((region_start, region_height))
        
        # Convert 1D text regions to 2D bounding boxes and apply morphological processing
        text_regions = []
        
        # Debug: Store horizontal projection data
        if debug_mode and debug_data is not None:
            debug_data["horizontal_projections"] = {
                f"{card_id}": {
                    "original_proj": horizontal_proj.tolist(),
                    "smoothed_proj": smoothed_proj.tolist(),
                    "threshold": text_threshold,
                    "text_regions_1d": text_regions_1d
                }
            }
        
        # Process each 1D text region
        for region_start, region_height in text_regions_1d:
            # Extract the text region for detailed analysis
            text_roi = filtered_roi[region_start:region_start + region_height, :]
            
            # Apply refined white text detection within this region
            _, white_mask = cv2.threshold(text_roi, self.WHITE_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
            
            # Apply morphological operations to connect text pixels
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=self.MORPH_ITERATIONS)
            
            # Debug: Store binary mask for first few regions
            if debug_mode and debug_data is not None and len(text_regions) < 3:
                if "binary_masks" not in debug_data:
                    debug_data["binary_masks"] = {}
                debug_data["binary_masks"][f"{card_id}_region_{len(text_regions)}"] = white_mask.copy()
        
        # Create full-size binary mask for compatibility with existing code
        full_white_mask = np.zeros_like(filtered_roi, dtype=np.uint8)
        
        for region_start, region_height in text_regions_1d:
            text_roi = filtered_roi[region_start:region_start + region_height, :]
            _, white_mask = cv2.threshold(text_roi, self.WHITE_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, self.MORPH_KERNEL_SIZE)
            white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel, iterations=self.MORPH_ITERATIONS)
            full_white_mask[region_start:region_start + region_height, :] = white_mask
        
        # Store debug masks
        if debug_mode and debug_data is not None:
            debug_data["binary_masks"][f"{card_id}_raw"] = full_white_mask.copy()
            debug_data["binary_masks"][f"{card_id}_processed"] = full_white_mask.copy()
            
            # Store enhanced morphology data
            debug_data["morphology_data"][card_id] = {
                "kernel_size": self.MORPH_KERNEL_SIZE,
                "iterations": self.MORPH_ITERATIONS,
                "projection_threshold": text_threshold,
                "text_regions_found": len(text_regions_1d),
                "horizontal_proj_method": True
            }
        
        # Use the full mask for contour detection
        white_mask = full_white_mask
        
        # Find contours of white regions
        contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Debug: Store contour data
        if debug_mode and debug_data is not None:
            debug_data["contour_data"][card_id] = {
                "total_contours": len(contours),
                "contour_areas": [cv2.contourArea(c) for c in contours],
                "contour_bboxes": [cv2.boundingRect(c) for c in contours]
            }
        
        # Convert contours to bounding rectangles (relative to search region)
        text_regions = []
        white_pixel_ratios = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if region has sufficient white pixels
            region_mask = white_mask[y:y+h, x:x+w]
            white_pixel_ratio = np.sum(region_mask > 0) / (w * h)
            white_pixel_ratios.append(white_pixel_ratio)
            
            if white_pixel_ratio >= self.MIN_WHITE_PIXEL_RATIO:
                # Convert to absolute coordinates
                abs_x = search_x + x
                abs_y = search_y + y
                text_regions.append((abs_x, abs_y, w, h))
        
        # Debug: Store white text regions data
        if debug_mode and debug_data is not None:
            debug_data["white_text_regions"][card_id] = {
                "total_regions": len(text_regions),
                "all_ratios": white_pixel_ratios,
                "filtered_regions": text_regions,
                "threshold_used": self.WHITE_THRESHOLD_MIN,
                "min_ratio_required": self.MIN_WHITE_PIXEL_RATIO
            }
        
        return text_regions
    
    def _filter_and_extract_boundaries(self, text_regions: List[Tuple[int, int, int, int]], 
                                     search_region: Tuple[int, int, int, int],
                                     debug_mode: bool = False, debug_data: Dict = None, card_id: int = 0) -> Optional[List[int]]:
        """
        Filter text regions by size constraints and extract the best name boundary
        
        Args:
            text_regions: List of detected text regions as (x, y, w, h)
            search_region: Original search region as (x, y, w, h)
            debug_mode: Enable debug data collection
            debug_data: Debug data dictionary to populate
            card_id: Card ID for debug tracking
            
        Returns:
            Best name boundary as [x, y, w, h] or None if no suitable region found
        """
        if not text_regions:
            if debug_mode and debug_data is not None:
                debug_data["processing_steps"].append(f"Card {card_id}: No text regions to filter")
            return None
        
        # Debug: Store filtering process
        if debug_mode and debug_data is not None:
            debug_data["processing_steps"].append(f"Card {card_id}: Filtering {len(text_regions)} text regions")
        
        # Filter by size constraints
        valid_regions = []
        rejected_regions = []
        for x, y, w, h in text_regions:
            if (self.MIN_NAME_WIDTH <= w <= self.MAX_NAME_WIDTH and 
                self.MIN_NAME_HEIGHT <= h <= self.MAX_NAME_HEIGHT):
                valid_regions.append((x, y, w, h))
            else:
                rejected_regions.append((x, y, w, h, f"size {w}x{h}"))
        
        # Debug: Store filtering results
        if debug_mode and debug_data is not None:
            if not hasattr(debug_data, 'filtering_results'):
                debug_data['filtering_results'] = {}
            debug_data['filtering_results'][card_id] = {
                "input_regions": text_regions,
                "valid_regions": valid_regions,
                "rejected_regions": rejected_regions,
                "size_constraints": {
                    "min_width": self.MIN_NAME_WIDTH,
                    "max_width": self.MAX_NAME_WIDTH,
                    "min_height": self.MIN_NAME_HEIGHT,
                    "max_height": self.MAX_NAME_HEIGHT
                }
            }
        
        if not valid_regions:
            if debug_mode and debug_data is not None:
                debug_data["processing_steps"].append(f"Card {card_id}: No regions passed size filtering")
            return None
        
        # If multiple valid regions, choose the largest one (most likely to be name)
        if len(valid_regions) == 1:
            selected_region = list(valid_regions[0])
        else:
            # Sort by area (width * height) and take the largest
            valid_regions.sort(key=lambda region: region[2] * region[3], reverse=True)
            selected_region = list(valid_regions[0])
            
            if debug_mode and debug_data is not None:
                areas = [r[2] * r[3] for r in valid_regions]
                debug_data["processing_steps"].append(f"Card {card_id}: Selected largest region (area: {areas[0]}) from {len(valid_regions)} candidates")
        
        return selected_region
    
    def _calculate_boundary_confidence(self, img: np.ndarray, name_boundary: List[int]) -> float:
        """
        Calculate confidence score for detected name boundary
        
        Args:
            img: Original image
            name_boundary: Name boundary as [x, y, w, h]
            
        Returns:
            Confidence score between 0.0 and 1.0
        """
        x, y, w, h = name_boundary
        
        # Extract name region
        name_roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate metrics for confidence
        # 1. Average brightness (higher for white text)
        avg_brightness = np.mean(gray_roi) / 255.0
        
        # 2. Contrast (difference between brightest and darkest pixels)
        contrast = (np.max(gray_roi) - np.min(gray_roi)) / 255.0
        
        # 3. Size appropriateness (penalty for very small or very large regions)
        size_score = min(1.0, w / 60.0)  # Normalize width to typical name width
        if w > 150:  # Penalty for very wide regions
            size_score *= 0.7
        
        # 4. Aspect ratio score (names are typically wider than tall)
        aspect_ratio = w / h
        if 2.0 <= aspect_ratio <= 8.0:  # Typical name aspect ratios
            aspect_score = 1.0
        else:
            aspect_score = 0.6
        
        # Combine metrics (weighted average)
        confidence = (avg_brightness * 0.4 + contrast * 0.3 + size_score * 0.2 + aspect_score * 0.1)
        
        return min(1.0, confidence)  # Cap at 1.0
    
    def _detect_grey_timestamp_left_edge(self, img: np.ndarray, card_bbox: List[int], avatar_data: Dict) -> Optional[int]:
        """
        Detect the left edge of grey timestamp text in the upper portion of the card
        
        This method extracts the upper portion of the card, isolates grey timestamp text,
        and finds its left edge position as the boundary between name and timestamp regions.
        
        Args:
            img: Original BGR image
            card_bbox: Card boundaries as [x, y, w, h] from CardBoundaryDetector
            avatar_data: Avatar information including bbox and center
            
        Returns:
            X-coordinate of the left edge of grey timestamp text, or None if not detected
        """
        rx, ry, rw, rh = map(int, card_bbox)
        avatar_bbox = avatar_data["bbox"]
        avatar_center = avatar_data["center"]
        
        # Phase 1: Extract upper card region (where timestamps are located)
        # Use precise card coordinates from CardBoundaryDetector
        upper_region_height = min(rh * 0.4, 35)  # Upper 40% of card or max 35px
        
        analysis_left = rx + 5  # Start from card left boundary with small margin
        analysis_right = rx + rw - 5  # End at card right boundary with small margin
        analysis_top = ry + 5  # Start from card top with small margin
        analysis_bottom = ry + int(upper_region_height)  # Upper portion only
        
        # Validate upper region
        if analysis_left >= analysis_right or analysis_top >= analysis_bottom:
            return None
        if analysis_right - analysis_left < 50 or analysis_bottom - analysis_top < 10:
            return None
            
        # Extract ROI for analysis
        roi = img[analysis_top:analysis_bottom, analysis_left:analysis_right]
        
        # Phase 2: Grey timestamp text isolation with HSV color space
        # Convert to HSV for better grey text detection
        roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define precise grey color range for WeChat timestamps
        # WeChat timestamps are typically light grey (#808080 to #AAAAAA)
        grey_lower = np.array([0, 0, 80])      # Lower bound for grey text (H, S, V)
        grey_upper = np.array([180, 50, 180])  # Upper bound for grey text (H, S, V)
        
        # Create mask for grey timestamp text
        grey_mask = cv2.inRange(roi_hsv, grey_lower, grey_upper)
        
        # Phase 3: Background inversion and text cleaning
        # Invert background: make black->white, keep grey text as-is
        roi_grey = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply morphological operations to clean up text connectivity
        kernel = np.ones((2, 3), np.uint8)  # Slightly wider kernel for text
        grey_mask_clean = cv2.morphologyEx(grey_mask, cv2.MORPH_CLOSE, kernel)
        grey_mask_clean = cv2.morphologyEx(grey_mask_clean, cv2.MORPH_OPEN, np.ones((2,2), np.uint8))
        
        # Phase 4: Left edge detection of grey timestamp text
        # Find leftmost grey pixels in each horizontal row
        left_edge_candidates = []
        
        for y in range(grey_mask_clean.shape[0]):
            row = grey_mask_clean[y, :]
            # Find first white pixel (grey text) from left
            white_pixels = np.where(row > 0)[0]
            if len(white_pixels) > 0:
                left_edge_candidates.append(white_pixels[0])
        
        if not left_edge_candidates:
            return None  # No grey text found
        
        # Use median of left edge positions for stability
        median_left_edge = int(np.median(left_edge_candidates))
        
        # Convert ROI coordinate back to full image coordinate
        timestamp_left_edge_x = analysis_left + median_left_edge
        
        return timestamp_left_edge_x
    
    def _create_search_region_left_of_timestamp(self, card: Dict, boundary_x: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Create search region for name detection LEFT of the detected timestamp boundary
        
        This approach searches to the LEFT of the detected grey timestamp text
        where contact names are typically located.
        
        Args:
            card: Card dictionary with boundary, avatar data
            boundary_x: X-coordinate of the left edge of grey timestamp text
            
        Returns:
            Search region as (x, y, w, h) or None if invalid
        """
        card_bbox = card["bbox"]  # [x, y, w, h]
        avatar_bbox = card["avatar"]["bbox"]  # [x, y, w, h]
        
        card_x, card_y, card_w, card_h = card_bbox
        avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
        
        # Create search region LEFT of the detected timestamp boundary
        # Vertical bounds: use the full card height for name detection
        search_top = card_y + self.SEARCH_MARGIN_TOP  # Start from card top
        search_bottom = card_y + card_h - self.SEARCH_MARGIN_TOP  # End before card bottom
        
        # Horizontal bounds: between avatar and timestamp
        search_left = avatar_x + avatar_w + self.SEARCH_MARGIN_LEFT  # Start right of avatar
        search_right = boundary_x - 5  # End before timestamp with small margin
        
        # Validate search region dimensions
        if search_right <= search_left or search_bottom <= search_top:
            return None
            
        search_width = search_right - search_left
        search_height = search_bottom - search_top
        
        # Ensure minimum search area
        if search_width < self.MIN_NAME_WIDTH or search_height < self.MIN_NAME_HEIGHT:
            return None
        
        return (search_left, search_top, search_width, search_height)
    
    def _calculate_detailed_confidence_breakdown(self, img: np.ndarray, name_boundary: List[int]) -> Dict:
        """
        Calculate detailed confidence breakdown for visualization
        
        Args:
            img: Original image
            name_boundary: Name boundary as [x, y, w, h]
            
        Returns:
            Dictionary with detailed confidence metrics
        """
        x, y, w, h = name_boundary
        
        # Extract name region
        name_roi = img[y:y+h, x:x+w]
        gray_roi = cv2.cvtColor(name_roi, cv2.COLOR_BGR2GRAY)
        
        # Calculate individual metrics
        avg_brightness = np.mean(gray_roi) / 255.0
        contrast = (np.max(gray_roi) - np.min(gray_roi)) / 255.0
        size_score = min(1.0, w / 60.0)
        if w > 150:
            size_score *= 0.7
        
        aspect_ratio = w / h
        if 2.0 <= aspect_ratio <= 8.0:
            aspect_score = 1.0
        else:
            aspect_score = 0.6
        
        # Calculate white pixel ratio
        _, white_mask = cv2.threshold(gray_roi, self.WHITE_THRESHOLD_MIN, 255, cv2.THRESH_BINARY)
        white_pixel_ratio = np.sum(white_mask > 0) / (w * h)
        
        # Overall confidence
        confidence = (avg_brightness * 0.4 + contrast * 0.3 + size_score * 0.2 + aspect_score * 0.1)
        
        return {
            "avg_brightness": avg_brightness,
            "contrast": contrast,
            "size_score": size_score,
            "aspect_score": aspect_score,
            "aspect_ratio": aspect_ratio,
            "white_pixel_ratio": white_pixel_ratio,
            "region_dimensions": [w, h],
            "overall_confidence": min(1.0, confidence),
            "weights": {
                "brightness": 0.4,
                "contrast": 0.3,
                "size": 0.2,
                "aspect": 0.1
            }
        }
    
    def _generate_statistical_analysis(self, debug_data: Dict, enhanced_cards: List[Dict]) -> Dict:
        """
        Generate comprehensive statistical analysis for debug visualization
        
        Args:
            debug_data: Debug data dictionary
            enhanced_cards: Cards with detection results
            
        Returns:
            Statistical analysis dictionary
        """
        stats = {
            "summary": {
                "total_cards": len(enhanced_cards),
                "successful_detections": len(debug_data["success_cards"]),
                "failed_detections": len(debug_data["failed_cards"]),
                "success_rate": len(debug_data["success_cards"]) / len(enhanced_cards) if enhanced_cards else 0
            },
            "confidence_statistics": {},
            "dimension_statistics": {},
            "processing_time_statistics": {},
            "contour_statistics": {},
            "white_text_statistics": {}
        }
        
        # Confidence statistics
        confidences = [debug_data["confidence_scores"].get(card_id, 0.0) 
                      for card_id in debug_data["success_cards"]]
        if confidences:
            stats["confidence_statistics"] = {
                "mean": np.mean(confidences),
                "median": np.median(confidences),
                "min": np.min(confidences),
                "max": np.max(confidences),
                "std": np.std(confidences)
            }
        
        # Dimension statistics for successful detections
        dimensions = []
        for card_id in debug_data["success_cards"]:
            result = debug_data["detection_results"].get(card_id, {})
            if result.get("bbox"):
                bbox = result["bbox"]
                dimensions.append({"width": bbox[2], "height": bbox[3], "area": bbox[2] * bbox[3]})
        
        if dimensions:
            widths = [d["width"] for d in dimensions]
            heights = [d["height"] for d in dimensions]
            areas = [d["area"] for d in dimensions]
            
            stats["dimension_statistics"] = {
                "width": {"mean": np.mean(widths), "min": np.min(widths), "max": np.max(widths)},
                "height": {"mean": np.mean(heights), "min": np.min(heights), "max": np.max(heights)},
                "area": {"mean": np.mean(areas), "min": np.min(areas), "max": np.max(areas)}
            }
        
        # Processing time statistics
        processing_times = list(debug_data["processing_time"].values())
        if processing_times:
            stats["processing_time_statistics"] = {
                "mean": np.mean(processing_times),
                "total": sum(processing_times),
                "min": np.min(processing_times),
                "max": np.max(processing_times)
            }
        
        # Contour and white text statistics
        total_contours = 0
        total_white_regions = 0
        for card_id, contour_data in debug_data["contour_data"].items():
            total_contours += contour_data.get("total_contours", 0)
        
        for card_id, white_data in debug_data["white_text_regions"].items():
            total_white_regions += white_data.get("total_regions", 0)
        
        stats["contour_statistics"]["total_contours_found"] = total_contours
        stats["white_text_statistics"]["total_white_regions_found"] = total_white_regions
        
        return stats
    
    def create_comprehensive_debug_visualization(self, enhanced_cards: List[Dict], detection_info: Dict) -> str:
        """
        Create comprehensive debug visualization matching time detection quality
        
        Args:
            enhanced_cards: Cards with name boundary detection results
            detection_info: Detection information with debug data
            
        Returns:
            Path to saved debug visualization image
        """
        from modules.visualization_engine import cDiagnosticVisualizationEngine
        
        debug_data = detection_info.get("debug_data")
        if not debug_data:
            raise ValueError("Debug data not available. Run detect_name_boundaries with debug_mode=True")
        
        # Prepare visualization data for the engine
        visualization_data = {
            "image_path": debug_data["image_path"],
            "enhanced_cards": enhanced_cards,
            "detection_info": detection_info,
            "debug_data": debug_data,
            "detector_type": "contact_name",
            "success_count": detection_info["names_detected"],
            "failed_count": detection_info["total_cards_processed"] - detection_info["names_detected"]
        }
        
        # Create visualization engine and generate comprehensive debug image
        viz_engine = cDiagnosticVisualizationEngine()
        output_path = viz_engine.create_comprehensive_debug_visualization(
            "contact_name", visualization_data
        )
        
        print(f"🎨 Comprehensive debug visualization saved: {output_path}")
        return output_path

    def create_name_boundary_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create visualization showing detected name boundaries with orange rectangles
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for visualization
            
        Returns:
            Filename of generated visualization
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating name boundary visualization...")
        
        # Get detection results
        enhanced_cards, detection_info = self.detect_name_boundaries(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Track visualization statistics
        names_visualized = 0
        search_regions_drawn = 0
        
        # Draw enhanced visualization with name boundaries
        for card in enhanced_cards:
            # Draw existing card and avatar boundaries (from CardBoundaryDetector)
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)
            
            # Draw name boundary if detected (orange rectangle)
            if card.get("name_boundary"):
                name_bbox = card["name_boundary"]["bbox"]
                name_x, name_y, name_w, name_h = name_bbox
                confidence = card["name_boundary"]["confidence"]
                
                # Draw name boundary (orange rectangle)
                cv2.rectangle(result, (name_x, name_y), (name_x + name_w, name_y + name_h), (0, 165, 255), 2)
                
                # Draw confidence score
                cv2.putText(result, f"{confidence:.2f}", (name_x, name_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1)
                
                names_visualized += 1
                
                # Optionally draw search region (light gray rectangle)
                if card["name_boundary"].get("search_region"):
                    search_region = card["name_boundary"]["search_region"]
                    search_x, search_y, search_w, search_h = search_region
                    cv2.rectangle(result, (search_x, search_y), (search_x + search_w, search_y + search_h), (128, 128, 128), 1)
                    search_regions_drawn += 1
            
            # Draw card ID
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add enhanced legend and statistics
        legend_y = 30
        cv2.putText(result, "5. Contact Name Boundary Detection Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers  Orange=Names",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection statistics
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        total_cards = detection_info.get('total_cards_processed', len(enhanced_cards))
        cv2.putText(result, f"Names Detected: {names_visualized}/{total_cards} cards ({success_rate:.1f}%)",
                   (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        detection_method = detection_info.get('detection_method', 'unknown')
        cv2.putText(result, f"Detection Method: {detection_method} | Gray=Search Regions",
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_05_ContactName_Boundaries_{names_visualized}names_{len(enhanced_cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Contact name boundary visualization saved: {output_path}")
        print(f"   📊 Visualization stats: {names_visualized} names, {search_regions_drawn} search regions drawn")
        
        return os.path.basename(output_path)


# =============================================================================
# PHASE 6: TIME BOX DETECTOR  
# =============================================================================

class cTimeBoxDetector:
    """
    Time Box Detector using visual density patterns (non-OCR approach)
    Detects timestamp regions using column projection and statistical analysis
    """
    
    def __init__(self, debug_mode: bool = False):
        self.card_boundary_detector = cCardBoundaryDetector(debug_mode=debug_mode)
        
        # WeChat-optimized layout parameters for upper-center time detection
        self.CARD_WIDTH_FRAC = 0.7      # Search up to 70% of card width
        self.VPAD_RATIO = 0.12          # Vertical padding ratio for text box sizing
        self.AVATAR_TIME_MARGIN = 15    # Gap between avatar and time box (pixels)
        self.MIN_W_PX = 10              # Minimum timestamp width (pixels) - reduced for smaller times
        
        # Image processing parameters
        self.BILATERAL_D = 5            # Bilateral filter diameter
        self.BILATERAL_SIGMA_COLOR = 35 # Color sigma for bilateral filter
        self.BILATERAL_SIGMA_SPACE = 35 # Space sigma for bilateral filter
        self.CLAHE_CLIP = 3.0          # CLAHE clip limit - increased for better gray text contrast
        self.CLAHE_GRID = 6            # CLAHE tile grid size - smaller tiles for finer contrast adjustment
        self.ADAPT_BLOCK = 21          # Adaptive threshold block size
        self.ADAPT_C = 8               # Adaptive threshold constant
        
        # Statistical analysis parameters
        self.K_MAD = 1.5               # MAD threshold multiplier - lowered for better detection of faint text
        self.K_MAD_FALLBACK = 1.0      # Even lower threshold for second pass if first fails
        
        # Timestamp-specific parameters (more lenient for small regions)
        self.K_MAD_TIMESTAMP = 1.0     # Lower threshold for timestamp region analysis
        self.K_MAD_TIMESTAMP_FALLBACK = 0.5  # More aggressive fallback for timestamps
        
    def detect_time_boundaries(self, image_path: str, coord_context: 'cWeChatCoordinateContext' = None,
                               return_context: bool = False) -> Tuple[List[Dict], Dict]:
        """
        Detect timestamp boundaries using visual density patterns
        
        📌 INPUT CONTRACT:
        - image_path: str - Path to WeChat screenshot (PNG/JPG, with visible timestamps)
        - coord_context: Optional[WeChatCoordinateContext] - Coordinate context for integration  
        - return_context: bool - Whether to return coordinate context with results
        
        📌 OUTPUT CONTRACT:
        Standard Mode (return_context=False):
        - Success: Tuple[List[Dict], Dict] - (enhanced_cards_with_times, detection_info)
        - enhanced_cards_with_times: List of cards with added timestamp components
        - detection_info: {"total_times": int, "success_rate": float, "method": "upper_region_density"}
        
        Context Mode (return_context=True):
        - Success: ((enhanced_cards_with_times, detection_info), cWeChatCoordinateContext)
        
        Side Effects:
        - Generates debug visualization files if debug_mode=True
        - Updates coordinate context with timestamp component data
        - Dependencies: Requires card boundary detection results
        """
        print(f"🎯 6. Time Box Detection: {os.path.basename(image_path)}")
        
        # Step 1: Get validated card and avatar data
        cards, card_detection_info = self.card_boundary_detector.detect_cards(image_path)
        if not cards:
            print("❌ No cards available for time detection")
            return [], {}
            
        # Load image for processing
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return [], {}
            
        # Step 3: Get panel right boundary from width detection
        width_boundaries = card_detection_info.get("width_boundaries")
        if not width_boundaries:
            print("❌ No width boundaries available for time detection")
            return [], {}
            
        # width_boundaries is a tuple (left_boundary, right_boundary)
        left_boundary, right_boundary = width_boundaries
        x_panel_right = right_boundary
        print(f"  📏 Using panel right boundary: {x_panel_right}px")
        print(f"  📄 Processing {len(cards)} cards for time boundary detection")
        
        # Process each card for time boundaries
        enhanced_cards = []
        total_times_detected = 0
        
        for card in cards:
            enhanced_card = self._detect_time_box_for_card(img, card, x_panel_right)
            enhanced_cards.append(enhanced_card)
            
            if enhanced_card.get("time_box"):
                total_times_detected += 1
                card_id = enhanced_card["card_id"]
                time_bbox = enhanced_card["time_box"]["bbox"]
                density_score = enhanced_card["time_box"]["density_score"]
                print(f"    ⏰ Card {card_id}: Time box {time_bbox[2]}×{time_bbox[3]}px at ({time_bbox[0]}, {time_bbox[1]}) | density={density_score:.1f}")
        
        # Step 5: Generate detection summary
        detection_info = {
            "total_cards_processed": len(cards),
            "times_detected": total_times_detected,
            "detection_success_rate": total_times_detected / len(cards) if cards else 0,
            "card_detection_info": card_detection_info,
            "detection_method": "visual_density_pattern",
            "panel_right_boundary": x_panel_right
        }
        
        print(f"  ✅ Time boundary detection complete: {total_times_detected}/{len(cards)} cards have detected timestamps")
        
        # Coordinate Context Integration
        results_tuple = (enhanced_cards, detection_info)
        
        # Create or initialize coordinate context if needed
        if return_context and coord_context is None:
            img = cv2.imread(image_path)
            if img is not None:
                H, W = img.shape[:2]
                coord_context = cWeChatCoordinateContext(image_path, (W, H))
        
        # Populate coordinate context with timestamp data if provided
        if coord_context is not None and enhanced_cards:
            for card in enhanced_cards:
                if "time_box" in card and card["time_box"]:
                    card_id = card["card_id"]
                    time_data = card["time_box"]
                    time_bbox = time_data["bbox"]
                    
                    coord_context.add_component(
                        card_id, "timestamp", time_bbox, 
                        "TimeBoxDetector.detect_time_boundaries",
                        time_data.get("density_score", 0.8),
                        ocr_suitable=True, expected_content="timestamp"
                    )
        
        # Return based on requested format
        if return_context:
            return results_tuple, coord_context
        else:
            return results_tuple

    def _detect_time_box_for_card(self, img: np.ndarray, card: Dict, x_panel_right: int) -> Dict:
        """
        Detect time box for a single card using boundary detection and upper region analysis
        
        Args:
            img: Original image as numpy array
            card: Card dictionary with boundary and avatar data
            x_panel_right: Right boundary of the panel (kept for compatibility)
            
        Returns:
            Enhanced card dictionary with time box and name-time boundary information
        """
        enhanced_card = card.copy()
        
        try:
            # Get card and avatar boundaries
            card_bbox = card["bbox"]  # [x, y, w, h]
            avatar_data = card["avatar"]  # Complete avatar information
            
            # Initialize debug info collection if requested
            debug_info = {} if hasattr(self, '_collect_debug') and self._collect_debug else None
            
            # Detect the boundary between name and timestamp regions
            boundary_y = self._detect_name_time_horizontal_boundary(img, card_bbox, avatar_data)
            
            if boundary_y is not None:
                enhanced_card["name_time_boundary"] = {
                    "y": boundary_y,
                    "detection_method": "horizontal_boundary_analysis",
                    "avatar_center_y": avatar_data["center"][1]
                }
                # Use detected boundary for time box detection
                time_result = self._upper_density_time_box_with_boundary(img, card_bbox, avatar_data, boundary_y, debug_info)
            else:
                # Fallback to original method using avatar center as boundary
                enhanced_card["name_time_boundary"] = {
                    "y": avatar_data["center"][1],  # Use avatar center as fallback
                    "detection_method": "avatar_center_fallback",
                    "avatar_center_y": avatar_data["center"][1]
                }
                # Apply original density-based time box detection in upper region
                time_result = self._upper_density_time_box(img, card_bbox, avatar_data, debug_info)
            
            if time_result:
                tx, ty, tw, th, density_score = time_result
                enhanced_card["time_box"] = {
                    "bbox": [tx, ty, tw, th],
                    "density_score": density_score,
                    "detection_method": "upper_region_density",
                    "search_region_info": {
                        "avatar_right": avatar_data["bbox"][0] + avatar_data["bbox"][2],
                        "avatar_center_y": avatar_data["center"][1],
                        "card_width_frac": self.CARD_WIDTH_FRAC,
                        "card_width": card_bbox[2]
                    }
                }
                
                # Add success info to debug data
                if debug_info is not None:
                    debug_info['detection_successful'] = True
            else:
                # Mark as failed if debug info exists
                if debug_info is not None:
                    debug_info['detection_successful'] = False
            
            # Store debug information in the card
            if debug_info is not None:
                enhanced_card["time_detection_debug"] = debug_info
        
        except Exception as e:
            print(f"    ⚠️ Card {card['card_id']}: Time detection failed - {e}")
            if hasattr(self, '_collect_debug') and self._collect_debug:
                enhanced_card["time_detection_debug"] = {
                    'detection_failed': True,
                    'failure_reason': f'exception: {str(e)}',
                    'detection_successful': False
                }
        
        return enhanced_card

    def _upper_density_time_box(self, bgr: np.ndarray, row_box: List[int], avatar_data: Dict, debug_info: Dict = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Core density-based time box detection algorithm for upper card region
        
        Args:
            bgr: Original BGR image
            row_box: Card boundaries as [x, y, w, h] 
            avatar_data: Avatar information including bbox and center
            
        Returns:
            Tuple of (x, y, w, h, density_score) or None if not found
        """
        rx, ry, rw, rh = map(int, row_box)
        avatar_bbox = avatar_data["bbox"]
        avatar_center = avatar_data["center"]
        
        # Calculate search region (upper portion, after avatar)
        avatar_right = avatar_bbox[0] + avatar_bbox[2]
        avatar_center_y = avatar_center[1]
        
        # Horizontal bounds: from avatar right edge to full card width (expanded coverage)
        search_left = avatar_right + 5  # Smaller margin for better coverage
        search_right = rx + rw - 5      # Full card width with small right margin
        
        # Vertical bounds: upper portion above avatar center line (divider)
        search_top = ry + 5  # Small margin from card top
        search_bottom = avatar_center_y  # Stop at avatar center (divider line)
        
        # Validate search region
        if search_left >= search_right or search_top >= search_bottom:
            return None
        if search_right - search_left < self.MIN_W_PX or search_bottom - search_top < 6:
            return None

        # Extract ROI for processing
        roi = bgr[search_top:search_bottom, search_left:search_right]
        
        # Store debug information if requested
        if debug_info is not None:
            debug_info.update({
                'search_region': {
                    'left': search_left, 'right': search_right,
                    'top': search_top, 'bottom': search_bottom,
                    'width': search_right - search_left,
                    'height': search_bottom - search_top
                },
                'roi_original': roi.copy(),
                'card_boundaries': {'x': rx, 'y': ry, 'w': rw, 'h': rh},
                'avatar_info': {'bbox': avatar_bbox, 'center': avatar_center}
            })
        
        # Simple color-based detection: Names are white/light, timestamps are gray
        # Convert to HSV for better color separation
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # Define color ranges for gray timestamps (the key distinguishing feature)
        gray_lower = np.array([0, 0, 80])      # Lower bound for gray text
        gray_upper = np.array([180, 60, 180])  # Upper bound for gray text
        
        # Create mask for gray timestamp text
        gray_mask = cv2.inRange(hsv, gray_lower, gray_upper)
        
        # Optional: Clean up the mask
        kernel = np.ones((2,2), np.uint8)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_CLOSE, kernel)
        gray_mask = cv2.morphologyEx(gray_mask, cv2.MORPH_OPEN, kernel)
        
        # Store preprocessing results in debug info
        if debug_info is not None:
            debug_info.update({
                'roi_original_hsv': hsv.copy(),
                'roi_gray_mask': gray_mask.copy(),
                'color_detection_method': 'HSV_gray_text_segmentation'
            })
        
        # Find contours in the gray timestamp mask
        contours, _ = cv2.findContours(gray_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            if debug_info is not None:
                debug_info.update({
                    'detection_failed': True,
                    'failure_reason': 'no_valid_timestamp_contour',
                    'contours_found': 0
                })
            return None
        
        # Find the best timestamp contour (rightmost, reasonable size)
        best_contour = None
        best_score = -1
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Validate timestamp characteristics
            if w < 8 or h < 6:  # Too small
                continue
            if w > search_right - search_left - 10:  # Too wide  
                continue
            if h > (search_bottom - search_top) * 0.8:  # Too tall
                continue
                
            # Prefer rightmost contours (timestamps are typically on the right)
            position_score = x / max(1, search_right - search_left)
            size_score = min(w * h / 100, 1.0)  # Size bonus but capped
            
            total_score = position_score * 0.7 + size_score * 0.3
            
            if total_score > best_score:
                best_score = total_score
                best_contour = contour
        
        if best_contour is None:
            if debug_info is not None:
                debug_info.update({
                    'detection_failed': True,
                    'failure_reason': 'no_valid_timestamp_contour',
                    'contours_found': len(contours)
                })
            return None
        
        # Get bounding box of best timestamp contour
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Convert to absolute coordinates
        tx0 = search_left + x
        ty = search_top + y
        
        # Extend width to card right boundary for full timestamp capture
        card_right = rx + rw - 5
        tw = max(w, card_right - tx0)  # At least contour width, but extend to card edge
        th = h
        
        # Calculate density score for compatibility
        contour_area = cv2.contourArea(best_contour)
        density_score = contour_area / max(1, w * h) * 100  # Percentage fill
        
        if debug_info is not None:
            debug_info.update({
                'detection_failed': False,
                'detection_successful': True,
                'contours_found': len(contours),
                'best_contour_score': best_score,
                'timestamp_bbox': [tx0, ty, tw, th],
                'density_score': density_score
            })
        
        return (int(tx0), int(ty), int(tw), int(th), float(density_score))
    
    def _detect_name_time_horizontal_boundary(self, img: np.ndarray, card_bbox: List[int], avatar_data: Dict) -> Optional[int]:
        """
        Detect the horizontal boundary between name and timestamp regions
        
        This method analyzes the region to the right of the avatar to find the
        visual separator between the contact name (below) and timestamp (above).
        
        Args:
            img: Original BGR image
            card_bbox: Card boundaries as [x, y, w, h]
            avatar_data: Avatar information including bbox and center
            
        Returns:
            Y-coordinate of the boundary line, or None if not detected
        """
        rx, ry, rw, rh = map(int, card_bbox)
        avatar_bbox = avatar_data["bbox"]
        avatar_center = avatar_data["center"]
        
        # Define analysis region: from avatar right edge to card edge
        avatar_right = avatar_bbox[0] + avatar_bbox[2]
        analysis_left = avatar_right + 5
        analysis_right = rx + rw - 5
        analysis_top = ry + 5
        analysis_bottom = min(ry + rh - 5, avatar_center[1] + 30)  # Focus on upper region
        
        # Validate region
        if analysis_left >= analysis_right or analysis_top >= analysis_bottom:
            return None
            
        # Extract ROI for analysis
        roi = img[analysis_top:analysis_bottom, analysis_left:analysis_right]
        
        # Convert to grayscale for intensity analysis
        gray_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to reduce noise while preserving edges
        filtered = cv2.bilateralFilter(gray_roi, d=5, sigmaColor=35, sigmaSpace=35)
        
        # Calculate horizontal projection (row-wise intensity average)
        horizontal_proj = np.mean(filtered, axis=1)
        
        # Look for significant gaps or transitions in horizontal projection
        # These indicate boundaries between text regions
        
        # Method 1: Find minimum intensity row (likely gap between name and time)
        if len(horizontal_proj) > 10:
            # Smooth the projection to reduce noise
            kernel_size = min(5, len(horizontal_proj) // 3)
            if kernel_size % 2 == 0:
                kernel_size += 1
            smoothed = cv2.GaussianBlur(horizontal_proj.reshape(-1, 1), (1, kernel_size), 0).ravel()
            
            # Find local minima (potential boundaries)
            minima = []
            for i in range(1, len(smoothed) - 1):
                if smoothed[i] < smoothed[i-1] and smoothed[i] < smoothed[i+1]:
                    # Check if it's a significant minimum
                    left_peak = max(smoothed[max(0, i-5):i])
                    right_peak = max(smoothed[i+1:min(len(smoothed), i+6)])
                    depth = min(left_peak - smoothed[i], right_peak - smoothed[i])
                    
                    if depth > np.std(smoothed) * 0.3:  # Significant dip
                        minima.append((i, depth))
            
            # Choose the most prominent minimum in the middle region
            if minima:
                # Prefer minima closer to avatar center
                avatar_center_relative = avatar_center[1] - analysis_top
                
                best_minimum = None
                best_score = -1
                
                for idx, depth in minima:
                    # Score based on depth and proximity to expected position
                    distance_from_center = abs(idx - avatar_center_relative)
                    score = depth * 100 / (distance_from_center + 10)  # Higher score for deeper, closer minima
                    
                    if score > best_score:
                        best_score = score
                        best_minimum = idx
                
                if best_minimum is not None:
                    # Convert to absolute Y coordinate
                    boundary_y = analysis_top + best_minimum
                    return boundary_y
        
        # Method 2: Use avatar center as fallback if no clear boundary found
        return None
    
    def _upper_density_time_box_with_boundary(self, bgr: np.ndarray, row_box: List[int], 
                                             avatar_data: Dict, boundary_y: int, 
                                             debug_info: Dict = None) -> Optional[Tuple[int, int, int, int, float]]:
        """
        Density-based time box detection using explicit boundary
        
        Args:
            bgr: Original BGR image
            row_box: Card boundaries as [x, y, w, h]
            avatar_data: Avatar information
            boundary_y: Y-coordinate of name-time boundary
            debug_info: Optional debug information dictionary
            
        Returns:
            Tuple of (x, y, w, h, density_score) or None if not found
        """
        rx, ry, rw, rh = map(int, row_box)
        avatar_bbox = avatar_data["bbox"]
        
        # Calculate search region (above the boundary)
        avatar_right = avatar_bbox[0] + avatar_bbox[2]
        
        # Horizontal bounds: from avatar right edge to full card width
        search_left = avatar_right + 5
        search_right = rx + rw - 5
        
        # Vertical bounds: from card top to the detected boundary
        search_top = ry + 5
        search_bottom = boundary_y
        
        # Validate search region
        if search_left >= search_right or search_top >= search_bottom:
            return None
        if search_right - search_left < self.MIN_W_PX or search_bottom - search_top < 6:
            return None
        
        # Extract ROI and process using existing method
        roi = bgr[search_top:search_bottom, search_left:search_right]
        
        # Store debug information if requested
        if debug_info is not None:
            debug_info.update({
                'search_region': {
                    'left': search_left,
                    'right': search_right,
                    'top': search_top,
                    'bottom': search_bottom,
                    'width': search_right - search_left,
                    'height': search_bottom - search_top
                },
                'boundary_y': boundary_y,
                'boundary_method': 'explicit_detection'
            })
        
        # Continue with existing density analysis logic
        # (Using the same processing as _upper_density_time_box but with new boundaries)
        
        # Pre-process ROI for timestamp detection
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter
        filtered = cv2.bilateralFilter(gray, d=self.BILATERAL_D, 
                                      sigmaColor=self.BILATERAL_SIGMA_COLOR, 
                                      sigmaSpace=self.BILATERAL_SIGMA_SPACE)
        
        # Apply CLAHE for contrast enhancement
        clahe = cv2.createCLAHE(clipLimit=self.CLAHE_CLIP, 
                               tileGridSize=(self.CLAHE_GRID, self.CLAHE_GRID))
        enhanced = clahe.apply(filtered)
        
        # Apply adaptive threshold
        binary = cv2.adaptiveThreshold(enhanced, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                      cv2.THRESH_BINARY_INV, self.ADAPT_BLOCK, self.ADAPT_C)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
        
        # Filter and score contours for timestamp characteristics
        best_contour = None
        best_score = 0
        
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Basic size filtering for timestamp
            if w < 10 or h < 6 or w > roi.shape[1] * 0.8 or h > roi.shape[0] * 0.8:
                continue
            
            # Calculate score based on position and size (timestamps are usually small and right-aligned)
            position_score = x / roi.shape[1]  # Prefer right-side position
            size_score = 1.0 - (w * h) / (roi.shape[0] * roi.shape[1])  # Prefer smaller sizes
            aspect_ratio = w / h
            aspect_score = 1.0 if 1.5 <= aspect_ratio <= 6.0 else 0.5  # Timestamps have specific aspect ratios
            
            score = position_score * 0.3 + size_score * 0.4 + aspect_score * 0.3
            
            if score > best_score:
                best_score = score
                best_contour = contour
        
        if best_contour is None:
            return None
        
        # Get bounding box of best timestamp contour
        x, y, w, h = cv2.boundingRect(best_contour)
        
        # Convert to absolute coordinates
        tx0 = search_left + x
        ty = search_top + y
        tw = w
        th = h
        
        # Calculate density score
        contour_area = cv2.contourArea(best_contour)
        density_score = contour_area / max(1, w * h) * 100
        
        if debug_info is not None:
            debug_info.update({
                'detection_successful': True,
                'timestamp_bbox': [tx0, ty, tw, th],
                'density_score': density_score
            })
        
        return (int(tx0), int(ty), int(tw), int(th), float(density_score))
    
    def _find_name_timestamp_boundary(self, roi: np.ndarray, column_projection: np.ndarray) -> int:
        """
        Find the boundary between name (left, larger/darker) and timestamp (right, smaller/lighter)
        
        Args:
            roi: Original ROI image (BGR)
            column_projection: Column-wise text density projection
            
        Returns:
            Boundary column index separating names from timestamps
        """
        if roi.shape[1] < 20:  # Too narrow for boundary detection
            return roi.shape[1] // 2
            
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        
        # Analyze text characteristics across columns
        roi_width = roi.shape[1]
        window_size = max(5, roi_width // 10)  # Sliding window for analysis
        
        color_scores = []  # Lower = darker text, Higher = lighter text
        size_scores = []   # Text density variations
        
        for col in range(0, roi_width - window_size, 2):  # Step by 2 for efficiency
            window = gray[:, col:col + window_size]
            
            # Color analysis: average intensity of text pixels
            text_pixels = window[window < 200]  # Assume text is darker than 200
            avg_intensity = text_pixels.mean() if len(text_pixels) > 0 else 255
            color_scores.append(avg_intensity)
            
            # Size analysis: variation in text density
            col_densities = column_projection[col:col + window_size]
            size_variation = col_densities.std() if len(col_densities) > 0 else 0
            size_scores.append(size_variation)
        
        if len(color_scores) < 3:
            return roi_width // 2
            
        # Find the transition point where text changes from dark to light
        color_scores = np.array(color_scores)
        
        # Look for the point where intensity increases significantly (dark → light)
        intensity_changes = np.diff(color_scores)
        
        # Find the largest positive change (transition to lighter text)
        max_change_idx = np.argmax(intensity_changes)
        
        # Convert back to column coordinate
        boundary_col = max_change_idx * 2 + window_size // 2
        
        # Ensure boundary is reasonable (not too close to edges)
        boundary_col = max(roi_width // 4, min(roi_width * 3 // 4, boundary_col))
        
        return boundary_col

    def create_time_box_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive visualization showing detected time boxes with purple rectangles
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for visualization
            
        Returns:
            Filename of generated visualization
        """
        # Load original image
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image for visualization: {image_path}")
            return None
        
        print(f"🎨 Creating comprehensive time box visualization...")
        
        # Get detection results
        enhanced_cards, detection_info = self.detect_time_boundaries(image_path)
        
        # Create visualization overlay
        result = img.copy()
        
        # Track visualization statistics
        times_visualized = 0
        search_regions_drawn = 0
        
        # Draw comprehensive visualization with all elements
        for card in enhanced_cards:
            # Draw existing card and avatar boundaries (from CardBoundaryDetector)
            card_x, card_y, card_w, card_h = card["bbox"]
            avatar_data = card["avatar"]
            avatar_bbox = avatar_data["bbox"]
            avatar_center = avatar_data["center"]
            
            # Draw card boundary (blue rectangle)
            cv2.rectangle(result, (card_x, card_y), (card_x + card_w, card_y + card_h), (255, 0, 0), 2)
            
            # Draw avatar boundary (green rectangle)  
            avatar_x, avatar_y, avatar_w, avatar_h = avatar_bbox
            cv2.rectangle(result, (avatar_x, avatar_y), (avatar_x + avatar_w, avatar_y + avatar_h), (0, 255, 0), 2)
            
            # Draw avatar center (red circle)
            cv2.circle(result, tuple(avatar_center), 5, (0, 0, 255), -1)
            
            # Draw horizontal divider line through avatar center
            avatar_center_y = avatar_center[1]
            cv2.line(result, (card_x, avatar_center_y), (card_x + card_w, avatar_center_y), (255, 255, 0), 1)
            
            # Draw name boundary if detected (orange rectangle)
            if card.get("name_boundary"):
                name_bbox = card["name_boundary"]["bbox"]
                name_x, name_y, name_w, name_h = name_bbox
                cv2.rectangle(result, (name_x, name_y), (name_x + name_w, name_y + name_h), (0, 165, 255), 2)
            
            # Draw time box if detected (purple rectangle)
            if card.get("time_box"):
                time_bbox = card["time_box"]["bbox"]
                time_x, time_y, time_w, time_h = time_bbox
                density_score = card["time_box"]["density_score"]
                
                # Draw time box boundary (purple rectangle)
                cv2.rectangle(result, (time_x, time_y), (time_x + time_w, time_y + time_h), (128, 0, 128), 2)
                
                # Draw density score
                cv2.putText(result, f"{density_score:.0f}", (time_x, time_y - 5), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.4, (128, 0, 128), 1)
                
                times_visualized += 1
                
                # Optionally draw search region (light gray rectangle) for debugging
                if card["time_box"].get("search_region_info"):
                    search_info = card["time_box"]["search_region_info"]
                    avatar_right = search_info["avatar_right"]
                    avatar_center_y = search_info["avatar_center_y"]
                    card_width_frac = search_info["card_width_frac"]
                    card_width = search_info["card_width"]
                    
                    # Calculate upper region search bounds
                    search_left = avatar_right + 15  # AVATAR_TIME_MARGIN
                    search_right = card_x + int(card_width * card_width_frac)
                    search_top = card_y + 5  # Small margin from card top
                    search_bottom = avatar_center_y  # Above divider line
                    
                    cv2.rectangle(result, (search_left, search_top), (search_right, search_bottom), 
                                (192, 192, 192), 1)  # Light gray
                    search_regions_drawn += 1
            
            # Draw card ID
            cv2.putText(result, f"Card {card['card_id']}", (card_x + 10, card_y + 25), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # Add comprehensive legend and statistics
        legend_y = 30
        cv2.putText(result, "Complete WeChat Card Analysis Results:", (10, legend_y), 
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(result, f"Blue=Cards  Green=Avatars  Red=Centers  Yellow=Dividers  Orange=Names  Purple=Times",
                   (10, legend_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add detection statistics
        success_rate = detection_info.get("detection_success_rate", 0) * 100
        cv2.putText(result, f"Times Detected: {times_visualized}/{detection_info['total_cards_processed']} cards ({success_rate:.1f}%)",
                   (10, legend_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(result, f"Method: {detection_info['detection_method']} | Panel Right: {detection_info['panel_right_boundary']}px",
                   (10, legend_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Generate output filename with timestamp-first format
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_06_Complete_Analysis_{times_visualized}times_{len(enhanced_cards)}cards.png"
        
        # Save visualization
        cv2.imwrite(output_path, result)
        print(f"✅ Complete card analysis visualization saved: {output_path}")
        print(f"   📊 Visualization stats: {times_visualized} time boxes, {search_regions_drawn} search regions drawn")
        
        return os.path.basename(output_path)


    def create_debug_visualization(self, image_path: str, output_path: str = None) -> str:
        """
        Create comprehensive debug visualization showing all detection details for every card
        
        Args:
            image_path: Path to screenshot image
            output_path: Optional output path for debug visualization
            
        Returns:
            Filename of generated debug visualization
        """
        import matplotlib.pyplot as plt
        import matplotlib.patches as patches
        from matplotlib.gridspec import GridSpec
        
        # Load original image
        img_bgr = cv2.imread(image_path)
        if img_bgr is None:
            print(f"❌ Failed to load image for debug visualization: {image_path}")
            return None
        
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        print(f"🔍 Creating debug visualization with detailed analysis...")
        
        # Enable debug collection and get detection results
        self._collect_debug = True
        enhanced_cards, detection_info = self.detect_time_boundaries(image_path)
        self._collect_debug = False
        
        # Separate successful and failed detections
        successful_cards = [card for card in enhanced_cards if card.get("time_box")]
        failed_cards = [card for card in enhanced_cards if not card.get("time_box")]
        
        print(f"  📊 Debug Analysis: {len(successful_cards)} successful, {len(failed_cards)} failed detections")
        
        # Create comprehensive matplotlib figure
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(4, 6, figure=fig, hspace=0.3, wspace=0.3)
        
        # Main image with all search regions
        ax_main = fig.add_subplot(gs[0:2, 0:3])
        ax_main.imshow(img_rgb)
        ax_main.set_title(f'Time Detection Overview - {len(enhanced_cards)} Cards Analyzed', fontsize=14, fontweight='bold')
        ax_main.set_xlabel(f'✅ {len(successful_cards)} Successful  ❌ {len(failed_cards)} Failed')
        
        # Draw search regions for ALL cards
        for i, card in enumerate(enhanced_cards):
            debug_info = card.get("time_detection_debug", {})
            search_region = debug_info.get("search_region", {})
            
            if search_region:
                left, right = search_region['left'], search_region['right']
                top, bottom = search_region['top'], search_region['bottom']
                
                # Color code: green for success, red for failure
                color = 'lime' if card.get("time_box") else 'red'
                alpha = 0.3 if card.get("time_box") else 0.5
                
                # Draw search region rectangle
                rect = patches.Rectangle((left, top), right-left, bottom-top, 
                                       linewidth=2, edgecolor=color, facecolor=color, alpha=alpha)
                ax_main.add_patch(rect)
                
                # Add card number
                ax_main.text(left+5, top+15, f'Card {card["card_id"]}', 
                           fontsize=10, color='white', fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=color, alpha=0.8))
        
        ax_main.set_xlim(0, img_rgb.shape[1])
        ax_main.set_ylim(img_rgb.shape[0], 0)
        
        # ROI previews for first few cards
        roi_cols = 3
        roi_start_row = 2
        for i, card in enumerate(enhanced_cards[:6]):  # Show first 6 cards
            debug_info = card.get("time_detection_debug", {})
            roi_original = debug_info.get("roi_original")
            
            if roi_original is not None:
                ax_roi = fig.add_subplot(gs[roi_start_row + i//roi_cols, i%roi_cols])
                roi_rgb = cv2.cvtColor(roi_original, cv2.COLOR_BGR2RGB) if len(roi_original.shape) == 3 else roi_original
                ax_roi.imshow(roi_rgb, cmap='gray' if len(roi_original.shape) == 2 else None)
                
                success = card.get("time_box") is not None
                status = "✅" if success else "❌"
                ax_roi.set_title(f'{status} Card {card["card_id"]} ROI', fontsize=10)
                ax_roi.set_xticks([])
                ax_roi.set_yticks([])
        
        # Binary processing results
        for i, card in enumerate(enhanced_cards[:3]):  # Show first 3 cards' processing
            debug_info = card.get("time_detection_debug", {})
            roi_binary = debug_info.get("roi_binary")
            
            if roi_binary is not None:
                ax_bin = fig.add_subplot(gs[roi_start_row + 1, i + 3])
                ax_bin.imshow(roi_binary, cmap='gray')
                ax_bin.set_title(f'Card {card["card_id"]} Binary', fontsize=10)
                ax_bin.set_xticks([])
                ax_bin.set_yticks([])
        
        # Column projection histograms for failed cards
        failed_count = 0
        for card in failed_cards:
            if failed_count >= 3:  # Show max 3 failed cards
                break
                
            debug_info = card.get("time_detection_debug", {})
            column_projection = debug_info.get("column_projection")
            stats = debug_info.get("statistics", {})
            
            if column_projection is not None:
                ax_hist = fig.add_subplot(gs[3, failed_count])
                ax_hist.bar(range(len(column_projection)), column_projection, alpha=0.7, color='red')
                
                # Draw threshold lines
                if 'threshold_primary' in stats:
                    ax_hist.axhline(y=stats['threshold_primary'], color='orange', linestyle='--', 
                                  label=f'Primary: {stats["threshold_primary"]:.1f}')
                if stats.get('threshold_fallback'):
                    ax_hist.axhline(y=stats['threshold_fallback'], color='red', linestyle=':', 
                                  label=f'Fallback: {stats["threshold_fallback"]:.1f}')
                
                ax_hist.set_title(f'❌ Card {card["card_id"]} Column Projection', fontsize=10)
                ax_hist.set_xlabel('Column Position')
                ax_hist.set_ylabel('Pixel Density')
                ax_hist.legend(fontsize=8)
                ax_hist.grid(True, alpha=0.3)
                
                failed_count += 1
        
        # Add overall statistics
        fig.suptitle(f'Time Detection Debug Analysis - {os.path.basename(image_path)}', fontsize=16, fontweight='bold')
        
        # Generate output filename
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"pic/screenshots/{timestamp}_06_Debug_TimeDetection_{len(successful_cards)}success_{len(failed_cards)}failed.png"
        
        # Save debug visualization
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Debug visualization saved: {output_path}")
        print(f"   📊 Analysis complete: {len(successful_cards)}/{len(enhanced_cards)} successful detections")
        
        # Print detailed debug info for failed cards
        for card in failed_cards:
            debug_info = card.get("time_detection_debug", {})
            stats = debug_info.get("statistics", {})
            search_region = debug_info.get("search_region", {})
            
            print(f"   ❌ Card {card['card_id']} Debug Info:")
            print(f"      Search region: {search_region.get('width', 0)}×{search_region.get('height', 0)}px")
            print(f"      Max projection: {stats.get('max_projection', 0)}")
            primary_thr = stats.get('threshold_primary', 0) or 0
            fallback_thr = stats.get('threshold_fallback', 0) or 0
            print(f"      Threshold: {primary_thr:.1f} (primary), {fallback_thr:.1f} (fallback)")
            print(f"      Reason: {debug_info.get('failure_reason', 'unknown')}")
        
        return os.path.basename(output_path)


# =============================================================================
# COMPLETE CARD PROCESSING PIPELINE
# =============================================================================

def fcomplete_card_analysis(image_path: str, debug_mode: bool = True) -> Optional[Dict]:
    """
    Complete 6-phase card processing pipeline with proper data flow
    
    Pipeline Order (Fixed):
    1. BoundaryCoordinator - Detect message card width boundaries
    2. RightBoundaryDetector - Enhanced right boundary detection (internal)  
    3. CardAvatarDetector - Detect avatar positions within cards
    4. CardBoundaryDetector - Detect individual card boundaries
    6. TimeBoxDetector - Detect timestamp boundaries (runs BEFORE names)
    5. ContactNameBoundaryDetector - Detect contact name boundaries (uses time data)
    
    Args:
        image_path: Path to WeChat screenshot image
        debug_mode: Enable debug visualization generation
        
    Returns:
        Dictionary with complete analysis results from all 6 phases or None if failed
    """
    try:
        print(f"🚀 Starting Complete Card Analysis Pipeline: {os.path.basename(image_path)}")
        print("=" * 80)
        
        # Initialize image and coordinate context
        import cv2
        img = cv2.imread(image_path)
        if img is None:
            print(f"❌ Failed to load image: {image_path}")
            return None
        
        img_height, img_width = img.shape[:2]
        print(f"📐 Image dimensions: {img_width}×{img_height}")
        
        # Initialize unified coordinate context
        coord_context = cWeChatCoordinateContext(image_path, (img_width, img_height))
        print("📊 Initialized unified coordinate context")
        
        # Initialize all processors
        width_detector = cBoundaryCoordinator(debug_mode=debug_mode)
        avatar_detector = cCardAvatarDetector(debug_mode=debug_mode) 
        boundary_detector = cCardBoundaryDetector(debug_mode=debug_mode)
        time_detector = cTimeBoxDetector(debug_mode=debug_mode)
        name_detector = cContactNameBoundaryDetector(debug_mode=debug_mode)
        
        results = {
            'pipeline_version': '6-phase-complete-with-coordinates',
            'image_processed': os.path.basename(image_path),
            'processing_successful': False,
            'coordinate_context': None  # Will be populated at the end
        }
        
        # Step 1: Width Detection
        print("📏 Phase 1: Width Detection")
        width_result = width_detector.detect_width(image_path)
        if width_result:
            left, right, width = width_result
            results['phase1_width_detected'] = width
            results['phase1_width_boundaries'] = {'left': left, 'right': right}
            
            # Add global conversation area to coordinate context
            conversation_bbox = [left, 0, width, img_height]
            coord_context.add_global_boundary(
                "conversation_area", 
                conversation_bbox, 
                "PHASE_1", 
                0.9
            )
            print(f"  ✅ Width detected: {width}px (left: {left}, right: {right})")
            print(f"  📊 Added global conversation area to coordinate context")
        else:
            results['phase1_width_detected'] = None
            results['phase1_width_boundaries'] = None
            print(f"  ❌ Width detection failed")
            
        # Step 3: Avatar Detection  
        print("👤 Phase 3: Avatar Detection")
        avatars, avatar_info = avatar_detector.detect_avatars(image_path)
        results['phase3_avatars_detected'] = len(avatars)
        results['phase3_avatar_list'] = avatars
        results['phase3_avatar_detection_info'] = avatar_info
        print(f"  ✅ Avatars detected: {len(avatars)}")
        
        # Step 4: Card Boundary Detection
        print("📋 Phase 4: Card Boundary Detection")
        cards, card_info = boundary_detector.detect_cards(image_path)
        results['phase4_cards_detected'] = len(cards)
        results['phase4_card_list'] = cards  
        results['phase4_card_detection_info'] = card_info
        
        # Populate coordinate context with card data
        for card in cards:
            card_id = card["card_id"]
            card_bbox = card["bbox"]
            
            # Add card to coordinate context
            coord_context.add_card(card_id, card_bbox, "PHASE_4", 0.95)
            
            # Add avatar component
            if "avatar" in card:
                avatar_data = card["avatar"]
                avatar_bbox = avatar_data["bbox"]
                avatar_center = avatar_data["center"]
                
                coord_context.add_component(
                    card_id, "avatar", avatar_bbox, "STEP_3", 0.98,
                    ocr_suitable=False, center=avatar_center,
                    avatar_id=avatar_data.get("avatar_id", card_id),
                    area=avatar_data.get("area", avatar_bbox[2] * avatar_bbox[3])
                )
        
        print(f"  ✅ Cards detected: {len(cards)}")
        print(f"  📊 Added {len(cards)} cards to coordinate context")
        
        # Phase 6: Time Box Detection (RUNS BEFORE Phase 5)
        print("🕒 Phase 6: Time Box Detection")
        cards_with_times, time_info = time_detector.detect_time_boundaries(image_path)
        results['phase6_times_detected'] = len([c for c in cards_with_times if 'time_box' in c])
        results['phase6_cards_with_times'] = cards_with_times
        results['phase6_time_detection_info'] = time_info
        
        # Add timestamp components to coordinate context
        for card in cards_with_times:
            if 'time_box' in card:
                card_id = card["card_id"]
                time_data = card["time_box"]
                time_bbox = time_data["bbox"]
                
                coord_context.add_component(
                    card_id, "timestamp", time_bbox, "PHASE_6", 
                    time_data.get("density_score", 0.8),
                    ocr_suitable=True, expected_content="timestamp",
                    density_score=time_data.get("density_score", 0.8),
                    detection_method=time_data.get("detection_method", "upper_region_density")
                )
        
        print(f"  ✅ Times detected: {results['phase6_times_detected']}/{len(cards_with_times)}")
        print(f"  📊 Added {results['phase6_times_detected']} timestamp regions to coordinate context")
        
        # Phase 5: Contact Name Detection (USES time data from Phase 6)
        print("📝 Phase 5: Contact Name Detection")
        cards_with_names, name_info = name_detector.detect_name_boundaries(
            image_path, 
            cards_with_times=cards_with_times,  # Use Phase 6 output as input
            debug_mode=debug_mode
        )
        results['phase5_names_detected'] = len([c for c in cards_with_names if 'name_boundary' in c])
        results['phase5_cards_with_names'] = cards_with_names
        results['phase5_name_detection_info'] = name_info
        
        # Add contact name components to coordinate context
        for card in cards_with_names:
            if 'name_boundary' in card:
                card_id = card["card_id"]
                name_data = card["name_boundary"]
                name_bbox = name_data["bbox"]
                
                coord_context.add_component(
                    card_id, "contact_name", name_bbox, "STEP_5",
                    name_data.get("confidence", 0.8),
                    ocr_suitable=True, expected_content="contact_name",
                    detection_method=name_data.get("detection_method", "grey_timestamp_edge")
                )
        
        print(f"  ✅ Names detected: {results['phase5_names_detected']}/{len(cards_with_names)}")
        print(f"  📊 Added {results['phase5_names_detected']} contact name regions to coordinate context")
        
        # Validate coordinate context
        validation_result = coord_context.validate_coordinates()
        if validation_result["errors"]:
            print(f"⚠️  Coordinate validation errors: {len(validation_result['errors'])}")
            for error in validation_result["errors"][:3]:  # Show first 3 errors
                print(f"    - {error}")
        else:
            print("✅ All coordinates validated successfully")
        
        # Final Results Summary with Coordinate Context
        results['processing_successful'] = True
        results['final_enhanced_cards'] = cards_with_names  # Most complete card data
        results['coordinate_context'] = coord_context.to_dict()  # Complete coordinate structure
        results['coordinate_validation'] = validation_result
        
        # Add OCR-ready extraction regions for easy access
        results['ocr_regions'] = {
            'contact_names': coord_context.extract_all_regions('contact_names'),
            'timestamps': coord_context.extract_all_regions('timestamps'), 
            'messages': coord_context.extract_all_regions('messages'),
            'total_regions': (
                len(coord_context.extract_all_regions('contact_names')) +
                len(coord_context.extract_all_regions('timestamps')) +
                len(coord_context.extract_all_regions('messages'))
            )
        }
        
        print("=" * 80)
        print("🎯 Pipeline Summary:")
        print(f"  Phase 1 - Width: {results['phase1_width_detected']}px" if results['phase1_width_detected'] else "  Phase 1 - Width: Failed")
        print(f"  Phase 3 - Avatars: {results['phase3_avatars_detected']}")
        print(f"  Phase 4 - Cards: {results['phase4_cards_detected']}")
        print(f"  Phase 6 - Times: {results['phase6_times_detected']}/{len(cards_with_times)}")
        print(f"  Phase 5 - Names: {results['phase5_names_detected']}/{len(cards_with_names)}")
        print("📊 Coordinate Context Summary:")
        print(f"  Total Cards: {coord_context.get_card_count()}")
        print(f"  OCR Regions: {results['ocr_regions']['total_regions']} (Names: {len(results['ocr_regions']['contact_names'])}, Times: {len(results['ocr_regions']['timestamps'])})")
        print(f"  Validation: {'✅ PASSED' if not validation_result['errors'] else '⚠️ ISSUES'}")
        print("✅ Complete Pipeline Analysis with Coordinate Context SUCCESSFUL")
        
        return results
        
    except Exception as e:
        print(f"❌ Complete pipeline analysis failed: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# ENHANCED API FUNCTIONS FOR LIVE PROCESSING
# =============================================================================

# Import screenshot processing functions from consolidated screenshot_processor module
if SCREENSHOT_AVAILABLE:
    capture_and_process_screenshot = m_screenshot_processor.fcapture_and_process_screenshot
    process_screenshot_file = m_screenshot_processor.fprocess_screenshot_file
    process_current_wechat_window = m_screenshot_processor.fprocess_current_wechat_window
    get_live_card_analysis = m_screenshot_processor.fget_live_card_analysis
else:
    # Legacy inline implementations for backward compatibility
    def capture_and_process_screenshot(output_dir: str = "pic/screenshots", 
                                      custom_filename: str = None) -> Optional[Tuple[str, Dict]]:
        """
        Capture a fresh WeChat screenshot and process it for card analysis
        
        Args:
            output_dir: Directory to save screenshot and visualizations
            custom_filename: Custom filename for screenshot, auto-generated if None
            
        Returns:
            Tuple of (screenshot_path, analysis_results) or None if failed
            
        Usage:
            screenshot_path, results = capture_and_process_screenshot()
            if results:
                print(f"Found {results['cards_detected']} cards")
        """
        if not SCREENSHOT_AVAILABLE:
            print("❌ Screenshot capture not available. Install required module.")
            return None
            
        try:
            print("\n🎯 Live WeChat Screenshot & Card Processing")
            print("=" * 50)
            
            # Step 1: Capture fresh screenshot
            print("\n📸 Phase 1: Capturing WeChat screenshot...")
            screenshot_path = m_screenshot_processor.fcapture_screenshot(output_dir=output_dir, filename=custom_filename)
            
            if not screenshot_path:
                print("❌ Failed to capture screenshot")
                return None
                
            print(f"✅ Screenshot captured: {os.path.basename(screenshot_path)}")
            
            # Step 2: Process with card analysis
            print("\n🔍 Phase 2: Processing card analysis...")
            results = process_screenshot_file(screenshot_path)
            
            if results:
                print(f"\n✅ Analysis complete:")
                print(f"   Width: {results.get('width_detected')}px")  
                print(f"   Avatars: {results.get('avatars_detected')}")
                print(f"   Cards: {results.get('cards_detected')}")
                return screenshot_path, results
            else:
                print("❌ Analysis failed")
                return None
                
        except Exception as e:
            print(f"❌ Error in capture_and_process_screenshot: {e}")
            return None

    def process_screenshot_file(image_path: str) -> Optional[Dict]:
        """
        Process a screenshot file using complete 6-phase pipeline
        
        Args:
            image_path: Path to screenshot file to analyze
            
        Returns:
            Dictionary with complete analysis results from all 6 phases or None if failed
        """
        # Use the complete pipeline with debug_mode=False for production processing
        return fcomplete_card_analysis(image_path, debug_mode=False)

    def process_current_wechat_window() -> Optional[Dict]:
        """
        Convenience function: Capture current WeChat window and analyze cards
        
        Returns:
            Dictionary with complete analysis results or None if failed
            
        Usage:
            results = process_current_wechat_window()
            if results:
                for i, card in enumerate(results['card_list'], 1):
                    print(f"Card {i}: {card['width']}×{card['height']}px")
        """
        result = capture_and_process_screenshot()
        if result:
            screenshot_path, analysis = result
            return analysis
        return None

    def get_live_card_analysis(include_visualizations: bool = True) -> Optional[Tuple[Dict, Dict]]:
        """
        Get comprehensive live card analysis with optional visualizations
        
        Args:
            include_visualizations: Whether to generate visualization files
            
        Returns:
            Tuple of (analysis_results, visualization_paths) or None if failed
        """
        result = capture_and_process_screenshot()
        if not result:
            return None
            
        screenshot_path, analysis = result
        
        visualization_paths = {}
        if include_visualizations:
            try:
                # Generate all visualizations (debug_mode=False for internal processing)
                width_detector = cBoundaryCoordinator(debug_mode=False)  
                avatar_detector = cCardAvatarDetector(debug_mode=False)
                boundary_detector = cCardBoundaryDetector(debug_mode=False)
                
                # Width visualization
                if analysis.get('width_detected'):
                    width_vis = width_detector.create_width_visualization(screenshot_path)
                    visualization_paths['width'] = width_vis
                    
                # Avatar visualization  
                if analysis.get('avatars_detected'):
                    avatar_vis = avatar_detector.create_visualization(screenshot_path) 
                    visualization_paths['avatars'] = avatar_vis
                    
                # Card boundary visualization
                if analysis.get('cards_detected'):
                    card_vis = boundary_detector.create_card_visualization(screenshot_path)
                    visualization_paths['cards'] = card_vis
                    
            except Exception as e:
                print(f"⚠️  Visualization generation failed: {e}")
                
        return analysis, visualization_paths


def f_save_coordinates_to_context(image_path: str, analysis_results: Dict) -> None:
    """
    Save coordinate analysis results to modules/wechat_ctx.json for software assistance and human inspection
    
    Args:
        image_path: Path to the processed screenshot
        analysis_results: Complete analysis results from the 6-phase pipeline
    """
    try:
        import json
        import os
        from datetime import datetime
        
        # Create context file path in the modules directory
        context_file = os.path.join(os.path.dirname(__file__), 'wechat_ctx.json')
        
        # Extract coordinate context if available
        coord_context = analysis_results.get('coordinate_context', {})
        image_metadata = coord_context.get('image_metadata', {})
        global_boundaries = coord_context.get('global_boundaries', {})
        cards_data = coord_context.get('cards', [])
        ocr_regions = coord_context.get('ocr_extraction_regions', {})
        
        # Extract image dimensions
        dimensions = image_metadata.get('dimensions', {})
        
        # Extract conversation area from global boundaries
        conversation_area = global_boundaries.get('conversation_area', {})
        conversation_bbox = conversation_area.get('bbox', [0, 0, 0, 0])
        
        # Build context data structure
        context_data = {
            "last_updated": datetime.now().isoformat(),
            "screenshot_path": os.path.basename(image_path),
            "processing_info": {
                "source_image": image_metadata.get('source_image', ''),
                "full_path": image_metadata.get('full_path', ''),
                "processing_timestamp": image_metadata.get('processing_timestamp', '')
            },
            "image_dimensions": {
                "width": dimensions.get('width'),
                "height": dimensions.get('height')
            },
            "detection_statistics": {
                "cards_detected": len(cards_data),
                "avatars_detected": len([card for card in cards_data if 'avatar' in card.get('components', {})]),
                "names_detected": len(ocr_regions.get('contact_names', [])),
                "times_detected": len(ocr_regions.get('timestamps', []))
            },
            "coordinates": {
                "conversation_area": {
                    "x": conversation_bbox[0] if len(conversation_bbox) >= 4 else None,
                    "y": conversation_bbox[1] if len(conversation_bbox) >= 4 else None,
                    "width": conversation_bbox[2] if len(conversation_bbox) >= 4 else None,
                    "height": conversation_bbox[3] if len(conversation_bbox) >= 4 else None,
                    "confidence": conversation_area.get('confidence')
                },
                "cards": [
                    {
                        "id": card.get('card_id', i + 1),
                        "region": {
                            "x": card.get('card_region', {}).get('bbox', [0, 0, 0, 0])[0],
                            "y": card.get('card_region', {}).get('bbox', [0, 0, 0, 0])[1],
                            "width": card.get('card_region', {}).get('bbox', [0, 0, 0, 0])[2],
                            "height": card.get('card_region', {}).get('bbox', [0, 0, 0, 0])[3],
                            "confidence": card.get('card_region', {}).get('confidence')
                        },
                        "components": {
                            "has_avatar": 'avatar' in card.get('components', {}),
                            "avatar": card.get('components', {}).get('avatar', {}),
                            "has_name": 'contact_name' in card.get('components', {}),
                            "contact_name": card.get('components', {}).get('contact_name', {}),
                            "has_time": 'timestamp' in card.get('components', {}),
                            "timestamp": card.get('components', {}).get('timestamp', {})
                        }
                    }
                    for i, card in enumerate(cards_data)
                ],
                "ocr_regions": {
                    "contact_names": ocr_regions.get('contact_names', []),
                    "timestamps": ocr_regions.get('timestamps', []),
                    "messages": ocr_regions.get('messages', []),
                    "avatars": ocr_regions.get('avatars', [])
                }
            }
        }
        
        # Save to JSON file (overwrites previous data)
        with open(context_file, 'w', encoding='utf-8') as f:
            json.dump(context_data, f, indent=2, ensure_ascii=False)
            
        print(f"💾 Coordinates saved to modules/wechat_ctx.json")
        
    except Exception as e:
        print(f"⚠️  Failed to save coordinates to context file: {e}")
        import traceback
        traceback.print_exc()


# =============================================================================
# MANUAL CODE TESTING
# =============================================================================
if __name__ == "__main__":
    print("=" * 60)
    print("Manual Code Testing - CARD PROCESSING MODULE")
    print("=" * 60)
    print("🔍 [DEBUG] Smoke test ENTRY")
    
    try:
        # Simply instantiate each class in order
        print("   🔧 Testing cWeChatCoordinateContext...")
        coord_context = cWeChatCoordinateContext("test_image.png", (800, 1200))
        print("   ✅ cWeChatCoordinateContext instantiated successfully")
        
        print("   🔧 Testing cBoundaryCoordinator...")
        boundary_coordinator = cBoundaryCoordinator()
        print("   ✅ cBoundaryCoordinator instantiated successfully")
        
        print("   🔧 Testing cCardAvatarDetector...")
        avatar_detector = cCardAvatarDetector()
        print("   ✅ cCardAvatarDetector instantiated successfully")
        
        print("   🔧 Testing cCardBoundaryDetector...")
        card_detector = cCardBoundaryDetector()
        print("   ✅ cCardBoundaryDetector instantiated successfully")
        
        print("   🔧 Testing cContactNameBoundaryDetector...")
        name_detector = cContactNameBoundaryDetector()
        print("   ✅ cContactNameBoundaryDetector instantiated successfully")
        
        print("   🔧 Testing cTimeBoxDetector...")
        time_detector = cTimeBoxDetector()
        print("   ✅ cTimeBoxDetector instantiated successfully")
        
        print("   🔧 Testing cCoordinateConverter...")
        converter = cCoordinateConverter()
        print("   ✅ cCoordinateConverter instantiated successfully")
        
        print("🏁 [DEBUG] Smoke test PASSED")
        
    except Exception as e:
        print(f"   ❌ [ERROR] Smoke test FAILED: {str(e)}")
        print("🏁 [DEBUG] Smoke test FAILED")