#!/usr/bin/env python3
"""
modules/m_photo_processor.py

HIGH-CONTRAST PHOTO CREATION TOOL

📋 PURPOSE:
Creates high-contrast photos from WeChat screenshots using levels adjustment for boundary
enhancement analysis. This tool is part of the WeChat card processing pipeline,
specifically designed to generate exactly two enhanced images with different contrast settings
optimized for Left and Right boundary detection algorithms.

🎯 INTEGRATION CONTEXT:
Integrates with the WeChat automation bot's 6-stage processing pipeline:
- Stage 1: Screenshot capture (handled by screenshot finder tool)
- Stage 2: Image preprocessing (THIS MODULE - high-contrast photo creation)
- Stages 3-6: OCR, detection, analysis, and response generation

📁 OUTPUT FILES:
- YYYYMMDD_HHMMSS_02_diag_photoshop_levels_gamma_Left.png  (Shadow=33, Midtone=1.1, Highlight=55)
- YYYYMMDD_HHMMSS_02_diag_photoshop_levels_gamma_Right.png (Shadow=16, Midtone=1.5, Highlight=126)

🔧 ALGORITHM:
Simple high-contrast photo creation process:
1. Convert color image to grayscale
2. Apply levels adjustment: clip to [input_black, input_white], normalize, apply gamma
3. Output range mapping (normalizes to 0-255 pixel values)

⚙️ DEPENDENCIES:
- opencv-python: Image I/O and array manipulation
- numpy: Mathematical operations and array processing
- m_screenshot_finder_tool: WeChat screenshot discovery and selection
"""

import cv2
import numpy as np
import os
import time
from datetime import datetime
from typing import Optional, Tuple

# Import screenshot capture functionality from m_screenshot_processor
try:
    from modules.m_screenshot_processor import fcapture_screenshot
except ImportError:
    # For direct execution, try relative import
    from m_screenshot_processor import fcapture_screenshot

# =============================================================================
# TOOL 1: HIGH-CONTRAST PHOTO CREATOR
# INPUT: Raw screenshot (PNG), enhancement parameters
# OUTPUT: High-contrast grayscale PNG files with Left/Right boundary optimization
# =============================================================================
# This standalone tool creates high-contrast photos from WeChat screenshots
# using levels adjustment with grayscale conversion for boundary enhancement.
# Generates exactly two output files with different contrast settings:
# - Left boundary: Conservative enhancement (Shadow=33, Midtone=1.1, Highlight=55)
# - Right boundary: Aggressive enhancement (Shadow=16, Midtone=1.5, Highlight=126)
# =============================================================================

class c_tool_Create_Highcontrast_Photo:
    """
    High-contrast photo creator for Left/Right boundary enhancement.
    
    📋 PURPOSE:
    Creates high-contrast photos from WeChat screenshots to enhance visibility of boundaries
    for improved boundary detection accuracy. Designed specifically for the WeChat automation
    bot's card processing pipeline with optimized contrast settings for different boundary types.
    
    🔧 ALGORITHM:
    High-Contrast Photo Creation (3-step process):
    1. Convert color image to grayscale
    2. Apply levels adjustment: clip to [input_black, input_white], normalize, apply gamma
    3. Output range mapping: scales result back to 0-255 pixel range
    
    📊 PARAMETER OPTIMIZATION:
    Left Boundary:  Shadow=33, Midtone=1.1, Highlight=55  (subtle enhancement)
    Right Boundary: Shadow=16, Midtone=1.5, Highlight=126 (aggressive enhancement)
    Dual Boundary:  Shadow=30, Midtone=1.0, Highlight=43  (balanced enhancement)
    
    These values were empirically determined for optimal WeChat interface boundary detection.
    
    🎨 OUTPUT FILES:
    - Generates timestamped PNG files with processed high-contrast photos
    - Uses standardized filename format for pipeline integration
    - Format: {timestamp}_02_diag_photoshop_levels_gamma{suffix}.png
    """
    
    def __init__(self):
        """Initialize high-contrast photo creator with timestamp."""
        print("🔍 [DEBUG] c_tool_Create_Highcontrast_Photo.__init__() ENTRY")
        
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        print(f"   ⏰ Timestamp generated: {self.timestamp}")
        
        self.output_dir = "pic/screenshots"
        print(f"   📁 Output directory: {self.output_dir}")
        
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"   ✅ Directory ensured: {os.path.abspath(self.output_dir)}")
        print("🏁 [DEBUG] c_tool_Create_Highcontrast_Photo.__init__() EXIT")
    
    def _apply_levels_adjustment(self, image, input_black, input_white, gamma, suffix):
        """
        Core levels adjustment method for creating high-contrast photos.
        
        📌 INPUT CONTRACT:
        - image: np.ndarray - Input image array (BGR format from cv2.imread)
        - input_black: int (0-255) - Shadow clipping point (darker values become black)
        - input_white: int (0-255) - Highlight clipping point (brighter values become white)
        - gamma: float (0.1-9.99) - Midtone adjustment (1.0=no change, <1=darker, >1=brighter)
        - suffix: str - File suffix for output image (e.g., "_Left", "_Right")
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (processed image array, filepath)
        - Failure: (None, None) - Returns (None, None) if input image is invalid
        - Side Effects: Saves processed image to pic/screenshots/ with timestamped filename
        - Format: {timestamp}_02_diag_photoshop_levels_gamma{suffix}.png
        
        🔧 ALGORITHM:
        1. Convert color image to grayscale
        2. Apply levels: clip to [input_black, input_white], normalize, apply gamma
        3. Convert back to 0-255 range
        """
        print(f"🔍 [DEBUG] _apply_levels_adjustment() ENTRY")
        print(f"   ⚙️ Parameters: shadow={input_black}, gamma={gamma}, highlight={input_white}, suffix={suffix}")
        
        start_time = time.time()
        
        # Validate input image
        if image is None or image.size == 0:
            print(f"   ❌ [ERROR] Invalid input image: {image}")
            print(f"🏁 [DEBUG] _apply_levels_adjustment() EXIT (FAILED)")
            return None, None
        
        print(f"   ✅ Input image validated: {image.shape} shape, {image.dtype} dtype")
        
        # Step 1: Convert color to grayscale
        print(f"   🔧 Step 1: Converting to grayscale...")
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            print(f"   ✅ Converted BGR to grayscale: {gray.shape}")
        else:
            gray = image
            print(f"   ✅ Already grayscale: {gray.shape}")
        
        print(f"   📊 Grayscale stats: min={gray.min()}, max={gray.max()}, mean={gray.mean():.1f}")
        
        # Step 2: Apply levels adjustment (input_black, input_white, gamma)
        print(f"   🔧 Step 2: Applying levels adjustment...")
        clipped = np.clip(gray, input_black, input_white)
        print(f"   📊 After clipping: min={clipped.min()}, max={clipped.max()}")
        
        normalized = (clipped - input_black) / (input_white - input_black)
        print(f"   📊 After normalization: min={normalized.min():.3f}, max={normalized.max():.3f}")
        
        gamma_corrected = np.power(normalized, 1.0 / gamma)
        print(f"   📊 After gamma correction: min={gamma_corrected.min():.3f}, max={gamma_corrected.max():.3f}")
        
        result = (gamma_corrected * 255).astype(np.uint8)
        print(f"   ✅ Final result: {result.shape}, min={result.min()}, max={result.max()}")
        
        # Step 3: Save result
        print(f"   💾 Step 3: Saving image...")
        filename_HighContrastPhoto = f"{self.timestamp}_02_diag_photoshop_levels_gamma{suffix}.png"
        filepath_HighContrastPhoto = os.path.join(self.output_dir, filename_HighContrastPhoto)
        
        print(f"   📂 Saving to: {filepath_HighContrastPhoto}")
        cv2.imwrite(filepath_HighContrastPhoto, result)
        
        # Verify file was written
        if os.path.exists(filepath_HighContrastPhoto):
            file_size = os.path.getsize(filepath_HighContrastPhoto)
            print(f"   ✅ File saved successfully: {filename_HighContrastPhoto} ({file_size} bytes)")
        else:
            print(f"   ❌ [ERROR] Failed to save file: {filepath_HighContrastPhoto}")
            print(f"🏁 [DEBUG] _apply_levels_adjustment() EXIT (FAILED)")
            return None, None
        
        # Performance logging
        processing_time = int((time.time() - start_time) * 1000)
        print(f"   ⏱️ Processing completed: {processing_time}ms")
        print(f"🏁 [DEBUG] _apply_levels_adjustment() EXIT (SUCCESS)")
        
        return result, filepath_HighContrastPhoto
    
    def create_left_boundary_photo(self, image):
        """
        Create Left boundary high-contrast photo with optimized parameters for subtle contrast.
        
        📌 INPUT CONTRACT:
        - image: np.ndarray - Input WeChat screenshot (BGR format from cv2.imread)
        - Expected: Screenshot containing WeChat interface with message cards
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (processed image array, filepath)
        - Failure: (None, None) - Returns (None, None) if processing fails
        - Side Effects: Saves enhanced image as YYYYMMDD_HHMMSS_02_diag_photoshop_levels_gamma_Left.png
        
        🔧 PARAMETER SELECTION:
        - Shadow=33: Moderate shadow clipping (preserves dark interface elements)
        - Highlight=55: Aggressive highlight clipping (enhances contrast in bright areas)
        - Midtone=1.1: Slight midtone darkening (subtle enhancement for clean detection)
        
        This conservative approach maintains interface visibility while enhancing boundaries.
        Optimized for detecting the left edge of WeChat's conversation area.
        """
        return self._apply_levels_adjustment(image, 33, 55, 1.1, "_Left")
    
    def create_right_boundary_photo(self, image):
        """
        Create Right boundary high-contrast photo with aggressive parameters for maximum contrast.
        
        📌 INPUT CONTRACT:
        - image: np.ndarray - Input WeChat screenshot (BGR format from cv2.imread)
        - Expected: Screenshot containing WeChat interface with message cards
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (processed image array, filepath)
        - Failure: (None, None) - Returns (None, None) if processing fails
        - Side Effects: Saves enhanced image as YYYYMMDD_HHMMSS_02_diag_photoshop_levels_gamma_Right.png
        
        🔧 PARAMETER SELECTION:
        - Shadow=16: Aggressive shadow clipping (removes subtle gradients)
        - Highlight=126: Moderate highlight preservation (maintains text readability)
        - Midtone=1.5: Strong midtone darkening (creates sharp contrast boundaries)
        
        This aggressive approach maximizes boundary visibility for challenging detection scenarios.
        Optimized for detecting the right edges of WeChat message cards and text boundaries.
        """
        return self._apply_levels_adjustment(image, 16, 126, 1.5, "_Right")
    
    def create_dual_photo(self, image):
        """
        Create Dual high-contrast photo with balanced parameters for general enhancement.
        
        📌 INPUT CONTRACT:
        - image: np.ndarray - Input WeChat screenshot (BGR format from cv2.imread)
        - Expected: Screenshot containing WeChat interface with message cards
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (processed image array, filepath)
        - Failure: (None, None) - Returns (None, None) if processing fails
        - Side Effects: Saves enhanced image as YYYYMMDD_HHMMSS_02_diag_photoshop_levels_gamma_Dual.png
        
        🔧 PARAMETER SELECTION:
        - Shadow=30: Balanced shadow clipping (preserves detail while enhancing contrast)
        - Highlight=43: Moderate highlight clipping (maintains good contrast range)
        - Midtone=1.0: No midtone adjustment (preserves natural gamma curve)
        
        This balanced approach provides general-purpose enhancement suitable for dual boundary detection.
        Optimized for detecting both left and right edges with equal effectiveness.
        """
        print("🔍 [DEBUG] create_dual_photo() ENTRY")
        
        if image is None:
            print("   ❌ [ERROR] Input image is None")
            print("🏁 [DEBUG] create_dual_photo() EXIT (FAILED)")
            return None, None
        
        print(f"   ✅ Input image received: {image.shape} shape")
        print("   ⚙️ Using Dual parameters: Shadow=30, Midtone=1.0, Highlight=43")
        
        start_time = time.time()
        result = self._apply_levels_adjustment(image, 30, 43, 1.0, "_Dual")
        
        processing_time = int((time.time() - start_time) * 1000)
        print(f"   ⏱️ Total dual photo processing: {processing_time}ms")
        print("🏁 [DEBUG] create_dual_photo() EXIT")
        
        return result
    
    def create_both_photos(self, image):
        """
        Create both Left and Right boundary high-contrast photos in one operation.
        
        📌 INPUT CONTRACT:
        - image: np.ndarray - Input WeChat screenshot (BGR format from cv2.imread)
        - Expected: Screenshot containing WeChat interface with message cards
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[Tuple[np.ndarray, str], Tuple[np.ndarray, str]] - ((left_img, left_path), (right_img, right_path))
        - Failure: ((None, None), (None, None)) - Returns ((None, None), (None, None)) if processing fails
        - Side Effects: Saves both enhanced images with timestamped filenames
        
        🔧 PROCESSING:
        Combines left and right boundary photo creation into single efficient operation.
        Uses same timestamp for both files ensuring they are processed together.
        """
        left_result = self.create_left_boundary_photo(image)
        right_result = self.create_right_boundary_photo(image)
        return left_result, right_result

    # =============================================================================
    # IMAGE ACQUISITION METHODS - Three ways to get images for processing
    # METHOD 1: Load from filepath (existing file)
    # METHOD 2: Load latest screenshot (most recent existing file)  
    # METHOD 3: Capture new screenshot (live WeChat window capture)
    # =============================================================================

    def load_image_from_filepath(self, filepath_screenshot: str) -> Optional[np.ndarray]:
        """
        METHOD 1: Load WeChat screenshot directly from provided filepath using cv2.imread()
        
        📌 INPUT CONTRACT:
        - filepath_screenshot: str - Full path to WeChat screenshot file (PNG/JPG)
        - Expected: Valid file path to existing WeChat interface screenshot
        
        📌 OUTPUT CONTRACT:
        - Success: np.ndarray - Loaded image in BGR format ready for processing
        - Failure: None - Returns None if file doesn't exist or can't be loaded
        - Side Effects: Prints debug information about loading process and file validation
        
        🔧 PROCESSING:
        Direct file loading with comprehensive validation and error handling.
        Validates file existence, format, and basic image properties before returning.
        """
        print(f"🔍 [DEBUG] load_image_from_filepath() ENTRY")
        print(f"   📄 Filepath: {filepath_screenshot}")
        
        start_time = time.time()
        
        # Validate file exists
        if not os.path.exists(filepath_screenshot):
            print(f"   ❌ [ERROR] File does not exist: {filepath_screenshot}")
            print(f"🏁 [DEBUG] load_image_from_filepath() EXIT (FAILED)")
            return None
        
        # Get file info
        file_size = os.path.getsize(filepath_screenshot)
        print(f"   📊 File size: {file_size} bytes ({file_size/1024:.1f} KB)")
        
        # Load image with cv2.imread
        print(f"   📂 Loading image with cv2.imread()...")
        image = cv2.imread(filepath_screenshot)
        
        if image is None:
            print(f"   ❌ [ERROR] cv2.imread() failed to load image: {filepath_screenshot}")
            print(f"🏁 [DEBUG] load_image_from_filepath() EXIT (FAILED)")
            return None
        
        # Validate image properties
        print(f"   ✅ Image loaded successfully: {image.shape} shape, {image.dtype} dtype")
        
        # Performance logging
        loading_time = int((time.time() - start_time) * 1000)
        print(f"   ⏱️ Image loading completed: {loading_time}ms")
        print(f"🏁 [DEBUG] load_image_from_filepath() EXIT (SUCCESS)")
        
        return image

    def load_image_from_latest_screenshot(self, screenshot_dir: str = None) -> Optional[Tuple[np.ndarray, str]]:
        """
        METHOD 2: Find and load the latest WeChat screenshot using m_screenshot_finder_tool.py
        
        📌 INPUT CONTRACT:
        - screenshot_dir: Optional[str] - Directory to search (uses default "pic/screenshots" if None)
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (loaded image, filepath to the screenshot)
        - Failure: (None, None) - Returns (None, None) if no screenshots found or loading fails
        - Side Effects: Uses screenshot finder tool, prints debug info about discovery and loading
        
        🔧 PROCESSING:
        Uses m_screenshot_finder_tool to find latest screenshot, then loads it with cv2.imread().
        Combines screenshot discovery with image loading in single efficient operation.
        """
        print(f"🔍 [DEBUG] load_image_from_latest_screenshot() ENTRY")
        print(f"   📁 Search directory: {screenshot_dir if screenshot_dir else 'default (pic/screenshots)'}")
        
        start_time = time.time()
        
        try:
            # Import and use screenshot finder tool
            print(f"   🔧 Importing screenshot finder tool...")
            try:
                from modules.m_screenshot_finder_tool import find_latest_screenshot
            except ImportError:
                # For direct execution, try relative import
                from m_screenshot_finder_tool import find_latest_screenshot
            print(f"   ✅ Screenshot finder tool imported")
            
            # Find latest screenshot
            print(f"   🔍 Searching for latest WeChat screenshot...")
            filepath_screenshot = find_latest_screenshot(screenshot_dir)
            
            if not filepath_screenshot:
                print(f"   ❌ [ERROR] No WeChat screenshots found")
                print(f"🏁 [DEBUG] load_image_from_latest_screenshot() EXIT (FAILED)")
                return None, None
            
            print(f"   ✅ Latest screenshot found: {os.path.basename(filepath_screenshot)}")
            
            # Load the found screenshot
            print(f"   📂 Loading found screenshot...")
            image = self.load_image_from_filepath(filepath_screenshot)
            
            if image is None:
                print(f"   ❌ [ERROR] Failed to load found screenshot")
                print(f"🏁 [DEBUG] load_image_from_latest_screenshot() EXIT (FAILED)")
                return None, None
            
            # Performance logging
            total_time = int((time.time() - start_time) * 1000)
            print(f"   ⏱️ Total discovery + loading time: {total_time}ms")
            print(f"🏁 [DEBUG] load_image_from_latest_screenshot() EXIT (SUCCESS)")
            
            return image, filepath_screenshot
            
        except ImportError as e:
            print(f"   ❌ [ERROR] Cannot import screenshot finder tool: {str(e)}")
            print(f"🏁 [DEBUG] load_image_from_latest_screenshot() EXIT (FAILED)")
            return None, None
        except Exception as e:
            print(f"   ❌ [ERROR] Unexpected error: {str(e)}")
            print(f"🏁 [DEBUG] load_image_from_latest_screenshot() EXIT (FAILED)")
            return None, None

    def capture_new_screenshot(self) -> Optional[Tuple[np.ndarray, str]]:
        """
        METHOD 3: Capture a fresh WeChat screenshot using m_screenshot_processor module
        
        📌 INPUT CONTRACT:
        - No parameters - captures live WeChat window screenshot
        
        📌 OUTPUT CONTRACT:
        - Success: Tuple[np.ndarray, str] - (loaded image, filepath to the screenshot)
        - Failure: (None, None) - Returns (None, None) if capture or loading fails
        - Side Effects: Captures new screenshot, saves to pic/screenshots/, prints debug info
        
        🔧 PROCESSING:
        Uses fcapture_screenshot from m_screenshot_processor to capture live WeChat window.
        Automatically detects WeChat window and captures current state.
        """
        print(f"🔍 [DEBUG] capture_new_screenshot() ENTRY")
        print(f"   📸 Capturing fresh WeChat screenshot...")
        
        start_time = time.time()
        
        try:
            # Capture new screenshot using m_screenshot_processor
            print(f"   🔧 Calling fcapture_screenshot()...")
            screenshot_path = fcapture_screenshot()
            
            if not screenshot_path:
                print(f"   ❌ [ERROR] Failed to capture screenshot")
                print(f"🏁 [DEBUG] capture_new_screenshot() EXIT (FAILED)")
                return None, None
            
            print(f"   ✅ Screenshot captured: {os.path.basename(screenshot_path)}")
            
            # Load the captured screenshot
            print(f"   📂 Loading captured screenshot...")
            image = self.load_image_from_filepath(screenshot_path)
            
            if image is None:
                print(f"   ❌ [ERROR] Failed to load captured screenshot")
                print(f"🏁 [DEBUG] capture_new_screenshot() EXIT (FAILED)")
                return None, None
            
            # Performance logging
            capture_time = int((time.time() - start_time) * 1000)
            print(f"   ⏱️ Total capture + loading time: {capture_time}ms")
            print(f"🏁 [DEBUG] capture_new_screenshot() EXIT (SUCCESS)")
            
            return image, screenshot_path
            
        except Exception as e:
            print(f"   ❌ [ERROR] Exception during screenshot capture: {str(e)}")
            print(f"🏁 [DEBUG] capture_new_screenshot() EXIT (FAILED)")
            return None, None

    def process_from_new_screenshot(self, processing_mode: str = "both") -> Optional[Tuple]:
        """
        METHOD 3 WORKFLOW: Capture new WeChat screenshot and process with specified mode
        
        📌 INPUT CONTRACT:
        - processing_mode: str - Processing mode: "left", "right", "dual", "both" (default: "both")
        
        📌 OUTPUT CONTRACT:
        - Success: Depends on processing_mode:
          * "left": Tuple[np.ndarray, str] - (left image, left filepath)
          * "right": Tuple[np.ndarray, str] - (right image, right filepath)  
          * "dual": Tuple[np.ndarray, str] - (dual image, dual filepath)
          * "both": Tuple[Tuple, Tuple] - ((left_img, left_path), (right_img, right_path))
        - Failure: None - Returns None if capture or processing fails
        - Side Effects: Captures new screenshot, saves processed images to output directory
        
        🔧 PROCESSING:
        Complete workflow combining live screenshot capture with processing.
        Perfect for real-time WeChat window analysis and boundary detection.
        """
        print(f"🔍 [DEBUG] process_from_new_screenshot() ENTRY")
        print(f"   ⚙️ Processing mode: {processing_mode}")
        
        start_time = time.time()
        
        # Step 1: Capture new screenshot
        print(f"   🔧 Step 1: Capturing new WeChat screenshot...")
        image, filepath_screenshot = self.capture_new_screenshot()
        
        if image is None:
            print(f"   ❌ [ERROR] No screenshot captured for processing")
            print(f"🏁 [DEBUG] process_from_new_screenshot() EXIT (FAILED)")
            return None
        
        print(f"   ✅ Using screenshot: {os.path.basename(filepath_screenshot)}")
        
        # Step 2: Process based on mode
        print(f"   🔧 Step 2: Processing with {processing_mode} mode...")
        
        if processing_mode == "left":
            result = self.create_left_boundary_photo(image)
        elif processing_mode == "right":
            result = self.create_right_boundary_photo(image)
        elif processing_mode == "dual":
            result = self.create_dual_photo(image)
        elif processing_mode == "both":
            result = self.create_both_photos(image)
        else:
            print(f"   ❌ [ERROR] Invalid processing mode: {processing_mode}")
            print(f"   💡 Valid modes: 'left', 'right', 'dual', 'both'")
            print(f"🏁 [DEBUG] process_from_new_screenshot() EXIT (FAILED)")
            return None
        
        # Performance logging
        total_time = int((time.time() - start_time) * 1000)
        print(f"   ⏱️ Total workflow time: {total_time}ms")
        print(f"   ✅ Complete workflow finished successfully")
        print(f"🏁 [DEBUG] process_from_new_screenshot() EXIT (SUCCESS)")
        
        return result

    def process_from_latest_screenshot(self, processing_mode: str = "both", screenshot_dir: str = None) -> Optional[Tuple]:
        """
        METHOD 4: Complete workflow - Find latest screenshot and process with specified mode
        
        📌 INPUT CONTRACT:
        - processing_mode: str - Processing mode: "left", "right", "dual", "both" (default: "both")
        - screenshot_dir: Optional[str] - Directory to search (uses default if None)
        
        📌 OUTPUT CONTRACT:
        - Success: Depends on processing_mode (same as process_from_filepath)
        - Failure: None - Returns None if no screenshots found or processing fails
        - Side Effects: Uses screenshot finder, loads latest screenshot, saves processed images
        
        🔧 PROCESSING:
        Ultimate convenience method combining screenshot discovery with processing.
        Perfect for automated workflows that need to process the most recent screenshot.
        """
        print(f"🔍 [DEBUG] process_from_latest_screenshot() ENTRY")
        print(f"   ⚙️ Processing mode: {processing_mode}")
        print(f"   📁 Search directory: {screenshot_dir if screenshot_dir else 'default'}")
        
        start_time = time.time()
        
        # Step 1: Find and load latest screenshot
        print(f"   🔧 Step 1: Finding and loading latest screenshot...")
        image, filepath_screenshot = self.load_image_from_latest_screenshot(screenshot_dir)
        
        if image is None:
            print(f"   ❌ [ERROR] No screenshot available for processing")
            print(f"🏁 [DEBUG] process_from_latest_screenshot() EXIT (FAILED)")
            return None
        
        print(f"   ✅ Using screenshot: {os.path.basename(filepath_screenshot)}")
        
        # Step 2: Process the loaded image
        print(f"   🔧 Step 2: Processing image with {processing_mode} mode...")
        
        if processing_mode == "left":
            result = self.create_left_boundary_photo(image)
        elif processing_mode == "right":
            result = self.create_right_boundary_photo(image)
        elif processing_mode == "dual":
            result = self.create_dual_photo(image)
        elif processing_mode == "both":
            result = self.create_both_photos(image)
        else:
            print(f"   ❌ [ERROR] Invalid processing mode: {processing_mode}")
            print(f"   💡 Valid modes: 'left', 'right', 'dual', 'both'")
            print(f"🏁 [DEBUG] process_from_latest_screenshot() EXIT (FAILED)")
            return None
        
        # Performance logging
        total_time = int((time.time() - start_time) * 1000)
        print(f"   ⏱️ Total workflow time: {total_time}ms")
        print(f"   ✅ Complete workflow finished successfully")
        print(f"🏁 [DEBUG] process_from_latest_screenshot() EXIT (SUCCESS)")
        
        return result


# =============================================================================
# COMPREHENSIVE MANUAL TESTING (Direct Run)
# =============================================================================
if __name__ == "__main__":
    print("=" * 80)
    print("🚀 COMPREHENSIVE PHOTO PROCESSOR TESTING")
    print("=" * 80)
    print("This test suite will validate all functions with actual WeChat screenshots")
    print("=" * 80)

    # Initialize the photo processor
    print("\n📦 INITIALIZATION")
    print("-" * 40)
    processor = c_tool_Create_Highcontrast_Photo()
    print("✅ Photo processor initialized successfully")
    
    # ==========================================================================
    # SECTION 1: IMAGE ACQUISITION TESTING (3 Methods)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📸 SECTION 1: IMAGE ACQUISITION METHODS")
    print("=" * 80)
    
    # METHOD 1: Capture new live screenshot
    print("\n🔴 METHOD 1: CAPTURE NEW SCREENSHOT (Live Capture)")
    print("-" * 40)
    try:
        print("📸 Capturing fresh WeChat window screenshot...")
        new_image, new_path = processor.capture_new_screenshot()
        if new_image is not None:
            print(f"✅ SUCCESS: Live screenshot captured")
            print(f"   📊 Image shape: {new_image.shape}")
            print(f"   📁 File saved: {os.path.basename(new_path)}")
            print(f"   💾 File size: {os.path.getsize(new_path) / 1024:.1f} KB")
            method1_success = True
            test_image = new_image  # Use this for further testing
            test_filepath = new_path
        else:
            print("⚠️  No screenshot captured (Is WeChat open?)")
            method1_success = False
            test_image = None
            test_filepath = None
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        method1_success = False
        test_image = None
        test_filepath = None
    
    # METHOD 2: Load latest existing screenshot
    print("\n🔵 METHOD 2: LOAD LATEST SCREENSHOT")
    print("-" * 40)
    try:
        print("🔍 Finding and loading latest screenshot...")
        latest_image, latest_path = processor.load_image_from_latest_screenshot()
        if latest_image is not None:
            print(f"✅ SUCCESS: Latest screenshot loaded")
            print(f"   📊 Image shape: {latest_image.shape}")
            print(f"   📁 File loaded: {os.path.basename(latest_path)}")
            print(f"   💾 File size: {os.path.getsize(latest_path) / 1024:.1f} KB")
            method2_success = True
            # Use as backup if METHOD 1 failed
            if test_image is None:
                test_image = latest_image
                test_filepath = latest_path
        else:
            print("⚠️  No screenshots found in directory")
            method2_success = False
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        method2_success = False
    
    # METHOD 3: Load from specific filepath
    print("\n🟢 METHOD 3: LOAD FROM FILEPATH")
    print("-" * 40)
    if test_filepath:
        try:
            print(f"📂 Loading from: {os.path.basename(test_filepath)}")
            filepath_image = processor.load_image_from_filepath(test_filepath)
            if filepath_image is not None:
                print(f"✅ SUCCESS: Image loaded from filepath")
                print(f"   📊 Image shape: {filepath_image.shape}")
                method3_success = True
            else:
                print("⚠️  Failed to load image")
                method3_success = False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            method3_success = False
    else:
        print("⏭️  SKIPPED: No test filepath available")
        method3_success = False
    
    # ==========================================================================
    # SECTION 2: CORE PROCESSING METHODS (High-Contrast Creation)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🎨 SECTION 2: HIGH-CONTRAST PHOTO CREATION")
    print("=" * 80)
    
    if test_image is not None:
        print(f"\n📊 Using test image: {test_image.shape} from {os.path.basename(test_filepath) if test_filepath else 'memory'}")
        
        # Test LEFT boundary processing
        print("\n🔴 LEFT BOUNDARY PROCESSING")
        print("-" * 40)
        try:
            print("⚙️  Parameters: Shadow=33, Midtone=1.1, Highlight=55")
            left_result = processor.create_left_boundary_photo(test_image)
            if left_result[0] is not None:
                print(f"✅ SUCCESS: Left boundary photo created")
                print(f"   📊 Output shape: {left_result[0].shape}")
                print(f"   📁 File saved: {os.path.basename(left_result[1])}")
                print(f"   💾 File size: {os.path.getsize(left_result[1]) / 1024:.1f} KB")
                left_success = True
            else:
                print("❌ FAILED: Processing returned None")
                left_success = False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            left_success = False
        
        # Test RIGHT boundary processing
        print("\n🔵 RIGHT BOUNDARY PROCESSING")
        print("-" * 40)
        try:
            print("⚙️  Parameters: Shadow=16, Midtone=1.5, Highlight=126")
            right_result = processor.create_right_boundary_photo(test_image)
            if right_result[0] is not None:
                print(f"✅ SUCCESS: Right boundary photo created")
                print(f"   📊 Output shape: {right_result[0].shape}")
                print(f"   📁 File saved: {os.path.basename(right_result[1])}")
                print(f"   💾 File size: {os.path.getsize(right_result[1]) / 1024:.1f} KB")
                right_success = True
            else:
                print("❌ FAILED: Processing returned None")
                right_success = False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            right_success = False
        
        # Test DUAL processing
        print("\n🟢 DUAL BOUNDARY PROCESSING")
        print("-" * 40)
        try:
            print("⚙️  Parameters: Shadow=30, Midtone=1.0, Highlight=43")
            dual_result = processor.create_dual_photo(test_image)
            if dual_result[0] is not None:
                print(f"✅ SUCCESS: Dual boundary photo created")
                print(f"   📊 Output shape: {dual_result[0].shape}")
                print(f"   📁 File saved: {os.path.basename(dual_result[1])}")
                print(f"   💾 File size: {os.path.getsize(dual_result[1]) / 1024:.1f} KB")
                dual_success = True
            else:
                print("❌ FAILED: Processing returned None")
                dual_success = False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            dual_success = False
        
        # Test BOTH photos creation
        print("\n🟡 BOTH PHOTOS CREATION")
        print("-" * 40)
        try:
            print("⚙️  Creating both Left and Right photos simultaneously...")
            left_res, right_res = processor.create_both_photos(test_image)
            if left_res[0] is not None and right_res[0] is not None:
                print(f"✅ SUCCESS: Both photos created")
                print(f"   📊 Left shape: {left_res[0].shape}")
                print(f"   📊 Right shape: {right_res[0].shape}")
                print(f"   📁 Files saved: 2 files with _Left and _Right suffixes")
                both_success = True
            else:
                print("❌ FAILED: One or both processings failed")
                both_success = False
        except Exception as e:
            print(f"❌ FAILED: {str(e)}")
            both_success = False
    else:
        print("\n⚠️  SKIPPING PROCESSING TESTS: No test image available")
        left_success = right_success = dual_success = both_success = False
    
    # ==========================================================================
    # SECTION 3: WORKFLOW METHODS (Combined Operations)
    # ==========================================================================
    print("\n" + "=" * 80)
    print("🔄 SECTION 3: WORKFLOW METHODS")
    print("=" * 80)
    
    # Test process_from_new_screenshot workflow
    print("\n🔴 WORKFLOW 1: CAPTURE + PROCESS (Live)")
    print("-" * 40)
    try:
        print("📸 Capturing new screenshot and processing as 'both'...")
        workflow1_result = processor.process_from_new_screenshot("both")
        if workflow1_result is not None:
            print(f"✅ SUCCESS: Live capture + processing workflow completed")
            print(f"   📊 Generated 2 high-contrast photos")
            workflow1_success = True
        else:
            print("⚠️  Workflow returned None")
            workflow1_success = False
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        workflow1_success = False
    
    # Test process_from_latest_screenshot workflow
    print("\n🔵 WORKFLOW 2: LOAD LATEST + PROCESS")
    print("-" * 40)
    try:
        print("🔍 Finding latest screenshot and processing as 'dual'...")
        workflow2_result = processor.process_from_latest_screenshot("dual")
        if workflow2_result is not None:
            print(f"✅ SUCCESS: Load latest + processing workflow completed")
            print(f"   📊 Generated dual high-contrast photo")
            workflow2_success = True
        else:
            print("⚠️  Workflow returned None")
            workflow2_success = False
    except Exception as e:
        print(f"❌ FAILED: {str(e)}")
        workflow2_success = False
    
    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("📊 COMPREHENSIVE TEST SUMMARY")
    print("=" * 80)
    
    # Count successes
    acquisition_tests = [method1_success, method2_success, method3_success]
    processing_tests = [left_success, right_success, dual_success, both_success]
    workflow_tests = [workflow1_success, workflow2_success]
    
    acquisition_passed = sum(acquisition_tests)
    processing_passed = sum(processing_tests)
    workflow_passed = sum(workflow_tests)
    total_passed = acquisition_passed + processing_passed + workflow_passed
    total_tests = len(acquisition_tests) + len(processing_tests) + len(workflow_tests)
    
    print(f"\n📸 Image Acquisition Methods: {acquisition_passed}/3 passed")
    print(f"   {'✅' if method1_success else '❌'} METHOD 1: Capture new screenshot (Live)")
    print(f"   {'✅' if method2_success else '❌'} METHOD 2: Load latest screenshot")
    print(f"   {'✅' if method3_success else '❌'} METHOD 3: Load from filepath")
    
    print(f"\n🎨 Processing Methods: {processing_passed}/4 passed")
    print(f"   {'✅' if left_success else '❌'} Left boundary processing")
    print(f"   {'✅' if right_success else '❌'} Right boundary processing")
    print(f"   {'✅' if dual_success else '❌'} Dual boundary processing")
    print(f"   {'✅' if both_success else '❌'} Both photos creation")
    
    print(f"\n🔄 Workflow Methods: {workflow_passed}/2 passed")
    print(f"   {'✅' if workflow1_success else '❌'} Capture + Process workflow")
    print(f"   {'✅' if workflow2_success else '❌'} Load Latest + Process workflow")
    
    print(f"\n📈 OVERALL: {total_passed}/{total_tests} tests passed ({total_passed*100//total_tests}%)")
    
    if total_passed == total_tests:
        print("\n🎉 PERFECT! All tests passed successfully!")
    elif total_passed >= total_tests * 0.7:
        print("\n✅ GOOD! Most tests passed successfully.")
    elif total_passed >= total_tests * 0.5:
        print("\n⚠️  PARTIAL SUCCESS: Some tests failed.")
    else:
        print("\n❌ CRITICAL: Most tests failed. Check WeChat window status.")
    
    print("\n" + "=" * 80)
    print("📁 Output Directory: pic/screenshots/")
    print("💡 Check the directory for generated high-contrast photos")
    print("=" * 80)
