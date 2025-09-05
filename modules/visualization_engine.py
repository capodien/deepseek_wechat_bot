#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Visualization Engine (visualization_engine.py)

Centralized visualization engine for all card processing diagnostics.
Provides consistent, professional diagnostic output across all detector modules.

Features:
- Comprehensive debug visualizations matching time detection quality
- Simple overlay visualizations for quick debugging
- Consistent styling and layout across all detectors
- Extensible framework for new detector types
"""

import os
import cv2
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any, Union
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

from .diagnostic_templates import (
    DiagnosticStyling, LayoutTemplates, VisualizationComponents, FilenameGenerator
)


class DiagnosticVisualizationEngine:
    """
    Centralized visualization engine for all card processing diagnostics
    
    Provides consistent, professional diagnostic output with multiple visualization
    types suitable for development, debugging, and documentation.
    """
    
    def __init__(self):
        self.styling = DiagnosticStyling()
        self.layouts = LayoutTemplates()
        self.components = VisualizationComponents()
        self.filename_gen = FilenameGenerator()
        
        # Maintain backward compatibility with existing VisualizationEngine
        self.colors = {
            'boundary': (0, 255, 0),        # Green for boundaries
            'avatar': (255, 0, 0),          # Red for avatars
            'card': (0, 0, 255),            # Blue for cards  
            'name': (255, 255, 0),          # Cyan for names
            'time': (0, 255, 255),          # Yellow for timestamps
            'detection': (255, 0, 255),     # Magenta for detection points
            'text': (255, 255, 255),        # White for text
            'error': (0, 0, 255),           # Red for errors
            'success': (0, 255, 0),         # Green for success
        }
        
        # Text settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        self.line_thickness = 2
    
    def create_comprehensive_debug_visualization(self, detector_type: str, 
                                               visualization_data: Dict[str, Any]) -> str:
        """
        Create comprehensive debug visualization matching time detection quality
        
        This is the main method for creating detailed, multi-panel diagnostic output
        similar to your time detection image with:
        - Main overview with success/failure indicators
        - Individual card ROI analysis
        - Statistical analysis charts
        - Processing pipeline visualization
        
        Args:
            detector_type: Type of detector ('contact_name', 'time_detection', etc.)
            visualization_data: Dictionary containing all necessary data for visualization
                Required keys:
                - 'image_path': Path to original WeChat screenshot
                - 'enhanced_cards': List of card dictionaries with detection results
                - 'detection_info': Detection summary statistics
                - 'debug_data': Optional detailed debug information per card
                
        Returns:
            Path to generated comprehensive debug visualization file
        """
        print(f"🎨 Creating comprehensive debug visualization for {detector_type}...")
        
        # Extract required data
        image_path = visualization_data['image_path']
        enhanced_cards = visualization_data['enhanced_cards']
        debug_data = visualization_data.get('debug_data', {})
        success_count = visualization_data.get('success_count', 0)
        failed_count = visualization_data.get('failed_count', 0)
        
        # Load and prepare image
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Create comprehensive debug visualization with enhanced layout
        fig = plt.figure(figsize=(24, 18), dpi=300)
        
        # Enhanced layout for comprehensive analysis:
        # Row 1: Main overview (2x2) + First 2 ROI panels
        # Row 2: Binary masks for first 3 cards + Algorithm parameters
        # Row 3: Statistical analysis charts (4 panels)
        gs = fig.add_gridspec(3, 4, height_ratios=[2, 1.5, 1.5], width_ratios=[2, 1, 1, 1])
        
        # Main panel - show the screenshot with detection overlays (larger)
        ax_main = fig.add_subplot(gs[0:2, 0])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        ax_main.imshow(img_rgb)
        ax_main.set_title(f'{detector_type.replace("_", " ").title()} Detection Results\n{success_count} Success, {failed_count} Failed', 
                         fontsize=16, fontweight='bold', pad=20)
        ax_main.axis('off')
        
        # Draw enhanced detection results on the main image
        for i, card in enumerate(enhanced_cards):
            card_bbox = card.get("bbox", [0, 0, 100, 100])
            x, y, w, h = card_bbox
            
            # Draw card boundary (blue)
            rect = plt.Rectangle((x, y), w, h, linewidth=2, 
                               edgecolor='#0066CC', facecolor='none', alpha=0.8)
            ax_main.add_patch(rect)
            
            # Draw avatar boundary if available (green)
            if card.get("avatar") and card["avatar"].get("bbox"):
                av = card["avatar"]["bbox"]
                avatar_rect = plt.Rectangle((av[0], av[1]), av[2], av[3],
                                          linewidth=1.5, edgecolor='#00AA00', 
                                          facecolor='none', alpha=0.7)
                ax_main.add_patch(avatar_rect)
            
            # Draw search region if available (orange, semi-transparent)
            if card.get("name_boundary") and card["name_boundary"].get("search_region"):
                sr = card["name_boundary"]["search_region"]
                search_rect = plt.Rectangle((sr[0], sr[1]), sr[2], sr[3], 
                                          linewidth=1, edgecolor='#FF8800', 
                                          facecolor='#FF8800', alpha=0.15)
                ax_main.add_patch(search_rect)
            
            # Draw detected name boundary if found (bright green with confidence)
            if card.get("name_boundary"):
                nb = card["name_boundary"]["bbox"]
                confidence = card["name_boundary"].get("confidence", 0.0)
                alpha_value = 0.4 + (confidence * 0.3)  # Variable transparency based on confidence
                name_rect = plt.Rectangle((nb[0], nb[1]), nb[2], nb[3],
                                        linewidth=3, edgecolor='#00FF00', 
                                        facecolor='#00FF00', alpha=alpha_value)
                ax_main.add_patch(name_rect)
                
                # Add confidence text
                ax_main.text(nb[0] + nb[2] + 5, nb[1] + nb[3]//2, f'{confidence:.2f}', 
                           color='white', fontsize=8, fontweight='bold',
                           bbox=dict(boxstyle="round,pad=0.2", facecolor='#00AA00', alpha=0.7))
            
            # Check if detection was successful and add enhanced status
            has_detection = card.get("name_boundary") is not None
            status_color = '#00AA00' if has_detection else '#CC0000'
            status_symbol = '✅' if has_detection else '❌'
            
            # Add enhanced card label with more info
            label_text = f'{status_symbol} Card {i+1}'
            if has_detection:
                nb = card["name_boundary"]["bbox"]
                label_text += f'\n{nb[2]}×{nb[3]}px'
            
            ax_main.text(x + 5, y + 15, label_text, 
                        color='white', fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.8))
        
        # Enhanced ROI Analysis Panels - First 3 cards with projection plots
        for idx in range(min(3, len(enhanced_cards))):
            card = enhanced_cards[idx]
            card_id = card.get("card_id", idx)
            
            # ROI Image Panel (Row 1)
            ax_roi = fig.add_subplot(gs[0, idx + 1])
            if debug_data and f"{card_id}" in debug_data.get("roi_images", {}):
                roi_img = debug_data["roi_images"][f"{card_id}"]
                ax_roi.imshow(cv2.cvtColor(roi_img, cv2.COLOR_BGR2RGB))
                
                # Add detection status indicator
                has_detection = card.get("name_boundary") is not None
                status_color = 'green' if has_detection else 'red'
                status_text = '✅ DETECTED' if has_detection else '❌ FAILED'
                ax_roi.text(0.02, 0.95, status_text, transform=ax_roi.transAxes, 
                           fontsize=9, fontweight='bold', color='white',
                           bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.8))
                
                if has_detection:
                    # Add confidence score
                    confidence = card["name_boundary"].get("confidence", 0.0)
                    ax_roi.text(0.02, 0.05, f'Conf: {confidence:.3f}', 
                               transform=ax_roi.transAxes, fontsize=8, fontweight='bold',
                               color='white', bbox=dict(boxstyle="round,pad=0.2", facecolor='green', alpha=0.8))
                
                ax_roi.set_title(f'Card {idx+1} Search ROI', fontsize=11, fontweight='bold')
            else:
                ax_roi.text(0.5, 0.5, f'Card {idx+1}\nNo ROI Data', 
                           ha='center', va='center', fontsize=10, fontweight='bold')
                ax_roi.set_title(f'Card {idx+1} ROI', fontsize=11)
            ax_roi.axis('off')
            
            # Horizontal Projection Panel (Row 2) - Key enhancement!
            ax_projection = fig.add_subplot(gs[1, idx + 1])
            if (debug_data and "horizontal_projections" in debug_data and 
                f"{card_id}" in debug_data["horizontal_projections"]):
                
                # Get projection data
                proj_data = debug_data["horizontal_projections"][f"{card_id}"]
                original_proj = proj_data.get("original_proj", [])
                smoothed_proj = proj_data.get("smoothed_proj", [])
                threshold = proj_data.get("threshold", 0)
                
                if original_proj and smoothed_proj:
                    x_positions = range(len(original_proj))
                    
                    # Plot original and smoothed projections
                    ax_projection.plot(original_proj, x_positions, 'lightblue', linewidth=1, alpha=0.7, label='Original')
                    ax_projection.plot(smoothed_proj, x_positions, 'red', linewidth=2, label='Smoothed') 
                    
                    # Add threshold line
                    ax_projection.axvline(threshold, color='orange', linestyle='--', linewidth=1.5, label=f'Threshold: {threshold:.1f}')
                    
                    # Formatting to match time detection style
                    ax_projection.set_xlabel('Pixel Density', fontsize=9)
                    ax_projection.set_ylabel('Row Position', fontsize=9)
                    ax_projection.set_title(f'Card {idx+1} Projection', fontsize=10, fontweight='bold')
                    ax_projection.legend(fontsize=7)
                    ax_projection.grid(True, alpha=0.3)
                    ax_projection.invert_yaxis()  # Match image coordinate system
                else:
                    ax_projection.text(0.5, 0.5, 'No Projection\nData', 
                                      ha='center', va='center', fontsize=10)
                    ax_projection.set_title(f'Card {idx+1} Projection', fontsize=10)
            else:
                # Show binary mask as fallback
                if debug_data and f"{card_id}_processed" in debug_data.get("binary_masks", {}):
                    binary_mask = debug_data["binary_masks"][f"{card_id}_processed"]
                    ax_projection.imshow(binary_mask, cmap='gray')
                    
                    # Add white pixel statistics
                    white_pixels = np.sum(binary_mask > 0)
                    total_pixels = binary_mask.shape[0] * binary_mask.shape[1]
                    white_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                    
                    ax_projection.text(0.02, 0.95, f'White: {white_ratio:.1%}', 
                                      transform=ax_projection.transAxes, fontsize=8, fontweight='bold',
                                      color='yellow', bbox=dict(boxstyle="round,pad=0.2", facecolor='black', alpha=0.7))
                    
                    ax_projection.set_title(f'Card {idx+1} Binary', fontsize=10, fontweight='bold')
                else:
                    ax_projection.text(0.5, 0.5, 'No Data\nAvailable', 
                                      ha='center', va='center', fontsize=10)
                    ax_projection.set_title(f'Card {idx+1} Analysis', fontsize=10)
                ax_projection.axis('off')
        
        # Fill remaining empty ROI panels if less than 3 cards
        for idx in range(len(enhanced_cards), 3):
            ax_empty_roi = fig.add_subplot(gs[0, idx + 1])
            ax_empty_roi.text(0.5, 0.5, f'Card {idx+1}\nNot Available', 
                             ha='center', va='center', fontsize=10, alpha=0.5)
            ax_empty_roi.set_title(f'Card {idx+1} ROI', fontsize=11)
            ax_empty_roi.axis('off')
            
            ax_empty_binary = fig.add_subplot(gs[1, idx + 1])
            ax_empty_binary.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=10, alpha=0.5)
            ax_empty_binary.set_title(f'Card {idx+1} Binary', fontsize=10)
            ax_empty_binary.axis('off')
        
        # Enhanced Statistical Analysis Panels (Bottom Row - 4 panels)
        
        # Panel 1: Algorithm Parameters & Detection Statistics
        ax_params = fig.add_subplot(gs[2, 0])
        params_text = ["🔧 ALGORITHM PARAMETERS", ""]
        if debug_data.get('algorithm_parameters'):
            params = debug_data['algorithm_parameters']
            if detector_type == "contact_name":
                params_text.extend([
                    f"• White Threshold: {params.get('WHITE_THRESHOLD_MIN', 155)}-{params.get('WHITE_THRESHOLD_MAX', 255)}",
                    f"• Name Size Constraints:",
                    f"  Width: {params.get('MIN_NAME_WIDTH', 20)}-{params.get('MAX_NAME_WIDTH', 180)}px",
                    f"  Height: {params.get('MIN_NAME_HEIGHT', 10)}-{params.get('MAX_NAME_HEIGHT', 30)}px", 
                    f"• Morphology: {params.get('MORPH_KERNEL_SIZE', (3, 2))} × {params.get('MORPH_ITERATIONS', 2)} iter",
                    f"• Min White Pixel Ratio: {params.get('MIN_WHITE_PIXEL_RATIO', 0.12):.2f}",
                    "",
                    "📊 DETECTION RESULTS", 
                    f"• Total Cards: {len(enhanced_cards)}",
                    f"• Successful: {success_count} ({success_count/len(enhanced_cards)*100:.1f}%)" if enhanced_cards else "• Successful: 0 (0%)",
                    f"• Failed: {failed_count} ({failed_count/len(enhanced_cards)*100:.1f}%)" if enhanced_cards else "• Failed: 0 (0%)"
                ])
        
        ax_params.text(0.05, 0.95, '\n'.join(params_text), fontsize=9, verticalalignment='top', 
                       fontfamily='monospace', transform=ax_params.transAxes)
        ax_params.set_title('Algorithm Parameters & Statistics', fontsize=12, fontweight='bold')
        ax_params.axis('off')
        
        # Panel 2: Confidence Distribution Chart
        ax_confidence = fig.add_subplot(gs[2, 1])
        if debug_data.get('confidence_scores') and success_count > 0:
            confidences = [debug_data['confidence_scores'][card_id] for card_id in debug_data.get('success_cards', [])]
            if confidences:
                # Create histogram of confidence scores
                bins = np.linspace(0, 1, 11)  # 10 bins from 0 to 1
                hist, bin_edges = np.histogram(confidences, bins=bins)
                ax_confidence.bar(bin_edges[:-1], hist, width=0.08, alpha=0.7, color='#00AA00', edgecolor='black')
                ax_confidence.axvline(np.mean(confidences), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(confidences):.3f}')
                ax_confidence.set_xlabel('Confidence Score', fontsize=10)
                ax_confidence.set_ylabel('Count', fontsize=10)
                ax_confidence.set_title(f'Confidence Distribution\n(n={len(confidences)} detections)', fontsize=11, fontweight='bold')
                ax_confidence.legend(fontsize=9)
                ax_confidence.grid(True, alpha=0.3)
            else:
                ax_confidence.text(0.5, 0.5, 'No Confidence\nData Available', ha='center', va='center', fontsize=11)
                ax_confidence.set_title('Confidence Distribution', fontsize=11, fontweight='bold')
        else:
            ax_confidence.text(0.5, 0.5, 'No Successful\nDetections', ha='center', va='center', fontsize=11, color='red')
            ax_confidence.set_title('Confidence Distribution', fontsize=11, fontweight='bold')
        
        # Panel 3: Processing Pipeline Analysis
        ax_pipeline = fig.add_subplot(gs[2, 2])
        pipeline_text = ["🔄 PROCESSING PIPELINE", ""]
        
        # Extract processing statistics from debug data
        if debug_data:
            total_contours = sum(data.get('total_contours', 0) for data in debug_data.get('contour_data', {}).values())
            total_white_regions = sum(data.get('total_regions', 0) for data in debug_data.get('white_text_regions', {}).values())
            roi_count = len(debug_data.get('roi_images', {}))
            mask_count = len(debug_data.get('binary_masks', {}))
            
            # Count boundary detections
            boundary_detections = 0
            fallback_detections = 0
            for card in enhanced_cards:
                if card.get("name_time_boundary"):
                    if card["name_time_boundary"]["detection_method"] == "horizontal_projection_analysis":
                        boundary_detections += 1
                    else:
                        fallback_detections += 1
            
            pipeline_text.extend([
                f"• Boundary Detections: {boundary_detections}",
                f"• Avatar Center Fallbacks: {fallback_detections}",
                f"• ROI Images Extracted: {roi_count}",
                f"• Horizontal Projections: {len(debug_data.get('horizontal_projections', {}))}",
                f"• Binary Masks Created: {mask_count//2 if mask_count > 0 else 0}",  # Divide by 2 (raw + processed)
                f"• Total Contours Found: {total_contours}",
                f"• White Text Regions: {total_white_regions}",
                f"• Regions After Filtering: {len(debug_data.get('filtered_boundaries', {}))}",
                "",
                "⏱️ PROCESSING TIMES",
            ])
            
            # Add processing time statistics if available
            processing_times = list(debug_data.get('processing_time', {}).values())
            if processing_times:
                pipeline_text.extend([
                    f"• Average: {np.mean(processing_times)*1000:.1f}ms",
                    f"• Total: {sum(processing_times)*1000:.1f}ms",
                    f"• Range: {np.min(processing_times)*1000:.1f}-{np.max(processing_times)*1000:.1f}ms"
                ])
            else:
                pipeline_text.append("• No timing data available")
        else:
            pipeline_text.append("No pipeline data available")
        
        ax_pipeline.text(0.05, 0.95, '\n'.join(pipeline_text), fontsize=9, verticalalignment='top',
                        fontfamily='monospace', transform=ax_pipeline.transAxes)
        ax_pipeline.set_title('Processing Pipeline Analysis', fontsize=12, fontweight='bold')
        ax_pipeline.axis('off')
        
        # Panel 4: Detection Dimensions Analysis
        ax_dimensions = fig.add_subplot(gs[2, 3])
        if success_count > 0 and debug_data.get('detection_results'):
            # Extract dimensions from successful detections
            widths, heights, areas = [], [], []
            for card_id in debug_data.get('success_cards', []):
                result = debug_data['detection_results'].get(card_id, {})
                if result.get('bbox'):
                    bbox = result['bbox']
                    widths.append(bbox[2])
                    heights.append(bbox[3])
                    areas.append(bbox[2] * bbox[3])
            
            if widths and heights:
                # Create scatter plot of width vs height
                colors = ['#00AA00' if area > np.median(areas) else '#FF8800' for area in areas]
                scatter = ax_dimensions.scatter(widths, heights, c=colors, alpha=0.7, s=60, edgecolors='black')
                ax_dimensions.set_xlabel('Width (px)', fontsize=10)
                ax_dimensions.set_ylabel('Height (px)', fontsize=10)
                ax_dimensions.set_title(f'Detection Dimensions\n(n={len(widths)} detections)', fontsize=11, fontweight='bold')
                ax_dimensions.grid(True, alpha=0.3)
                
                # Add statistics text
                stats_text = f'W: {np.mean(widths):.1f}±{np.std(widths):.1f}px\nH: {np.mean(heights):.1f}±{np.std(heights):.1f}px'
                ax_dimensions.text(0.02, 0.98, stats_text, transform=ax_dimensions.transAxes, 
                                  fontsize=9, verticalalignment='top', 
                                  bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
            else:
                ax_dimensions.text(0.5, 0.5, 'No Dimension\nData Available', ha='center', va='center', fontsize=11)
                ax_dimensions.set_title('Detection Dimensions', fontsize=11, fontweight='bold')
        else:
            ax_dimensions.text(0.5, 0.5, 'No Successful\nDetections', ha='center', va='center', fontsize=11, color='red')
            ax_dimensions.set_title('Detection Dimensions', fontsize=11, fontweight='bold')
        
        plt.tight_layout()
        
        # Generate filename and save
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        section_num = "05" if detector_type == "contact_name" else "06"
        detector_name = "ContactNameDetection" if detector_type == "contact_name" else "TimeDetection"
        filename = f"{timestamp}_{section_num}_Debug_{detector_name}_{success_count}success_{failed_count}failed.png"
        
        output_dir = "pic/screenshots"
        os.makedirs(output_dir, exist_ok=True)
        filepath = os.path.join(output_dir, filename)
        
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Comprehensive debug visualization saved: {filename}")
        return filepath
    
    def _categorize_detections(self, enhanced_cards: List[Dict], detector_type: str) -> Tuple[List[Dict], List[Dict]]:
        """
        Categorize cards into successful and failed detections
        
        Args:
            enhanced_cards: List of card dictionaries with detection results
            detector_type: Type of detector to categorize for
            
        Returns:
            Tuple of (successful_cards, failed_cards)
        """
        successful_cards = []
        failed_cards = []
        
        if detector_type == "contact_name":
            # Contact name detection success is based on name_boundary presence
            for card in enhanced_cards:
                if card.get("name_boundary"):
                    successful_cards.append(card)
                else:
                    failed_cards.append(card)
        elif detector_type == "time_detection":
            # Time detection success is based on time_box presence
            for card in enhanced_cards:
                if card.get("time_box"):
                    successful_cards.append(card)
                else:
                    failed_cards.append(card)
        else:
            # Generic detection - check for any detection results
            for card in enhanced_cards:
                has_detection = any(key in card for key in ["name_boundary", "time_box", "detection_result"])
                if has_detection:
                    successful_cards.append(card)
                else:
                    failed_cards.append(card)
        
        return successful_cards, failed_cards

    def create_base_overlay(self, image_path: str) -> Optional[np.ndarray]:
        """
        Create base image overlay for visualizations
        
        Args:
            image_path: Path to source image
            
        Returns:
            Base image array or None if failed
        """
        try:
            img = cv2.imread(image_path)
            if img is None:
                print(f"❌ Failed to load image: {image_path}")
                return None
            return img.copy()
        except Exception as e:
            print(f"❌ Error creating base overlay: {e}")
            return None
    
    def draw_vertical_line(self, img: np.ndarray, x: int, color: str = 'boundary', 
                          label: str = None, offset: int = 0) -> np.ndarray:
        """
        Draw vertical line with optional label
        
        Args:
            img: Image to draw on
            x: X coordinate for line
            color: Color key from self.colors
            label: Optional text label
            offset: Y offset for label positioning
        """
        if img is None:
            return img
            
        height = img.shape[0]
        color_bgr = self.colors.get(color, self.colors['boundary'])
        
        # Draw vertical line
        cv2.line(img, (x, 0), (x, height), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            y_pos = max(30 + offset, 30)
            cv2.putText(img, label, (x + 5, y_pos), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_horizontal_line(self, img: np.ndarray, y: int, x1: int = 0, 
                           x2: int = None, color: str = 'boundary', 
                           label: str = None) -> np.ndarray:
        """
        Draw horizontal line with optional label
        
        Args:
            img: Image to draw on
            y: Y coordinate for line
            x1: Start X coordinate
            x2: End X coordinate (image width if None)
            color: Color key from self.colors
            label: Optional text label
        """
        if img is None:
            return img
            
        if x2 is None:
            x2 = img.shape[1]
            
        color_bgr = self.colors.get(color, self.colors['boundary'])
        
        # Draw horizontal line
        cv2.line(img, (x1, y), (x2, y), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x1 + 5, y - 5), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_rectangle(self, img: np.ndarray, x: int, y: int, w: int, h: int,
                      color: str = 'card', label: str = None, filled: bool = False) -> np.ndarray:
        """
        Draw rectangle with optional label
        
        Args:
            img: Image to draw on
            x, y: Top-left corner coordinates
            w, h: Width and height
            color: Color key from self.colors
            label: Optional text label
            filled: Whether to fill rectangle
        """
        if img is None:
            return img
            
        color_bgr = self.colors.get(color, self.colors['card'])
        
        # Draw rectangle
        if filled:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, -1)
        else:
            cv2.rectangle(img, (x, y), (x + w, y + h), color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x + 5, y + 20), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def draw_circle(self, img: np.ndarray, x: int, y: int, radius: int = 5,
                   color: str = 'detection', label: str = None) -> np.ndarray:
        """
        Draw circle with optional label
        
        Args:
            img: Image to draw on
            x, y: Center coordinates
            radius: Circle radius
            color: Color key from self.colors
            label: Optional text label
        """
        if img is None:
            return img
            
        color_bgr = self.colors.get(color, self.colors['detection'])
        
        # Draw circle
        cv2.circle(img, (x, y), radius, color_bgr, self.line_thickness)
        
        # Add label if provided
        if label:
            cv2.putText(img, label, (x + radius + 5, y + 5), self.font, 
                       self.font_scale, color_bgr, self.font_thickness)
        
        return img
    
    def add_info_panel(self, img: np.ndarray, info: Dict[str, Any], 
                      x: int = 10, y: int = 10) -> np.ndarray:
        """
        Add information panel to image
        
        Args:
            img: Image to draw on
            info: Dictionary of information to display
            x, y: Top-left position for panel
        """
        if img is None or not info:
            return img
            
        line_height = 25
        current_y = y
        
        for key, value in info.items():
            text = f"{key}: {value}"
            cv2.putText(img, text, (x, current_y), self.font, 
                       self.font_scale, self.colors['text'], self.font_thickness)
            current_y += line_height
        
        return img
    
    def generate_heatmap(self, data: np.ndarray, width: int, height: int, 
                        title: str = "Heatmap") -> np.ndarray:
        """
        Generate heatmap visualization from 1D or 2D data
        
        Args:
            data: Data array to visualize
            width: Output width
            height: Output height  
            title: Heatmap title
            
        Returns:
            Heatmap image as BGR array
        """
        try:
            if data.ndim == 1:
                # Convert 1D to 2D by repeating rows
                data_2d = np.tile(data, (height // 4, 1))
            else:
                data_2d = data
            
            # Normalize to 0-255 range
            normalized = cv2.normalize(data_2d, None, 0, 255, cv2.NORM_MINMAX)
            normalized = normalized.astype(np.uint8)
            
            # Apply colormap
            heatmap = cv2.applyColorMap(normalized, cv2.COLORMAP_JET)
            
            # Resize to target dimensions
            if heatmap.shape[:2] != (height, width):
                heatmap = cv2.resize(heatmap, (width, height))
            
            # Add title
            cv2.putText(heatmap, title, (10, 30), self.font, 
                       0.8, self.colors['text'], self.font_thickness)
            
            return heatmap
            
        except Exception as e:
            print(f"❌ Error generating heatmap: {e}")
            # Return blank image on error
            blank = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.putText(blank, f"Heatmap Error: {str(e)[:50]}", (10, height//2), 
                       self.font, 0.6, self.colors['error'], 1)
            return blank
    
    def save_visualization(self, img: np.ndarray, base_path: str, 
                          suffix: str = "visualization", 
                          output_dir: str = "pic/screenshots") -> Optional[str]:
        """
        Save visualization with timestamped filename
        
        Args:
            img: Image to save
            base_path: Base image path for naming
            suffix: Filename suffix
            output_dir: Output directory
            
        Returns:
            Saved filename or None if failed
        """
        try:
            if img is None:
                return None
                
            # Ensure output directory exists
            os.makedirs(output_dir, exist_ok=True)
            
            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(base_path))[0]
            filename = f"{timestamp}_{base_name}_{suffix}.png"
            output_path = os.path.join(output_dir, filename)
            
            # Save image
            success = cv2.imwrite(output_path, img)
            if success:
                print(f"✅ Visualization saved: {filename}")
                return filename
            else:
                print(f"❌ Failed to save visualization: {output_path}")
                return None
                
        except Exception as e:
            print(f"❌ Error saving visualization: {e}")
            return None
    
    def create_composite_visualization(self, images: List[np.ndarray], 
                                     labels: List[str] = None,
                                     arrangement: str = 'horizontal') -> Optional[np.ndarray]:
        """
        Create composite visualization from multiple images
        
        Args:
            images: List of images to combine
            labels: Optional labels for each image
            arrangement: 'horizontal' or 'vertical'
            
        Returns:
            Composite image or None if failed
        """
        try:
            if not images:
                return None
                
            # Filter out None images
            valid_images = [img for img in images if img is not None]
            if not valid_images:
                return None
            
            if arrangement == 'horizontal':
                composite = np.hstack(valid_images)
            else:  # vertical
                composite = np.vstack(valid_images)
            
            # Add labels if provided
            if labels and len(labels) >= len(valid_images):
                label_height = 30
                for i, label in enumerate(labels[:len(valid_images)]):
                    if arrangement == 'horizontal':
                        x = sum(img.shape[1] for img in valid_images[:i]) + 10
                        y = label_height
                    else:  # vertical
                        x = 10
                        y = sum(img.shape[0] for img in valid_images[:i]) + label_height
                    
                    cv2.putText(composite, label, (x, y), self.font, 
                               0.8, self.colors['text'], self.font_thickness)
            
            return composite
            
        except Exception as e:
            print(f"❌ Error creating composite visualization: {e}")
            return None


# Utility functions for backward compatibility and convenience
def create_width_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for width detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect width using the detector
    detection_result = detector.detect_width(image_path)
    if detection_result is None:
        return None
    
    left_boundary, right_boundary, width = detection_result
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw boundaries
    engine.draw_vertical_line(img, left_boundary, 'boundary', f"L:{left_boundary}")
    engine.draw_vertical_line(img, right_boundary, 'boundary', f"R:{right_boundary}", offset=30)
    
    # Add width info
    info = {'Width': f"{width}px"}
    engine.add_info_panel(img, info, left_boundary, 80)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"SimpleWidth_{width}px")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


def create_avatar_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for avatar detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect avatars using the detector
    avatars, avatar_info = detector.detect_avatars(image_path)
    if not avatars:
        return None
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw avatar detections
    for i, avatar in enumerate(avatars):
        x, y, w, h = avatar['x'], avatar['y'], avatar['width'], avatar['height']
        engine.draw_rectangle(img, x, y, w, h, 'avatar', f"Avatar {i+1}")
        
        # Mark center point
        center_x, center_y = x + w//2, y + h//2
        engine.draw_circle(img, center_x, center_y, 3, 'detection')
    
    # Add info panel
    info = {
        'Avatars': len(avatars),
        'Method': avatar_info.get('method', 'Unknown'),
        'Processing': f"{avatar_info.get('processing_time', 0):.2f}s"
    }
    engine.add_info_panel(img, info)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"Avatars_{len(avatars)}_detected")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


def create_card_visualization(detector, image_path: str, output_path: str = None) -> Optional[str]:
    """
    Wrapper for card detector visualization using visualization engine
    """
    engine = VisualizationEngine()
    
    # Detect cards using the detector
    cards, card_info = detector.detect_cards(image_path)
    if not cards:
        return None
    
    # Create base overlay
    img = engine.create_base_overlay(image_path)
    if img is None:
        return None
    
    # Draw card detections
    for i, card in enumerate(cards):
        x, y, w, h = card['x'], card['y'], card['width'], card['height']
        engine.draw_rectangle(img, x, y, w, h, 'card', f"Card {i+1}")
    
    # Add info panel
    info = {
        'Cards': len(cards),
        'Method': card_info.get('method', 'Unknown'),
        'Processing': f"{card_info.get('processing_time', 0):.2f}s"
    }
    engine.add_info_panel(img, info)
    
    # Save visualization
    if output_path is None:
        return engine.save_visualization(img, image_path, f"Cards_{len(cards)}_detected")
    else:
        success = cv2.imwrite(output_path, img)
        return os.path.basename(output_path) if success else None


    # Helper Methods for DiagnosticVisualizationEngine
    
    def _categorize_detections(self, enhanced_cards: List[Dict], detector_type: str) -> Tuple[List[Dict], List[Dict]]:
        """Categorize cards into successful and failed detections"""
        successful_cards = []
        failed_cards = []
        
        for card in enhanced_cards:
            if self._is_detection_successful(card, detector_type):
                successful_cards.append(card)
            else:
                failed_cards.append(card)
        
        return successful_cards, failed_cards
    
    def _is_detection_successful(self, card: Dict, detector_type: str) -> bool:
        """Check if detection was successful for a given card and detector type"""
        if detector_type == 'contact_name':
            return card.get("name_boundary") is not None
        elif detector_type == 'time_detection':
            return card.get("time_box") is not None
        else:
            return False  # Unknown detector type
    
    def _get_section_number(self, detector_type: str) -> int:
        """Get section number for filename generation"""
        section_map = {
            'contact_name': 5,
            'time_detection': 6,
            'card_boundary': 4,
            'avatar_detection': 3
        }
        return section_map.get(detector_type, 0)
    
    def _get_detector_display_name(self, detector_type: str) -> str:
        """Get display name for detector type"""
        name_map = {
            'contact_name': 'Contact Name Detection',
            'time_detection': 'Time Detection', 
            'card_boundary': 'Card Boundary Detection',
            'avatar_detection': 'Avatar Detection'
        }
        return name_map.get(detector_type, 'Unknown Detection')
    
    def _create_main_overview_panel(self, ax: plt.Axes, img_rgb: np.ndarray, 
                                  enhanced_cards: List[Dict], detector_type: str) -> None:
        """Create the main overview panel with WeChat image and detection overlays"""
        ax.imshow(img_rgb)
        
        # Count successful and failed detections
        successful_count = 0
        failed_count = 0
        
        # Draw card annotations and search regions
        for i, card in enumerate(enhanced_cards):
            success = self._is_detection_successful(card, detector_type)
            if success:
                successful_count += 1
            else:
                failed_count += 1
            
            # Draw card boundaries and annotations
            self.components.draw_card_annotations(ax, card, i+1, success)
            
            # Draw search regions if available in debug data
            if detector_type == 'contact_name':
                # For contact name detection, show search regions right of avatars
                avatar_data = card.get("avatar", {})
                card_bbox = card.get("bbox", [0, 0, 0, 0])
                if avatar_data and card_bbox:
                    avatar_bbox = avatar_data.get("bbox", [0, 0, 0, 0])
                    # Calculate search region (right of avatar)
                    search_left = avatar_bbox[0] + avatar_bbox[2] + 5
                    search_right = card_bbox[0] + card_bbox[2] - 10
                    search_top = avatar_bbox[1] - 12
                    search_bottom = avatar_bbox[1] + avatar_bbox[3] // 2
                    
                    if search_right > search_left and search_bottom > search_top:
                        self.components.draw_search_region(
                            ax, (search_left, search_top, 
                                search_right - search_left, search_bottom - search_top),
                            success, alpha=0.2
                        )
        
        # Set title with success/failure summary
        total_cards = len(enhanced_cards)
        title = f'{self._get_detector_display_name(detector_type)} Overview - {total_cards} Cards Analyzed'
        subtitle = f'✅ {successful_count} Successful  ❌ {failed_count} Failed'
        
        ax.set_title(title, **self.styling.FONTS['title_medium'])
        ax.set_xlabel(subtitle, **self.styling.FONTS['label_large'])
        
        ax.set_xlim(0, img_rgb.shape[1])
        ax.set_ylim(img_rgb.shape[0], 0)
    
    def _create_roi_analysis_panels(self, axes: Dict[str, plt.Axes], cards: List[Dict],
                                  debug_data: Dict, detector_type: str) -> None:
        """Create individual card ROI analysis panels"""
        for i, card in enumerate(cards):
            if i >= 6:  # Limit to 6 ROI panels
                break
                
            ax_key = f'roi_{i+1}'
            if ax_key not in axes:
                continue
                
            ax = axes[ax_key]
            
            # Get ROI image from debug data if available
            card_debug = debug_data.get(f'card_{card.get("card_id", i+1)}', {})
            roi_image = card_debug.get('roi_original')
            
            if roi_image is not None:
                # Display ROI image
                if len(roi_image.shape) == 3:
                    roi_rgb = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
                else:
                    roi_rgb = roi_image
                ax.imshow(roi_rgb, cmap='gray' if len(roi_image.shape) == 2 else None)
                
                # Add success/failure indicator
                success = self._is_detection_successful(card, detector_type)
                status = "✅" if success else "❌"
                ax.set_title(f'{status} Card {card.get("card_id", i+1)} ROI', 
                           **self.styling.FONTS['title_small'])
            else:
                # No ROI data available, show placeholder
                ax.text(0.5, 0.5, f'Card {card.get("card_id", i+1)}\nROI\n(No debug data)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=self.styling.FONTS['label_medium']['size'])
                ax.set_title(f'Card {card.get("card_id", i+1)} ROI', 
                           **self.styling.FONTS['title_small'])
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    def _create_binary_processing_panels(self, axes: Dict[str, plt.Axes], cards: List[Dict],
                                       debug_data: Dict, detector_type: str) -> None:
        """Create binary processing visualization panels"""
        for i, card in enumerate(cards[:3]):  # Show first 3 cards' processing
            ax_key = f'binary_{i+1}'
            if ax_key not in axes:
                continue
                
            ax = axes[ax_key]
            
            # Get binary processing image from debug data
            card_debug = debug_data.get(f'card_{card.get("card_id", i+1)}', {})
            
            # Try to get appropriate binary image based on detector type
            if detector_type == 'contact_name':
                binary_image = card_debug.get('roi_white_threshold') or card_debug.get('roi_morphological')
            elif detector_type == 'time_detection':
                binary_image = card_debug.get('roi_binary')
            else:
                binary_image = None
            
            if binary_image is not None:
                ax.imshow(binary_image, cmap='gray')
                ax.set_title(f'Card {card.get("card_id", i+1)} Binary', 
                           **self.styling.FONTS['title_small'])
            else:
                ax.text(0.5, 0.5, f'Card {card.get("card_id", i+1)}\nBinary\n(No debug data)',
                       ha='center', va='center', transform=ax.transAxes,
                       fontsize=self.styling.FONTS['label_medium']['size'])
                ax.set_title(f'Card {card.get("card_id", i+1)} Binary', 
                           **self.styling.FONTS['title_small'])
            
            ax.set_xticks([])
            ax.set_yticks([])
    
    def _create_statistical_analysis_panels(self, axes: Dict[str, plt.Axes], enhanced_cards: List[Dict],
                                          debug_data: Dict, detector_type: str) -> None:
        """Create statistical analysis panels with histograms and projections"""
        
        # Panel 1: Detection confidence histogram
        if 'stats_1' in axes:
            ax = axes['stats_1']
            confidence_scores = []
            
            for card in enhanced_cards:
                if detector_type == 'contact_name' and card.get("name_boundary"):
                    confidence_scores.append(card["name_boundary"].get("confidence", 0.5))
                elif detector_type == 'time_detection' and card.get("time_box"):
                    confidence_scores.append(card["time_box"].get("density_score", 0.5))
            
            if confidence_scores:
                self.components.create_statistics_histogram(
                    ax, np.array(confidence_scores), 
                    'Detection Confidence Distribution', 
                    threshold=0.5
                )
            else:
                ax.text(0.5, 0.5, 'No confidence data available',
                       ha='center', va='center', transform=ax.transAxes)
        
        # Panel 2: Success rate analysis
        if 'stats_2' in axes:
            ax = axes['stats_2']
            successful_cards, failed_cards = self._categorize_detections(enhanced_cards, detector_type)
            
            # Simple bar chart showing success vs failure
            categories = ['Successful', 'Failed']
            counts = [len(successful_cards), len(failed_cards)]
            colors = [self.styling.COLORS['success'], self.styling.COLORS['failure']]
            
            bars = ax.bar(categories, counts, color=colors, alpha=0.7, edgecolor='black')
            
            # Add count labels on bars
            for bar, count in zip(bars, counts):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       f'{count}', ha='center', va='bottom',
                       **self.styling.FONTS['label_medium'])
            
            ax.set_title('Detection Results Summary', **self.styling.FONTS['title_small'])
            ax.set_ylabel('Card Count', **self.styling.FONTS['label_medium'])
            ax.grid(True, alpha=0.3)


# Maintain backward compatibility by aliasing the old class name
VisualizationEngine = DiagnosticVisualizationEngine


if __name__ == "__main__":
    print("🎨 Diagnostic Visualization Engine Module Test")
    print("=" * 50)
    
    # Test diagnostic visualization engine
    engine = DiagnosticVisualizationEngine()
    print("✅ Diagnostic Visualization Engine initialized")
    
    # Test basic functionality (backward compatibility)
    test_img = np.zeros((400, 600, 3), dtype=np.uint8)
    
    # Draw some test elements
    engine.draw_vertical_line(test_img, 150, 'boundary', "Left Boundary")
    engine.draw_vertical_line(test_img, 450, 'boundary', "Right Boundary", offset=30)
    engine.draw_rectangle(test_img, 200, 100, 100, 80, 'card', "Test Card")
    engine.draw_circle(test_img, 250, 140, 5, 'avatar', "Avatar")
    
    info = {'Test': 'Diagnostic Visualization Engine', 'Status': 'Working'}
    engine.add_info_panel(test_img, info, 10, 300)
    
    # Save test image
    test_path = engine.save_visualization(test_img, "test_image.png", "engine_test")
    if test_path:
        print(f"✅ Test visualization created: {test_path}")
    else:
        print("❌ Test visualization failed")