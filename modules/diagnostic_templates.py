#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Diagnostic Templates Module (diagnostic_templates.py)

Reusable visualization components and templates for consistent diagnostic output
across all card processing detectors.

Provides:
- Standardized color schemes and styling
- Layout templates and GridSpec configurations  
- Common visualization elements and annotations
- Filename generation patterns
"""

import os
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend for server compatibility
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

class cDiagnosticStyling:
    """Centralized styling constants for consistent visualization"""
    
    # Color Schemes
    COLORS = {
        # Detection status colors
        'success': '#00FF00',      # Bright green for successful detections
        'failure': '#FF0000',      # Bright red for failed detections  
        'warning': '#FFA500',      # Orange for warnings
        'info': '#00BFFF',         # Deep sky blue for information
        
        # Component colors (matching existing system)
        'card_boundary': '#0000FF',        # Blue for card boundaries
        'avatar_boundary': '#00FF00',      # Green for avatar boundaries  
        'avatar_center': '#FF0000',        # Red for avatar centers
        'name_boundary': '#00A5FF',        # Orange for name boundaries
        'time_boundary': '#800080',        # Purple for time boundaries
        'search_region': '#808080',        # Gray for search regions
        'divider_line': '#FFFF00',         # Yellow for divider lines
        
        # Chart colors
        'histogram_primary': '#FF4444',    # Red for primary histograms
        'histogram_secondary': '#4444FF',  # Blue for secondary histograms
        'threshold_line': '#FFA500',       # Orange for threshold lines
        'profile_line': '#0066CC',         # Blue for profile lines
        
        # Background colors
        'success_alpha': (0, 1, 0, 0.3),   # Green with 30% transparency
        'failure_alpha': (1, 0, 0, 0.5),   # Red with 50% transparency
        'info_alpha': (0, 0, 1, 0.1),      # Blue with 10% transparency
    }
    
    # Font Settings
    FONTS = {
        'title_large': {'size': 16, 'weight': 'bold'},
        'title_medium': {'size': 14, 'weight': 'bold'},
        'title_small': {'size': 12, 'weight': 'bold'},
        'label_large': {'size': 12, 'weight': 'normal'},
        'label_medium': {'size': 10, 'weight': 'normal'},
        'label_small': {'size': 8, 'weight': 'normal'},
        'annotation': {'size': 9, 'weight': 'normal'},
        'legend': {'size': 10, 'weight': 'normal'}
    }
    
    # Layout Settings
    LAYOUT = {
        'figure_dpi': 300,
        'figure_size_large': (20, 16),
        'figure_size_medium': (16, 12),
        'figure_size_small': (12, 8),
        'margins': {'left': 0.08, 'right': 0.95, 'top': 0.92, 'bottom': 0.08},
        'spacing': {'wspace': 0.3, 'hspace': 0.3},
        'bbox_tight': True
    }

class cLayoutTemplates:
    """Pre-configured layout templates for different visualization types"""
    
    @staticmethod
    def comprehensive_debug_layout(fig: plt.Figure) -> Dict[str, plt.Axes]:
        """
        Create comprehensive debug layout matching time detection format
        
        Layout:
        - Main overview (top 2 rows, left 3 columns)
        - ROI previews (middle section, 6 smaller panels)
        - Statistical analysis (bottom section, 3 wider panels)
        
        Returns:
            Dictionary of axis objects with descriptive keys
        """
        gs = GridSpec(4, 6, figure=fig, 
                     hspace=cDiagnosticStyling.LAYOUT['spacing']['hspace'],
                     wspace=cDiagnosticStyling.LAYOUT['spacing']['wspace'])
        
        axes = {}
        
        # Main overview panel (spans top 2 rows, left 3 columns)
        axes['main_overview'] = fig.add_subplot(gs[0:2, 0:3])
        
        # ROI preview panels (6 smaller panels in grid)
        for i in range(6):
            row = 2 + i // 3  # Start from row 2
            col = i % 3
            axes[f'roi_{i+1}'] = fig.add_subplot(gs[row, col])
        
        # Binary processing panels (3 panels in row 3, right side)
        for i in range(3):
            axes[f'binary_{i+1}'] = fig.add_subplot(gs[2, i + 3])
        
        # Statistical analysis panels (bottom row, 3 wider panels)
        for i in range(3):
            axes[f'stats_{i+1}'] = fig.add_subplot(gs[3, i*2:(i+1)*2])
            
        return axes
    
    @staticmethod  
    def simple_overlay_layout(fig: plt.Figure) -> Dict[str, plt.Axes]:
        """Create simple overlay layout with main image and legend"""
        gs = GridSpec(1, 1, figure=fig)
        return {'main': fig.add_subplot(gs[0, 0])}
    
    @staticmethod
    def comparison_layout(fig: plt.Figure) -> Dict[str, plt.Axes]:
        """Create before/after comparison layout"""
        gs = GridSpec(1, 2, figure=fig,
                     wspace=cDiagnosticStyling.LAYOUT['spacing']['wspace'])
        return {
            'before': fig.add_subplot(gs[0, 0]),
            'after': fig.add_subplot(gs[0, 1])
        }

class cVisualizationComponents:
    """Reusable visualization components"""
    
    @staticmethod
    def draw_card_annotations(ax: plt.Axes, card_data: Dict, card_index: int, 
                            success: bool = True) -> None:
        """
        Draw standardized card annotations (boundaries, centers, labels)
        
        Args:
            ax: Matplotlib axis object
            card_data: Dictionary containing card boundary and avatar info
            card_index: Card number for labeling
            success: Whether detection was successful (affects color coding)
        """
        colors = cDiagnosticStyling.COLORS
        
        # Extract card and avatar data
        card_bbox = card_data["bbox"]  # (x, y, w, h)
        avatar_data = card_data["avatar"]
        avatar_bbox = avatar_data["bbox"]
        avatar_center = avatar_data["center"]
        
        # Draw card boundary (blue rectangle)
        card_rect = patches.Rectangle(
            (card_bbox[0], card_bbox[1]), card_bbox[2], card_bbox[3],
            linewidth=2, edgecolor=colors['card_boundary'], facecolor='none'
        )
        ax.add_patch(card_rect)
        
        # Draw avatar boundary (green rectangle)
        avatar_rect = patches.Rectangle(
            (avatar_bbox[0], avatar_bbox[1]), avatar_bbox[2], avatar_bbox[3],
            linewidth=2, edgecolor=colors['avatar_boundary'], facecolor='none'
        )
        ax.add_patch(avatar_rect)
        
        # Draw avatar center (red circle)
        ax.scatter(*avatar_center, c=colors['avatar_center'], s=50, zorder=10)
        
        # Draw horizontal divider line through avatar center
        ax.axhline(y=avatar_center[1], xmin=card_bbox[0]/ax.get_xlim()[1], 
                  xmax=(card_bbox[0]+card_bbox[2])/ax.get_xlim()[1],
                  color=colors['divider_line'], linewidth=1)
        
        # Add card label with success/failure indicator
        status_color = colors['success'] if success else colors['failure']
        status_symbol = '✅' if success else '❌'
        
        ax.text(card_bbox[0] + 5, card_bbox[1] + 20, f'{status_symbol} Card {card_index}',
                fontsize=cDiagnosticStyling.FONTS['label_medium']['size'],
                color='white', fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.3", facecolor=status_color, alpha=0.8))
    
    @staticmethod
    def draw_search_region(ax: plt.Axes, region_bbox: Tuple[int, int, int, int],
                          success: bool = True, alpha: float = 0.3) -> None:
        """Draw search region overlay with appropriate color coding"""
        colors = cDiagnosticStyling.COLORS
        color = colors['success'] if success else colors['failure']
        
        search_rect = patches.Rectangle(
            (region_bbox[0], region_bbox[1]), region_bbox[2], region_bbox[3],
            linewidth=1, edgecolor=color, 
            facecolor=color, alpha=alpha
        )
        ax.add_patch(search_rect)
    
    @staticmethod
    def create_detection_legend(ax: plt.Axes, detector_type: str) -> None:
        """Create standardized legend for detection visualizations"""
        colors = cDiagnosticStyling.COLORS
        
        legend_elements = [
            patches.Patch(color=colors['card_boundary'], label='Card Boundaries'),
            patches.Patch(color=colors['avatar_boundary'], label='Avatar Boundaries'), 
            patches.Patch(color=colors['avatar_center'], label='Avatar Centers'),
            patches.Patch(color=colors['divider_line'], label='Divider Lines'),
        ]
        
        # Add detector-specific elements
        if detector_type == 'contact_name':
            legend_elements.append(
                patches.Patch(color=colors['name_boundary'], label='Name Boundaries')
            )
            legend_elements.append(
                patches.Patch(color=colors['search_region'], label='Search Regions')
            )
        elif detector_type == 'time_detection':
            legend_elements.append(
                patches.Patch(color=colors['time_boundary'], label='Time Boundaries')
            )
        
        legend_elements.extend([
            patches.Patch(color=colors['success'], label='✅ Successful Detection'),
            patches.Patch(color=colors['failure'], label='❌ Failed Detection')
        ])
        
        ax.legend(handles=legend_elements, 
                 loc='upper right', 
                 fontsize=cDiagnosticStyling.FONTS['legend']['size'],
                 framealpha=0.9)
    
    @staticmethod
    def create_statistics_histogram(ax: plt.Axes, data: np.ndarray, 
                                  title: str, threshold: Optional[float] = None) -> None:
        """Create standardized histogram for statistical analysis"""
        colors = cDiagnosticStyling.COLORS
        
        # Create histogram
        ax.hist(data, bins=30, alpha=0.7, color=colors['histogram_primary'],
               edgecolor='black', linewidth=0.5)
        
        # Add threshold line if provided
        if threshold is not None:
            ax.axvline(x=threshold, color=colors['threshold_line'], 
                      linestyle='--', linewidth=2, 
                      label=f'Threshold: {threshold:.2f}')
            ax.legend()
        
        # Styling
        ax.set_title(title, **cDiagnosticStyling.FONTS['title_small'])
        ax.grid(True, alpha=0.3)
        ax.set_ylabel('Frequency', **cDiagnosticStyling.FONTS['label_medium'])

class cFilenameGenerator:
    """Standardized filename generation for diagnostic outputs"""
    
    @staticmethod
    def generate_debug_filename(detector_section: int, detector_name: str, 
                              success_count: int, failed_count: int) -> str:
        """
        Generate standardized debug visualization filename
        
        Args:
            detector_section: Section number (5 for contact name, 6 for time)
            detector_name: Name of detector (e.g., 'ContactNameDetection', 'TimeDetection')  
            success_count: Number of successful detections
            failed_count: Number of failed detections
            
        Returns:
            Standardized filename with timestamp
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{detector_section:02d}_Debug_{detector_name}_{success_count}success_{failed_count}failed.png"
    
    @staticmethod
    def generate_simple_filename(detector_section: int, detector_name: str,
                               detection_count: int, total_count: int) -> str:
        """Generate filename for simple overlay visualizations"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return f"{timestamp}_{detector_section:02d}_{detector_name}_{detection_count}of{total_count}.png"
    
    @staticmethod
    def get_output_directory() -> str:
        """Get standardized output directory for diagnostic files"""
        output_dir = "pic/screenshots"
        os.makedirs(output_dir, exist_ok=True)
        return output_dir