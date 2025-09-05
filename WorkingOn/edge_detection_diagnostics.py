#!/usr/bin/env python3
"""
Comprehensive Visual Diagnostic Tool for Right Edge Detection
Provides step-by-step analysis of white-to-black transition detection method
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional

class EdgeDetectionDiagnostics:
    """Comprehensive diagnostic tool for analyzing right edge detection accuracy"""
    
    def __init__(self):
        self.output_dir = "pic/diagnostics"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Detection parameters (matching RightBoundaryDetector)
        self.PREPROCESSED_THRESHOLD = 0.005  # 0.5% threshold
        self.SMOOTHING_SIZE = 5
        self.MIN_BOUNDARY_PX = 800
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
    def analyze_right_edge_detection(self, preprocessed_image_path: str) -> Dict:
        """
        Comprehensive step-by-step analysis of right edge detection method
        
        Args:
            preprocessed_image_path: Path to high-contrast preprocessed image
            
        Returns:
            Dictionary with analysis results and metrics
        """
        print("🔬 COMPREHENSIVE RIGHT EDGE DETECTION ANALYSIS")
        print("=" * 60)
        
        results = {
            'input_path': preprocessed_image_path,
            'timestamp': self.timestamp,
            'steps': {},
            'final_result': None,
            'metrics': {}
        }
        
        # Step 1: Input Image Analysis
        print("\n📊 STEP 1: INPUT IMAGE ANALYSIS")
        print("-" * 40)
        adjusted, step1_results = self._step1_input_analysis(preprocessed_image_path)
        results['steps']['step1'] = step1_results
        
        # Step 2: Horizontal Difference Calculation  
        print("\n📊 STEP 2: HORIZONTAL DIFFERENCE CALCULATION")
        print("-" * 40)
        diff_x, step2_results = self._step2_horizontal_differences(adjusted)
        results['steps']['step2'] = step2_results
        
        # Step 3: Vertical Projection Creation
        print("\n📊 STEP 3: VERTICAL PROJECTION CREATION")
        print("-" * 40)
        raw_projection, step3_results = self._step3_vertical_projection(diff_x)
        results['steps']['step3'] = step3_results
        
        # Step 4: Smoothing and Enhancement
        print("\n📊 STEP 4: SMOOTHING AND ENHANCEMENT")
        print("-" * 40)
        smoothed_projection, step4_results = self._step4_smoothing(raw_projection)
        results['steps']['step4'] = step4_results
        
        # Step 5: Threshold Calculation
        print("\n📊 STEP 5: THRESHOLD CALCULATION")
        print("-" * 40)
        threshold, step5_results = self._step5_threshold_calculation(smoothed_projection)
        results['steps']['step5'] = step5_results
        
        # Step 6: Peak Detection
        print("\n📊 STEP 6: PEAK DETECTION")
        print("-" * 40)
        candidates, step6_results = self._step6_peak_detection(smoothed_projection, threshold, adjusted.shape[1])
        results['steps']['step6'] = step6_results
        
        # Step 7: Boundary Selection
        print("\n📊 STEP 7: BOUNDARY SELECTION")
        print("-" * 40)
        final_boundary, step7_results = self._step7_boundary_selection(candidates)
        results['steps']['step7'] = step7_results
        results['final_result'] = final_boundary
        
        # Generate comprehensive visual report
        self._generate_visual_report(results, adjusted, raw_projection, smoothed_projection, 
                                   threshold, candidates, final_boundary)
        
        # Calculate final metrics
        results['metrics'] = self._calculate_accuracy_metrics(results)
        
        return results
    
    def _step1_input_analysis(self, image_path: str) -> Tuple[np.ndarray, Dict]:
        """Step 1: Analyze input preprocessed image"""
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        
        results = {
            'image_shape': img.shape,
            'pixel_min': int(img.min()),
            'pixel_max': int(img.max()),
            'pixel_mean': float(img.mean()),
            'pixel_std': float(img.std()),
            'white_pixels': int(np.sum(img > 200)),
            'black_pixels': int(np.sum(img < 50)),
            'contrast_ratio': float(img.max() / max(img.min(), 1))
        }
        
        print(f"📐 Image dimensions: {results['image_shape'][1]}x{results['image_shape'][0]}")
        print(f"📊 Pixel value range: {results['pixel_min']}-{results['pixel_max']}")
        print(f"📊 Mean/Std: {results['pixel_mean']:.1f} ± {results['pixel_std']:.1f}")
        print(f"⚫ Black pixels (<50): {results['black_pixels']:,} ({results['black_pixels']/img.size*100:.1f}%)")
        print(f"⚪ White pixels (>200): {results['white_pixels']:,} ({results['white_pixels']/img.size*100:.1f}%)")
        print(f"🔍 Contrast ratio: {results['contrast_ratio']:.1f}:1")
        
        # Save input visualization
        self._save_step_visualization(img, "01_input_analysis", 
                                    "Input Preprocessed Image Analysis",
                                    colormap='gray')
        
        return img, results
    
    def _step2_horizontal_differences(self, img: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Step 2: Calculate horizontal pixel differences to find transitions"""
        diff_x = np.diff(img.astype(np.int16), axis=1)
        
        results = {
            'diff_shape': diff_x.shape,
            'diff_min': int(diff_x.min()),
            'diff_max': int(diff_x.max()),
            'diff_mean': float(diff_x.mean()),
            'positive_transitions': int(np.sum(diff_x > 0)),  # black-to-white
            'negative_transitions': int(np.sum(diff_x < 0)),  # white-to-black
            'strong_transitions': int(np.sum(np.abs(diff_x) > 50))
        }
        
        print(f"📐 Difference array shape: {results['diff_shape']}")
        print(f"📊 Difference range: {results['diff_min']} to {results['diff_max']}")
        print(f"➡️ Black-to-white transitions: {results['positive_transitions']:,}")
        print(f"⬅️ White-to-black transitions: {results['negative_transitions']:,}")
        print(f"💪 Strong transitions (|diff|>50): {results['strong_transitions']:,}")
        
        # Save horizontal differences visualization
        self._save_step_visualization(diff_x, "02_horizontal_differences",
                                    "Horizontal Pixel Differences (Transitions)",
                                    colormap='RdBu', center_zero=True)
        
        return diff_x, results
    
    def _step3_vertical_projection(self, diff_x: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Step 3: Create vertical projection by summing transitions"""
        # Sum differences vertically, then invert to make right edges positive
        transition_profile = np.sum(diff_x, axis=0)
        edge_projection = -transition_profile  # Invert so white-to-black becomes positive
        
        results = {
            'projection_length': len(edge_projection),
            'projection_min': float(edge_projection.min()),
            'projection_max': float(edge_projection.max()),
            'projection_mean': float(edge_projection.mean()),
            'projection_std': float(edge_projection.std()),
            'significant_peaks': int(np.sum(edge_projection > edge_projection.mean() + edge_projection.std()))
        }
        
        print(f"📏 Projection length: {results['projection_length']} pixels")
        print(f"📊 Projection range: {results['projection_min']:.0f} to {results['projection_max']:.0f}")
        print(f"📊 Mean/Std: {results['projection_mean']:.1f} ± {results['projection_std']:.1f}")
        print(f"🏔️ Significant peaks (>mean+std): {results['significant_peaks']}")
        
        # Save projection graph
        self._save_projection_graph(edge_projection, "03_raw_projection", 
                                  "Raw Vertical Projection (Right Edge Signals)")
        
        return edge_projection, results
    
    def _step4_smoothing(self, projection: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """Step 4: Apply smoothing to enhance true boundaries"""
        smoothed = uniform_filter1d(projection, size=self.SMOOTHING_SIZE, mode='nearest')
        
        # Compare smoothing effectiveness
        noise_reduction = np.std(projection) - np.std(smoothed)
        peak_preservation = np.corrcoef(projection, smoothed)[0, 1]
        
        results = {
            'smoothing_kernel': self.SMOOTHING_SIZE,
            'original_std': float(np.std(projection)),
            'smoothed_std': float(np.std(smoothed)),
            'noise_reduction': float(noise_reduction),
            'peak_preservation': float(peak_preservation),
            'max_value_change': float(np.max(smoothed) - np.max(projection))
        }
        
        print(f"🔧 Smoothing kernel size: {results['smoothing_kernel']}")
        print(f"📊 Noise reduction: {results['noise_reduction']:.1f} (std: {results['original_std']:.1f} → {results['smoothed_std']:.1f})")
        print(f"📊 Peak preservation: {results['peak_preservation']:.3f} correlation")
        print(f"📊 Max value change: {results['max_value_change']:+.1f}")
        
        # Save smoothing comparison
        self._save_smoothing_comparison(projection, smoothed, "04_smoothing_comparison")
        
        return smoothed, results
    
    def _step5_threshold_calculation(self, smoothed_projection: np.ndarray) -> Tuple[float, Dict]:
        """Step 5: Calculate adaptive threshold for peak detection"""
        max_strength = np.max(smoothed_projection)
        threshold = max_strength * self.PREPROCESSED_THRESHOLD
        
        # Analyze threshold sensitivity
        above_threshold = np.sum(smoothed_projection > threshold)
        threshold_percentile = (1 - np.sum(smoothed_projection < threshold) / len(smoothed_projection)) * 100
        
        results = {
            'max_strength': float(max_strength),
            'threshold_percentage': self.PREPROCESSED_THRESHOLD,
            'threshold_value': float(threshold),
            'pixels_above_threshold': int(above_threshold),
            'threshold_percentile': float(threshold_percentile)
        }
        
        print(f"📊 Maximum projection strength: {results['max_strength']:.0f}")
        print(f"📊 Threshold percentage: {results['threshold_percentage']*100:.1f}%")
        print(f"📊 Threshold value: {results['threshold_value']:.0f}")
        print(f"📊 Pixels above threshold: {results['pixels_above_threshold']} ({results['threshold_percentile']:.1f}%)")
        
        return threshold, results
    
    def _step6_peak_detection(self, smoothed_projection: np.ndarray, threshold: float, img_width: int) -> Tuple[List[Tuple[int, float]], Dict]:
        """Step 6: Find peaks above threshold in search range"""
        search_start = int(img_width * 0.4)
        search_end = int(img_width * 0.95)
        
        candidates = []
        for x in range(search_start, search_end):
            if smoothed_projection[x] > threshold:
                # Check for local maximum
                is_peak = True
                if x > 0 and smoothed_projection[x] < smoothed_projection[x-1]:
                    is_peak = False
                if x < len(smoothed_projection) - 1 and smoothed_projection[x] < smoothed_projection[x+1]:
                    is_peak = False
                    
                if is_peak:
                    candidates.append((x, smoothed_projection[x]))
        
        results = {
            'search_start': search_start,
            'search_end': search_end,
            'search_width': search_end - search_start,
            'total_candidates': len(candidates),
            'strongest_candidate': max(candidates, key=lambda x: x[1]) if candidates else None,
            'rightmost_candidate': max(candidates, key=lambda x: x[0]) if candidates else None,
            'candidates_above_800': len([c for c in candidates if c[0] >= 800])
        }
        
        print(f"🔍 Search range: {search_start}-{search_end}px ({results['search_width']}px wide)")
        print(f"🏔️ Total candidates found: {results['total_candidates']}")
        if results['strongest_candidate']:
            print(f"💪 Strongest: x={results['strongest_candidate'][0]}px, strength={results['strongest_candidate'][1]:.0f}")
        if results['rightmost_candidate']:
            print(f"➡️ Rightmost: x={results['rightmost_candidate'][0]}px, strength={results['rightmost_candidate'][1]:.0f}")
        print(f"✅ Valid candidates (≥800px): {results['candidates_above_800']}")
        
        # Save peak detection visualization
        self._save_peak_detection(smoothed_projection, threshold, candidates, search_start, search_end, "06_peak_detection")
        
        return candidates, results
    
    def _step7_boundary_selection(self, candidates: List[Tuple[int, float]]) -> Tuple[int, Dict]:
        """Step 7: Select final boundary using selection criteria"""
        if not candidates:
            final_boundary = 800  # Fallback
            selection_reason = "No candidates found, using fallback"
        else:
            # Sort by position (rightmost first)
            candidates.sort(key=lambda c: c[0], reverse=True)
            
            # Find rightmost candidate that meets minimum boundary requirement
            selected = None
            for x, strength in candidates:
                if x >= self.MIN_BOUNDARY_PX:
                    selected = (x, strength)
                    selection_reason = f"Rightmost valid candidate ≥{self.MIN_BOUNDARY_PX}px"
                    break
            
            if selected is None:
                selected = candidates[0]  # Take rightmost available
                selection_reason = "Rightmost available candidate (below minimum threshold)"
            
            final_boundary = selected[0]
        
        results = {
            'final_boundary': final_boundary,
            'selection_reason': selection_reason,
            'min_boundary_requirement': self.MIN_BOUNDARY_PX,
            'total_candidates': len(candidates),
            'selected_candidate': selected if 'selected' in locals() else None
        }
        
        print(f"🎯 Final selected boundary: {results['final_boundary']}px")
        print(f"📋 Selection reason: {results['selection_reason']}")
        if results['selected_candidate']:
            print(f"📊 Selected strength: {results['selected_candidate'][1]:.0f}")
        
        return final_boundary, results
    
    def _save_step_visualization(self, img: np.ndarray, filename: str, title: str, 
                               colormap: str = 'gray', center_zero: bool = False):
        """Save visualization for a processing step"""
        plt.figure(figsize=(12, 8))
        
        if center_zero:
            vmax = max(abs(img.min()), abs(img.max()))
            plt.imshow(img, cmap=colormap, vmin=-vmax, vmax=vmax)
        else:
            plt.imshow(img, cmap=colormap)
        
        plt.colorbar(label='Pixel Value')
        plt.title(title)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Y Position (pixels)')
        
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}.png")
    
    def _save_projection_graph(self, projection: np.ndarray, filename: str, title: str):
        """Save projection graph visualization"""
        plt.figure(figsize=(15, 6))
        plt.plot(projection, linewidth=1.5, color='blue')
        plt.title(title)
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Projection Strength')
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}.png")
    
    def _save_smoothing_comparison(self, original: np.ndarray, smoothed: np.ndarray, filename: str):
        """Save smoothing before/after comparison"""
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        plt.plot(original, linewidth=1, color='red', alpha=0.7, label='Original')
        plt.plot(smoothed, linewidth=2, color='blue', label='Smoothed')
        plt.title('Projection Smoothing Comparison')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Projection Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.subplot(2, 1, 2)
        difference = original - smoothed
        plt.plot(difference, linewidth=1, color='green')
        plt.title('Difference (Original - Smoothed)')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Difference')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}.png")
    
    def _save_peak_detection(self, projection: np.ndarray, threshold: float, 
                           candidates: List[Tuple[int, float]], search_start: int, 
                           search_end: int, filename: str):
        """Save peak detection visualization"""
        plt.figure(figsize=(15, 8))
        
        # Plot projection
        plt.plot(projection, linewidth=1.5, color='blue', label='Smoothed Projection')
        
        # Plot threshold line
        plt.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold ({threshold:.0f})')
        
        # Plot search range
        plt.axvspan(search_start, search_end, alpha=0.2, color='yellow', label='Search Range')
        
        # Plot candidates
        if candidates:
            candidate_x = [c[0] for c in candidates]
            candidate_y = [c[1] for c in candidates]
            plt.scatter(candidate_x, candidate_y, color='red', s=50, zorder=5, label='Detected Peaks')
            
            # Annotate top 5 candidates
            top_candidates = sorted(candidates, key=lambda x: x[1], reverse=True)[:5]
            for i, (x, strength) in enumerate(top_candidates):
                plt.annotate(f'#{i+1}: {x}px\n{strength:.0f}', 
                           xy=(x, strength), xytext=(10, 10), 
                           textcoords='offset points', fontsize=8,
                           bbox=dict(boxstyle='round,pad=0.3', fc='yellow', alpha=0.7))
        
        plt.title('Peak Detection Analysis')
        plt.xlabel('X Position (pixels)')
        plt.ylabel('Projection Strength')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_{filename}.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: {filename}.png")
    
    def _generate_visual_report(self, results: Dict, original_img: np.ndarray,
                              raw_projection: np.ndarray, smoothed_projection: np.ndarray,
                              threshold: float, candidates: List[Tuple[int, float]], 
                              final_boundary: int):
        """Generate comprehensive visual report"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Top left: Original image with detected boundary
        axes[0, 0].imshow(original_img, cmap='gray')
        axes[0, 0].axvline(x=final_boundary, color='red', linewidth=3, label=f'Detected Edge: {final_boundary}px')
        axes[0, 0].set_title('Final Result: Detected Right Edge')
        axes[0, 0].set_xlabel('X Position (pixels)')
        axes[0, 0].set_ylabel('Y Position (pixels)')
        axes[0, 0].legend()
        
        # Top right: Raw vs smoothed projection
        axes[0, 1].plot(raw_projection, alpha=0.5, color='gray', label='Raw Projection')
        axes[0, 1].plot(smoothed_projection, color='blue', linewidth=2, label='Smoothed Projection')
        axes[0, 1].axhline(y=threshold, color='red', linestyle='--', label=f'Threshold: {threshold:.0f}')
        axes[0, 1].axvline(x=final_boundary, color='red', linewidth=2, label=f'Final Boundary: {final_boundary}px')
        axes[0, 1].set_title('Projection Analysis')
        axes[0, 1].set_xlabel('X Position (pixels)')
        axes[0, 1].set_ylabel('Projection Strength')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Bottom left: Candidate analysis
        if candidates:
            candidate_x = [c[0] for c in candidates]
            candidate_strength = [c[1] for c in candidates]
            colors = ['red' if x == final_boundary else 'blue' for x in candidate_x]
            axes[1, 0].scatter(candidate_x, candidate_strength, c=colors, s=60)
            axes[1, 0].set_title(f'Boundary Candidates ({len(candidates)} found)')
            axes[1, 0].set_xlabel('X Position (pixels)')
            axes[1, 0].set_ylabel('Candidate Strength')
            axes[1, 0].grid(True, alpha=0.3)
            
            # Annotate final selection
            final_strength = next((s for x, s in candidates if x == final_boundary), 0)
            axes[1, 0].annotate(f'SELECTED\n{final_boundary}px', 
                              xy=(final_boundary, final_strength), 
                              xytext=(20, 20), textcoords='offset points',
                              bbox=dict(boxstyle='round,pad=0.5', fc='red', alpha=0.7),
                              arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
        
        # Bottom right: Summary metrics
        axes[1, 1].axis('off')
        summary_text = f"""
DETECTION SUMMARY
{'='*30}

Final Boundary: {final_boundary}px
Total Candidates: {len(candidates)}
Threshold: {threshold:.0f} ({self.PREPROCESSED_THRESHOLD*100:.1f}%)
Smoothing Kernel: {self.SMOOTHING_SIZE}px

IMAGE ANALYSIS
{'='*30}
Dimensions: {original_img.shape[1]}×{original_img.shape[0]}
Contrast Ratio: {results['steps']['step1']['contrast_ratio']:.1f}:1
White/Black Ratio: {results['steps']['step1']['white_pixels']}/{results['steps']['step1']['black_pixels']}

ACCURACY METRICS
{'='*30}
Search Range: {results['steps']['step6']['search_start']}-{results['steps']['step6']['search_end']}px
Valid Candidates (≥800px): {results['steps']['step6']['candidates_above_800']}
Peak Preservation: {results['steps']['step4']['peak_preservation']:.3f}
Noise Reduction: {results['steps']['step4']['noise_reduction']:.1f}
        """
        
        axes[1, 1].text(0.1, 0.9, summary_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Right Edge Detection Analysis Report - {self.timestamp}', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_comprehensive_report.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved: comprehensive_report.png")
    
    def _calculate_accuracy_metrics(self, results: Dict) -> Dict:
        """Calculate comprehensive accuracy metrics"""
        return {
            'detection_confidence': 'High' if results['steps']['step6']['candidates_above_800'] > 0 else 'Low',
            'method_efficiency': results['steps']['step4']['peak_preservation'],
            'noise_reduction_score': results['steps']['step4']['noise_reduction'],
            'candidate_quality': len([c for c in results['steps']['step6'].get('candidates', []) if c[1] > results['steps']['step5']['threshold_value'] * 1.5]),
            'boundary_validity': 'Valid' if results['final_result'] >= 800 else 'Below minimum threshold'
        }

def main():
    """Main diagnostic execution"""
    diagnostics = EdgeDetectionDiagnostics()
    
    # Target preprocessed image
    target_image = "pic/screenshots/20250905_124900_02_photoshop_levels_gamma.png"
    
    if os.path.exists(target_image):
        results = diagnostics.analyze_right_edge_detection(target_image)
        
        print(f"\n🎉 ANALYSIS COMPLETE!")
        print(f"📂 Results saved in: {diagnostics.output_dir}/")
        print(f"🎯 Final boundary detected: {results['final_result']}px")
        print(f"📊 Confidence: {results['metrics']['detection_confidence']}")
    else:
        print(f"❌ Target image not found: {target_image}")

if __name__ == "__main__":
    main()