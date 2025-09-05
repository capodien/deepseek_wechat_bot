#!/usr/bin/env python3
"""
Parameter Optimization Analysis for Right Edge Detection
Tests different parameters to improve accuracy
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d
import os
from datetime import datetime
from typing import Dict, List, Tuple

class ParameterOptimization:
    """Optimize detection parameters for maximum accuracy"""
    
    def __init__(self):
        self.output_dir = "pic/optimization"
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        os.makedirs(self.output_dir, exist_ok=True)
    
    def optimize_parameters(self, preprocessed_image_path: str) -> Dict:
        """
        Test different parameter combinations to optimize accuracy
        
        Args:
            preprocessed_image_path: Path to preprocessed image
            
        Returns:
            Dictionary with optimization results and recommendations
        """
        print("🔧 PARAMETER OPTIMIZATION ANALYSIS")
        print("=" * 50)
        
        # Load image
        img = cv2.imread(preprocessed_image_path, cv2.IMREAD_GRAYSCALE)
        
        # Calculate base projection
        diff_x = np.diff(img.astype(np.int16), axis=1)
        transition_profile = np.sum(diff_x, axis=0)
        raw_projection = -transition_profile
        
        results = {
            'image_path': preprocessed_image_path,
            'threshold_tests': {},
            'smoothing_tests': {},
            'search_range_tests': {},
            'recommendations': {}
        }
        
        # Test 1: Threshold Optimization
        print("\n🎯 TEST 1: THRESHOLD OPTIMIZATION")
        print("-" * 40)
        threshold_results = self._test_thresholds(raw_projection, img.shape[1])
        results['threshold_tests'] = threshold_results
        
        # Test 2: Smoothing Optimization
        print("\n🎯 TEST 2: SMOOTHING OPTIMIZATION")
        print("-" * 40)
        smoothing_results = self._test_smoothing(raw_projection, img.shape[1])
        results['smoothing_tests'] = smoothing_results
        
        # Test 3: Search Range Optimization
        print("\n🎯 TEST 3: SEARCH RANGE OPTIMIZATION")
        print("-" * 40)
        search_results = self._test_search_ranges(raw_projection, img.shape[1])
        results['search_range_tests'] = search_results
        
        # Generate recommendations
        results['recommendations'] = self._generate_recommendations(results)
        
        # Create optimization report
        self._create_optimization_report(results, img, raw_projection)
        
        return results
    
    def _test_thresholds(self, raw_projection: np.ndarray, img_width: int) -> Dict:
        """Test different threshold percentages"""
        threshold_percentages = [0.003, 0.005, 0.007, 0.01, 0.015, 0.02]  # 0.3% to 2%
        
        results = {}
        best_score = 0
        best_threshold = 0.005
        
        for threshold_pct in threshold_percentages:
            # Apply smoothing
            smoothed = uniform_filter1d(raw_projection, size=5, mode='nearest')
            
            # Calculate threshold
            max_strength = np.max(smoothed)
            threshold = max_strength * threshold_pct
            
            # Find candidates
            candidates = self._find_candidates(smoothed, threshold, img_width)
            
            # Score this threshold
            score = self._score_threshold(candidates, threshold_pct)
            
            results[threshold_pct] = {
                'threshold_value': threshold,
                'candidate_count': len(candidates),
                'valid_candidates': len([c for c in candidates if c[0] >= 800]),
                'rightmost_candidate': max(candidates, key=lambda x: x[0])[0] if candidates else None,
                'score': score
            }
            
            print(f"  📊 {threshold_pct*100:.1f}%: {len(candidates)} candidates, "
                  f"{results[threshold_pct]['valid_candidates']} valid, "
                  f"score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_threshold = threshold_pct
        
        results['best_threshold'] = best_threshold
        results['best_score'] = best_score
        
        print(f"  ✅ Best threshold: {best_threshold*100:.1f}% (score: {best_score:.2f})")
        
        return results
    
    def _test_smoothing(self, raw_projection: np.ndarray, img_width: int) -> Dict:
        """Test different smoothing kernel sizes"""
        kernel_sizes = [3, 5, 7, 9, 11, 13]
        
        results = {}
        best_score = 0
        best_kernel = 5
        
        for kernel_size in kernel_sizes:
            # Apply smoothing
            smoothed = uniform_filter1d(raw_projection, size=kernel_size, mode='nearest')
            
            # Use best threshold from previous test
            max_strength = np.max(smoothed)
            threshold = max_strength * 0.005  # 0.5%
            
            # Find candidates
            candidates = self._find_candidates(smoothed, threshold, img_width)
            
            # Calculate smoothing quality metrics
            noise_reduction = np.std(raw_projection) - np.std(smoothed)
            peak_preservation = np.corrcoef(raw_projection, smoothed)[0, 1]
            
            # Score this kernel
            score = self._score_smoothing(candidates, noise_reduction, peak_preservation)
            
            results[kernel_size] = {
                'candidate_count': len(candidates),
                'valid_candidates': len([c for c in candidates if c[0] >= 800]),
                'noise_reduction': noise_reduction,
                'peak_preservation': peak_preservation,
                'score': score
            }
            
            print(f"  📊 Kernel {kernel_size}: {len(candidates)} candidates, "
                  f"noise reduction: {noise_reduction:.0f}, "
                  f"correlation: {peak_preservation:.3f}, score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_kernel = kernel_size
        
        results['best_kernel'] = best_kernel
        results['best_score'] = best_score
        
        print(f"  ✅ Best kernel size: {best_kernel} (score: {best_score:.2f})")
        
        return results
    
    def _test_search_ranges(self, raw_projection: np.ndarray, img_width: int) -> Dict:
        """Test different search range configurations"""
        search_configs = [
            {'start_pct': 0.35, 'end_pct': 0.90, 'name': 'Narrow'},
            {'start_pct': 0.40, 'end_pct': 0.95, 'name': 'Current'},
            {'start_pct': 0.45, 'end_pct': 0.98, 'name': 'Extended'},
            {'start_pct': 0.30, 'end_pct': 0.95, 'name': 'Wide'}
        ]
        
        results = {}
        best_score = 0
        best_config = None
        
        # Apply optimal smoothing
        smoothed = uniform_filter1d(raw_projection, size=5, mode='nearest')
        max_strength = np.max(smoothed)
        threshold = max_strength * 0.005
        
        for config in search_configs:
            search_start = int(img_width * config['start_pct'])
            search_end = int(img_width * config['end_pct'])
            
            # Find candidates in this range
            candidates = []
            for x in range(search_start, min(search_end, len(smoothed))):
                if smoothed[x] > threshold:
                    # Check for local maximum
                    is_peak = True
                    if x > 0 and smoothed[x] < smoothed[x-1]:
                        is_peak = False
                    if x < len(smoothed) - 1 and smoothed[x] < smoothed[x+1]:
                        is_peak = False
                    
                    if is_peak:
                        candidates.append((x, smoothed[x]))
            
            # Score this range
            score = self._score_search_range(candidates, search_start, search_end, img_width)
            
            results[config['name']] = {
                'start_px': search_start,
                'end_px': search_end,
                'range_width': search_end - search_start,
                'candidate_count': len(candidates),
                'valid_candidates': len([c for c in candidates if c[0] >= 800]),
                'rightmost_candidate': max(candidates, key=lambda x: x[0])[0] if candidates else None,
                'score': score
            }
            
            print(f"  📊 {config['name']}: {search_start}-{search_end}px, "
                  f"{len(candidates)} candidates, score: {score:.2f}")
            
            if score > best_score:
                best_score = score
                best_config = config['name']
        
        results['best_config'] = best_config
        results['best_score'] = best_score
        
        print(f"  ✅ Best search range: {best_config} (score: {best_score:.2f})")
        
        return results
    
    def _find_candidates(self, smoothed_projection: np.ndarray, threshold: float, img_width: int) -> List[Tuple[int, float]]:
        """Find peak candidates using current method"""
        search_start = int(img_width * 0.4)
        search_end = int(img_width * 0.95)
        
        candidates = []
        for x in range(search_start, min(search_end, len(smoothed_projection))):
            if smoothed_projection[x] > threshold:
                # Check for local maximum
                is_peak = True
                if x > 0 and smoothed_projection[x] < smoothed_projection[x-1]:
                    is_peak = False
                if x < len(smoothed_projection) - 1 and smoothed_projection[x] < smoothed_projection[x+1]:
                    is_peak = False
                
                if is_peak:
                    candidates.append((x, smoothed_projection[x]))
        
        return candidates
    
    def _score_threshold(self, candidates: List[Tuple[int, float]], threshold_pct: float) -> float:
        """Score threshold based on candidate quality"""
        if not candidates:
            return 0.0
        
        valid_candidates = [c for c in candidates if c[0] >= 800]
        
        # Scoring factors
        candidate_count_score = min(len(candidates) / 20, 1.0)  # Prefer 10-20 candidates
        valid_ratio_score = len(valid_candidates) / len(candidates) if candidates else 0
        threshold_score = 1.0 - abs(threshold_pct - 0.005) * 100  # Prefer around 0.5%
        
        # Rightmost candidate bonus
        rightmost_bonus = 0
        if valid_candidates:
            rightmost = max(valid_candidates, key=lambda x: x[0])[0]
            if 900 <= rightmost <= 1000:  # Sweet spot for message boundaries
                rightmost_bonus = 0.2
        
        return candidate_count_score * 0.3 + valid_ratio_score * 0.4 + threshold_score * 0.2 + rightmost_bonus
    
    def _score_smoothing(self, candidates: List[Tuple[int, float]], noise_reduction: float, peak_preservation: float) -> float:
        """Score smoothing based on noise reduction and peak preservation"""
        if not candidates:
            return 0.0
        
        valid_candidates = [c for c in candidates if c[0] >= 800]
        
        # Scoring factors
        candidate_score = min(len(valid_candidates) / 10, 1.0)
        noise_score = min(noise_reduction / 5000, 1.0)  # Normalize noise reduction
        preservation_score = max(peak_preservation, 0)  # Correlation should be positive
        
        return candidate_score * 0.4 + noise_score * 0.3 + preservation_score * 0.3
    
    def _score_search_range(self, candidates: List[Tuple[int, float]], start: int, end: int, img_width: int) -> float:
        """Score search range based on coverage and candidate quality"""
        if not candidates:
            return 0.0
        
        valid_candidates = [c for c in candidates if c[0] >= 800]
        
        # Scoring factors
        coverage_score = (end - start) / (img_width * 0.6)  # Prefer good coverage
        candidate_score = min(len(valid_candidates) / 8, 1.0)
        
        # Penalty for being too narrow or too wide
        range_penalty = 0
        range_width = end - start
        if range_width < img_width * 0.3:  # Too narrow
            range_penalty = 0.2
        elif range_width > img_width * 0.7:  # Too wide
            range_penalty = 0.1
        
        return candidate_score * 0.6 + coverage_score * 0.4 - range_penalty
    
    def _generate_recommendations(self, results: Dict) -> Dict:
        """Generate optimization recommendations based on test results"""
        recommendations = {}
        
        # Threshold recommendation
        best_threshold = results['threshold_tests']['best_threshold']
        current_threshold = 0.005
        
        if best_threshold != current_threshold:
            improvement = (results['threshold_tests']['best_score'] - 
                         results['threshold_tests'].get(current_threshold, {}).get('score', 0))
            recommendations['threshold'] = {
                'current': current_threshold,
                'recommended': best_threshold,
                'improvement': improvement,
                'confidence': 'High' if improvement > 0.1 else 'Medium'
            }
        
        # Smoothing recommendation
        best_kernel = results['smoothing_tests']['best_kernel']
        current_kernel = 5
        
        if best_kernel != current_kernel:
            improvement = (results['smoothing_tests']['best_score'] - 
                         results['smoothing_tests'].get(current_kernel, {}).get('score', 0))
            recommendations['smoothing'] = {
                'current': current_kernel,
                'recommended': best_kernel,
                'improvement': improvement,
                'confidence': 'High' if improvement > 0.1 else 'Medium'
            }
        
        # Search range recommendation
        best_range = results['search_range_tests']['best_config']
        if best_range != 'Current':
            improvement = (results['search_range_tests']['best_score'] - 
                         results['search_range_tests'].get('Current', {}).get('score', 0))
            recommendations['search_range'] = {
                'current': 'Current (40%-95%)',
                'recommended': best_range,
                'improvement': improvement,
                'confidence': 'Medium'
            }
        
        return recommendations
    
    def _create_optimization_report(self, results: Dict, img: np.ndarray, raw_projection: np.ndarray):
        """Create comprehensive optimization report"""
        fig, axes = plt.subplots(2, 2, figsize=(20, 12))
        
        # Threshold comparison
        threshold_data = results['threshold_tests']
        thresholds = [k for k in threshold_data.keys() if isinstance(k, float)]
        scores = [threshold_data[t]['score'] for t in thresholds]
        valid_counts = [threshold_data[t]['valid_candidates'] for t in thresholds]
        
        ax1 = axes[0, 0]
        ax1_twin = ax1.twinx()
        
        line1 = ax1.plot([t*100 for t in thresholds], scores, 'b-o', label='Score')
        line2 = ax1_twin.plot([t*100 for t in thresholds], valid_counts, 'r-s', label='Valid Candidates')
        ax1.set_xlabel('Threshold (%)')
        ax1.set_ylabel('Score', color='b')
        ax1_twin.set_ylabel('Valid Candidates', color='r')
        ax1.set_title('Threshold Optimization')
        ax1.grid(True, alpha=0.3)
        
        # Mark best threshold
        best_t = results['threshold_tests']['best_threshold']
        ax1.axvline(x=best_t*100, color='green', linestyle='--', alpha=0.7, label=f'Best: {best_t*100:.1f}%')
        ax1.legend(loc='upper left')
        ax1_twin.legend(loc='upper right')
        
        # Smoothing comparison
        smoothing_data = results['smoothing_tests']
        kernels = [k for k in smoothing_data.keys() if isinstance(k, int)]
        smooth_scores = [smoothing_data[k]['score'] for k in kernels]
        correlations = [smoothing_data[k]['peak_preservation'] for k in kernels]
        
        ax2 = axes[0, 1]
        ax2_twin = ax2.twinx()
        
        ax2.plot(kernels, smooth_scores, 'b-o', label='Score')
        ax2_twin.plot(kernels, correlations, 'g-^', label='Correlation')
        ax2.set_xlabel('Kernel Size')
        ax2.set_ylabel('Score', color='b')
        ax2_twin.set_ylabel('Peak Preservation', color='g')
        ax2.set_title('Smoothing Optimization')
        ax2.grid(True, alpha=0.3)
        
        # Mark best kernel
        best_k = results['smoothing_tests']['best_kernel']
        ax2.axvline(x=best_k, color='green', linestyle='--', alpha=0.7, label=f'Best: {best_k}')
        ax2.legend(loc='upper left')
        ax2_twin.legend(loc='upper right')
        
        # Search range comparison
        search_data = results['search_range_tests']
        ranges = [k for k in search_data.keys() if k not in ['best_config', 'best_score']]
        range_scores = [search_data[r]['score'] for r in ranges]
        range_candidates = [search_data[r]['valid_candidates'] for r in ranges]
        
        ax3 = axes[1, 0]
        bars = ax3.bar(ranges, range_scores, alpha=0.7)
        ax3.set_ylabel('Score')
        ax3.set_title('Search Range Optimization')
        
        # Color the best bar
        best_range = results['search_range_tests']['best_config']
        for i, r in enumerate(ranges):
            if r == best_range:
                bars[i].set_color('green')
        
        # Add candidate counts as text
        for i, (r, count) in enumerate(zip(ranges, range_candidates)):
            ax3.text(i, range_scores[i] + 0.01, f'{count} candidates', ha='center', fontsize=9)
        
        # Summary and recommendations
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_text = "OPTIMIZATION SUMMARY\\n"
        summary_text += "=" * 30 + "\\n\\n"
        
        # Add recommendations
        recs = results['recommendations']
        if recs:
            summary_text += "RECOMMENDED CHANGES:\\n"
            summary_text += "-" * 20 + "\\n"
            
            for param, rec in recs.items():
                summary_text += f"{param.upper()}:\\n"
                summary_text += f"  Current: {rec['current']}\\n"
                summary_text += f"  Recommended: {rec['recommended']}\\n"
                summary_text += f"  Improvement: {rec['improvement']:+.3f}\\n"
                summary_text += f"  Confidence: {rec['confidence']}\\n\\n"
        else:
            summary_text += "✅ CURRENT PARAMETERS OPTIMAL\\n"
            summary_text += "No significant improvements found.\\n\\n"
        
        summary_text += "CURRENT PERFORMANCE:\\n"
        summary_text += "-" * 20 + "\\n"
        summary_text += f"Threshold: 0.5%\\n"
        summary_text += f"Smoothing: 5px kernel\\n"
        summary_text += f"Search Range: 40%-95%\\n"
        
        ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, fontsize=10, 
                verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Parameter Optimization Analysis - {self.timestamp}', fontsize=16)
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, f"{self.timestamp}_optimization_report.png")
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"💾 Saved optimization report: {output_path}")

def main():
    """Run parameter optimization"""
    optimizer = ParameterOptimization()
    
    # Use available preprocessed image
    target_image = "pic/screenshots/20250905_130426_02_photoshop_levels_gamma.png"
    
    if os.path.exists(target_image):
        results = optimizer.optimize_parameters(target_image)
        
        print(f"\\n🎉 OPTIMIZATION COMPLETE!")
        print(f"📂 Results saved in: {optimizer.output_dir}/")
        
        # Print recommendations
        if results['recommendations']:
            print(f"\\n🔧 RECOMMENDATIONS:")
            for param, rec in results['recommendations'].items():
                print(f"  {param}: {rec['current']} → {rec['recommended']} "
                      f"(improvement: {rec['improvement']:+.3f}, confidence: {rec['confidence']})")
        else:
            print(f"\\n✅ Current parameters are optimal!")
    else:
        print(f"❌ Target image not found: {target_image}")

if __name__ == "__main__":
    main()