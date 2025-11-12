"""
Vectorization Quality Analysis
Shows where swiggly lines come from in the stroke extraction → SVG conversion pipeline

Compares:
1. Pixel-level mask (from ML - should be smooth)
2. Extracted stroke contours (from OpenCV findContours)
3. Smoothed stroke points (after Gaussian smoothing)
4. Final SVG paths (after Bezier curve fitting)

This pinpoints EXACTLY where smoothness is lost!
"""

import sys
from pathlib import Path
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2

# Add scanner backend to path
SCANNER_DIR = Path(__file__).parent.parent / "OneNote-Whiteboard-Scanner" / "local-ai-backend"
sys.path.insert(0, str(SCANNER_DIR))

from ai.hybrid_extractor import HybridStrokeExtractor
from ai.stroke_extract import extract_strokes, smooth_stroke
from ai.vectorize import points_to_path_data, points_to_bezier

print("=" * 80)
print("VECTORIZATION QUALITY ANALYSIS")
print("=" * 80)


def analyze_single_stroke(mask, stroke_idx=0):
    """Analyze a single stroke through the vectorization pipeline"""
    
    print(f"\nAnalyzing stroke #{stroke_idx}...")
    
    # Find contours (same as scanner does)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    
    if stroke_idx >= len(contours):
        print(f"❌ Only {len(contours)} contours found, can't analyze stroke {stroke_idx}")
        return None
    
    # Sort by area to get consistent indexing
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contour = contours[stroke_idx]
    
    print(f"  Contour area: {cv2.contourArea(contour):.0f} pixels")
    print(f"  Contour perimeter: {cv2.arcLength(contour, True):.0f} pixels")
    
    # Step 1: Raw contour points
    raw_points = contour.reshape(-1, 2).astype(float)
    print(f"  Raw points: {len(raw_points)}")
    
    # Step 2: Apply multi-pass smoothing (as scanner does)
    if len(raw_points) >= 10:
        smooth_pass1 = smooth_stroke(raw_points, window_size=11)
        smooth_pass2 = smooth_stroke(smooth_pass1, window_size=7)
        smooth_pass3 = smooth_stroke(smooth_pass2, window_size=5)
        print(f"  After 3-pass smoothing: {len(smooth_pass3)} points")
    else:
        smooth_pass3 = raw_points
        print(f"  Skipped smoothing (too few points)")
    
    # Step 3: Simplify curve
    epsilon = 0.5
    simplified_contour = cv2.approxPolyDP(smooth_pass3.reshape(-1, 1, 2).astype(np.float32), epsilon, False)
    simplified_points = simplified_contour.reshape(-1, 2).astype(float)
    print(f"  After simplification (epsilon={epsilon}): {len(simplified_points)} points")
    
    # Step 4: Convert to SVG path
    svg_path = points_to_path_data(simplified_points, smooth=True)
    
    # Calculate jaggedness metric (sum of angle changes)
    def calculate_jaggedness(points):
        if len(points) < 3:
            return 0
        
        angles = []
        for i in range(1, len(points) - 1):
            v1 = points[i] - points[i-1]
            v2 = points[i+1] - points[i]
            
            # Angle between vectors
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            angle = np.arccos(np.clip(cos_angle, -1, 1))
            angles.append(angle)
        
        # Average angle change (higher = more jagged)
        return np.mean(angles) if angles else 0
    
    raw_jagged = calculate_jaggedness(raw_points)
    smooth_jagged = calculate_jaggedness(smooth_pass3)
    simple_jagged = calculate_jaggedness(simplified_points)
    
    print(f"\n  Jaggedness (avg angle change):")
    print(f"    Raw contour:        {raw_jagged:.4f} rad")
    print(f"    After smoothing:    {smooth_jagged:.4f} rad ({(1-smooth_jagged/raw_jagged)*100:.1f}% reduction)")
    print(f"    After simplifying:  {simple_jagged:.4f} rad")
    
    return {
        'raw_points': raw_points,
        'smooth_points': smooth_pass3,
        'simplified_points': simplified_points,
        'svg_path': svg_path,
        'contour': contour,
        'jaggedness': {
            'raw': raw_jagged,
            'smoothed': smooth_jagged,
            'simplified': simple_jagged
        }
    }


def visualize_vectorization_pipeline(test_image_path):
    """Visualize the complete vectorization pipeline"""
    
    # Load and process image
    print(f"\nProcessing: {test_image_path.name}")
    img = cv2.imread(str(test_image_path))
    
    # Run scanner pipeline to get mask
    print("Running scanner pipeline...")
    extractor = HybridStrokeExtractor()
    result = extractor.process_image(img)
    mask = result['mask']
    
    print(f"Mask shape: {mask.shape}")
    print(f"Stroke pixels: {np.count_nonzero(mask):,}")
    print(f"Number of strokes extracted: {len(result.get('strokes', []))}")
    
    # Analyze first few strokes
    num_strokes_to_show = min(3, len(result.get('strokes', [])))
    
    fig = plt.figure(figsize=(20, 5 * num_strokes_to_show))
    
    for stroke_idx in range(num_strokes_to_show):
        analysis = analyze_single_stroke(mask, stroke_idx)
        
        if analysis is None:
            continue
        
        # Create 4-panel visualization for this stroke
        base_idx = stroke_idx * 4
        
        # Panel 1: Original mask region (zoomed to stroke bbox)
        ax1 = plt.subplot(num_strokes_to_show, 4, base_idx + 1)
        x, y, w, h = cv2.boundingRect(analysis['contour'])
        padding = 20
        x1, y1 = max(0, x-padding), max(0, y-padding)
        x2, y2 = min(mask.shape[1], x+w+padding), min(mask.shape[0], y+h+padding)
        mask_crop = mask[y1:y2, x1:x2]
        
        ax1.imshow(mask_crop, cmap='gray')
        ax1.set_title(f'Stroke #{stroke_idx}: ML Mask Output\n{np.count_nonzero(mask_crop)} pixels', 
                     fontweight='bold')
        ax1.axis('off')
        
        # Panel 2: Raw contour points
        ax2 = plt.subplot(num_strokes_to_show, 4, base_idx + 2)
        ax2.imshow(mask_crop, cmap='gray', alpha=0.3)
        raw_pts = analysis['raw_points']
        ax2.plot(raw_pts[:, 0] - x1, raw_pts[:, 1] - y1, 'r.-', linewidth=1, markersize=2, 
                label=f'Raw: {len(raw_pts)} pts')
        ax2.set_title(f'Raw Contour Points\nJagged: {analysis["jaggedness"]["raw"]:.3f}', 
                     fontweight='bold', color='red')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.axis('off')
        
        # Panel 3: Smoothed points
        ax3 = plt.subplot(num_strokes_to_show, 4, base_idx + 3)
        ax3.imshow(mask_crop, cmap='gray', alpha=0.3)
        smooth_pts = analysis['smooth_points']
        ax3.plot(smooth_pts[:, 0] - x1, smooth_pts[:, 1] - y1, 'g.-', linewidth=2, markersize=3,
                label=f'Smoothed: {len(smooth_pts)} pts')
        ax3.set_title(f'After 3-Pass Gaussian Smoothing\nJagged: {analysis["jaggedness"]["smoothed"]:.3f} '
                     f'({(1-analysis["jaggedness"]["smoothed"]/analysis["jaggedness"]["raw"])*100:.0f}% better)',
                     fontweight='bold', color='green')
        ax3.legend(loc='upper right', fontsize=8)
        ax3.axis('off')
        
        # Panel 4: Simplified + SVG
        ax4 = plt.subplot(num_strokes_to_show, 4, base_idx + 4)
        ax4.imshow(mask_crop, cmap='gray', alpha=0.3)
        simple_pts = analysis['simplified_points']
        ax4.plot(simple_pts[:, 0] - x1, simple_pts[:, 1] - y1, 'b.-', linewidth=2, markersize=5,
                label=f'Simplified: {len(simple_pts)} pts')
        ax4.set_title(f'Final: Simplified + Bezier Curves\nJagged: {analysis["jaggedness"]["simplified"]:.3f}',
                     fontweight='bold', color='blue')
        ax4.legend(loc='upper right', fontsize=8)
        ax4.axis('off')
    
    plt.suptitle('Vectorization Pipeline: Pixel Mask → SVG Curves', 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    output_path = Path("vectorization_quality_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved visualization: {output_path}")
    plt.close()
    
    # Also create a detailed comparison for one stroke
    if num_strokes_to_show > 0:
        create_detailed_stroke_comparison(mask, 0)


def create_detailed_stroke_comparison(mask, stroke_idx):
    """Create detailed side-by-side comparison of one stroke"""
    
    analysis = analyze_single_stroke(mask, stroke_idx)
    if analysis is None:
        return
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Detailed Stroke Analysis: Stroke #{stroke_idx}', fontsize=16, fontweight='bold')
    
    # Get stroke region
    x, y, w, h = cv2.boundingRect(analysis['contour'])
    padding = 50
    x1, y1 = max(0, x-padding), max(0, y-padding)
    x2, y2 = min(mask.shape[1], x+w+padding), min(mask.shape[0], y+h+padding)
    mask_crop = mask[y1:y2, x1:x2]
    
    # Row 1: Different stages
    for idx, (name, points, color) in enumerate([
        ('Raw Contour', analysis['raw_points'], 'red'),
        ('Smoothed (3-pass)', analysis['smooth_points'], 'green'),
        ('Simplified', analysis['simplified_points'], 'blue')
    ]):
        axes[0, idx].imshow(mask_crop, cmap='gray', alpha=0.5)
        axes[0, idx].plot(points[:, 0] - x1, points[:, 1] - y1, 
                         color=color, linewidth=2, marker='o', markersize=3)
        axes[0, idx].set_title(f'{name}\n{len(points)} points', fontweight='bold')
        axes[0, idx].axis('off')
    
    # Row 2: Overlays showing improvement
    # Before/after smoothing
    axes[1, 0].imshow(mask_crop, cmap='gray', alpha=0.3)
    axes[1, 0].plot(analysis['raw_points'][:, 0] - x1, 
                   analysis['raw_points'][:, 1] - y1, 
                   'r-', linewidth=1, alpha=0.5, label='Raw')
    axes[1, 0].plot(analysis['smooth_points'][:, 0] - x1,
                   analysis['smooth_points'][:, 1] - y1,
                   'g-', linewidth=2, label='Smoothed')
    axes[1, 0].set_title('Raw vs Smoothed', fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].axis('off')
    
    # Smoothed vs simplified
    axes[1, 1].imshow(mask_crop, cmap='gray', alpha=0.3)
    axes[1, 1].plot(analysis['smooth_points'][:, 0] - x1,
                   analysis['smooth_points'][:, 1] - y1,
                   'g-', linewidth=1, alpha=0.5, label='Smoothed')
    axes[1, 1].plot(analysis['simplified_points'][:, 0] - x1,
                   analysis['simplified_points'][:, 1] - y1,
                   'b-', linewidth=2, marker='o', markersize=5, label='Simplified')
    axes[1, 1].set_title('Smoothed vs Simplified', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].axis('off')
    
    # Final quality metrics
    metrics_text = f"""QUALITY METRICS:

Point Count:
  Raw:        {len(analysis['raw_points'])}
  Smoothed:   {len(analysis['smooth_points'])}
  Simplified: {len(analysis['simplified_points'])}
  
Jaggedness (lower = smoother):
  Raw:        {analysis['jaggedness']['raw']:.4f}
  Smoothed:   {analysis['jaggedness']['smoothed']:.4f}
  Simplified: {analysis['jaggedness']['simplified']:.4f}
  
Improvement:
  Smoothing:  {(1 - analysis['jaggedness']['smoothed']/analysis['jaggedness']['raw'])*100:.1f}% smoother
  
SVG Path Length:
  {len(analysis['svg_path'])} characters
"""
    
    axes[1, 2].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                   verticalalignment='center')
    axes[1, 2].axis('off')
    
    plt.tight_layout()
    output_path = Path("stroke_detail_analysis.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved detailed analysis: {output_path}")
    plt.close()


def main():
    """Run vectorization quality analysis"""
    
    # Find test image
    test_images_dir = Path(__file__).parent / "dataset" / "test-images" / "images"
    image_files = list(test_images_dir.glob("*.jpg")) + list(test_images_dir.glob("*.png"))
    
    if len(image_files) == 0:
        print(f"❌ No test images found in {test_images_dir}")
        return
    
    test_image = image_files[0]
    visualize_vectorization_pipeline(test_image)
    
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nGenerated visualizations:")
    print("  1. vectorization_quality_analysis.png - Full pipeline for 3 strokes")
    print("  2. stroke_detail_analysis.png - Detailed analysis of one stroke")
    print("\nThese show EXACTLY where smoothness is gained/lost in the pipeline!")


if __name__ == "__main__":
    main()
