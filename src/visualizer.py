"""
Visualization Module
Creates visualizations for measurements
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import io
from PIL import Image as PILImage


class MeasurementVisualizer:
    """Visualizes measurements on images"""
    
    def __init__(self):
        """Initialize visualizer"""
        self.colors = {
            'point': (0, 255, 0),      # Green
            'line': (255, 0, 0),       # Red
            'area': (0, 0, 255),       # Blue
            'text': (255, 255, 255),   # White
            'bbox': (255, 255, 0)      # Yellow
        }
    
    def draw_distance_measurement(self, image: np.ndarray, point1: Tuple[int, int],
                                 point2: Tuple[int, int], distance: float,
                                 unit: str = 'm') -> np.ndarray:
        """
        Draw distance measurement on image
        
        Args:
            image: Input image
            point1: First point
            point2: Second point
            distance: Measured distance
            unit: Unit of measurement
            
        Returns:
            numpy array: Annotated image
        """
        annotated = image.copy()
        x1, y1 = point1
        x2, y2 = point2
        
        # Draw line
        cv2.line(annotated, point1, point2, self.colors['line'], 2)
        
        # Draw points
        cv2.circle(annotated, point1, 5, self.colors['point'], -1)
        cv2.circle(annotated, point2, 5, self.colors['point'], -1)
        
        # Draw distance label
        mid_x, mid_y = (x1 + x2) // 2, (y1 + y2) // 2
        label = f"{distance:.2f} {unit}"
        cv2.putText(annotated, label, (mid_x - 50, mid_y - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return annotated
    
    def draw_area_measurement(self, image: np.ndarray, points: List[Tuple[int, int]],
                             area: float, unit: str = 'm²') -> np.ndarray:
        """
        Draw area measurement on image
        
        Args:
            image: Input image
            points: Polygon vertices
            area: Measured area
            unit: Unit of measurement
            
        Returns:
            numpy array: Annotated image
        """
        annotated = image.copy()
        
        # Draw polygon
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.fillPoly(annotated, [pts], self.colors['area'])
        cv2.polylines(annotated, [pts], True, self.colors['line'], 2)
        
        # Draw vertices
        for point in points:
            cv2.circle(annotated, point, 5, self.colors['point'], -1)
        
        # Draw area label
        center = self._polygon_center(points)
        label = f"Area: {area:.2f} {unit}"
        cv2.putText(annotated, label, (center[0] - 60, center[1]),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return annotated
    
    def draw_volume_measurement(self, image: np.ndarray, base_points: List[Tuple[int, int]],
                               height_point: Tuple[int, int], volume: float,
                               unit: str = 'm³') -> np.ndarray:
        """
        Draw volume measurement on image
        
        Args:
            image: Input image
            base_points: Base polygon
            height_point: Height point
            volume: Measured volume
            unit: Unit of measurement
            
        Returns:
            numpy array: Annotated image
        """
        annotated = image.copy()
        
        # Draw base area
        annotated = self.draw_area_measurement(annotated, base_points, 0)
        
        # Draw height line
        base_center = self._polygon_center(base_points)
        cv2.line(annotated, base_center, height_point, self.colors['line'], 2)
        cv2.circle(annotated, height_point, 5, self.colors['point'], -1)
        
        # Draw volume label
        label = f"Volume: {volume:.2f} {unit}"
        cv2.putText(annotated, label, (base_center[0] - 60, base_center[1] - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.colors['text'], 2)
        
        return annotated
    
    def draw_object_dimensions(self, image: np.ndarray, bbox: Tuple[int, int, int, int],
                              dimensions: Dict) -> np.ndarray:
        """
        Draw object dimensions on image
        
        Args:
            image: Input image
            bbox: Bounding box
            dimensions: Dimension measurements
            
        Returns:
            numpy array: Annotated image
        """
        annotated = image.copy()
        x, y, w, h = bbox
        
        # Draw bounding box
        cv2.rectangle(annotated, (x, y), (x + w, y + h), self.colors['bbox'], 2)
        
        # Draw width
        cv2.line(annotated, (x, y + h + 20), (x + w, y + h + 20), self.colors['line'], 2)
        width_label = f"W: {dimensions['width']:.2f}m"
        cv2.putText(annotated, width_label, (x + w // 2 - 40, y + h + 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        
        # Draw height
        cv2.line(annotated, (x + w + 20, y), (x + w + 20, y + h), self.colors['line'], 2)
        height_label = f"H: {dimensions['height']:.2f}m"
        cv2.putText(annotated, height_label, (x + w + 25, y + h // 2),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.colors['text'], 2)
        
        return annotated
    
    def create_measurement_report(self, image: np.ndarray, measurements: Dict) -> PILImage.Image:
        """
        Create comprehensive measurement report visualization
        
        Args:
            image: Original image
            measurements: Dictionary of measurements
            
        Returns:
            PIL Image
        """
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # Original image with measurements
        axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Measurement Overlay')
        axes[0].axis('off')
        
        # Measurement summary
        axes[1].axis('off')
        summary_text = "Measurement Summary\n\n"
        
        if 'distance' in measurements:
            summary_text += f"Distance: {measurements['distance']:.2f} m\n"
        if 'area' in measurements:
            summary_text += f"Area: {measurements['area']:.2f} m²\n"
        if 'volume' in measurements:
            summary_text += f"Volume: {measurements['volume']:.2f} m³\n"
        if 'dimensions' in measurements:
            dims = measurements['dimensions']
            summary_text += f"Width: {dims['width']:.2f} m\n"
            summary_text += f"Height: {dims['height']:.2f} m\n"
            summary_text += f"Depth: {dims['depth']:.2f} m\n"
        
        axes[1].text(0.1, 0.5, summary_text, fontsize=14, 
                    verticalalignment='center', family='monospace')
        
        plt.tight_layout()
        
        # Convert to PIL Image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img = PILImage.open(buf)
        plt.close()
        
        return img
    
    def _polygon_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate polygon center"""
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (int(x_sum / len(points)), int(y_sum / len(points)))





