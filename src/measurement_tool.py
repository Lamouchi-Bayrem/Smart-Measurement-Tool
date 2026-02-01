"""
Measurement Tool Module
Performs various measurements using depth and CV
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean
import math


class MeasurementTool:
    """Smart measurement tool using computer vision"""
    
    def __init__(self, depth_estimator):
        """
        Initialize measurement tool
        
        Args:
            depth_estimator: DepthEstimator instance
        """
        self.depth_estimator = depth_estimator
        self.reference_object = None
        self.reference_distance = 1.0  # meters
        self.calibration_factor = 1.0
    
    def calibrate_with_reference(self, reference_points: List[Tuple[int, int]], 
                                 real_distance: float):
        """
        Calibrate using reference object of known size
        
        Args:
            reference_points: List of (x, y) points defining reference object
            real_distance: Real-world distance in meters
        """
        if len(reference_points) < 2:
            return
        
        # Calculate pixel distance
        pixel_distance = euclidean(reference_points[0], reference_points[1])
        
        # Calculate calibration factor
        self.calibration_factor = real_distance / pixel_distance
        self.reference_distance = real_distance
        self.reference_object = reference_points
    
    def measure_distance(self, point1: Tuple[int, int], point2: Tuple[int, int],
                        depth_map: np.ndarray) -> Dict:
        """
        Measure distance between two points
        
        Args:
            point1: First point (x, y)
            point2: Second point (x, y)
            depth_map: Depth map
            
        Returns:
            dict: Measurement results
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Pixel distance
        pixel_distance = euclidean(point1, point2)
        
        # Get depth at both points
        depth1 = self.depth_estimator.get_depth_at_point(depth_map, x1, y1)
        depth2 = self.depth_estimator.get_depth_at_point(depth_map, x2, y2)
        avg_depth = (depth1 + depth2) / 2.0
        
        # Convert to real distance
        real_distance = pixel_distance * self.calibration_factor
        
        # Adjust for depth (objects at different depths)
        depth_adjustment = abs(depth1 - depth2) * self.reference_distance
        adjusted_distance = real_distance * (1 + depth_adjustment)
        
        return {
            'pixel_distance': pixel_distance,
            'real_distance': adjusted_distance,
            'depth1': depth1,
            'depth2': depth2,
            'avg_depth': avg_depth
        }
    
    def measure_area(self, points: List[Tuple[int, int]], 
                    depth_map: np.ndarray) -> Dict:
        """
        Measure area of polygon
        
        Args:
            points: List of polygon vertices (x, y)
            depth_map: Depth map
            
        Returns:
            dict: Area measurement results
        """
        if len(points) < 3:
            return {'area_pixels': 0, 'area_real': 0}
        
        # Calculate pixel area using shoelace formula
        area_pixels = self._polygon_area(points)
        
        # Get average depth
        avg_depth = np.mean([
            self.depth_estimator.get_depth_at_point(depth_map, x, y)
            for x, y in points
        ])
        
        # Convert to real area
        # Area scales with distance squared
        depth_factor = self.depth_estimator.depth_to_distance(avg_depth, self.reference_distance)
        area_real = area_pixels * (self.calibration_factor ** 2) * (depth_factor ** 2)
        
        return {
            'area_pixels': area_pixels,
            'area_real': area_real,
            'avg_depth': avg_depth
        }
    
    def measure_volume(self, base_points: List[Tuple[int, int]],
                      height_point: Tuple[int, int],
                      depth_map: np.ndarray) -> Dict:
        """
        Estimate volume (area × height)
        
        Args:
            base_points: Base polygon vertices
            height_point: Top point for height measurement
            depth_map: Depth map
            
        Returns:
            dict: Volume measurement results
        """
        # Measure base area
        area_result = self.measure_area(base_points, depth_map)
        
        # Measure height
        base_center = self._polygon_center(base_points)
        height_result = self.measure_distance(base_center, height_point, depth_map)
        
        # Calculate volume (simplified as area × height)
        volume = area_result['area_real'] * height_result['real_distance']
        
        return {
            'volume': volume,
            'base_area': area_result['area_real'],
            'height': height_result['real_distance'],
            'area_result': area_result,
            'height_result': height_result
        }
    
    def measure_object_dimensions(self, bbox: Tuple[int, int, int, int],
                                 depth_map: np.ndarray) -> Dict:
        """
        Measure object dimensions (width, height, depth)
        
        Args:
            bbox: Bounding box (x, y, w, h)
            depth_map: Depth map
            
        Returns:
            dict: Dimension measurements
        """
        x, y, w, h = bbox
        
        # Measure width
        width_result = self.measure_distance(
            (x, y + h // 2),
            (x + w, y + h // 2),
            depth_map
        )
        
        # Measure height
        height_result = self.measure_distance(
            (x + w // 2, y),
            (x + w // 2, y + h),
            depth_map
        )
        
        # Estimate depth (using depth variation in bbox)
        bbox_depth = depth_map[y:y+h, x:x+w]
        depth_variation = bbox_depth.max() - bbox_depth.min()
        estimated_depth = depth_variation * self.reference_distance
        
        return {
            'width': width_result['real_distance'],
            'height': height_result['real_distance'],
            'depth': estimated_depth,
            'width_pixels': w,
            'height_pixels': h
        }
    
    def _polygon_area(self, points: List[Tuple[int, int]]) -> float:
        """Calculate polygon area using shoelace formula"""
        n = len(points)
        area = 0.0
        for i in range(n):
            j = (i + 1) % n
            area += points[i][0] * points[j][1]
            area -= points[j][0] * points[i][1]
        return abs(area) / 2.0
    
    def _polygon_center(self, points: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Calculate polygon center"""
        x_sum = sum(p[0] for p in points)
        y_sum = sum(p[1] for p in points)
        return (int(x_sum / len(points)), int(y_sum / len(points)))


class ObjectDetector:
    """Detects objects for automatic measurement"""
    
    def __init__(self):
        """Initialize object detector"""
        # Use OpenCV's object detection or MediaPipe
        pass
    
    def detect_objects(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects in image
        
        Args:
            image: Input image
            
        Returns:
            list: Detected objects with bounding boxes
        """
        # Simplified object detection using contours
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 1000:  # Filter small objects
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'contour': contour
                })
        
        return objects





