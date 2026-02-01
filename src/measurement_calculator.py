"""
Measurement Calculator Module
Calculates distances, areas, volumes from depth and image data
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
from scipy.spatial.distance import euclidean
import math


class MeasurementCalculator:
    """Calculates measurements from depth and image data"""
    
    def __init__(self, focal_length: float = 700, sensor_width: float = 0.036):
        """
        Initialize measurement calculator
        
        Args:
            focal_length: Camera focal length in pixels
            sensor_width: Camera sensor width in meters
        """
        self.focal_length = focal_length
        self.sensor_width = sensor_width
    
    def measure_distance(self, depth_map: np.ndarray, 
                        point1: Tuple[int, int],
                        point2: Tuple[int, int]) -> Dict:
        """
        Measure distance between two points in 3D space
        
        Args:
            depth_map: Depth map
            point1: First point (x, y)
            point2: Second point (x, y)
            
        Returns:
            dict: Distance measurement results
        """
        x1, y1 = point1
        x2, y2 = point2
        
        # Get depths
        depth1 = depth_map[y1, x1] if self._is_valid_point(depth_map, point1) else 0
        depth2 = depth_map[y2, x2] if self._is_valid_point(depth_map, point2) else 0
        
        # Convert pixel coordinates to 3D coordinates
        # Assuming pinhole camera model
        cx, cy = depth_map.shape[1] // 2, depth_map.shape[0] // 2
        
        # 3D coordinates
        X1 = (x1 - cx) * depth1 / self.focal_length
        Y1 = (y1 - cy) * depth1 / self.focal_length
        Z1 = depth1
        
        X2 = (x2 - cx) * depth2 / self.focal_length
        Y2 = (y2 - cy) * depth2 / self.focal_length
        Z2 = depth2
        
        # Calculate 3D distance
        distance_3d = math.sqrt((X2 - X1)**2 + (Y2 - Y1)**2 + (Z2 - Z1)**2)
        
        # 2D pixel distance
        distance_2d = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        
        return {
            'distance_3d': distance_3d,  # meters
            'distance_2d': distance_2d,  # pixels
            'point1_depth': depth1,
            'point2_depth': depth2,
            'point1_3d': (X1, Y1, Z1),
            'point2_3d': (X2, Y2, Z2)
        }
    
    def measure_area(self, depth_map: np.ndarray,
                    polygon_points: List[Tuple[int, int]]) -> Dict:
        """
        Measure area of a polygon in 3D space
        
        Args:
            depth_map: Depth map
            polygon_points: List of polygon vertices (x, y)
            
        Returns:
            dict: Area measurement results
        """
        if len(polygon_points) < 3:
            return {'area_2d': 0, 'area_3d': 0, 'average_depth': 0}
        
        # 2D area (pixel area)
        points_array = np.array(polygon_points, dtype=np.int32)
        area_2d = cv2.contourArea(points_array)
        
        # Get average depth
        mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points_array], 255)
        depths_in_region = depth_map[mask > 0]
        average_depth = np.mean(depths_in_region) if len(depths_in_region) > 0 else 0
        
        # Convert 2D area to 3D area
        # Using average depth to approximate
        if average_depth > 0:
            # Scale factor based on depth
            pixel_size = (average_depth * self.sensor_width) / (self.focal_length * depth_map.shape[1])
            area_3d = area_2d * (pixel_size ** 2)  # square meters
        else:
            area_3d = 0
        
        return {
            'area_2d': area_2d,  # square pixels
            'area_3d': area_3d,  # square meters
            'average_depth': average_depth,
            'perimeter': cv2.arcLength(points_array, True)
        }
    
    def measure_volume(self, depth_map: np.ndarray,
                      polygon_points: List[Tuple[int, int]],
                      reference_depth: float = None) -> Dict:
        """
        Measure volume of an object
        
        Args:
            depth_map: Depth map
            polygon_points: Base polygon vertices
            reference_depth: Reference depth (background)
            
        Returns:
            dict: Volume measurement results
        """
        # Get area measurement
        area_result = self.measure_area(depth_map, polygon_points)
        
        # Create mask for polygon
        points_array = np.array(polygon_points, dtype=np.int32)
        mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
        cv2.fillPoly(mask, [points_array], 255)
        
        # Get depths in region
        depths_in_region = depth_map[mask > 0]
        
        if len(depths_in_region) == 0:
            return {'volume': 0, 'height': 0, 'base_area': 0}
        
        # Calculate volume
        if reference_depth is None:
            # Use minimum depth as reference (closest point)
            reference_depth = np.min(depths_in_region)
        
        # Height map (difference from reference)
        height_map = reference_depth - depths_in_region
        height_map = np.maximum(height_map, 0)  # Only positive heights
        
        # Average height
        average_height = np.mean(height_map)
        
        # Volume = base area * average height
        volume = area_result['area_3d'] * average_height
        
        return {
            'volume': volume,  # cubic meters
            'height': average_height,  # meters
            'base_area': area_result['area_3d'],
            'max_height': np.max(height_map) if len(height_map) > 0 else 0
        }
    
    def measure_object_dimensions(self, depth_map: np.ndarray,
                                  bounding_box: Tuple[int, int, int, int]) -> Dict:
        """
        Measure dimensions of object in bounding box
        
        Args:
            depth_map: Depth map
            bounding_box: (x, y, width, height)
            
        Returns:
            dict: Object dimensions
        """
        x, y, w, h = bounding_box
        
        # Get depth in bounding box
        roi = depth_map[y:y+h, x:x+w]
        average_depth = np.mean(roi[roi > 0]) if np.any(roi > 0) else 0
        
        # Convert pixel dimensions to real-world dimensions
        if average_depth > 0:
            pixel_size = (average_depth * self.sensor_width) / (self.focal_length * depth_map.shape[1])
            width_m = w * pixel_size
            height_m = h * pixel_size
        else:
            width_m = 0
            height_m = 0
        
        return {
            'width_pixels': w,
            'height_pixels': h,
            'width_meters': width_m,
            'height_meters': height_m,
            'average_depth': average_depth,
            'diagonal_meters': math.sqrt(width_m**2 + height_m**2)
        }
    
    def calibrate_with_reference(self, image: np.ndarray,
                                 reference_size_meters: float,
                                 reference_pixels: float) -> Dict:
        """
        Calibrate measurement system using reference object
        
        Args:
            image: Image with reference object
            reference_size_meters: Known size of reference in meters
            reference_pixels: Size of reference in pixels
            
        Returns:
            dict: Calibration parameters
        """
        # Calculate pixel-to-meter ratio
        pixel_to_meter = reference_size_meters / reference_pixels
        
        # Update focal length estimate
        # This is simplified - real calibration would be more complex
        self.pixel_to_meter_ratio = pixel_to_meter
        
        return {
            'pixel_to_meter_ratio': pixel_to_meter,
            'calibrated': True
        }
    
    def _is_valid_point(self, depth_map: np.ndarray, point: Tuple[int, int]) -> bool:
        """Check if point is valid in depth map"""
        x, y = point
        return 0 <= y < depth_map.shape[0] and 0 <= x < depth_map.shape[1]





