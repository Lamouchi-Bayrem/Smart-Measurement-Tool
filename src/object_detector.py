"""
Object Detection Module
Detects objects for automatic measurement
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional
import mediapipe as mp


class ObjectDetector:
    """Detects objects in images for measurement"""
    
    def __init__(self):
        """Initialize object detector"""
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.5
        )
    
    def detect_objects_contour(self, image: np.ndarray) -> List[Dict]:
        """
        Detect objects using contour detection
        
        Args:
            image: Input image
            
        Returns:
            list: Detected objects with bounding boxes
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        objects = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Filter small contours
                x, y, w, h = cv2.boundingRect(contour)
                objects.append({
                    'bbox': (x, y, w, h),
                    'contour': contour,
                    'area': area,
                    'type': 'contour'
                })
        
        return objects
    
    def detect_hand_reference(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect hand for reference measurement
        
        Args:
            image: Input image
            
        Returns:
            dict: Hand detection result with reference size
        """
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(rgb_image)
        
        if results.multi_hand_landmarks:
            hand_landmarks = results.multi_hand_landmarks[0]
            
            # Get hand bounding box
            h, w = image.shape[:2]
            x_coords = [lm.x * w for lm in hand_landmarks.landmark]
            y_coords = [lm.y * h for lm in hand_landmarks.landmark]
            
            x_min, x_max = int(min(x_coords)), int(max(x_coords))
            y_min, y_max = int(min(y_coords)), int(max(y_coords))
            
            # Estimate hand width (average adult hand width ~0.08-0.1m)
            hand_width_pixels = x_max - x_min
            hand_width_meters = 0.09  # Average
            
            return {
                'detected': True,
                'bbox': (x_min, y_min, x_max - x_min, y_max - y_min),
                'width_pixels': hand_width_pixels,
                'width_meters': hand_width_meters,
                'pixel_to_meter_ratio': hand_width_meters / hand_width_pixels if hand_width_pixels > 0 else 0
            }
        
        return None
    
    def detect_aruco_marker(self, image: np.ndarray) -> Optional[Dict]:
        """
        Detect ArUco marker for calibration
        
        Args:
            image: Input image
            
        Returns:
            dict: ArUco marker detection result
        """
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Initialize ArUco detector
        aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        parameters = cv2.aruco.DetectorParameters()
        detector = cv2.aruco.ArucoDetector(aruco_dict, parameters)
        
        # Detect markers
        corners, ids, _ = detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            # Get first marker
            marker_corners = corners[0][0]
            
            # Calculate marker size in pixels
            side_lengths = []
            for i in range(4):
                p1 = marker_corners[i]
                p2 = marker_corners[(i + 1) % 4]
                length = np.linalg.norm(p2 - p1)
                side_lengths.append(length)
            
            avg_side_length = np.mean(side_lengths)
            
            # Standard ArUco marker size (example: 0.05m = 5cm)
            marker_size_meters = 0.05
            
            return {
                'detected': True,
                'corners': marker_corners,
                'id': ids[0][0],
                'side_length_pixels': avg_side_length,
                'side_length_meters': marker_size_meters,
                'pixel_to_meter_ratio': marker_size_meters / avg_side_length if avg_side_length > 0 else 0
            }
        
        return None





