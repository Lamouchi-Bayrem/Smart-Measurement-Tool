# Smart Measurement Tool - AI-Powered CV Measurement
<img width="1920" height="869" alt="test" src="https://github.com/user-attachments/assets/ea01b21b-dbd3-4611-96d9-bf26f92fd067" />

An intelligent measurement tool using computer vision and depth estimation to measure distances, areas, volumes, and object dimensions from images.

## Features


- ✅ **Depth Estimation**: Monocular depth estimation using gradient-based and model-based methods
- ✅ **Distance Measurement**: Measure distances between two points
- ✅ **Area Measurement**: Calculate area of polygons
- ✅ **Volume Estimation**: Estimate volume from base area and height
- ✅ **Object Dimensions**: Automatic width, height, depth measurement
- ✅ **Reference Calibration**: Calibrate using objects of known size
- ✅ **Multiple Units**: Metric (meters) and Imperial (feet) support
- ✅ **Visual Overlays**: Real-time measurement visualization
- ✅ **Depth Visualization**: Color-coded depth maps
- ✅ **Export Capabilities**: CSV export of measurements

## Measurement Modes

### 1. Distance Measurement
- Click two points on image
- Measures real-world distance
- Accounts for depth differences

### 2. Area Measurement
- Click multiple points to form polygon
- Calculates area in m² or ft²
- Adjusts for depth variations

### 3. Volume Estimation
- Define base polygon
- Specify height point
- Estimates volume in m³ or ft³

### 4. Object Dimensions
- Automatic object detection
- Measures width, height, depth
- Bounding box visualization

### 5. Auto Detect
- Automatic object detection
- Batch measurement of multiple objects

## Requirements

- Python 3.8+
- Webcam/Camera (optional)
- Modern web browser

## Installation

1. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Run the application**:
   ```bash
   streamlit run app.py
   ```
   
   Or use the run script:
   ```bash
   python run.py
   ```

3. **Open in browser**: The app will automatically open at `http://localhost:8501`

## Usage

### Basic Measurement

1. **Upload or Capture Image**:
   - Upload an image file
   - Or use camera to capture

2. **Select Measurement Mode**:
   - Choose from Distance, Area, Volume, Object Dimensions, or Auto Detect

3. **Add Measurement Points**:
   - Enter coordinates or click on image
   - Add points based on measurement type

4. **Calibrate (Recommended)**:
   - Measure a known reference object
   - Enter real-world distance
   - Click "Calibrate with Reference"

5. **View Results**:
   - See measurements with visual overlays
   - Check depth map visualization
   - Export measurements to CSV

### Calibration

For accurate measurements:
1. Identify an object of known size in the image
2. Select two points on that object
3. Enter the real-world distance
4. Click "Calibrate"
5. All subsequent measurements will use this calibration

## Technical Details

### Depth Estimation

- **Monocular Depth**: Uses gradient-based depth estimation
- **Stereo Vision**: Support for stereo camera pairs (optional)
- **Model-Based**: Can integrate MiDaS or similar models
- **Fallback Method**: Gradient magnitude for depth approximation

### Measurement Algorithms

- **Distance**: Euclidean distance with depth adjustment
- **Area**: Shoelace formula for polygon area
- **Volume**: Area × Height estimation
- **Dimensions**: Bounding box analysis with depth variation

### Computer Vision Techniques

- Edge detection for depth estimation
- Contour detection for object recognition
- Bounding box calculation
- Polygon area calculation

## Project Structure

```
smart_measurement/
├── src/
│   ├── __init__.py
│   ├── depth_estimator.py    # Depth estimation algorithms
│   ├── measurement_tool.py    # Measurement calculations
│   └── visualizer.py          # Visualization and overlays
├── app.py                     # Streamlit main app
├── run.py                     # Entry point
├── requirements.txt
└── README.md
```

## Accuracy Notes

⚠️ **Important**: 
- Measurements are estimates based on depth perception
- Accuracy depends on:
  - Image quality and resolution
  - Calibration accuracy
  - Depth estimation quality
  - Camera angle and distance
- For precise measurements, use calibrated reference objects
- Results are approximations suitable for estimation purposes

## Use Cases

- **Construction**: Measure distances and areas in construction sites
- **Real Estate**: Room dimensions and area calculations
- **Interior Design**: Space planning and furniture placement
- **Architecture**: Building measurements and planning
- **E-commerce**: Product dimension estimation
- **AR/VR Applications**: Spatial measurement in augmented reality

## Limitations

- Depth estimation accuracy depends on image quality
- Requires calibration for accurate measurements
- Monocular depth estimation has limitations
- Best results with good lighting and clear images
- Measurements are estimates, not precise

## Future Enhancements

- [ ] Integration with actual MiDaS model for better depth
- [ ] Stereo camera support
- [ ] 3D reconstruction
- [ ] AR overlay for real-time measurement
- [ ] Mobile app version
- [ ] Multiple reference point calibration
- [ ] Machine learning for better depth estimation
- [ ] Point cloud generation
- [ ] Integration with LiDAR data

## Troubleshooting

### Poor Depth Estimation
- Ensure good lighting
- Use high-resolution images
- Avoid flat, textureless surfaces
- Try different camera angles

### Inaccurate Measurements
- Calibrate with known reference object
- Ensure reference object is in same plane as measured object
- Check that points are accurately selected
- Verify calibration distance is correct

### No Depth Map Generated
- Check image format and quality
- Ensure image is properly loaded
- Try different image
- Check console for errors

## License

This project is provided as-is for educational and portfolio purposes.

## Acknowledgments

- OpenCV for computer vision
- Streamlit for web framework
- Depth estimation research community







