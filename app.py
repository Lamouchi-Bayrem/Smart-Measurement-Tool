"""
Streamlit Web App for Smart Measurement Tool
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd
from typing import List, Tuple

from src.depth_estimator import DepthEstimator
from src.measurement_tool import MeasurementTool, ObjectDetector
from src.visualizer import MeasurementVisualizer


# Page configuration
st.set_page_config(
    page_title="Smart Measurement Tool",
    page_icon="📏",
    layout="wide"
)

# Initialize session state
if 'depth_estimator' not in st.session_state:
    st.session_state.depth_estimator = DepthEstimator()
if 'measurement_tool' not in st.session_state:
    st.session_state.measurement_tool = MeasurementTool(st.session_state.depth_estimator)
if 'visualizer' not in st.session_state:
    st.session_state.visualizer = MeasurementVisualizer()
if 'object_detector' not in st.session_state:
    st.session_state.object_detector = ObjectDetector()
if 'current_image' not in st.session_state:
    st.session_state.current_image = None
if 'depth_map' not in st.session_state:
    st.session_state.depth_map = None
if 'measurement_points' not in st.session_state:
    st.session_state.measurement_points = []
if 'measurement_mode' not in st.session_state:
    st.session_state.measurement_mode = 'distance'
if 'calibrated' not in st.session_state:
    st.session_state.calibrated = False


def reset_measurements():
    """Reset measurement state"""
    st.session_state.measurement_points = []
    st.session_state.calibrated = False


def main():
    """Main application"""
    st.title("📏 Smart Measurement Tool")
    st.markdown("**AI-Powered Measurement Using Computer Vision & Depth Estimation**")
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Settings")
        
        # Measurement mode
        st.subheader("📐 Measurement Mode")
        measurement_mode = st.radio(
            "Select Mode",
            ["Distance", "Area", "Volume", "Object Dimensions", "Auto Detect"],
            index=0
        )
        st.session_state.measurement_mode = measurement_mode.lower().replace(' ', '_')
        
        # Calibration
        st.subheader("🎯 Calibration")
        st.markdown("**Reference Object Calibration**")
        ref_distance = st.number_input("Reference Distance (meters)", 
                                      min_value=0.01, max_value=10.0, 
                                      value=1.0, step=0.01)
        
        if st.button("📏 Calibrate with Reference"):
            if len(st.session_state.measurement_points) >= 2:
                st.session_state.measurement_tool.calibrate_with_reference(
                    st.session_state.measurement_points[:2],
                    ref_distance
                )
                st.session_state.calibrated = True
                st.success("✅ Calibrated!")
            else:
                st.warning("Please select 2 points for calibration")
        
        # Unit selection
        st.subheader("📊 Units")
        unit_system = st.selectbox("Unit System", ["Metric (m)", "Imperial (ft)"])
        unit_factor = 3.28084 if "Imperial" in unit_system else 1.0
        unit_label = "ft" if "Imperial" in unit_system else "m"
        
        # Clear measurements
        if st.button("🗑️ Clear Measurements"):
            reset_measurements()
            st.rerun()
        
        # Instructions
        st.subheader("📖 Instructions")
        st.markdown("""
        1. Upload or capture image
        2. Select measurement mode
        3. Click points on image
        4. Calibrate with known reference
        5. View measurements
        """)
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📷 Image Input")
        
        # Image upload or camera
        input_method = st.radio("Input Method", ["Upload Image", "Camera"], horizontal=True)
        
        if input_method == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image",
                type=['jpg', 'jpeg', 'png'],
                help="Upload an image for measurement"
            )
            
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                image_np = np.array(image)
                if len(image_np.shape) == 3:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                else:
                    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
                
                st.session_state.current_image = image_bgr
        else:
            camera_input = st.camera_input("Capture image")
            if camera_input:
                image = Image.open(camera_input)
                image_np = np.array(image)
                image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                st.session_state.current_image = image_bgr
        
        # Process image if available
        if st.session_state.current_image is not None:
            # Estimate depth
            if st.session_state.depth_map is None:
                with st.spinner("Estimating depth..."):
                    st.session_state.depth_map = st.session_state.depth_estimator.estimate_depth(
                        st.session_state.current_image
                    )
            
            # Display image with clickable points
            st.markdown("### Click on image to add measurement points")
            
            # Create clickable image
            display_image = st.session_state.current_image.copy()
            
            # Draw existing points
            for i, point in enumerate(st.session_state.measurement_points):
                cv2.circle(display_image, point, 8, (0, 255, 0), -1)
                cv2.putText(display_image, str(i+1), (point[0]+10, point[1]-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display image
            st.image(cv2.cvtColor(display_image, cv2.COLOR_BGR2RGB), use_container_width=True)
            
            # Point selection (simplified - in real app would use click events)
            col_a, col_b = st.columns(2)
            with col_a:
                x_coord = st.number_input("X coordinate", min_value=0, 
                                         max_value=display_image.shape[1], value=100)
            with col_b:
                y_coord = st.number_input("Y coordinate", min_value=0,
                                         max_value=display_image.shape[0], value=100)
            
            if st.button("➕ Add Point"):
                st.session_state.measurement_points.append((int(x_coord), int(y_coord)))
                st.rerun()
            
            # Perform measurements
            if len(st.session_state.measurement_points) > 0:
                st.markdown("### Measurement Results")
                
                annotated_image = display_image.copy()
                measurements = {}
                
                if st.session_state.measurement_mode == 'distance':
                    if len(st.session_state.measurement_points) >= 2:
                        result = st.session_state.measurement_tool.measure_distance(
                            st.session_state.measurement_points[0],
                            st.session_state.measurement_points[1],
                            st.session_state.depth_map
                        )
                        distance = result['real_distance'] * unit_factor
                        measurements['distance'] = distance
                        
                        annotated_image = st.session_state.visualizer.draw_distance_measurement(
                            annotated_image,
                            st.session_state.measurement_points[0],
                            st.session_state.measurement_points[1],
                            distance,
                            unit_label
                        )
                        
                        st.metric("Distance", f"{distance:.2f} {unit_label}")
                
                elif st.session_state.measurement_mode == 'area':
                    if len(st.session_state.measurement_points) >= 3:
                        result = st.session_state.measurement_tool.measure_area(
                            st.session_state.measurement_points,
                            st.session_state.depth_map
                        )
                        area = result['area_real'] * (unit_factor ** 2)
                        measurements['area'] = area
                        
                        annotated_image = st.session_state.visualizer.draw_area_measurement(
                            annotated_image,
                            st.session_state.measurement_points,
                            area,
                            f"{unit_label}²"
                        )
                        
                        st.metric("Area", f"{area:.2f} {unit_label}²")
                
                elif st.session_state.measurement_mode == 'volume':
                    if len(st.session_state.measurement_points) >= 4:
                        base_points = st.session_state.measurement_points[:-1]
                        height_point = st.session_state.measurement_points[-1]
                        
                        result = st.session_state.measurement_tool.measure_volume(
                            base_points,
                            height_point,
                            st.session_state.depth_map
                        )
                        volume = result['volume'] * (unit_factor ** 3)
                        measurements['volume'] = volume
                        
                        annotated_image = st.session_state.visualizer.draw_volume_measurement(
                            annotated_image,
                            base_points,
                            height_point,
                            volume,
                            f"{unit_label}³"
                        )
                        
                        st.metric("Volume", f"{volume:.2f} {unit_label}³")
                
                # Display annotated image
                st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), 
                       use_container_width=True, caption="Measurement Overlay")
                
                # Depth map visualization
                st.markdown("### Depth Map")
                depth_colored = cv2.applyColorMap(
                    (st.session_state.depth_map * 255).astype(np.uint8),
                    cv2.COLORMAP_JET
                )
                st.image(depth_colored, use_container_width=True, 
                        caption="Depth Estimation (Red=Close, Blue=Far)")
    
    with col2:
        st.subheader("📊 Measurement Data")
        
        # Current points
        if st.session_state.measurement_points:
            st.markdown("**Selected Points:**")
            points_df = pd.DataFrame(
                st.session_state.measurement_points,
                columns=['X', 'Y']
            )
            st.dataframe(points_df, use_container_width=True)
            
            # Measurement history
            st.markdown("**Measurement History:**")
            if measurements:
                measurements_df = pd.DataFrame([measurements])
                st.dataframe(measurements_df, use_container_width=True)
        
        # Calibration status
        st.markdown("**Calibration Status:**")
        if st.session_state.calibrated:
            st.success("✅ Calibrated")
            st.info(f"Reference: {st.session_state.measurement_tool.reference_distance}m")
        else:
            st.warning("⚠️ Not calibrated")
            st.info("Calibrate with a known reference object for accurate measurements")
        
        # Export options
        st.subheader("💾 Export")
        if st.button("📥 Export Measurements"):
            if measurements:
                df = pd.DataFrame([measurements])
                csv = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV",
                    data=csv,
                    file_name="measurements.csv",
                    mime="text/csv"
                )


if __name__ == "__main__":
    main()





