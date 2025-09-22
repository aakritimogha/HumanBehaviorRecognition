import cv2
import numpy as np
import streamlit as st

class CameraHandler:
    def __init__(self):
        self.cap = None
        self.is_initialized = False
    
    def initialize_camera(self, camera_index=0):
        """Initialize camera with given index"""
        try:
            if self.cap is not None:
                self.cap.release()
            
            self.cap = cv2.VideoCapture(camera_index)
            
            if not self.cap.isOpened():
                st.error(f"❌ Could not open camera {camera_index}")
                return False
            
            # Set camera properties for better performance
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.is_initialized = True
            st.success(f"✅ Camera {camera_index} initialized successfully")
            return True
            
        except Exception as e:
            st.error(f"❌ Error initializing camera: {str(e)}")
            return False
    
    def get_frame(self):
        """Get current frame from camera"""
        if not self.is_initialized or self.cap is None:
            return None
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                return None
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            return frame
            
        except Exception as e:
            st.error(f"❌ Error capturing frame: {str(e)}")
            return None
    
    def release_camera(self):
        """Release camera resources"""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.is_initialized = False
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.release_camera()
