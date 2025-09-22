import cv2
import numpy as np
import mediapipe as mp
import math

class BehaviorDetector:
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Store pose history for temporal analysis
        self.pose_history = []
        self.max_history = 10
    
    def detect(self, frame):
        """Detect behavior from pose landmarks"""
        try:
            # Convert BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.pose.process(rgb_frame)
            
            if not results.pose_landmarks:
                return None
            
            landmarks = results.pose_landmarks.landmark
            
            # Extract key points
            keypoints = self._extract_keypoints(landmarks)
            
            # Store in history
            self.pose_history.append(keypoints)
            if len(self.pose_history) > self.max_history:
                self.pose_history.pop(0)
            
            # Classify behavior
            behavior, confidence = self._classify_behavior(keypoints)
            
            return {
                'behavior': behavior,
                'confidence': confidence,
                'landmarks': results.pose_landmarks,
                'keypoints': keypoints
            }
            
        except Exception as e:
            print(f"Error in behavior detection: {str(e)}  behavior_logic.py:54 - pose_behavior.py:54")
            return None
    
    def _extract_keypoints(self, landmarks):
        """Extract key pose points for analysis"""
        keypoints = {}
        
        # Define key landmark indices
        landmark_indices = {
            'nose': 0,
            'left_shoulder': 11,
            'right_shoulder': 12,
            'left_elbow': 13,
            'right_elbow': 14,
            'left_wrist': 15,
            'right_wrist': 16,
            'left_hip': 23,
            'right_hip': 24,
            'left_knee': 25,
            'right_knee': 26,
            'left_ankle': 27,
            'right_ankle': 28
        }
        
        for name, idx in landmark_indices.items():
            if idx < len(landmarks):
                landmark = landmarks[idx]
                keypoints[name] = {
                    'x': landmark.x,
                    'y': landmark.y,
                    'z': landmark.z,
                    'visibility': landmark.visibility
                }
        
        return keypoints
    
    def _classify_behavior(self, keypoints):
        """Classify behavior based on pose keypoints"""
        behaviors = []
        
        # Check for waving
        wave_confidence = self._detect_waving(keypoints)
        if wave_confidence > 0.25:
            behaviors.append(('waving', wave_confidence))
        
        # Check for standing/sitting
        posture_behavior, posture_confidence = self._detect_posture(keypoints)
        if posture_confidence > 0.3:
            behaviors.append((posture_behavior, posture_confidence))
        
        # Check for walking
        walk_confidence = self._detect_walking()
        if walk_confidence > 0.2:
            behaviors.append(('walking', walk_confidence))
        
        # Return highest confidence behavior
        if behaviors:
            behavior, confidence = max(behaviors, key=lambda x: x[1])
            return behavior, confidence
        else:
            return 'standing', 0.5  # Default behavior
    
    def _detect_waving(self, keypoints):
        """Detect waving gesture"""
        try:
            # Check if hands are raised
            if 'left_wrist' not in keypoints or 'right_wrist' not in keypoints:
                return 0.0
            
            left_wrist = keypoints['left_wrist']
            right_wrist = keypoints['right_wrist']
            left_shoulder = keypoints.get('left_shoulder', {})
            right_shoulder = keypoints.get('right_shoulder', {})
            
            confidence = 0.0
            
            # Check if left hand is raised above shoulder
            if (left_shoulder and left_wrist['y'] < left_shoulder['y'] and 
                left_wrist['visibility'] > 0.5):
                confidence += 0.5
            
            # Check if right hand is raised above shoulder
            if (right_shoulder and right_wrist['y'] < right_shoulder['y'] and 
                right_wrist['visibility'] > 0.5):
                confidence += 0.5
            
            # Check for hand movement in history
            if len(self.pose_history) >= 3:
                movement_score = self._calculate_hand_movement()
                confidence += movement_score * 0.3
            
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0
    
    def _detect_posture(self, keypoints):
        try:
        # Check if hips are available
            if 'left_hip' not in keypoints or 'right_hip' not in keypoints:
                return 'unknown', 0.3

            avg_hip_y = (keypoints['left_hip']['y'] + keypoints['right_hip']['y']) / 2

        # Check if knees are visible
            left_knee_visible = keypoints.get('left_knee', {}).get('visibility', 0) > 0.5
            right_knee_visible = keypoints.get('right_knee', {}).get('visibility', 0) > 0.5

            if left_knee_visible and right_knee_visible:
                avg_knee_y = (keypoints['left_knee']['y'] + keypoints['right_knee']['y']) / 2
                hip_knee_ratio = abs(avg_knee_y - avg_hip_y)

                if hip_knee_ratio < 0.15:
                    return 'sitting', 0.9
                else:
                    return 'standing', 0.9
            else:
            # Knees not visible → cannot determine posture
                return 'unknown', 0.5

        except Exception:
            return 'unknown', 0.3

   
    
    def _detect_walking(self):
        """Detect walking based on pose history"""
        if len(self.pose_history) < 5:
            return 0.0
        # Check if knees are visible in latest frame
        latest_pose = self.pose_history[-1]
        if ('left_knee' not in latest_pose) or ('right_knee' not in latest_pose):
            return 0.0   # force unknown if knees missing


        try:
            # Analyze ankle movement patterns
            ankle_movements = []
            for i in range(1, len(self.pose_history)):
                prev_pose = self.pose_history[i-1]
                curr_pose = self.pose_history[i]
                
                if ('left_ankle' in prev_pose and 'left_ankle' in curr_pose and
                    'right_ankle' in prev_pose and 'right_ankle' in curr_pose):
                    
                    left_movement = abs(curr_pose['left_ankle']['x'] - prev_pose['left_ankle']['x'])
                    right_movement = abs(curr_pose['right_ankle']['x'] - prev_pose['right_ankle']['x'])
                    ankle_movements.append(left_movement + right_movement)
            
            if ankle_movements:
                avg_movement = sum(ankle_movements) / len(ankle_movements)
                # Normalize movement score
                walking_confidence = min(avg_movement * 10, 1.0)
                return walking_confidence
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def _calculate_hand_movement(self):
        """Calculate hand movement score from pose history"""
        if len(self.pose_history) < 3:
            return 0.0
        
        try:
            movements = []
            for i in range(1, len(self.pose_history)):
                prev_pose = self.pose_history[i-1]
                curr_pose = self.pose_history[i]
                
                for hand in ['left_wrist', 'right_wrist']:
                    if hand in prev_pose and hand in curr_pose:
                        dx = curr_pose[hand]['x'] - prev_pose[hand]['x']
                        dy = curr_pose[hand]['y'] - prev_pose[hand]['y']
                        movement = math.sqrt(dx*dx + dy*dy)
                        movements.append(movement)
            
            if movements:
                avg_movement = sum(movements) / len(movements)
                return min(avg_movement * 20, 1.0)
            
            return 0.0
            
        except Exception:
            return 0.0
    
    def draw_landmarks(self, frame, landmarks):
        """Draw pose landmarks on frame"""
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=self.mp_drawing_styles.get_default_pose_landmarks_style()
            )
            return frame
        except Exception:
            return frame
