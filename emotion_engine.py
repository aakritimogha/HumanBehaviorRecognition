import mediapipe as mp
import math

class EmotionDetector:
    def __init__(self):
        

        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        self.key_landmarks = {
            'left_eye': [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
            'right_eye': [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
            'mouth': [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 415, 310, 311, 312, 13, 82, 81, 80, 61],
            'eyebrows': [70, 63, 105, 66, 107, 55, 65, 52, 53, 46, 296, 334, 293, 300, 276, 283, 282, 295, 285]
        }

    # ------------------ DETECT FUNCTION ------------------
    def detect(self, frame):
        import cv2
        try:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)

            if not results.multi_face_landmarks:
                return None

            landmarks = results.multi_face_landmarks[0].landmark
            features = self._extract_facial_features(landmarks, frame.shape)
            emotion, confidence = self._classify_emotion(features)

            return {
                'emotion': emotion,
                'confidence': confidence,
                'landmarks': results.multi_face_landmarks[0],
                'features': features
            }

        except Exception as e:
            print(f"Error in emotion detection: {str(e)} - emotion_engine.py:48")
            return None

    # ------------------ CLASSIFY EMOTION ------------------
    def _classify_emotion(self, features):
        try:
            emotions = []

        # Happy
            happy_score = self._detect_happy(features)
            if happy_score > 0.35:
                emotions.append(('happy', happy_score))


        # Sad
            sad_score = self._detect_sad(features)
            if sad_score > 0.35:
                emotions.append(('sad', sad_score))

        # Surprised
            surprised_score = self._detect_surprised(features)
            if surprised_score > 0.2:
                emotions.append(('surprised', surprised_score))

        # Sleepy
            sleepy_score = self._detect_sleepy(features)
            if sleepy_score > 0.25:
                emotions.append(('sleepy', sleepy_score))

        # Cry
            cry_score = self._detect_cry(features)
            if cry_score > 0.35:
                emotions.append(('cry', cry_score))

        # Flu
            flu_score = self._detect_flu(features)
            if flu_score > 0.25:
                emotions.append(('flu', flu_score))

        # Smoking
            smoking_score = self._detect_smoking(features)
            if smoking_score > 0.25:
                emotions.append(('smoking', smoking_score))

        # Anger
            anger_score = self._detect_anger(features)
            if anger_score > 0.35:
                emotions.append(('anger', anger_score))


        # Determine final emotion
            if emotions:
            # Pick highest confidence
                emotion, confidence = max(emotions, key=lambda x: x[1])
            # Fallback to neutral if highest confidence is still low
                if confidence < 0.4:
                    return 'neutral', confidence
                return emotion, confidence
            else:
                return 'neutral', 0.6

        except Exception:
            return 'neutral', 0.5


    # ------------------ FEATURE EXTRACTION ------------------
    def _extract_facial_features(self, landmarks, frame_shape):
        h, w = frame_shape[:2]
        features = {}
        points = {}

        for region, indices in self.key_landmarks.items():
            region_points = []
            for idx in indices:
                if idx < len(landmarks):
                    l = landmarks[idx]
                    region_points.append((int(l.x*w), int(l.y*h)))
            points[region] = region_points

        features['eye_aspect_ratio'] = self._calculate_eye_aspect_ratio(points)
        features['mouth_aspect_ratio'] = self._calculate_mouth_aspect_ratio(points)
        features['eyebrow_position'] = self._calculate_eyebrow_position(points)
        features['mouth_curve'] = self._calculate_mouth_curve(points)
        return features

    # ------------------ CALCULATIONS ------------------
    def _calculate_eye_aspect_ratio(self, points):
        left_eye = points.get('left_eye', [])
        right_eye = points.get('right_eye', [])
        if len(left_eye)<6 or len(right_eye)<6: return 0.3
        left_ear = self._calculate_single_ear(left_eye)
        right_ear = self._calculate_single_ear(right_eye)
        return (left_ear + right_ear)/2

    def _calculate_single_ear(self, eye_points):
        try:
            vertical_1 = self._euclidean_distance(eye_points[1], eye_points[5])
            vertical_2 = self._euclidean_distance(eye_points[2], eye_points[4])
            horizontal = self._euclidean_distance(eye_points[0], eye_points[3])
            if horizontal==0: return 0.3
            return (vertical_1 + vertical_2)/(2.0*horizontal)
        except: return 0.3

    def _calculate_mouth_aspect_ratio(self, points):
        mouth = points.get('mouth', [])
        if len(mouth)<10: return 0.5
        width = self._euclidean_distance(mouth[0], mouth[10])
        height = self._euclidean_distance(mouth[5], mouth[15])
        if width==0: return 0.5
        return height/width

    def _calculate_eyebrow_position(self, points):
        eyebrows = points.get('eyebrows', [])
        left_eye = points.get('left_eye', [])
        if len(eyebrows)<4 or len(left_eye)<4: return 0.5
        eyebrow_y = sum([p[1] for p in eyebrows[:4]])/4
        eye_y = sum([p[1] for p in left_eye[:4]])/4
        relative_pos = abs(eyebrow_y-eye_y)/100.0
        return min(relative_pos,1.0)

    def _calculate_mouth_curve(self, points):
        mouth = points.get('mouth', [])
        if len(mouth)<10: return 0.0
        left_corner = mouth[0]
        right_corner = mouth[10]
        center_top = mouth[5]
        center_bottom = mouth[15]
        curve = (center_top[1]+center_bottom[1])/2 - (left_corner[1]+right_corner[1])/2
        return max(-1.0, min(1.0, curve/50.0))

    def _euclidean_distance(self, p1,p2):
        import math
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

    # ------------------ EMOTION DETECTORS ------------------
    def _detect_happy(self, features):
        score = 0.0
        if features.get('mouth_curve',0) > 0.05: score+=0.4
        ear = features.get('eye_aspect_ratio',0.3)
        if 0.18 < ear < 0.38: score+=0.3
        if features.get('mouth_aspect_ratio',0.5) > 0.38: score+=0.3
        return min(score,1.0)

    def _detect_sad(self, features):
        score = 0.0
        if features.get('mouth_curve',0) < -0.05: score+=0.4
        if features.get('eyebrow_position',0.5) < 0.35: score+=0.3
        if features.get('eye_aspect_ratio',0.3) < 0.28: score+=0.3
        return min(score,1.0)

    def _detect_surprised(self, features):
        score = 0.0
        if features.get('eye_aspect_ratio',0.3) > 0.38: score+=0.4
        if features.get('mouth_aspect_ratio',0.5) > 0.75: score+=0.4
        if features.get('eyebrow_position',0.5) > 0.65: score+=0.2
        return min(score,1.0)

    def _detect_sleepy(self, features):
        return 0.6 if features.get('eye_aspect_ratio',0.3) < 0.25 else 0.0

    def _detect_cry(self, features):
        score=0.0
        if features.get('mouth_curve',0) < -0.15: score+=0.4
        if features.get('eye_aspect_ratio',0.3) < 0.26: score+=0.3
        if features.get('eyebrow_position',0.5) < 0.3: score+=0.3
        return min(score,1.0)

    def _detect_flu(self, features):
        score=0.0
        if features.get('mouth_aspect_ratio',0.5) > 0.55: score+=0.4
        if features.get('eye_aspect_ratio',0.3) < 0.28: score+=0.3
        return min(score,1.0)

    def _detect_smoking(self, features):
        score=0.0
        mar = features.get('mouth_aspect_ratio',0.5)
        if 0.3 < mar < 0.6: score+=0.5
        return score
    
    def _detect_anger(self, features):
        score = 0.0
        if features.get('eyebrow_position', 0.5) < 0.3:  # eyebrows niche
            score += 0.4
        if features.get('eye_aspect_ratio', 0.3) < 0.28:  # aankh thodi band
            score += 0.3
        if features.get('mouth_curve', 0) < 0.05:  # mouth flat/tight
            score += 0.3
        return min(score, 1.0)



    # ------------------ DRAWING ------------------
    def draw_landmarks(self, frame, landmarks):
        try:
            self.mp_drawing.draw_landmarks(
                frame,
                landmarks,
                self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles.get_default_face_mesh_contours_style()
            )
            return frame
        except:
            return frame
