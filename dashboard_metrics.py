from datetime import datetime, timedelta
from collections import defaultdict, deque
import pandas as pd

class AnalyticsTracker:
    def __init__(self):
        self.session_start = datetime.now()
        self.behavior_counts = defaultdict(int)
        self.emotion_counts = defaultdict(int)
        self.behavior_confidences = defaultdict(list)
        self.emotion_confidences = defaultdict(list)
        self.recent_activity = deque(maxlen=100)
        self.total_detections = 0
    
    def add_behavior_detection(self, behavior, confidence):
        """Add a behavior detection to analytics"""
        self.behavior_counts[behavior] += 1
        self.behavior_confidences[behavior].append(confidence)
        self.total_detections += 1
        
        self.recent_activity.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'Behavior',
            'detection': behavior.title(),
            'confidence': f"{confidence:.2f}"
        })
    
    def add_emotion_detection(self, emotion, confidence):
        """Add an emotion detection to analytics"""
        self.emotion_counts[emotion] += 1
        self.emotion_confidences[emotion].append(confidence)
        self.total_detections += 1
        
        self.recent_activity.append({
            'timestamp': datetime.now().strftime('%H:%M:%S'),
            'type': 'Emotion',
            'detection': emotion.title(),
            'confidence': f"{confidence:.2f}"
        })
    
    def get_session_stats(self):
        """Get overall session statistics"""
        duration = datetime.now() - self.session_start
        duration_minutes = duration.total_seconds() / 60
        
        return {
            'total_detections': self.total_detections,
            'unique_behaviors': len(self.behavior_counts),
            'unique_emotions': len(self.emotion_counts),
            'duration_minutes': duration_minutes,
            'session_start': self.session_start.strftime('%H:%M:%S')
        }
    
    def get_behavior_distribution(self):
        """Get behavior count distribution"""
        return dict(self.behavior_counts)
    
    def get_emotion_distribution(self):
        """Get emotion count distribution"""
        return dict(self.emotion_counts)
    
    def get_recent_activity(self, limit=10):
        """Get recent detection activity"""
        return list(self.recent_activity)[-limit:]
    
    def get_top_behaviors(self, limit=5):
        """Get top detected behaviors"""
        sorted_behaviors = sorted(
            self.behavior_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_behaviors[:limit]
    
    def get_top_emotions(self, limit=5):
        """Get top detected emotions"""
        sorted_emotions = sorted(
            self.emotion_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_emotions[:limit]
    
    def get_average_confidence(self, detection_type='all'):
        """Get average confidence scores"""
        if detection_type == 'behavior' or detection_type == 'all':
            behavior_confidences = []
            for confidences in self.behavior_confidences.values():
                behavior_confidences.extend(confidences)
            avg_behavior = sum(behavior_confidences) / len(behavior_confidences) if behavior_confidences else 0
        else:
            avg_behavior = 0
        
        if detection_type == 'emotion' or detection_type == 'all':
            emotion_confidences = []
            for confidences in self.emotion_confidences.values():
                emotion_confidences.extend(confidences)
            avg_emotion = sum(emotion_confidences) / len(emotion_confidences) if emotion_confidences else 0
        else:
            avg_emotion = 0
        
        if detection_type == 'all':
            return {
                'behavior': avg_behavior,
                'emotion': avg_emotion,
                'overall': (avg_behavior + avg_emotion) / 2 if (avg_behavior or avg_emotion) else 0
            }
        elif detection_type == 'behavior':
            return avg_behavior
        else:
            return avg_emotion
    
    def export_session_data(self):
        """Export session data as dictionary for saving"""
        return {
            'session_start': self.session_start.isoformat(),
            'session_end': datetime.now().isoformat(),
            'behavior_counts': dict(self.behavior_counts),
            'emotion_counts': dict(self.emotion_counts),
            'behavior_confidences': {k: v for k, v in self.behavior_confidences.items()},
            'emotion_confidences': {k: v for k, v in self.emotion_confidences.items()},
            'total_detections': self.total_detections,
            'recent_activity': list(self.recent_activity)
        }
    
    def reset_session(self):
        """Reset all analytics data"""
        self.session_start = datetime.now()
        self.behavior_counts.clear()
        self.emotion_counts.clear()
        self.behavior_confidences.clear()
        self.emotion_confidences.clear()
        self.recent_activity.clear()
        self.total_detections = 0
