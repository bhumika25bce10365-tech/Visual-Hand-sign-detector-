import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time

class SignLanguageInterpreter:
    def __init__(self):
        # Initialize MediaPipe hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        
        # Store recent landmarks for gesture recognition
        self.landmark_history = deque(maxlen=30)
        
        # Recognized text
        self.recognized_text = ""
        self.last_gesture = ""
        self.gesture_start_time = None
        self.gesture_hold_threshold = 1.5  # seconds
        
    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)
    
    def is_finger_extended(self, landmarks, finger_tip_idx, finger_pip_idx):
        """Check if a finger is extended based on tip and PIP joint positions"""
        tip = landmarks[finger_tip_idx]
        pip = landmarks[finger_pip_idx]
        return tip[1] < pip[1]  # Tip is above PIP (extended)
    
    def count_extended_fingers(self, landmarks):
        """Count how many fingers are extended"""
        count = 0
        # Index finger
        if self.is_finger_extended(landmarks, 8, 6):
            count += 1
        # Middle finger
        if self.is_finger_extended(landmarks, 12, 10):
            count += 1
        # Ring finger
        if self.is_finger_extended(landmarks, 16, 14):
            count += 1
        # Pinky finger
        if self.is_finger_extended(landmarks, 20, 18):
            count += 1
        # Thumb (different logic - check x-axis)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        if abs(thumb_tip[0] - thumb_ip[0]) > 0.04:
            count += 1
        return count
    
    def recognize_gesture(self, landmarks):
        """
        Enhanced gesture recognition for more ASL letters
        Returns the recognized letter/gesture
        """
        if not landmarks:
            return None
        
        # Extract key landmark positions (normalized coordinates)
        thumb_tip = landmarks[4]
        thumb_ip = landmarks[3]
        thumb_mcp = landmarks[2]
        
        index_tip = landmarks[8]
        index_dip = landmarks[7]
        index_pip = landmarks[6]
        index_mcp = landmarks[5]
        
        middle_tip = landmarks[12]
        middle_pip = landmarks[10]
        middle_mcp = landmarks[9]
        
        ring_tip = landmarks[16]
        ring_pip = landmarks[14]
        ring_mcp = landmarks[13]
        
        pinky_tip = landmarks[20]
        pinky_pip = landmarks[18]
        pinky_mcp = landmarks[17]
        
        wrist = landmarks[0]
        
        # Calculate distances
        thumb_index_dist = self.calculate_distance(thumb_tip, index_tip)
        thumb_middle_dist = self.calculate_distance(thumb_tip, middle_tip)
        index_middle_dist = self.calculate_distance(index_tip, middle_tip)
        middle_ring_dist = self.calculate_distance(middle_tip, ring_tip)
        ring_pinky_dist = self.calculate_distance(ring_tip, pinky_tip)
        
        # Count extended fingers
        extended_count = self.count_extended_fingers(landmarks)
        
        # Check finger states
        index_extended = self.is_finger_extended(landmarks, 8, 6)
        middle_extended = self.is_finger_extended(landmarks, 12, 10)
        ring_extended = self.is_finger_extended(landmarks, 16, 14)
        pinky_extended = self.is_finger_extended(landmarks, 20, 18)
        
        # ASL Letter Recognition
        
        # 'A' - Closed fist with thumb alongside
        if extended_count == 0 and thumb_tip[1] > index_mcp[1]:
            return 'A'
        
        # 'B' - Open hand, all fingers extended together, thumb tucked
        elif extended_count == 4 and index_middle_dist < 0.05 and thumb_tip[0] < thumb_mcp[0]:
            return 'B'
        
        # 'C' - Curved hand shape
        elif thumb_index_dist > 0.08 and thumb_index_dist < 0.15 and extended_count >= 3:
            return 'C'
        
        # 'D' - Index finger up, other fingers touching thumb
        elif index_extended and not middle_extended and not ring_extended and thumb_middle_dist < 0.06:
            return 'D'
        
        # 'E' - All fingers curled, thumb across
        elif extended_count == 0 and thumb_tip[1] < index_mcp[1]:
            return 'E'
        
        # 'F' - Index and thumb touching in circle, other fingers extended
        elif thumb_index_dist < 0.05 and middle_extended and ring_extended and pinky_extended:
            return 'F'
        
        # 'G' - Index finger and thumb pointing horizontally
        elif index_extended and not middle_extended and thumb_tip[0] > thumb_mcp[0] and abs(thumb_tip[1] - index_tip[1]) < 0.08:
            return 'G'
        
        # 'H' - Index and middle fingers extended horizontally
        elif index_extended and middle_extended and not ring_extended and index_middle_dist < 0.06:
            return 'H'
        
        # 'I' - Only pinky extended
        elif pinky_extended and not index_extended and not middle_extended and not ring_extended:
            return 'I'
        
        # 'K' - Index up, middle out, thumb between
        elif index_extended and middle_extended and not ring_extended and index_middle_dist > 0.08:
            return 'K'
        
        # 'L' - L shape with thumb and index
        elif index_extended and not middle_extended and thumb_tip[0] > thumb_mcp[0] and thumb_tip[1] < index_mcp[1]:
            return 'L'
        
        # 'O' - Circle with all fingers
        elif thumb_index_dist < 0.05 and extended_count <= 2:
            return 'O'
        
        # 'R' - Index and middle crossed
        elif index_extended and middle_extended and not ring_extended and index_middle_dist < 0.03:
            return 'R'
        
        # 'U' - Index and middle together pointing up
        elif index_extended and middle_extended and not ring_extended and index_middle_dist < 0.04:
            return 'U'
        
        # 'V' - Index and middle separated (peace sign)
        elif index_extended and middle_extended and not ring_extended and index_middle_dist > 0.06:
            return 'V'
        
        # 'W' - Three fingers extended and separated
        elif index_extended and middle_extended and ring_extended and not pinky_extended:
            return 'W'
        
        # 'Y' - Thumb and pinky extended (shaka sign)
        elif pinky_extended and not index_extended and not middle_extended and thumb_tip[0] > thumb_mcp[0]:
            return 'Y'
        
        # Number '1' or 'ONE'
        elif extended_count == 1 and index_extended:
            return '1'
        
        # Number '2' or 'TWO'
        elif extended_count == 2 and index_extended and middle_extended:
            return '2'
        
        # Number '3' or 'THREE'
        elif extended_count == 3 and index_extended and middle_extended and ring_extended:
            return '3'
        
        # Number '4' or 'FOUR'
        elif extended_count == 4:
            return '4'
        
        # Number '5' or 'FIVE' - All fingers extended
        elif extended_count == 5:
            return '5'
        
        return None
    
    def process_frame(self, frame):
        """Process a single frame and detect gestures"""
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.hands.process(rgb_frame)
        
        # Draw hand landmarks and recognize gestures
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw landmarks
                self.mp_draw.draw_landmarks(
                    frame, 
                    hand_landmarks, 
                    self.mp_hands.HAND_CONNECTIONS
                )
                
                # Extract landmark coordinates
                landmarks = []
                for lm in hand_landmarks.landmark:
                    landmarks.append([lm.x, lm.y, lm.z])
                
                # Recognize gesture
                gesture = self.recognize_gesture(landmarks)
                
                if gesture:
                    current_time = time.time()
                    
                    # Check if same gesture is held
                    if gesture == self.last_gesture:
                        if self.gesture_start_time is None:
                            self.gesture_start_time = current_time
                        elif current_time - self.gesture_start_time > self.gesture_hold_threshold:
                            # Gesture held long enough, add to text
                            if not self.recognized_text.endswith(gesture):
                                self.recognized_text += gesture
                            self.gesture_start_time = None
                    else:
                        # New gesture detected
                        self.last_gesture = gesture
                        self.gesture_start_time = current_time
                    
                    # Display current gesture
                    cv2.putText(frame, f"Gesture: {gesture}", (10, 100), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Display recognized text
        cv2.putText(frame, f"Text: {self.recognized_text}", (10, 50), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Instructions
        cv2.putText(frame, "Hold gesture for 1.5s | Press 'c' to clear | 'q' to quit", 
                   (10, frame.shape[0] - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def run(self):
        """Main loop to run the interpreter"""
        cap = cv2.VideoCapture(0)
        
        print("Sign Language Interpreter Started!")
        print("Hold a gesture for 1.5 seconds to recognize it")
        print("Press 'c' to clear text, 'q' to quit")
        
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("Failed to capture frame")
                break
            
            # Process frame
            frame = self.process_frame(frame)
            
            # Display
            cv2.imshow('Sign Language Interpreter', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('c'):
                self.recognized_text = ""
                print("Text cleared")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.hands.close()

if __name__ == "__main__":
    interpreter = SignLanguageInterpreter()
    interpreter.run()