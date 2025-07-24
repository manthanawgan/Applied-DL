import cv2
import google.generativeai as genai
import os
from PIL import Image
import textwrap
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp
from collections import deque

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_VIDEO_FILE = "swim.mp4"
OUTPUT_VIDEO_FILE = "swim_feedback.mp4"
PROMPT = "You are a world-class swimming coach. Analyze this single frame of a swimmer. Provide specific, actionable feedback on their technique, focusing on stroke quality, body posture, breathing, and hand entry. Keep the feedback concise and limited to 2-3 short sentences."

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Define which landmarks to track (hands and feet only)
TRACKED_LANDMARKS = [
    mp_pose.PoseLandmark.LEFT_WRIST,
    mp_pose.PoseLandmark.RIGHT_WRIST,
    mp_pose.PoseLandmark.LEFT_ANKLE,
    mp_pose.PoseLandmark.RIGHT_ANKLE,
]

# Trail settings
TRAIL_LENGTH = 15  # Number of previous positions to keep
TRAIL_COLORS = {
    mp_pose.PoseLandmark.LEFT_WRIST: (0, 255, 0),    # Green for left hand
    mp_pose.PoseLandmark.RIGHT_WRIST: (0, 255, 255),  # Yellow for right hand
    mp_pose.PoseLandmark.LEFT_ANKLE: (255, 0, 0),     # Blue for left foot
    mp_pose.PoseLandmark.RIGHT_ANKLE: (255, 0, 255),  # Magenta for right foot
}

class LandmarkTracker:
    def __init__(self, trail_length=TRAIL_LENGTH):
        self.trails = {}
        self.trail_length = trail_length
        
        # Initialize trails for each tracked landmark
        for landmark in TRACKED_LANDMARKS:
            self.trails[landmark] = deque(maxlen=trail_length)
    
    def update(self, landmarks, frame_width, frame_height):
        """Update trails with new landmark positions"""
        if landmarks:
            for landmark_type in TRACKED_LANDMARKS:
                landmark = landmarks.landmark[landmark_type.value]
                if landmark.visibility > 0.5:  # Only track visible landmarks
                    x = int(landmark.x * frame_width)
                    y = int(landmark.y * frame_height)
                    self.trails[landmark_type].append((x, y))
    
    def draw_trails(self, frame):
        """Draw trailing effects for tracked landmarks"""
        # Create a temporary overlay for smooth blending
        overlay = np.zeros_like(frame, dtype=np.uint8)
        
        for landmark_type, trail in self.trails.items():
            if len(trail) < 2:
                continue
                
            color = TRAIL_COLORS[landmark_type]
            
            # Convert trail points to numpy array for smooth curve drawing
            if len(trail) >= 3:
                points = np.array(trail, dtype=np.int32)
                
                # Draw trail with varying thickness and smooth curves
                for i in range(1, len(trail)):
                    # Calculate alpha and thickness based on position in trail
                    alpha = i / len(trail)
                    thickness = max(4, int(12 * alpha))  # Thicker lines, min 4px
                    
                    # Calculate color intensity (fade effect)
                    fade_factor = alpha * 0.8  # Max 80% opacity
                    faded_color = tuple(int(c * fade_factor) for c in color)
                    
                    # Draw smooth line segment
                    cv2.line(overlay, trail[i-1], trail[i], faded_color, thickness, cv2.LINE_AA)
            
            # Draw current position as a larger circle with glow effect
            if trail:
                current_pos = trail[-1]
                
                # Draw glow effect (larger, semi-transparent circle)
                glow_color = tuple(int(c * 0.3) for c in color)
                cv2.circle(overlay, current_pos, 12, glow_color, -1)
                
                # Draw main circle
                cv2.circle(overlay, current_pos, 6, color, -1)
                # Add a subtle white border
                cv2.circle(overlay, current_pos, 6, (255, 255, 255), 1)
        
        # Blend the overlay with the original frame
        mask = overlay.astype(float) / 255.0
        frame_float = frame.astype(float)
        
        # Combine using alpha blending
        result = frame_float * (1 - mask) + overlay.astype(float) * mask
        
        # Convert back to uint8 and update frame
        frame[:] = result.astype(np.uint8)

def get_gemini_feedback(frame_image: Image.Image) -> str:
    """
    Sends a single frame to the Gemini API and returns the feedback.
    """
    try:
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content([PROMPT, frame_image])
        return response.text
    except Exception as e:
        print(f"Error communicating with Gemini API: {e}")
        return "Could not get feedback from API."

def wrap_text(text: str, width: int) -> list[str]:
    """
    Wraps text to fit within a specified pixel width.
    """
    wrapper = textwrap.TextWrapper(width=width)
    return wrapper.wrap(text=text)

def main():
    """
    Main function to process the video, add pose landmarks, and generate feedback.
    """
    if not API_KEY:
        print("Error: GEMINI_API_KEY not found in .env file or environment variables.")
        return

    genai.configure(api_key=API_KEY)

    if not os.path.exists(INPUT_VIDEO_FILE):
        print(f"Error: Input video file not found at '{INPUT_VIDEO_FILE}'")
        return

    cap = cv2.VideoCapture(INPUT_VIDEO_FILE)
    if not cap.isOpened():
        print(f"Error: Could not open video file '{INPUT_VIDEO_FILE}'.")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        print("Warning: Video FPS is 0. Defaulting to 30.")
        fps = 30

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_FILE, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    feedback_text = "Analyzing... Please wait for initial feedback."
    
    # Initialize landmark tracker
    tracker = LandmarkTracker(TRAIL_LENGTH)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                # Update tracker with new landmarks
                if results.pose_landmarks:
                    tracker.update(results.pose_landmarks, frame_width, frame_height)
                
                # Draw trailing effects for hands and legs
                tracker.draw_trails(frame)

                # Get feedback every second
                if frame_count == 0:
                    print(f"Getting feedback for frame {frame_count}...")
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    feedback_text = get_gemini_feedback(pil_img)
                    print(f"  Feedback: {feedback_text}")

                # Overlay Feedback Text (Bottom Center)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.7
                font_color = (255, 255, 255)
                line_type = 2
                
                wrapped_text = wrap_text(feedback_text, width=80)

                # Calculate the total height of the text block
                text_block_height = 0
                line_heights = []
                for line in wrapped_text:
                    (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, line_type)
                    line_heights.append(text_height)
                    text_block_height += text_height + 10

                # Add a semi-transparent background at the bottom
                rect_y_start = frame_height - text_block_height - 20
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, rect_y_start), (frame_width, frame_height), (0, 0, 0), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw the wrapped text line by line
                text_y = rect_y_start + 15
                for i, line in enumerate(wrapped_text):
                    (text_width, _), _ = cv2.getTextSize(line, font, font_scale, line_type)
                    text_x = (frame_width - text_width) // 2
                    cv2.putText(frame, line, (text_x, text_y + line_heights[i]), font, font_scale, font_color, line_type)
                    text_y += line_heights[i] + 10

                # Write the frame
                out.write(frame)
                frame_count += 1

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete. Output video saved as '{OUTPUT_VIDEO_FILE}'")

if __name__ == "__main__":
    main()
