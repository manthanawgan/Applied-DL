import cv2
import google.generativeai as genai
import os
from PIL import Image
import textwrap
from dotenv import load_dotenv
import numpy as np
import mediapipe as mp


load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")
INPUT_VIDEO_FILE = "swim.mp4"
OUTPUT_VIDEO_FILE = "swim_feedback.mp4"
PROMPT = "You are a world-class swimming coach. Analyze this single frame of a swimmer. Provide specific, actionable feedback on their technique, focusing on stroke quality, body posture, breathing, and hand entry. Keep the feedback concise and limited to 2-3 short sentences."

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

SWIM_CONNECTIONS = [
    # Arms
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW),
    (mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST),
    (mp_pose.PoseLandmark.RIGHT_SHOULDER, mp_pose.PoseLandmark.RIGHT_ELBOW),
    (mp_pose.PoseLandmark.RIGHT_ELBOW, mp_pose.PoseLandmark.RIGHT_WRIST),
    (mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.RIGHT_SHOULDER),
    # Legs
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE),
    (mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE),
    (mp_pose.PoseLandmark.RIGHT_HIP, mp_pose.PoseLandmark.RIGHT_KNEE),
    (mp_pose.PoseLandmark.RIGHT_KNEE, mp_pose.PoseLandmark.RIGHT_ANKLE),
    (mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.RIGHT_HIP),
]

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

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(image_rgb)

                if results.pose_landmarks:
                    mp_drawing.draw_landmarks(
                        frame, 
                        results.pose_landmarks, 
                        SWIM_CONNECTIONS, # Using custom connections
                        landmark_drawing_spec=mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                        connection_drawing_spec=mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=1)
                    )

                if frame_count % int(fps) == 0:
                    print(f"Getting feedback for frame {frame_count}...")
                    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    feedback_text = get_gemini_feedback(pil_img)
                    print(f"  Feedback: {feedback_text}")

                # Overlay Feedback Text (Bottom Center)
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0 # Increased font size
                font_color = (255, 255, 255) # White
                line_type = 2
                
                wrapped_text = wrap_text(feedback_text, width=80) # Wider text block

                #Calculate the total height of the text block
                text_block_height = 0
                line_heights = []
                for line in wrapped_text:
                    (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, line_type)
                    line_heights.append(text_height)
                    text_block_height += text_height + 10 # 10px padding between lines

                # Add a semi-transparent background at the bottom
                rect_y_start = frame_height - text_block_height - 20 # 20px padding at bottom
                overlay = frame.copy()
                cv2.rectangle(overlay, (0, rect_y_start), (frame_width, frame_height), (0,0,0), -1)
                alpha = 0.6
                frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

                # Draw the wrapped text line by line
                text_y = rect_y_start + 15 # Start drawing text with some padding
                for i, line in enumerate(wrapped_text):
                    (text_width, _), _ = cv2.getTextSize(line, font, font_scale, line_type)
                    text_x = (frame_width - text_width) // 2 # Center horizontally
                    cv2.putText(frame, line, (text_x, text_y + line_heights[i]), font, font_scale, font_color, line_type)
                    text_y += line_heights[i] + 10

                # --- 5. Write the Frame ---
                out.write(frame)
                frame_count += 1

        finally:
            cap.release()
            out.release()
            cv2.destroyAllWindows()
            print(f"\nProcessing complete. Output video saved as '{OUTPUT_VIDEO_FILE}'")

if __name__ == "__main__":
    main()