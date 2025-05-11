import cv2
import numpy as np
import mediapipe as mp
import colorsys

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Initialize webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

# Get video properties for the output window
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_pythagorean = (frame_height**2 + frame_width**2)**.5

# Define landmark display properties
# LANDMARK_COLOR = (0, 255, 0)
LANDMARK_SIZE = 3  # Size of landmark points

def rainbow_color(idx, total):
    h = idx / total
    r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
    return (int(b * 255), int(g * 255), int(r * 255))

color_function = rainbow_color

print("MediaPipe Face Mesh running. Press 'q' to quit.")

window_name = 'MediaPipe_Face_Mesh'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
cv2.setWindowProperty(window_name, cv2.WND_PROP_AUTOSIZE, cv2.WINDOW_AUTOSIZE)

while True:
    # Read frame from webcam
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture frame.")
        break
    
    # Flip the frame horizontally for a more natural mirror view
    frame = cv2.flip(frame, 1)
    
    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process the frame with MediaPipe Face Mesh
    results = face_mesh.process(rgb_frame)
    
    # Create a black image for display
    display_image = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
    
    # Draw facial landmarks on the black image
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            total_landmarks = len(face_landmarks.landmark)
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Convert normalized landmark position to pixel coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Get color for this landmark
                landmark_color = color_function((x**2 + y**2)**.5, frame_pythagorean)
                
                # Draw the landmark point
                cv2.circle(display_image, (x, y), LANDMARK_SIZE, landmark_color, -1)

    cv2.imshow(window_name, display_image)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
