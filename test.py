import cv2
import numpy as np
import mediapipe as mp

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

# Define landmark display properties
LANDMARK_COLOR = (0, 255, 0)  # Green color for landmarks
LANDMARK_SIZE = 2  # Size of landmark points

print("MediaPipe Face Mesh running. Press 'q' to quit.")

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
            for idx, landmark in enumerate(face_landmarks.landmark):
                # Convert normalized landmark position to pixel coordinates
                x = int(landmark.x * frame_width)
                y = int(landmark.y * frame_height)
                
                # Draw the landmark point
                cv2.circle(display_image, (x, y), LANDMARK_SIZE, LANDMARK_COLOR, -1)
    
    # Display the result
    cv2.imshow('MediaPipe Face Mesh for OBS', display_image)
    
    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
