import cv2
import numpy as np
import mediapipe as mp

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Initial position and size of the image
image_x, image_y = 100, 100
initial_image_size = 100
image_scale = 1.0  # Initial scale factor
move_speed = 5  # Speed of image movement

# Load the image with transparency
original_image = cv2.imread('C:\\Users\\Suman Mondal\\Downloads\\lungs 1.jpg', cv2.IMREAD_UNCHANGED)
if original_image is None:
    print("Error: Image file not found.")
    cap.release()
    cv2.destroyAllWindows()
    exit()

# Resize the image initially
resized_image = cv2.resize(original_image, (initial_image_size, initial_image_size))

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from webcam.")
        break

    # Convert the frame to RGB (MediaPipe uses RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    # Check if hand landmarks are detected
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get the hand's index finger tip coordinates
            hand_x = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * frame.shape[1])
            hand_y = int(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * frame.shape[0])

            # Calculate distance from hand to center of frame for zooming
            center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
            distance = np.sqrt((hand_x - center_x) ** 2 + (hand_y - center_y) ** 2)

            # Update the scale based on distance with a stronger zoom effect
            image_scale = max(0.3, min(3.0, 1.0 + (300 - distance) / 150))  # Increased zoom range
            resized_image = cv2.resize(original_image, 
                                       (int(initial_image_size * image_scale), int(initial_image_size * image_scale)))

            # Move the image based on hand position
            if hand_x > center_x:
                image_x = min(image_x + move_speed, frame.shape[1] - resized_image.shape[1])
            elif hand_x < center_x:
                image_x = max(image_x - move_speed, 0)
            if hand_y > center_y:
                image_y = min(image_y + move_speed, frame.shape[0] - resized_image.shape[0])
            elif hand_y < center_y:
                image_y = max(image_y - move_speed, 0)

            # Draw hand landmarks for reference (optional)
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Overlay the resized image with transparency
    if resized_image.shape[2] == 4:  # If image has an alpha channel
        alpha_s = resized_image[:, :, 3] / 255.0  # Normalize alpha values to [0, 1]
        alpha_l = 1.0 - alpha_s

        # Calculate the region where the image will be placed within the frame
        end_x = min(image_x + resized_image.shape[1], frame.shape[1])
        end_y = min(image_y + resized_image.shape[0], frame.shape[0])

        # Calculate overlay dimensions
        overlay_width = end_x - image_x
        overlay_height = end_y - image_y

        # Adjust resized_image and alpha_s to fit the overlay region exactly
        adjusted_resized_image = resized_image[:overlay_height, :overlay_width, :]
        adjusted_alpha_s = alpha_s[:overlay_height, :overlay_width]
        adjusted_alpha_l = 1.0 - adjusted_alpha_s

        # Apply the overlay with broadcasting compatibility
        for c in range(3):  # Loop over the color channels (B, G, R)
            frame[image_y:end_y, image_x:end_x, c] = (
                adjusted_alpha_s * adjusted_resized_image[:, :, c] +
                adjusted_alpha_l * frame[image_y:end_y, image_x:end_x, c]
            )
    else:  # If the image does not have transparency
        frame[image_y:image_y + resized_image.shape[0], image_x:image_x + resized_image.shape[1]] = resized_image

    # Display the frame
    cv2.imshow('Hologram Simulation with Enhanced Zoom and Movement', frame)

    # Exit condition
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()