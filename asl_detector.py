"""
ASL Alphabet Real Time Detection
---------------------------------
This script performs real-time ASL alphabet recognition using a webcam feed.
It detects hand landmarks and predicts the corresponding letter using a trained MLP model.

The script displays the predicted letter and confidence score on the video feed.
"""

# Import the required libraries image processing, hand tracking, drawing, and model prediction
import cv2
import mediapipe as mp
from draw import draw_hand_landmarks, draw_text, print_landmark_coordinates
from train_model import retrieve_model_and_label_encoder, predict_letter

def main():
    """
    The main function of the program.
    """

    # Initialize the webcam/video connection (Connect to the default webcam: 0)
    capture = cv2.VideoCapture(0)

    # Access MediaPipe's built-in modules and create an instance of the Hands class
    mp_hands = mp.solutions.hands # type: ignore
    hand = mp_hands.Hands(static_image_mode = False, max_num_hands = 1)

    # Retrieve the trained model and label encoder from the saved files
    model, label_encoder = retrieve_model_and_label_encoder()

    # Start an infinite loop to continuously read frames from the webcam
    while True:
        # Read one frame from the webcam
        success, frame = capture.read()

        # Skip iteration if the webcam frame could not be read
        if not success or frame is None:
            print("Failed to read frame from webcam.")
            continue

        # Store the height and width of the current frame
        frame_height = frame.shape[0]
        frame_width = frame.shape[1]

        # Convert the captured frame from BGR (OpenCV) to RGB (MediaPipe) and process it
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hand.process(frame_rgb)

        # Check if MediaPipe detected any hand landmarks in the current frame
        if results.multi_hand_landmarks:
            # Iterate through each detected hand's landmark set (which contains 21 landmarks)
            for landmark_set in results.multi_hand_landmarks:
                # Draw the hand landmarks and connections
                draw_hand_landmarks(frame, landmark_set)

                # Print landmark coordinates for debugging
                print_landmark_coordinates(landmark_set, frame_width, frame_height)

                # Convert MediaPipe landmarks to list of (x, y, z) tuples
                landmarks_list = [(lm.x, lm.y, lm.z) for lm in landmark_set.landmark]

                # Detect the current hand sign with its respective confidence
                predicted_letter, confidence = predict_letter(model, label_encoder, landmarks_list)

                # Determine if the confidence of the letter is greater than 95%
                if confidence > 0.95:
                    # Draw the current hand sign with its respective confidence on the frame
                    draw_text(frame, predicted_letter, position=(10, 70), font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=3, color=(0, 0, 255), thickness=3)

        # Display the current frame in a window titled "Webcam"
        cv2.imshow("Webcam", frame)

        # Exit when q is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    capture.release()
    cv2.destroyAllWindows()

# Run main() only if this file is executed directly (prevents it from running when imported)
if __name__ == "__main__":
    # Call the main function
    main()