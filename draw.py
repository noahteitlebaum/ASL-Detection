"""
Drawing
----------------------------------
This module provides functions for drawing hand landmarks, text overlays,
and debugging information on video frames during ASL detection.

Functions include landmark visualization, text rendering, and coordinate printing.
"""

# Import libraries for image processing and hand tracking
import cv2
import mediapipe as mp

# Initialize MediaPipe drawing utilities
mp_hands = mp.solutions.hands # type: ignore
mp_draw = mp.solutions.drawing_utils # type: ignore

def draw_hand_landmarks(frame, landmark_set):
    """
    Draw hand landmarks and connections on the frame.
        
    Arguments:
        frame: The video frame to draw on
        landmark_set: MediaPipe hand landmark set containing 21 landmarks
    """

    # Draw the hand landmarks and connections on the current hand on the frame
    mp_draw.draw_landmarks(frame, landmark_set, mp_hands.HAND_CONNECTIONS)

def draw_text(frame, text, position=(10, 70), font=cv2.FONT_HERSHEY_SIMPLEX, 
              font_scale=3, color=(0, 0, 255), thickness=3):
    """
    Draw text on the frame.
        
    Arguments:
        frame: The video frame to draw on
        text: The text string to display
        position: (x, y) coordinates for text position
        font: OpenCV font type
        font_scale: Font size multiplier
        color: BGR color tuple
        thickness: Text thickness in pixels
    """

    # Draw and display the text on the current frame
    cv2.putText(frame, text, position, font, font_scale, color, thickness)
    print(text)

def print_landmark_coordinates(landmark_set, frame_width, frame_height):
    """
    Print all landmark coordinates to console (for debugging).
            
    Arguments:
        landmark_set: MediaPipe hand landmark set containing 21 landmarks
        frame_width: Frame width for coordinate conversion
        frame_height: Frame height for coordinate conversion
    """

    # Iterate through all hand landmarks
    for id, landmark in enumerate(landmark_set.landmark):
        # Print the index and respective coordinates of each landmark
        print(f"{id}: ({landmark.x * frame_width}, {landmark.y * frame_height}, {landmark.z})")
    
    # Print a divider to separate outputs visually
    print("------------------------------")
