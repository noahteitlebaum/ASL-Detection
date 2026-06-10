"""
Landmark Normalization
---------------------------------
Functions to normalize hand landmarks for consistent ASL recognition
across different hand positions, sizes, and orientations.
"""

# Import the required library for numerical operations / array manipulation
import numpy as np

def normalize_landmarks(landmarks):
    """
    Normalize hand landmarks to be invariant to translation, scale, and rotation.
    
    This function:
    1. Translates landmarks so wrist (landmark 0) is at origin
    2. Scales landmarks based on hand size (max distance from wrist)
    3. Returns normalized coordinates
    
    Arguments:
        landmarks: list of 21 landmarks, each with [x, y, z] coordinates
    
    Returns:
        normalized_landmarks: numpy array of shape (21, 3) with normalized coordinates
    """
    
    # Convert to numpy array for easier manipulation
    landmarks_array = np.array(landmarks)
    
    # Step 1: Translate so wrist (landmark 0) is at origin
    wrist = landmarks_array[0]
    translated = landmarks_array - wrist # Subtract wrist from all landmarks (Wrist is now at [0, 0, 0])
    
    # Step 2: Calculate hand size (max distance from wrist to any landmark)
    distances = np.linalg.norm(translated, axis=1)
    max_distance = np.max(distances)
    
    # Determine if all landmark points are at the same location
    if max_distance == 0:
        # Set max_distance to 1.0 so division does not change the values and avoids errors (division by zero)
        max_distance = 1.0
    
    # Step 3: Scale by hand size
    normalized = translated / max_distance
    
    # Return the normalized landmarks as a numpy array
    return normalized

def flatten_landmarks(landmarks):
    """
    Flatten normalized landmarks into a 1D array for model input.
    
    Arguments:
        landmarks: numpy array of shape (21, 3)
    
    Returns:
        flattened: numpy array of shape (63,)
    """
    
    # Flatten the 21 landmarks × 3 coordinates into a single list (63 x 1 matrix)
    flattened_landmarks = []
    for landmark in landmarks:
        flattened_landmarks.extend(landmark)  # [x, y, z]

    # Return the flattened landmarks as a numpy array
    return np.array(flattened_landmarks)

def normalize_and_flatten(landmarks):
    """
    Convenience function to normalize and flatten landmarks in one step.
    
    Arguments:
        landmarks: list of 21 landmarks, each with [x, y, z] coordinates
    
    Returns:
        flattened_normalized: numpy array of shape (63,)
    """

    # Normalize and flatten the given landmarks
    return flatten_landmarks(normalize_landmarks(landmarks))
