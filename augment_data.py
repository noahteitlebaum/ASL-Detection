"""
Data Augmentation for ASL Hand Landmarks
-----------------------------------------
This module provides functions to augment hand landmark data with various
transformations to improve model robustness and accuracy.

Augmentations include:
- Rotation around x, y, z axes
- Random noise addition
- Scaling variations
- Translation variations
"""

# Import libraries for numerical operations / array manipulation and normalizing /flattening landmarks
import numpy as np
from normalize_flatten_landmarks import normalize_and_flatten

def rotate_landmarks_x(landmarks, angle_degrees):
    """
    Rotate landmarks around the x-axis for simulating tilting the hand up and down.
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape 21 x 3
    """

    # Convert angle to radians (for numpy functions) and compute cosine/sine to control how coordinates change during rotation
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Create a rotation matrix around y-axis given x' = x, y' = y*cos(a) - z*sin(a), z' = y*sin(a) + z*cos(a)
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])
    
    # Use matrix multiplication to multiply (21×3) landmarks by the (3×3) rotation matrix to get rotated (21×3) points
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T) # Use the transpose so the row landmark coordinates [x,y,z] are rotated correctly

    # Return the rotated landmarks around the x-axis
    return rotated_landmarks

def rotate_landmarks_y(landmarks, angle_degrees):
    """
    Rotate landmarks around the y-axis for simulating turning the hand left and right.
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape 21 x 3
    """

    # Convert angle to radians (for numpy functions) and compute cosine/sine to control how coordinates change during rotation
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Create a rotation matrix around y-axis given x' = x*cos(a) + z*sin(a), y' = y, z' = -x*sin(a) + z*cos(a)
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Use matrix multiplication to multiply (21×3) landmarks by the (3×3) rotation matrix to get rotated (21×3) points
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T) # Use the transpose so the row landmark coordinates [x,y,z] are rotated correctly

    # Return the rotated landmarks around the y-axis
    return rotated_landmarks

def rotate_landmarks_z(landmarks, angle_degrees):
    """
    Rotate landmarks around the z-axis for simulating twisting the hand clockwise or counterclockwise.    
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape 21 x 3
    """

    # Convert angle to radians (for numpy functions) and compute cosine/sine to control how coordinates change during rotation
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Create a rotation matrix around z-axis given x' = x*cos(a) - y*sin(a), y' = x*sin(a) + y*cos(a), z' = z
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Use matrix multiplication to multiply (21×3) landmarks by the (3×3) rotation matrix to get rotated (21×3) points
    rotated_landmarks = np.dot(landmarks, rotation_matrix.T) # Use the transpose so the row landmark coordinates [x,y,z] are rotated correctly

    # Return the rotated landmarks around the z-axis
    return rotated_landmarks

def apply_landmarks_noise(landmarks, noise_level=0.01):
    """
    Apply random noise to landmarks for making the overall data more realistic.
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        noise_level: standard deviation controlling how much noise varies around a mean of 0    
    
    Returns:
        noisy_landmarks: numpy array of shape 21 x 3
    """

    # Generate random noise where 68% of values are within [-SD, +SD] around the mean 0
    noise = np.random.normal(0, noise_level, landmarks.shape)
    noisy_landmarks = landmarks + noise # Add the random noise to the exisiting landmarks

    # Return the noisy landmarks
    return noisy_landmarks

def scale_landmarks(landmarks, scale_factor):
    """
    Scale landmarks by a factor for simulating variations in hand size.    
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        scale_factor: scaling factor (e.g., 0.9 for 90%)
    
    Returns:
        scaled_landmarks: numpy array of shape 21 x 3
    """

    # Scale all the landmark coordinates by the scale factor
    scaled_landmarks = landmarks * scale_factor

    # Return the scaled landmarks
    return scaled_landmarks

def translate_landmarks(landmarks, translation):
    """
    Translate landmarks by a vector for simulating variations in hand position.
    
    Arguments:
        landmarks: numpy array of shape 21 x 3 with [x, y, z] coordinates
        translation: numpy array of 3 values where [dx, dy, dz]
    
    Returns:
        translated_landmarks: numpy array of shape 21 x 3
    """

    # Translate all the landmark coordinates by the translation vector
    translated_landmarks = landmarks + translation

    # Return the translated landmarks
    return translated_landmarks

def augment_sample(landmarks, augmentation_params):
    """
    Apply multiple augmentations (rotate, noise, scale, translate) to a single sample (hand).
    Each hand has 21 landmark points
    
    Arguments:
        landmarks: list or numpy array of 21 landmarks with [x, y, z] coordinates
        augmentation_params: dictionary with augmentation parameters
            - 'rotate_x': rotation angle around x-axis in degrees (tilts the hand up/down)
            - 'rotate_y': rotation angle around y-axis in degrees (turns the hand left/right)
            - 'rotate_z': rotation angle around z-axis in degrees (twists the hand)
            - 'noise': standard deviation controlling the magnitude of random variation added to each coordinate
            - 'scale': multiplicative factor controlling how much the hand size is increased / decreased
            - 'translate': translation vector [dx, dy, dz] specifying how much to shift the hand along each axis
    
    Returns:
        augmented_landmarks: numpy array of shape 21 x 3
    """

    # Convert to numpy array to prepare for augmentation operations
    augmented_landmarks = np.array(landmarks)
    
    # Rotate landmarks around the x-axis
    if 'rotate_x' in augmentation_params:
        # Extract the rotation angle in degrees from the dictionary and apply the rotation function
        angle_degrees = augmentation_params['rotate_x']
        augmented_landmarks = rotate_landmarks_x(augmented_landmarks, angle_degrees)
    
    # Rotate landmarks around the y-axis
    if 'rotate_y' in augmentation_params:
        # Extract the rotation angle in degrees from the dictionary and apply the rotation function
        angle_degrees = augmentation_params['rotate_y']
        augmented_landmarks = rotate_landmarks_y(augmented_landmarks, angle_degrees)
    
    # Rotate landmarks around the z-axis
    if 'rotate_z' in augmentation_params:
        # Extract the rotation angle in degrees from the dictionary and apply the rotation function
        angle_degrees = augmentation_params['rotate_z']       
        augmented_landmarks = rotate_landmarks_z(augmented_landmarks, angle_degrees)
    
    # Add random noise to landmarks
    if 'noise' in augmentation_params:
        # Extract the noise level from the dictionary and apply the noise function
        noise_level = augmentation_params['noise']
        augmented_landmarks = apply_landmarks_noise(augmented_landmarks, noise_level)
    
    # Scale landmarks
    if 'scale' in augmentation_params:
        # Extract the scale factor from the dictionary and apply the scaling function
        scale_factor = augmentation_params['scale']
        augmented_landmarks = scale_landmarks(augmented_landmarks, scale_factor)
    
    # Translate landmarks
    if 'translate' in augmentation_params:
        # Extract the translation vector from the dictionary and apply the translation function
        translation = augmentation_params['translate']
        augmented_landmarks = translate_landmarks(augmented_landmarks, translation)
    
    # Return the augmented landmarks
    return augmented_landmarks

def generate_augmentation_configurations():
    """
    Generate a list of configurations for data augmentation.
 
    Returns:
        configurations: list of dictionaries, each representing a different combination of augmentation parameters
    """

    # Initialize the list of configurations (5) for data augmentation
    configurations = []

    # 1. Original sample (no augmentation)
    configurations.append({})

    # 2. Mild rotation (turn + twist)
    configurations.append({
        'rotate_y': 15,
        'rotate_z': 10
    })

    # 3. Opposite rotation + slight scale change
    configurations.append({
        'rotate_y': -15,
        'rotate_x': -10,
        'scale': 0.95
    })

    # 4. Tilt + small translation shift
    configurations.append({
        'rotate_x': 10,
        'translate': [0.05, 0.0, 0.0]
    })

    # 5. Full realistic combination (best augmentation combination)
    configurations.append({
        'rotate_x': 10,
        'rotate_y': 10,
        'noise': 0.005,
        'scale': 1.05,
    }) 

    # Return the list of augmentation configurations
    return configurations

def augment_dataset(x, y):
    """
    Augment the entire .json dataset by creating multiple augmented versions of each sample (hand).
    
    Arguments:
        x: numpy array of shape (n x 63 matrix) - flattened (one list) landmark coordinates
        y: numpy array of shape (n x 1 martrix) - letter labels
    
    Returns:
        x_augmented: numpy array with original + augmented samples
        y_augmented: numpy array with corresponding labels for original + augmented samples
    """

    # Get the list of configurations (5) for data augmentation
    configurations = generate_augmentation_configurations()
    
    # Initialize ists to store the augmented data
    x_augmented = []
    y_augmented = []
    
    # Iterate over each sample (63 flattened landmarks) in the dataset
    for i in range(len(x)):
        # Reshape the flattened landmarks back to 21 x 3 and store the corresponding letter label
        landmarks = x[i].reshape(21, 3)
        label = y[i]
        
        # Apply each augmentation configuration per sample (hand)
        for config in configurations:
            # Turn the current sample into 5 augmented configurations
            augmented_landmarks = augment_sample(landmarks, config)
            
            # Normalize augmented landmarks (remove position/size differences) and re-flatten into 63 features for model input
            flattened_landmarks = normalize_and_flatten(augmented_landmarks)
            
            # Add each of the augmented samples (5) and their corresponding labels
            x_augmented.append(flattened_landmarks)
            y_augmented.append(label)
    
    # Convert list into a NumPy array so the model can process it properly
    x_augmented = np.array(x_augmented)
    y_augmented = np.array(y_augmented)
    
    # Display the size of the dataset before and after augmentation
    print(f"Original dataset size: {len(x)}")
    print(f"Augmented dataset size: {len(x_augmented)}")
    print(f"Augmentation factor: {len(x_augmented) / len(x):.1f}x") # Should be 5x larger than original dataset
    
    # Return the flattened (one list) augmented landmark coordinates and the corresponding letter labels
    return x_augmented, y_augmented