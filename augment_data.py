"""
Data Augmentation for ASL Hand Landmarks
-----------------------------------------
This module provides functions to augment hand landmark data with various
transformations to improve model robustness and accuracy.

Augmentations include:
- Rotation around X, Y, Z axes
- Random noise addition
- Scaling variations
- Translation variations
"""

import numpy as np
from normalize_landmarks import normalize_landmarks

def rotate_landmarks_x(landmarks, angle_degrees):
    """
    Rotate landmarks around the X-axis (pitch - nodding motion).
    
    Arguments:
        landmarks: numpy array of shape (21, 3) with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape (21, 3)
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around X-axis
    rotation_matrix = np.array([
        [1, 0, 0],
        [0, cos_a, -sin_a],
        [0, sin_a, cos_a]
    ])
    
    # Apply rotation to each landmark
    rotated = np.dot(landmarks, rotation_matrix.T)
    return rotated

def rotate_landmarks_y(landmarks, angle_degrees):
    """
    Rotate landmarks around the Y-axis (yaw - shaking head motion).
    
    Arguments:
        landmarks: numpy array of shape (21, 3) with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape (21, 3)
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around Y-axis
    rotation_matrix = np.array([
        [cos_a, 0, sin_a],
        [0, 1, 0],
        [-sin_a, 0, cos_a]
    ])
    
    # Apply rotation to each landmark
    rotated = np.dot(landmarks, rotation_matrix.T)
    return rotated

def rotate_landmarks_z(landmarks, angle_degrees):
    """
    Rotate landmarks around the Z-axis (roll - tilting hand side to side).
    
    Arguments:
        landmarks: numpy array of shape (21, 3) with [x, y, z] coordinates
        angle_degrees: rotation angle in degrees
    
    Returns:
        rotated_landmarks: numpy array of shape (21, 3)
    """
    angle_rad = np.radians(angle_degrees)
    cos_a = np.cos(angle_rad)
    sin_a = np.sin(angle_rad)
    
    # Rotation matrix around Z-axis
    rotation_matrix = np.array([
        [cos_a, -sin_a, 0],
        [sin_a, cos_a, 0],
        [0, 0, 1]
    ])
    
    # Apply rotation to each landmark
    rotated = np.dot(landmarks, rotation_matrix.T)
    return rotated

def add_noise(landmarks, noise_level=0.01):
    """
    Add random Gaussian noise to landmarks.
    
    Arguments:
        landmarks: numpy array of shape (21, 3)
        noise_level: standard deviation of Gaussian noise
    
    Returns:
        noisy_landmarks: numpy array of shape (21, 3)
    """
    noise = np.random.normal(0, noise_level, landmarks.shape)
    return landmarks + noise

def scale_landmarks(landmarks, scale_factor):
    """
    Scale landmarks by a factor (simulates hand size variation).
    
    Arguments:
        landmarks: numpy array of shape (21, 3)
        scale_factor: scaling factor (e.g., 0.9 for 90%, 1.1 for 110%)
    
    Returns:
        scaled_landmarks: numpy array of shape (21, 3)
    """
    return landmarks * scale_factor

def translate_landmarks(landmarks, translation):
    """
    Translate landmarks by a vector (simulates hand position variation).
    
    Arguments:
        landmarks: numpy array of shape (21, 3)
        translation: numpy array of shape (3,) with [dx, dy, dz]
    
    Returns:
        translated_landmarks: numpy array of shape (21, 3)
    """
    return landmarks + translation

def augment_sample(landmarks, augmentation_params):
    """
    Apply multiple augmentations to a single sample.
    
    Arguments:
        landmarks: list or numpy array of 21 landmarks with [x, y, z] coordinates
        augmentation_params: dict with augmentation parameters
            - 'rotate_x': rotation angle around X-axis in degrees
            - 'rotate_y': rotation angle around Y-axis in degrees
            - 'rotate_z': rotation angle around Z-axis in degrees
            - 'noise': noise level (standard deviation)
            - 'scale': scaling factor
            - 'translate': translation vector [dx, dy, dz]
    
    Returns:
        augmented_landmarks: numpy array of shape (21, 3)
    """
    # Convert to numpy array if needed
    landmarks_array = np.array(landmarks)
    
    # Apply rotations
    if 'rotate_x' in augmentation_params and augmentation_params['rotate_x'] != 0:
        landmarks_array = rotate_landmarks_x(landmarks_array, augmentation_params['rotate_x'])
    
    if 'rotate_y' in augmentation_params and augmentation_params['rotate_y'] != 0:
        landmarks_array = rotate_landmarks_y(landmarks_array, augmentation_params['rotate_y'])
    
    if 'rotate_z' in augmentation_params and augmentation_params['rotate_z'] != 0:
        landmarks_array = rotate_landmarks_z(landmarks_array, augmentation_params['rotate_z'])
    
    # Apply noise
    if 'noise' in augmentation_params and augmentation_params['noise'] > 0:
        landmarks_array = add_noise(landmarks_array, augmentation_params['noise'])
    
    # Apply scaling
    if 'scale' in augmentation_params and augmentation_params['scale'] != 1.0:
        landmarks_array = scale_landmarks(landmarks_array, augmentation_params['scale'])
    
    # Apply translation
    if 'translate' in augmentation_params:
        landmarks_array = translate_landmarks(landmarks_array, augmentation_params['translate'])
    
    return landmarks_array

def generate_augmentation_configs(num_augmentations=5):
    """
    Generate a list of augmentation configurations for data augmentation.
    
    Arguments:
        num_augmentations: number of augmented versions to create per sample
    
    Returns:
        configs: list of augmentation parameter dictionaries
    """
    configs = []
    
    # Original (no augmentation)
    configs.append({})
    
    # Rotation variations (important for side-to-side hand movement)
    rotation_angles = [-30, -15, 15, 30]  # Degrees
    
    for angle in rotation_angles:
        # Rotate around Y-axis (side to side)
        configs.append({'rotate_y': angle})
        
        # Rotate around Z-axis (tilting)
        configs.append({'rotate_z': angle})
        
        # Rotate around X-axis (up and down)
        configs.append({'rotate_x': angle})
    
    # Combined rotations (more realistic hand movements)
    configs.append({'rotate_y': 15, 'rotate_z': 10})
    configs.append({'rotate_y': -15, 'rotate_z': -10})
    configs.append({'rotate_x': 10, 'rotate_y': 15})
    configs.append({'rotate_x': -10, 'rotate_y': -15})
    
    # Small noise additions
    configs.append({'noise': 0.005})
    configs.append({'noise': 0.01})
    
    # Scale variations
    configs.append({'scale': 0.95})
    configs.append({'scale': 1.05})
    
    # Combined augmentations
    configs.append({'rotate_y': 20, 'noise': 0.005})
    configs.append({'rotate_z': 15, 'scale': 0.98})
    configs.append({'rotate_x': 10, 'rotate_y': 10, 'noise': 0.005})
    
    # Return the requested number of configurations
    return configs[:num_augmentations] if num_augmentations < len(configs) else configs

def augment_dataset(X, y, num_augmentations=5):
    """
    Augment an entire dataset by creating multiple augmented versions of each sample.
    
    Arguments:
        X: numpy array of shape (n, 63) - flattened landmark coordinates
        y: numpy array of shape (n,) - letter labels
        num_augmentations: number of augmented versions per sample
    
    Returns:
        X_augmented: numpy array with original + augmented samples
        y_augmented: numpy array with corresponding labels
    """
    # Get augmentation configurations
    configs = generate_augmentation_configs(num_augmentations)
    
    # Lists to store augmented data
    X_augmented = []
    y_augmented = []
    
    # Process each sample
    for i in range(len(X)):
        # Reshape flattened landmarks back to (21, 3)
        landmarks = X[i].reshape(21, 3)
        label = y[i]
        
        # Apply each augmentation configuration
        for config in configs:
            # Augment the landmarks
            augmented_landmarks = augment_sample(landmarks, config)
            
            # Normalize and flatten the augmented landmarks
            from normalize_landmarks import normalize_and_flatten
            flattened = normalize_and_flatten(augmented_landmarks)
            
            # Add to augmented dataset
            X_augmented.append(flattened)
            y_augmented.append(label)
    
    # Convert to numpy arrays
    X_augmented = np.array(X_augmented)
    y_augmented = np.array(y_augmented)
    
    print(f"Original dataset size: {len(X)}")
    print(f"Augmented dataset size: {len(X_augmented)}")
    print(f"Augmentation factor: {len(X_augmented) / len(X):.1f}x")
    
    return X_augmented, y_augmented
