"""
ASL Alphabet Data Collection Tool
---------------------------------------------------------
This script processes imported images for ASL alphabet recognition training data.

The script will save hand landmark data to 'data/asl_data.json'
"""

# Import libraries for image processing, hand tracking, and data storage
import cv2
import mediapipe as mp
import json
import glob

# Access MediaPipe's built-in modules and create an instance of the Hands class
mp_hands = mp.solutions.hands # type: ignore
hand = mp_hands.Hands(static_image_mode = True, max_num_hands = 1) # Process each image independently (static) and detect at most one hand per image

# Store the file paths
DATA_FILE = "data\\asl_data.json"
IMAGE_FILE = "images"

# Store the list of ASL letters
LETTERS = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

# Initialize list to store all collected data
all_data = []

# Display instructions
print("=" * 60)
print("ASL ALPHABET DATA COLLECTION TOOL (IMAGE IMPORT)")
print("=" * 60)
print("\nThis tool processes images to extract hand landmark data.")
print("=" * 60)

# Iterate through each letter
for letter in LETTERS:
    # Initialize an empty list to store the image files paths for the current letter
    image_files = []
    # Iterate through the valid image file extensions
    for extension in ("*.jpg", "*.jpeg"):
        # Store all .jpeg & .jpg image file paths for the current letter in a list
        image_files.extend(glob.glob(IMAGE_FILE + "\\" + letter + "\\" + extension)) # Use glob for wildcarding

    # Iterate through each image file path
    for img_file in image_files:
        # Read the image
        image = cv2.imread(img_file)

        # Determine if the image failed to load
        if image is None:
            # Display that the image failed to load and skip to the next image
            print(f"Could not read image: {img_file}")
            continue

        # Convert the image from BGR (OpenCV) to RGB (MediaPipe format)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image to detect hand landmarks
        results = hand.process(image_rgb)
        
        # Check if hand landmarks were detected
        if results.multi_hand_landmarks:
            # Iterate through each detected hand's landmark set (which contains 21 landmarks)
            for landmark_set in results.multi_hand_landmarks:
                # Initialize an empty list to store the landmarks data for the current sample
                landmarks_data = []

                # Iterate through each landmark in the current hand's landmark set
                for landmark in landmark_set.landmark:
                    # Append the x, y, and z coordinates of the current landmark to the sample data
                    landmarks_data.append((landmark.x, landmark.y, landmark.z))
            
                # Store the current sample
                sample = {"letter": letter, "landmarks": landmarks_data}
                all_data.append(sample)

                # Display that the image was processed successfully
                print(f"Processed: {img_file}")
        else:
            # Display that no hand was detected in the image
            print(f"No hand detected in: {img_file}")

# Display the total number of samples collected per letter
print("=" * 60)
print("\nSamples per letter:")
print("=" * 60)

# Initialize a dictionary to store the count of samples per letter
letter_counts = {}

# Iterate through each sample
for sample in all_data:
    # Store the letter of the current sample
    letter = sample["letter"]

    # Determine if the letter is already in the dictionary
    if letter not in letter_counts:
        # Initialize the count for the letter
        letter_counts[letter] = 0
    else:
        # Increment the count for the letter
        letter_counts[letter] += 1

# Iterate through each letter and its count
for letter in letter_counts.keys():
    # Display the letter and its corresponding count
    print(f"{letter}: {letter_counts[letter]} samples")

# Save all collected data by appending to existing file (if it exists)
try:
    # Read the existing data from the json file
    with open(DATA_FILE, "r") as f:
        existing_data = json.load(f)

except (FileNotFoundError, json.JSONDecodeError):
    # Start fresh if file doesn't exist or is empty
    existing_data = []

# Combine the old data with new data
existing_data.extend(all_data)

# Write the new file to the json file
with open(DATA_FILE, "w") as f:
    json.dump(existing_data, f, indent=4)
        
# Display that the data is saved to the json file and end the program
print(f"\nData saved to {DATA_FILE}")
