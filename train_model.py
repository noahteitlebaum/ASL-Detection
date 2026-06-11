"""
ASL Letter Recognition - MLP Model Training
-------------------------------------------
This script trains a Multi-Layer Perceptron (MLP) neural network to recognize
ASL alphabet letters from hand landmark coordinates.

Input: 21 landmarks × 3 coordinates (x, y, z) = 63 features per sample
Output: Letter classification labels (A-Z)
"""

# Import libraries for data handling, model training, evaluation, and saving/loading
import json
import numpy as np
from normalize_flatten_landmarks import normalize_and_flatten
from augment_data import augment_dataset
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Store the file paths for the dataset, model, and label encoder
DATA_FILE = "data/asl_data.json" # Path to the dataset file containing the ASL hand landmark data
MODEL_FILE = "models/asl_mlp_model.pkl" # Path to the file where the trained AI model will be saved
LABEL_ENCODER_FILE = "models/label_encoder.pkl" # Path to the file where the label encoder will be saved

# Seed for randomness to ensure consistent and reproducible results across runs
RANDOM_STATE = 0

def load_and_preprocess_data(data_file):
    """
    Load ASL data from JSON and preprocess it.
    
    Returns: (Let n be the number of samples)
        x: numpy array of shape (n x 63 matrix) - flattened (one list) landmark coordinates
        y: numpy array of shape (n x 1 martrix) - letter labels
    """

    # Load data from JSON file
    with open(data_file, 'r') as f:
        # Data is a list of samples, where each sample (Letter to coordinates) is a dictionary
        data = json.load(f)
    
    # Create two empty lists where x will be the flattened (one list) landmark coordinates and y will be the letter labels
    x = []
    y = []
    
    # Iterate through each sample (letter to coordinates) in the data
    for sample in data:
        # Store the current letter label and its respective 63 landmark coordinates
        letter = sample['letter']
        landmarks = sample['landmarks']
        
        # Normalize landmarks (remove position/size differences) and flatten into 63 features for model input
        flattened_landmarks = normalize_and_flatten(landmarks)
        
        # Add another training example: the flattened landmarks (features) and its letter label
        x.append(flattened_landmarks)
        y.append(letter)
    
    # Convert list into a NumPy array so the model can process it properly
    x = np.array(x)
    y = np.array(y)
    
    # Return the flattened (one list) landmark coordinates and the corresponding letter labels
    return x, y

def encode_labels(y):
    """
    Encode letter labels (A-Z) as numeric classes (0-25).
    
    Returns:
        y_encoded: numpy array of encoded letter labels for model training - A-Z encoded as 0-25
        label_encoder: fitted LabelEncoder object for future decoding
    """

    # Initialize the LabelEncoder object and encode the letter labels to numbers (0-25)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y) # label_encoder is updated here (learns the mapping of letters to numbers)

    # Return the encoded letter labels for model training and the encoder to convert predictions back to letters     
    return y_encoded, label_encoder

def build_mlp_model():
    """
    Build and return an MLP classifier.
    
    Architecture:
    - Input layer: 63 features (21 landmarks × 3 coords)
    - Hidden layers: (128, 64, 32) neurons with ReLU activation
    - Output layer: 26 classes (A-Z)
    """

    # Initialize the MLPClassifier model with the specified parameters
    model = MLPClassifier(
        hidden_layer_sizes = (128, 64, 32),  # Structure of the network
        activation = 'relu',                 # How neurons behave
        solver = 'adam',                     # Learning algorithm
        alpha = 0.0001,                      # Prevents overfitting
        batch_size = 'auto',                 # Training batch size
        learning_rate = 'adaptive',          # Adjusts learning speed
        learning_rate_init = 0.001,          # Starting learning speed
        max_iter = 500,                      # Max training steps
        random_state = RANDOM_STATE,         # Ensures the split is reproducible (same result each run)
        verbose = True,                      # Print training progress
        early_stopping = True,               # Stop if no improvement
        validation_fraction = 0.1,           # The % of data used for validating (rest is training)
        n_iter_no_change = 20                # Stop after no improvement
    )

    # Return the configured MLP model (ready for training)
    return model

def train_model(model, x_train, y_train):
    """
    Train the MLP model.
    """

    # Train the configured MLP model (tell the model what our input and output are)
    model.fit(x_train, y_train)
    print("Training complete!")

    # Return the trained model (now a learned AI capable of making predictions)
    return model

def evaluate_model(model, x_test, y_test, label_encoder):
    """
    Evaluate the trained model on test data.
    This function tests how well the model performs.
    Calculates the accuracy of the mode, a classification report, and a confusion matrix.
    """

    # Use the trained MLP model to predict labels for the test data
    y_predicted = model.predict(x_test)
    
    # Calculate the accuracy of the model
    accuracy = accuracy_score(y_test, y_predicted)
    print(f"\nTest Accuracy: {accuracy * 100:.2f}%")
    
    # Calculate the classification report of the model (precision, recall, f1-score, support)
    print("\nClassification Report:")
    print(classification_report(y_test, y_predicted, target_names = label_encoder.classes_, zero_division = "warn"))
    
    # Calculate the confusion matrix of the model
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_predicted))

def save_model(model, label_encoder):
    """
    Save the trained MLP model and label_encoder.
    label_encoder is a fitted LabelEncoder object for future decoding
    """

    # Add a try block in case there's an error saving the model or encoder
    try:
        # Save the trained model so we don't have to retrain it every time I run the program
        with open(MODEL_FILE, 'wb') as f:
            pickle.dump(model, f)
        
        # Save the encoder so we can convert predictions (numbers) back into letters later
        with open(LABEL_ENCODER_FILE, 'wb') as f:
            pickle.dump(label_encoder, f)
    except Exception as e:
        # Display there is an error saving the model or encoder to their respective files
        print(f"Error saving model or encoder: {e}")

def retrieve_model_and_label_encoder():
    """
    Retrieve the saved MLP model and label_encoder.
    label_encoder is a fitted LabelEncoder object for future decoding
    """

    # Initialize the model and label_encoder
    model, label_encoder = None, None

    # Add a try block in case there's an error loading the model or encoder
    try:
        # Load the trained model from the file
        with open(MODEL_FILE, "rb") as f:
            model = pickle.load(f)

        # Load the encoder from the file
        with open(LABEL_ENCODER_FILE, "rb") as f:
            label_encoder = pickle.load(f)
    except Exception as e:
        # Display there is an error loading the model or encoder from their respective files
        print(f"Error loading model or encoder: {e}")

    # Return the model and label_encoder
    return model, label_encoder

def predict_letter(model, label_encoder, landmarks):
    """
    Predict the ASL letter from hand landmarks.
    
    Arguments:
        model: trained MLPClassifier
        label_encoder: fitted LabelEncoder
        landmarks: list of 21 landmarks representing 1 letter, each with [x, y, z] coordinates
    
    Returns:
        predicted_letter: string (A-Z)
        confidence: float (0-1)
    """
    
    # Normalize landmarks (remove position/size differences) and flatten into 63 features for model input
    flattened_landmarks = normalize_and_flatten(landmarks)
    
    # Convert to NumPy array and reshape for single sample prediction
    x = np.array([flattened_landmarks])
    
    # Predict the letter from 63 input features and convert the numeric prediction to a letter
    y_predicted = model.predict(x)
    predicted_letter = label_encoder.inverse_transform(y_predicted)[0]

    # Get probabilities for each letter and select the highest as the confidence the model was correct
    y_probabilities = model.predict_proba(x)    
    confidence = np.max(y_probabilities)
    
    # Return the predicted letter and its confidence score
    return predicted_letter, confidence

def main():
    """
    The main function of the program.
    # Main pipeline: load / preproccess data, train model, evaluate, and test predictions.
    """

    # Display that the model training is starting
    print("=" * 60)
    print("ASL LETTER RECOGNITION - MLP MODEL TRAINING")
    print("=" * 60)
    
    # Load the dataset (json) and convert it into features (x) and letter labels (y)
    x, y = load_and_preprocess_data(DATA_FILE)
    
    # Augment the dataset to create more training samples with variations
    x_augmented, y_augmented = augment_dataset(x, y) # Create 5 augmented samples
    
    # Encode the letter labels (y) by converting letters to numbers (y_encoded) and learn the mapping for later decoding (label_encoder)
    y_encoded, label_encoder = encode_labels(y_augmented)
    
    # Split data into train sets (80% - x_train, y_train) and test sets (20% - x_test, y_test)
    x_train, x_test, y_train, y_test = train_test_split(
        x_augmented,                    # Augmented feature data (flattened landmark coordinates)
        y_encoded,                      # Encoded letter labels (numeric class for each sample)
        test_size = 0.2,                # The % of data used for testing (rest is training)
        random_state = RANDOM_STATE,    # Ensures the split is reproducible (same result each run)
        shuffle = True,                 # Shuffle the data before splitting to ensure randomness (no bias)
        stratify = y_encoded            # Use stratified sampling to ensure the same labels distribution in both sets (each letter is a stratum)
    )

    # Display number of samples in training and test sets
    print(f"Training samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    
    # Build the MLP model
    model = build_mlp_model()
    
    # Train the MLP model with the training sets
    model = train_model(model, x_train, y_train)
    
    # Evaluate the MLP model with the testing set
    evaluate_model(model, x_test, y_test, label_encoder)
    
    # Save the MLP model and label_encoder for reusability to their respective files
    save_model(model, label_encoder)
    
    # Display that the model training is complete
    print("\n" + "=" * 60)
    print(f"TRAINING COMPLETE!!!!!")
    print("=" * 60)
    
    # Display that an example prediction is being made from the testing set
    print("\nExample prediction on the first test sample:")

    # Make a prediction on the first test sample
    predicted_letter, confidence = predict_letter(model, label_encoder, x_test[0].reshape(21, 3)) # 1) Get first test sample features, 2) Reshape from flattend list into 21x3 landmark format
    actual_letter = label_encoder.inverse_transform([y_test[0]])[0] # 1) Get first test sample label, 2) Wrap in list for inverse_transform, 3) Extract decoded letter (returns a list)

    # Display the predicted and actual letters with their respective confidence
    print(f"Predicted: {predicted_letter} (confidence: {confidence:.2%})")
    print(f"Actual: {actual_letter}")

# Run main() only if this file is executed directly (prevents it from running when imported)
if __name__ == "__main__":
    # Call the main function
    main()