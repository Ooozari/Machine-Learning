import numpy as np # Import NumPy for numerical operations
import tensorflow as tf # Import TensorFlow for building neural networks
from tensorflow import keras # Import Keras from TensorFlow for a highlevel neural network API
from tensorflow.keras import layers # Import layers module from Keras
from sklearn.datasets import load_iris # Import function to load the Iris dataset
from sklearn.model_selection import train_test_split # Import function to split datasets
from sklearn.preprocessing import StandardScaler # Import scaler for data standardization
# Function to run multi-class classification with Softmax activation in the output layer
def run_multi_class_classification(hidden_activation):
 # Load the Iris dataset
 iris = load_iris()
 X, y = iris.data, iris.target # Separate features (X) and labels (y)
 # Convert labels to categorical format (one-hot encoding)
 y = keras.utils.to_categorical(y, num_classes=3)
 # Standardize the feature data
 scaler = StandardScaler()
 X = scaler.fit_transform(X) # Fit the scaler and transform the data
 # Split the dataset into training and testing sets
 X_train, X_test, y_train, y_test = train_test_split(X, y,
test_size=0.2, random_state=42)
 # Build the neural network model
 model = keras.Sequential([
 layers.Input(shape=(X_train.shape[1],)), # Input layer with shape equal to number of features
 layers.Dense(64, activation=hidden_activation), # Hidden layer with specified activation function
 layers.Dense(3, activation='softmax') # Output layer with Softmax activation for multi-class classification
 ])
 # Compile the model with optimizer, loss function, and metrics
 model.compile(optimizer='adam', loss='categorical_crossentropy',
metrics=['accuracy'])
 # Train the model on the training data
 model.fit(X_train, y_train, epochs=50, verbose=0) # Suppress verbose output during training
 # Evaluate the model on the test data and get accuracy
 accuracy = model.evaluate(X_test, y_test, verbose=0)[1] # Extract accuracy from the evaluation result
 return accuracy # Return the accuracy
# List of hidden layer activation functions to apply
activation_functions = {
 'Tanh': 'tanh', # Hyperbolic tangent activation function
 'Sigmoid': 'sigmoid', # Sigmoid activation function
 'ReLU': 'relu', # Rectified Linear Unit activation function
 'Leaky ReLU': layers.LeakyReLU(alpha=0.2), # Leaky ReLU with a small slope for negative values
 'Linear': 'linear' # Linear activation function
}
# Run the classification for each hidden layer activation function and print the accuracy
print("=== Multi-Class Classification with Different Hidden Layer Activations on Iris Dataset ===")
for name, activation in activation_functions.items():
 accuracy = run_multi_class_classification(activation) # Run classification with the current activation
 print(f'Accuracy with {name} activation in hidden layer (Softmax inoutput): {accuracy:.4f}') # Print the accuracy