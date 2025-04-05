import numpy as np
import pandas as pd
import scipy
import scipy.signal
import pickle
import os
import math
import cv2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Base Layer Class for all layer types
class Layer:
    def __init__(self):
        self.input = None
        self.output = None
        
    def forward(self, input_data):
        # Forward pass - to be implemented by subclasses
        raise NotImplementedError
    
    def backward(self, output_gradient, learning_rate):
        # Backward pass - to be implemented by subclasses
        raise NotImplementedError

# Convolutional Layer
class Conv2D(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        """
        Initialize convolutional layer
        
        :param input_shape: (height, width, channels)
        :param kernel_size: Size of the convolution kernel (height, width)
        :param depth: Number of kernels/filters
        """
        super().__init__()
        self.input_shape = input_shape
        self.input_height, self.input_width, self.input_channels = input_shape
        self.kernel_size = kernel_size
        self.depth = depth
        
        # Initialize filters with Xavier/Glorot initialization
        self.kernels_shape = (kernel_size[0], kernel_size[1], self.input_channels, depth)
        limit = np.sqrt(6 / (np.prod(kernel_size) * self.input_channels + np.prod(kernel_size) * depth))
        self.kernels = np.random.uniform(-limit, limit, self.kernels_shape)
        self.biases = np.zeros(depth)
        
        # For Adam optimizer
        self.m_kernels = np.zeros_like(self.kernels)
        self.v_kernels = np.zeros_like(self.kernels)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        
        # Calculate output dimensions
        self.output_shape = (
            self.input_height - kernel_size[0] + 1,
            self.input_width - kernel_size[1] + 1,
            depth
        )
    
    def forward(self, input_data):
        """
        Forward pass for convolutional layer
        
        :param input_data: Input data of shape (batch_size, height, width, channels)
        :return: Output of shape (batch_size, new_height, new_width, depth)
        """
        self.input = input_data
        batch_size = input_data.shape[0]
        
        # Initialize output array
        self.output = np.zeros((batch_size, *self.output_shape))
        
        # Perform convolution for each sample in batch
        for i in range(batch_size):
            for d in range(self.depth):
                for c in range(self.input_channels):
                    # Convolve each channel with corresponding kernel
                    self.output[i, :, :, d] += scipy.signal.convolve2d(
                        self.input[i, :, :, c], 
                        self.kernels[:, :, c, d], 
                        mode='valid'
                    )
                # Add bias
                self.output[i, :, :, d] += self.biases[d]
        
        return self.output
    
    def backward(self, output_gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """
        Backward pass for convolutional layer using Adam optimizer
        
        :param output_gradient: Gradient from next layer of shape (batch_size, height, width, depth)
        :param learning_rate: Learning rate for optimizer
        :param beta1: Exponential decay rate for 1st moment estimates
        :param beta2: Exponential decay rate for 2nd moment estimates
        :param epsilon: Small constant for numerical stability
        :param t: Timestep (for bias correction in Adam)
        :return: Gradient with respect to input
        """
        batch_size = output_gradient.shape[0]
        kernels_gradient = np.zeros_like(self.kernels)
        biases_gradient = np.zeros_like(self.biases)
        input_gradient = np.zeros_like(self.input)
        
        # Calculate gradients for each sample in batch
        for i in range(batch_size):
            for d in range(self.depth):
                # Gradient for biases - simple sum over height and width dimensions
                biases_gradient[d] += np.sum(output_gradient[i, :, :, d])
                
                for c in range(self.input_channels):
                    # Gradient for kernels - correlation between input and output gradient
                    kernels_gradient[:, :, c, d] += scipy.signal.correlate2d(
                        self.input[i, :, :, c],
                        output_gradient[i, :, :, d],
                        mode='valid'
                    )
                    
                    # Gradient for input - full convolution with rotated kernel
                    rotated_kernel = np.rot90(self.kernels[:, :, c, d], 2)
                    input_gradient[i, :, :, c] += scipy.signal.convolve2d(
                        output_gradient[i, :, :, d],
                        rotated_kernel,
                        mode='full'
                    )
        
        # Update kernels and biases using Adam optimizer
        # For kernels
        self.m_kernels = beta1 * self.m_kernels + (1 - beta1) * kernels_gradient
        self.v_kernels = beta2 * self.v_kernels + (1 - beta2) * (kernels_gradient ** 2)
        m_hat_kernels = self.m_kernels / (1 - beta1 ** t)
        v_hat_kernels = self.v_kernels / (1 - beta2 ** t)
        self.kernels -= learning_rate * m_hat_kernels / (np.sqrt(v_hat_kernels) + epsilon)
        
        # For biases
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * biases_gradient
        self.v_biases = beta2 * self.v_biases + (1 - beta2 ** t) * (biases_gradient ** 2)
        m_hat_biases = self.m_biases / (1 - beta1 ** t)
        v_hat_biases = self.v_biases / (1 - beta2 ** t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon)
        
        return input_gradient

# MaxPooling Layer
class MaxPool2D(Layer):
    def __init__(self, pool_size=(2, 2), stride=None):
        """
        Initialize max pooling layer
        
        :param pool_size: Size of the pooling window (height, width)
        :param stride: Stride of the pooling operation, defaults to pool_size
        """
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride if stride is not None else pool_size
        self.max_indices = None  # To store indices of max values for backprop
    
    def forward(self, input_data):
        """
        Forward pass for max pooling layer
        
        :param input_data: Input data of shape (batch_size, height, width, channels)
        :return: Output after max pooling
        """
        self.input = input_data
        batch_size, h_in, w_in, channels = input_data.shape
        h_pool, w_pool = self.pool_size
        h_stride, w_stride = self.stride
        
        # Calculate output dimensions
        h_out = (h_in - h_pool) // h_stride + 1
        w_out = (w_in - w_pool) // w_stride + 1
        
        output = np.zeros((batch_size, h_out, w_out, channels))
        self.max_indices = np.zeros((batch_size, h_out, w_out, channels, 2), dtype=int)
        
        # Perform max pooling
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for c in range(channels):
                        h_start = i * h_stride
                        h_end = h_start + h_pool
                        w_start = j * w_stride
                        w_end = w_start + w_pool
                        
                        # Get the region to pool from
                        pool_region = input_data[b, h_start:h_end, w_start:w_end, c]
                        
                        # Find max value and its position within the pool region
                        max_val = np.max(pool_region)
                        max_pos = np.unravel_index(np.argmax(pool_region), pool_region.shape)
                        
                        # Store max value and its position for backprop
                        output[b, i, j, c] = max_val
                        self.max_indices[b, i, j, c] = max_pos
        
        self.output = output
        return output
    
    def backward(self, output_gradient, learning_rate=None, **kwargs):
        """
        Backward pass for max pooling layer
        
        :param output_gradient: Gradient from next layer
        :param learning_rate: Not used for pooling layer
        :return: Gradient with respect to input
        """
        batch_size, h_out, w_out, channels = output_gradient.shape
        h_in, w_in = self.input.shape[1:3]
        h_pool, w_pool = self.pool_size
        h_stride, w_stride = self.stride
        
        input_gradient = np.zeros_like(self.input)
        
        # Distribute gradient only to max elements
        for b in range(batch_size):
            for i in range(h_out):
                for j in range(w_out):
                    for c in range(channels):
                        h_start = i * h_stride
                        w_start = j * w_stride
                        h_max, w_max = self.max_indices[b, i, j, c]
                        
                        # Add gradient to the position where the max was found
                        input_gradient[b, h_start + h_max, w_start + w_max, c] += output_gradient[b, i, j, c]
        
        return input_gradient

# Flatten Layer
class Flatten(Layer):
    def __init__(self):
        super().__init__()
        self.input_shape = None
    
    def forward(self, input_data):
        """
        Forward pass for flatten layer
        
        :param input_data: Input data of shape (batch_size, height, width, channels)
        :return: Flattened data of shape (batch_size, height*width*channels)
        """
        self.input = input_data
        self.input_shape = input_data.shape
        batch_size = input_data.shape[0]
        flattened_dim = np.prod(input_data.shape[1:])
        
        self.output = input_data.reshape(batch_size, flattened_dim)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None, **kwargs):
        """
        Backward pass for flatten layer
        
        :param output_gradient: Gradient from next layer
        :param learning_rate: Not used for flatten layer
        :return: Gradient with respect to input
        """
        return output_gradient.reshape(self.input_shape)

# Dense (Fully Connected) Layer
class Dense(Layer):
    def __init__(self, input_size, output_size, activation='relu'):
        """
        Initialize dense (fully connected) layer
        
        :param input_size: Number of input features
        :param output_size: Number of output features
        :param activation: Activation function ('sigmoid', 'relu', 'leaky_relu', 'linear')
        """
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.activation_type = activation
        
        # Xavier/Glorot initialization
        limit = np.sqrt(6 / (input_size + output_size))
        self.weights = np.random.uniform(-limit, limit, (input_size, output_size))
        self.biases = np.zeros(output_size)
        
        # For Adam optimizer
        self.m_weights = np.zeros_like(self.weights)
        self.v_weights = np.zeros_like(self.weights)
        self.m_biases = np.zeros_like(self.biases)
        self.v_biases = np.zeros_like(self.biases)
        
        # Set activation function
        self.activation_func = self._get_activation(self.activation_type)
        self.activation_derivative = self._get_activation_derivative(self.activation_type)
    
    def _get_activation(self, name):
        if name == 'sigmoid':
            def sigmoid_activation(x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sigmoid_activation
        elif name == 'relu':
            def relu_activation(x):
                return np.maximum(0, x)
            return relu_activation
        elif name == 'leaky_relu':
            def leaky_relu_activation(x):
                return np.where(x > 0, x, x * 0.01)
            return leaky_relu_activation
        elif name == 'softmax':
            def softmax_activation(x):
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)
            return softmax_activation
        elif name == 'linear':
            def linear_activation(x):
                return x
            return linear_activation
        else:
            raise ValueError(f"Unknown activation function: '{name}'")
    
    def _get_activation_derivative(self, name):
        if name == 'sigmoid':
            def sigmoid_derivative(x):
                return x * (1 - x)
            return sigmoid_derivative
        elif name == 'relu':
            def relu_derivative(x):
                return np.where(x > 0, 1, 0)
            return relu_derivative
        elif name == 'leaky_relu':
            def leaky_relu_derivative(x):
                return np.where(x > 0, 1, 0.01)
            return leaky_relu_derivative
        elif name == 'linear':
            def linear_derivative(x):
                return np.ones_like(x)
            return linear_derivative
        elif name == 'softmax':
            def softmax_derivative(x):
                return x * (1 - x)  # Simplified for when used with cross-entropy
            return softmax_derivative
        else:
            raise ValueError(f"Unknown activation function derivative: '{name}'")
    
    def forward(self, input_data):
        """
        Forward pass for dense layer
        
        :param input_data: Input data of shape (batch_size, input_size)
        :return: Output after dense layer and activation
        """
        self.input = input_data
        self.z = np.dot(input_data, self.weights) + self.biases
        self.output = self.activation_func(self.z)
        return self.output
    
    def backward(self, output_gradient, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8, t=1):
        """
        Backward pass for dense layer using Adam optimizer
        
        :param output_gradient: Gradient from next layer
        :param learning_rate: Learning rate for optimizer
        :param beta1: Exponential decay rate for 1st moment estimates
        :param beta2: Exponential decay rate for 2nd moment estimates
        :param epsilon: Small constant for numerical stability
        :param t: Timestep (for bias correction in Adam)
        :return: Gradient with respect to input
        """
        # Calculate gradient through activation function
        if self.activation_type == 'softmax':
            # Special case for softmax (assuming cross-entropy loss)
            delta = output_gradient
        else:
            delta = output_gradient * self.activation_derivative(self.output)
        
        # Calculate gradients for weights and biases
        weights_gradient = np.dot(self.input.T, delta)
        biases_gradient = np.sum(delta, axis=0)
        
        # Calculate gradient to pass to previous layer
        input_gradient = np.dot(delta, self.weights.T)
        
        # Update weights and biases using Adam optimizer
        # For weights
        self.m_weights = beta1 * self.m_weights + (1 - beta1) * weights_gradient
        self.v_weights = beta2 * self.v_weights + (1 - beta2) * (weights_gradient ** 2)
        m_hat_weights = self.m_weights / (1 - beta1 ** t)
        v_hat_weights = self.v_weights / (1 - beta2 ** t)
        self.weights -= learning_rate * m_hat_weights / (np.sqrt(v_hat_weights) + epsilon)
        
        # For biases
        self.m_biases = beta1 * self.m_biases + (1 - beta1) * biases_gradient
        self.v_biases = beta2 * self.v_biases + (1 - beta2 ** t) * (biases_gradient ** 2)
        m_hat_biases = self.m_biases / (1 - beta1 ** t)
        v_hat_biases = self.v_biases / (1 - beta2 ** t)
        self.biases -= learning_rate * m_hat_biases / (np.sqrt(v_hat_biases) + epsilon)
        
        return input_gradient

# Activation Layer as a separate layer
class Activation(Layer):
    def __init__(self, activation):
        """
        Initialize activation layer
        
        :param activation: Activation function name
        """
        super().__init__()
        self.activation_type = activation
        self.activation_func = self._get_activation(activation)
        self.activation_derivative = self._get_activation_derivative(activation)
    
    def _get_activation(self, name):
        if name == 'sigmoid':
            def sigmoid_activation(x):
                return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
            return sigmoid_activation
        elif name == 'relu':
            def relu_activation(x):
                return np.maximum(0, x)
            return relu_activation
        elif name == 'leaky_relu':
            def leaky_relu_activation(x):
                return np.where(x > 0, x, x * 0.01)
            return leaky_relu_activation
        elif name == 'softmax':
            def softmax_activation(x):
                exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
                return exp_x / (np.sum(exp_x, axis=-1, keepdims=True) + 1e-9)
            return softmax_activation
        elif name == 'linear':
            def linear_activation(x):
                return x
            return linear_activation
        else:
            raise ValueError(f"Unknown activation function: '{name}'")
    
    def _get_activation_derivative(self, name):
        if name == 'sigmoid':
            def sigmoid_derivative(x):
                return x * (1 - x)
            return sigmoid_derivative
        elif name == 'relu':
            def relu_derivative(x):
                return np.where(x > 0, 1, 0)
            return relu_derivative
        elif name == 'leaky_relu':
            def leaky_relu_derivative(x):
                return np.where(x > 0, 1, 0.01)
            return leaky_relu_derivative
        elif name == 'linear':
            def linear_derivative(x):
                return np.ones_like(x)
            return linear_derivative
        elif name == 'softmax':
            def softmax_derivative(x):
                return x * (1 - x)  # Simplified for when used with cross-entropy
            return softmax_derivative
        else:
            raise ValueError(f"Unknown activation function derivative: '{name}'")
    
    def forward(self, input_data):
        """
        Forward pass for activation layer
        
        :param input_data: Input data
        :return: Output after activation
        """
        self.input = input_data
        self.output = self.activation_func(input_data)
        return self.output
    
    def backward(self, output_gradient, learning_rate=None, **kwargs):
        """
        Backward pass for activation layer
        
        :param output_gradient: Gradient from next layer
        :param learning_rate: Not used for activation layer
        :return: Gradient with respect to input
        """
        if self.activation_type == 'softmax':
            # Special case for softmax (assuming cross-entropy loss)
            return output_gradient
        return output_gradient * self.activation_derivative(self.output)

# CNN Model
class CNN:
    def __init__(self, learning_rate=0.001):
        """
        Initialize CNN model
        
        :param learning_rate: Learning rate for optimizer
        """
        self.layers = []
        self.learning_rate = learning_rate
        self.t = 0  # Time step for Adam optimizer
    
    def add(self, layer):
        """
        Add a layer to the model
        
        :param layer: Layer to add
        """
        self.layers.append(layer)
    
    def predict(self, input_data):
        """
        Make predictions with the model
        
        :param input_data: Input data
        :return: Model predictions
        """
        # Ensure input data is in batch format
        if input_data.ndim == 3:  # Single image (height, width, channels)
            input_data = np.expand_dims(input_data, axis=0)
        
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        
        return output
    
    def train(self, X_train, y_train, epochs, batch_size=32, verbose=True):
        """
        Train the model
        
        :param X_train: Training data
        :param y_train: Training labels
        :param epochs: Number of epochs
        :param batch_size: Batch size
        :param verbose: Whether to print progress
        """
        # Ensure X_train is in batch format
        if X_train.ndim == 3:  # Single image (height, width, channels)
            X_train = np.expand_dims(X_train, axis=0)
        
        # Ensure y_train is in batch format
        if y_train.ndim == 1:  
            y_train = np.expand_dims(y_train, axis=0)
        
        n_samples = X_train.shape[0]
        
        for epoch in range(epochs):
            # Shuffle the data
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]
            
            loss = 0
            
            # Train in batches
            for i in range(0, n_samples, batch_size):
                # Get batch
                X_batch = X_shuffled[i:min(i + batch_size, n_samples)]
                y_batch = y_shuffled[i:min(i + batch_size, n_samples)]
                
                # Forward pass
                output = self.predict(X_batch)
                
                # Compute loss
                loss += self._compute_loss(y_batch, output)
                
                # Backward pass
                self.t += 1  # Increment time step for Adam optimizer
                grad = self._compute_loss_gradient(y_batch, output)
                for layer in reversed(self.layers):
                    grad = layer.backward(grad, self.learning_rate, t=self.t)
            
            # Average loss over all batches
            loss /= n_samples / batch_size
            
            if verbose and (epoch % 10 == 0 or epoch == epochs - 1):
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss:.4f}")
    
    def _compute_loss(self, y_true, y_pred):
        """
        Compute loss
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Loss value
        """
        # Mean Squared Error
        return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))
    
    def _compute_loss_gradient(self, y_true, y_pred):
        """
        Compute gradient of loss
        
        :param y_true: True labels
        :param y_pred: Predicted labels
        :return: Gradient of loss
        """
        # Gradient of Mean Squared Error
        return 2 * (y_pred - y_true) / y_true.shape[0]
    
    def save_model(self, filename):
        """
        Save model to file
        :param filename: Filename to save to
        """
        with open(filename, 'wb') as f:
            pickle.dump(self, f)
    
    @classmethod
    def load_model(cls, filename):
        """
        Load model from file
        
        :param filename: Filename to load from
        :return: Loaded model
        """
        with open(filename, 'rb') as f:
            return pickle.load(f)


def save_model(model, filename):
    """
    Save the model to a file.
    :param model: The CNN model to save.
    :param filename: The filename to save the model to.
    """
    with open(filename, 'wb') as f:
        pickle.dump(model, f)

def load_model(filename):
    """
    Load a model from a file.
    :param filename: The filename to load the model from.
    :return: The loaded CNN model.
    """
    with open(filename, 'rb') as f:
        return pickle.load(f)

def train_model(model, X_train, y_train, epochs=50, batch_size=32):
    """
    Train the CNN model.
    :param model: The CNN model to train.
    :param X_train: Training data.
    :param y_train: Training labels.
    :param epochs: Number of epochs to train for.
    :param batch_size: Batch size for training.
    """
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)

def test_model(model, image_path, target_size):
    """
    Test the model on a single image.
    :param model: The CNN model to test.
    :param image_path: Path to the image to test.
    :param target_size: Tuple specifying the target size for resizing the image.
    :return: The model's prediction.
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f'Could not load image from {image_path}')
    img = cv2.resize(img, target_size)
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    prediction = model.predict(img)
    return prediction

def load_and_preprocess_images(image_folder, target_size=(128, 128)):
    """
    Load and preprocess images from a folder.
    :param image_folder: Path to the folder containing images.
    :param target_size: Tuple specifying the target size for resizing the images.
    :return: Tuple of (images, labels).
    """
    images = []
    labels = []
    
    # Check if the folder exists
    if not os.path.exists(image_folder):
        print(f"Error: Folder {image_folder} does not exist")
        return np.array([]), np.array([])
    
    # Check if this is a folder directly contains images
    contents = os.listdir(image_folder)
    has_subfolders = any(os.path.isdir(os.path.join(image_folder, item)) for item in contents)
    
    if has_subfolders:
        # Process folder with class subfolders
        class_names = [item for item in contents if os.path.isdir(os.path.join(image_folder, item))]
        class_to_index = {class_name: idx for idx, class_name in enumerate(sorted(class_names))}
        
        for class_name in class_names:
            class_folder = os.path.join(image_folder, class_name)
            class_idx = class_to_index[class_name]
            
            for filename in os.listdir(class_folder):
                if filename.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif')):
                    image_path = os.path.join(class_folder, filename)
                    img = cv2.imread(image_path)
                    if img is None:
                        print(f"Warning: Could not load image {filename}")
                        continue
                    
                    if len(img.shape) == 2:  # If grayscale, convert to RGB
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                    else:
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                    
                    img = cv2.resize(img, target_size)
                    img = img.astype(np.float32) / 255.0
                    images.append(img)
                    labels.append(class_idx)
    else:
        # Process folder with images directly (single class)
        for filename in contents:
            if filename.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif')):
                image_path = os.path.join(image_folder, filename)
                img = cv2.imread(image_path)
                if img is None:
                    print(f"Warning: Could not load image {filename}")
                    continue
                
                if len(img.shape) == 2:  # If grayscale, convert to RGB
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
                else:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                
                img = cv2.resize(img, target_size)
                img = img.astype(np.float32) / 255.0
                images.append(img)
                labels.append(0)  # Single class, assign label 0
    
    if not images:
        print("No valid images found in the folder.")
        return np.array([]), np.array([])
    
    return np.array(images), np.array(labels)

def train_model_with_images(model, image_folder, test_size=0.2, epochs=10, batch_size=32):
    """
    Train the CNN model using images from a folder.
    :param model: The CNN model to train.
    :param image_folder: Path to the folder containing images.
    :param test_size: Proportion of the dataset to include in the test split.
    :param epochs: Number of epochs to train for.
    :param batch_size: Batch size for training.
    """
    # Load and preprocess images
    X, y = load_and_preprocess_images(image_folder)
    
    if len(X) == 0:
        print("No images found for training")
        return
        
    # Dynamically determine input size from the first image
    input_shape = X.shape[1:]  # (height, width, channels)
    
    # Count unique classes and prepare appropriate labels format
    unique_classes = np.unique(y)
    num_classes = len(unique_classes)
    
    print(f"Found {num_classes} unique classes in the dataset")
    
    # Format labels appropriately
    # For binary classification (sigmoid output)
    if num_classes <= 2:
        y = np.array(y).reshape(-1, 1)
    # For multi-class classification (softmax output)
    else:
        # One-hot encode the labels
        y_onehot = np.zeros((len(y), num_classes))
        for i, label in enumerate(y):
            y_onehot[i, label] = 1
        y = y_onehot
    
    # Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Testing set: {X_test.shape[0]} samples")
    
    # Train the model
    model.train(X_train, y_train, epochs=epochs, batch_size=batch_size)
    
    # Evaluate the model
    predictions = model.predict(X_test)
    
    # Calculate accuracy based on classification type
    if num_classes <= 2:
        # Binary classification
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes == y_test)
    else:
        # Multi-class classification
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(y_test, axis=1)
        accuracy = np.mean(pred_classes == true_classes)
    
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    
    return model

def create_dynamic_cnn(input_shape, num_classes=1):
    """
    Create a CNN model dynamically based on input shape
    
    :param input_shape: Input shape (height, width, channels)
    :param num_classes: Number of classes for classification
    :return: CNN model
    """
    model = CNN(learning_rate=0.001)
    
    # First convolutional layer
    model.add(Conv2D(input_shape=input_shape, kernel_size=(3, 3), depth=16))
    model.add(Activation('relu'))
    
    # MaxPooling layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Second convolutional layer
    h, w, c = input_shape
    # Calculate dimensions after first conv and pool
    new_h = (h - 3 + 1) // 2
    new_w = (w - 3 + 1) // 2
    model.add(Conv2D(input_shape=(new_h, new_w, 16), kernel_size=(3, 3), depth=32))
    model.add(Activation('relu'))
    
    # MaxPooling layer
    model.add(MaxPool2D(pool_size=(2, 2)))
    
    # Flatten layer
    model.add(Flatten())
    
    # Calculate input size for dense layer
    dense_input_size = 32 * ((new_h - 3 + 1) // 2) * ((new_w - 3 + 1) // 2)
    
    # Dense layer
    model.add(Dense(dense_input_size, 128, activation='relu'))
    
    # Output layer
    if num_classes == 1:
        model.add(Dense(128, 1, activation='sigmoid'))  # Binary classification
    else:
        model.add(Dense(128, num_classes, activation='softmax'))  # Multi-class classification
    
    return model

def test_model_on_images(model, test_folder, target_size=(128, 128)):
    """
    Test the model on a folder of images.
    
    :param model: The CNN model to test.
    :param test_folder: Path to the folder containing test images.
    :param target_size: Tuple specifying the target size for resizing the images.
    :return: Dictionary with filenames as keys and predictions as values.
    """
    results = {}
    
    if not os.path.exists(test_folder):
        print(f"Error: Test folder {test_folder} does not exist")
        return results
    
    # Get all image files in the test folder
    files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.jpeg', '.jpg', '.png', '.bmp', '.gif'))]
    
    if not files:
        print("No image files found in the test folder")
        return results
    
    for filename in files:
        image_path = os.path.join(test_folder, filename)
        img = cv2.imread(image_path)
        
        if img is None:
            print(f"Warning: Could not load image {filename}")
            continue
        
        # Preprocess image
        if len(img.shape) == 2:  # If grayscale, convert to RGB
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        
        img = cv2.resize(img, target_size)
        img = img.astype(np.float32) / 255.0
        
        # Make prediction
        prediction = model.predict(np.expand_dims(img, axis=0))[0]
        
        # Store result
        results[filename] = prediction
        
        # Print result for this image
        print(f"Image: {filename}")
        
        # Handle binary vs multi-class prediction
        if len(prediction) == 1:
            # Binary classification
            pred_class = 1 if prediction[0] > 0.5 else 0
            confidence = prediction[0] if pred_class == 1 else 1 - prediction[0]
            print(f"  Predicted class: {pred_class} with confidence: {confidence*100:.2f}%")
        else:
            # Multi-class classification
            pred_class = np.argmax(prediction)
            confidence = prediction[pred_class]
            print(f"  Predicted class: {pred_class} with confidence: {confidence*100:.2f}%")
    
    return results

def visualize_predictions(model, images, true_labels, num_samples=5, figsize=(15, 10)):
    """
    Visualize model predictions with matplotlib
    
    :param model: The trained CNN model
    :param images: Array of images
    :param true_labels: Array of true labels
    :param num_samples: Number of samples to visualize
    :param figsize: Figure size for the plot
    """
    # Select a random subset of images
    num_samples = min(num_samples, len(images))
    indices = np.random.choice(len(images), num_samples, replace=False)
    
    # Set up the plot
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]  # Make axes iterable if only one sample
    
    # Plot each image with its prediction
    for i, idx in enumerate(indices):
        img = images[idx]
        true_label = true_labels[idx][0] if true_labels[idx].shape == (1,) else true_labels[idx]
        
        # Make prediction
        pred = model.predict(np.expand_dims(img, axis=0))[0]
        
        # Determine prediction class and confidence
        if len(pred) == 1:  # Binary classification
            pred_class = 1 if pred[0] > 0.5 else 0
            confidence = pred[0] if pred_class == 1 else 1 - pred[0]
        else:  # Multi-class classification
            pred_class = np.argmax(pred)
            confidence = pred[pred_class]
        
        # Convert from RGB back to BGR for display (optional)
        display_img = img.copy()
        
        # Add a colored border based on prediction correctness
        if (len(true_label) == 1 and pred_class == true_label[0]) or (len(true_label) > 1 and pred_class == np.argmax(true_label)):
            # Green border for correct predictions
            bordered_img = cv2.copyMakeBorder(display_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(0, 1, 0))
        else:
            # Red border for incorrect predictions
            bordered_img = cv2.copyMakeBorder(display_img, 5, 5, 5, 5, cv2.BORDER_CONSTANT, value=(1, 0, 0))
        
        # Display the image
        axes[i].imshow(bordered_img)
        axes[i].set_title(f"True: {true_label[0] if len(true_label.shape) > 0 else true_label}\nPred: {pred_class}\nConf: {confidence:.2f}")
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def test_with_visualization(model, test_images, test_labels, num_samples=5):
    """
    Test model and visualize results
    
    :param model: The trained CNN model
    :param test_images: Array of test images
    :param test_labels: Array of test labels
    :param num_samples: Number of samples to visualize
    """
    # Evaluate on all test data
    predictions = model.predict(test_images)
    
    # Calculate accuracy
    if predictions.shape[1] == 1:  # Binary classification
        pred_classes = (predictions > 0.5).astype(int)
        true_classes = test_labels
        accuracy = np.mean(pred_classes == true_classes)
    else:  # Multi-class classification
        pred_classes = np.argmax(predictions, axis=1)
        true_classes = np.argmax(test_labels, axis=1) if test_labels.shape[1] > 1 else test_labels
        accuracy = np.mean(pred_classes == true_classes)
    
    print(f"Overall Test Accuracy: {accuracy * 100:.2f}%")
    
    # Visualize results
    visualize_predictions(model, test_images, test_labels, num_samples)
    
    return accuracy

# Example usage
if __name__ == "__main__":
    # Define the folder containing images
    image_folder = r"c:\Users\223146145\Downloads\Neural-Network-XOR\n01443537" # Need to change this as soon as i use Google colab..
    print(f"Loading images from: {image_folder}")
    
    # First load all images to get dimensions and count classes
    sample_images, sample_labels = load_and_preprocess_images(image_folder, target_size=(128, 128))
    
    if len(sample_images) == 0:
        print("Error: No images found. Please check the path.")
    else:
        print(f"Found {len(sample_images)} images")
        input_shape = sample_images[0].shape
        print(f"Image shape: {input_shape}")
        
        # Since we're using a single folder with all the same class images, assign half to class 0 and half to class 1
        # just to create a binary classification scenario for testing
        
        # Divide images into two "virtual" classes for demonstration
        half_idx = len(sample_images) // 2
        labels = np.zeros(len(sample_images))
        labels[half_idx:] = 1  # Second half gets label 1
        
        # Prepare datasets with balanced classes
        X_train, X_test, y_train, y_test = train_test_split(
            sample_images, labels, test_size=0.2, random_state=42, stratify=labels 
        )
        
        # Format labels for binary classification
        y_train = y_train.reshape(-1, 1)
        y_test = y_test.reshape(-1, 1)
        
        print(f"Training set: {X_train.shape[0]} samples")
        print(f"Testing set: {X_test.shape[0]} samples")
        
        # Create a binary classification CNN model
        model = create_dynamic_cnn(input_shape, num_classes=1)
        
        # Train the model
        print("\nTraining model...")
        model.train(X_train, y_train, epochs=10, batch_size=4)
        
        # Evaluate model
        print("\nEvaluating model...")
        predictions = model.predict(X_test)
        pred_classes = (predictions > 0.5).astype(int)
        accuracy = np.mean(pred_classes == y_test)
        print(f"Test Accuracy: {accuracy * 100:.2f}%")
        
        # Save the model
        save_model(model, "fish_classifier.pkl") # Buggy
        print("Model saved as fish_classifier.pkl")
        
        # Load the model and test on a few images
        print("\nLoading model and testing on sample images...")
        loaded_model = load_model("fish_classifier.pkl")
        
        # Test on a few random images
        num_test = min(5, len(X_test))
        for i in range(num_test):
            test_img = X_test[i]
            true_label = y_test[i][0]
            
            pred = model.predict(np.expand_dims(test_img, axis=0))[0][0] 
            pred_class = 1 if pred > 0.5 else 0
            
            print(f"Sample {i+1}: True class={true_label}, Predicted class={pred_class}, Confidence={pred*100:.2f}% if class 1, {(1-pred)*100:.2f}% if class 0")
        
        # Visualize predictions
        # print("\nVisualizing predictions...")
        # test_with_visualization(model, X_test, y_test, num_samples=5)
