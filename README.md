# Neural-Network-XOR
This repo has the Nueral Network which was trained to perform an XOR Operation on a 2 inputted array. 
> The Documentation and Upgrades from this is still pending.

- A changelog will be attached to track changes made, TO-DO list to list all the functionalities to be introduced in future 

## Adam Optimiser 
- An optimisation algorithm used to update the learning rate, biases and the weights of the nueral network by moving the learning rate to find the best global minimum instead of being stuck on the local minimum.. This is really useful for cases whereby you might be having more than 1 minimum for your dataset. This algorithm uses the combinations of Momemntum and root mean square propagation.
### Formula

$m_t = 0   \text{First Moment Vector} $

$v_t = 0   \text{Second Momemnt Vector}$

$ t = 0  \text{Time step} $

1. To update the rules for each time step $t$ :
   - Compute the gradient $nabla \theta$
  
     
        $m_t = \beta_1 \times m_{t-1} +(1-\beta_2)\times \nabla \theta_t$

   
        $v_t = \beta_2 \times v_{t-1}+ (1-\beta_2)\times \(\nabla \theta_t\)^2$
``
import numpy as np
import mne # This will be goog for data loading
import pandas as pd # Reading the csv or an spreadsheet format (CSV)
import scipy
class NeuralNetwork():
    def __init__(self, input_size, hidden_nodes, output_size, learning_rate=0.1):
        """
        :param input_size: Number of input neurons
        :param hidden_nodes: List specifying number of neurons in each hidden layer
        :param output_size: Number of output neurons
        :param learning_rate: Learning rate for weight updates

        """
        self.input_size = input_size
        self.hidden_nodes = hidden_nodes  # List specifying neurons per hidden layer
        self.output_size = output_size
        self.learning_rate = learning_rate

        # Define the model
        layer_sizes = [input_size] + hidden_nodes + [output_size]

        # Initialize weights and biases dynamically
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i+1]) - 0.5 for i in range(len(layer_sizes) - 1)]
        self.biases = [np.random.rand(layer_sizes[i+1]) - 0.5 for i in range(len(layer_sizes) - 1)]

    def sigmoid(self, x):
      return 1 / (1 + np.exp(-x))


    def sigmoid_derivative(self, x):
        return x * (1 - x)
    
    def Relu(x, gradient_factor = 0.001):
        return np.max(x*gradient_factor, x)

    def feedForward(self, inputs):
        # Forward propagation through all layers.......
        self.layers = [inputs]  # Store outputs of all layers
        for i in range(len(self.weights)):
            inputs = self.sigmoid(np.dot(inputs, self.weights[i]) + self.biases[i])
            self.layers.append(inputs)  # Save outputs of [i+1] layer for backpropagation
        return inputs

    def featureExtraction(self, epochs): # Need to work on Extraction of features.
        feauture = []
        for channel in epochs:
            feauture.append(np.mean(channel))
            feauture.append(np.std(channel))

    def SoftMax(x):
        """
        Transforms the input scores into probabilities
        """
        Numerator = np.exp(x)
        result = Numerator/np.sum(Numerator)
        return result
    # Back Prop, to update the weights and Biases of the nueral network
    def Adam_Optimiser(weights, gradients, first_moment, second_moment, time_step, learning_rate = 0.001, beta1 = 0.9, beta2 = 0.999, epsilon = 1e-8):
        """
        Adam Optimiser function to update the parameters based on te inputed gradient.
           Arguments:
        - params: Current parameters (weights) of the model (numpy array)
        - grads: Gradients of the parameters (numpy array)
        - m: First moment (mean of gradients) (numpy array)
        - v: Second moment (uncentered variance of gradients) (numpy array)
        - t: Time step or iteration counter (int)
        - learning_rate: Learning rate (float)
        - beta1: Exponential decay rate for the first moment estimate (float)
        - beta2: Exponential decay rate for the second moment estimate (float)
        - epsilon: Small constant to avoid division by zero (float)
        
        Returns:
        - Updated parameters (weights) after applying Adam update rule
        - Updated first and second moment estimates
        - Updated time step
        
        """
        time_step += 1 # Update teh Time step

        #Update the First Moment
        first_moment = first_moment*beta1 + (1-beta1)*gradients
         
        # Update the second moment
        second_moment = beta2*second_moment +(1-beta2)*(gradients**2)


        ## Update the firstmoment hat 
        first_hat = (first_moment)/(1-beta1**time_step)
        #Update the second hat
        second_hat = (second_moment)/(1-beta2**time_step)

        ## Update the weights now
        weights -= learning_rate * (first_hat/(np.sqrt(second_hat)+epsilon))
        return weights, first_moment, second_hat, time_step

    def backpropagation(self, target_output):
        errors = [target_output - self.layers[-1]]  # Output layer error
        deltas = [errors[0] * self.sigmoid_derivative(self.layers[-1])]  # Output layer delta

        # Get Dltas For each hidden layer in reverse order
        for i in range(len(self.hidden_nodes), 0, -1):
            errors.insert(0, np.dot(deltas[0], self.weights[i].T))  # Error of previous layer
            deltas.insert(0, errors[0] * self.sigmoid_derivative(self.layers[i]))  # Delta, previous layer

        # Update weights and biases
        for i in range(len(self.weights)):
            self.weights[i] += np.dot(self.layers[i].reshape(-1, 1), deltas[i].reshape(1, -1)) * self.learning_rate
            self.biases[i] += deltas[i] * self.learning_rate

    def train(self, X, y, epochs=10000):
        for epoch in range(epochs):
            total_loss = 0
            for i in range(len(X)):
                self.feedForward(X[i])
                self.backpropagation(y[i])
                total_loss += np.sum(np.abs(y[i] - self.layers[-1]))

            if epoch % 3000 == 0:
                print(f"Epoch {epoch}, Loss: {(total_loss / len(X)):.2f}")

    def predict(self, X):
        return [self.feedForward(x) for x in X]


    def save_model(self, filename):
        with open(filename, 'wb') as file:
            np.save(file, self.input_size)
            np.save(file, self.hidden_nodes)
            np.save(file, self.output_size)


            # Save all weights and biases
            for weight in self.weights:
                np.save(file, weight)

            for bias in self.biases:
                np.save(file, bias)


    def Load_Model(self, filename):
      # Open the file in read mode
      with open(filename, 'rb') as file:
          self.input_size = np.load(file)
          self.hidden_nodes = np.load(file)
          self.output_size = np.load(file)
          self.weights = []
          self.biases = []
          size = [self.input_size] + self.hidden_nodes + [self.output_size]
          self.weights = [np.load(file) for _ in range(len(size) + 1)]
          self.biases = [np.load(file) for _ in range(len(size) + 1)]
          print(f'Model : {filename} has been Loaded successfully')


# OR dataset
X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])
#  Desired Output/ Target Output
y = np.array([
    [0],
    [1],
    [1],
    [1]
])

# Create A Nueral Network with 2 inputs, and 2 Hidden Layers with nodes each,
nn = NeuralNetwork(input_size=2, hidden_nodes=[20], output_size=1, learning_rate=0.2)
nn.train(X, y, epochs=10000)

"""
The Model is too small and the learning rate is pretty quick
So, The 50 Thousands epochs are not tha much of a deal,
ever since the model has to solve a basic problem

"""
nn.save_model('model.txt')
# EEG Processing
From the previously designed Nueral Network system on the `Model.ipynb` file, we will be building upon that to create fundamental concepts of EEG signal Processing.

There are a couple of things to note here when transitioning to the Nueral Network that should be used for EEG interpretation.

- The EEG data consists of time series recording of electrical activities in the brain, , they are measured from multiple electrodes(From the scalp). 
- The recording of data has a sampling rate, which should the speed at which data is being recorded per unit second.. This Frequency determines our temporal resolution of the data.
- Labeling the data you use for training, 

- Use a ReLu Function or the parameterised ReLu function for better approximation, and avoiding the vanishing gradient...

- The number of the outputs should vary according to the number of the outputs desired for the application.. Example, output will be binary if you are trying to classsify the binary problem, `The patient is experiencing a seizure or not..`

- Venture through the Convulational ueral Network, Recurrent Nueral Network or Long short term memory  Networks.. 


- Use Category Cross Entropy for Multiple class for error finding*(Using Softmax function), or Binary Cross entropy for Binary output. 

-- Adams Optimisation 

-- Fill in the equations for the adams and then later on do the research on how it actually works..



# Resources Used
- [MIT LECTURE](https://www.youtube.com/watch?v=wrEcHhoJxjM)
