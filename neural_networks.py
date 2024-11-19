import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
import os
from functools import partial

result_dir = "results"
os.makedirs(result_dir, exist_ok=True)

# Define activation functions and their derivatives
def activation_function(x, activation):
    if activation == 'tanh':
        return np.tanh(x)
    elif activation == 'relu':
        return np.maximum(0, x)
    elif activation == 'sigmoid':
        return 1 / (1 + np.exp(-x))
    else:
        raise ValueError("Unsupported activation function")

def activation_derivative(x, activation):
    if activation == 'tanh':
        return 1 - np.tanh(x) ** 2
    elif activation == 'relu':
        return np.where(x > 0, 1, 0)
    elif activation == 'sigmoid':
        sigmoid = 1 / (1 + np.exp(-x))
        return sigmoid * (1 - sigmoid)
    else:
        raise ValueError("Unsupported activation function")

# Define a simple MLP class
class MLP:
    def __init__(self, input_dim, hidden_dim, output_dim, lr, activation='tanh'):
        np.random.seed(0)
        self.lr = lr
        self.activation_fn = activation
        # Initialize weights and biases
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.01
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.01
        self.b2 = np.zeros((1, output_dim))
        # For visualization
        self.hidden_activations = None
        self.gradients = None

    def forward(self, X, return_hidden=False):
        # Forward pass: Input → Hidden → Output
        self.Z1 = np.dot(X, self.W1) + self.b1
        self.A1 = activation_function(self.Z1, self.activation_fn)
        self.Z2 = np.dot(self.A1, self.W2) + self.b2
        self.A2 = activation_function(self.Z2, 'sigmoid')  # Output layer uses sigmoid for binary classification
        if return_hidden:
            return self.A1
        return self.A2

    def backward(self, X, y):
        # Backpropagation
        m = X.shape[0]  # Number of samples
        dZ2 = self.A2 - y
        dW2 = np.dot(self.A1.T, dZ2) / m
        db2 = np.sum(dZ2, axis=0, keepdims=True) / m

        dA1 = np.dot(dZ2, self.W2.T)
        dZ1 = dA1 * activation_derivative(self.Z1, self.activation_fn)
        dW1 = np.dot(X.T, dZ1) / m
        db1 = np.sum(dZ1, axis=0, keepdims=True) / m

        # Gradient descent update
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

        # For visualization
        self.gradients = [np.linalg.norm(dW1), np.linalg.norm(dW2)]

def generate_data(n_samples=100):
    np.random.seed(0)
    X = np.random.randn(n_samples, 2)
    y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).astype(int).reshape(-1, 1)  # Circular boundary
    return X, y

# Visualization update function
def update(frame, mlp, ax_input, ax_hidden, ax_gradient, X, y):
    ax_hidden.clear()
    ax_input.clear()
    ax_gradient.clear()

    # Perform training steps
    for _ in range(10):  # Train for 10 steps per frame
        mlp.forward(X)
        mlp.backward(X, y)

    # Plot hidden layer features in 3D
    hidden_features = mlp.forward(X, return_hidden=True)
    ax_hidden = plt.subplot(131, projection='3d')
    ax_hidden.scatter(hidden_features[:, 0], hidden_features[:, 1], hidden_features[:, 2], c=y.ravel(), cmap='bwr', alpha=0.7)
    ax_hidden.set_title("Hidden Space")
    ax_hidden.set_xlabel("h1")
    ax_hidden.set_ylabel("h2")
    ax_hidden.set_zlabel("h3")

    # Decision boundary in input space
    xx, yy = np.meshgrid(np.linspace(-3, 3, 200), np.linspace(-3, 3, 200))  # High-resolution grid
    grid = np.c_[xx.ravel(), yy.ravel()]
    predictions = mlp.forward(grid)
    ax_input.contourf(xx, yy, predictions.reshape(xx.shape), alpha=0.5, cmap='bwr')
    ax_input.scatter(X[:, 0], X[:, 1], c=y.ravel(), cmap='bwr', edgecolors='k')
    ax_input.set_title("Input Space")

    # Gradients visualization as bar chart
    ax_gradient.bar(["W1", "W2"], mlp.gradients, color="blue", alpha=0.6)
    ax_gradient.set_title("Gradients")
    ax_gradient.set_ylabel("Magnitude")

def visualize(activation, lr, step_num):
    X, y = generate_data()
    mlp = MLP(input_dim=2, hidden_dim=3, output_dim=1, lr=lr, activation=activation)

    # Set up visualization
    matplotlib.use('agg')
    fig = plt.figure(figsize=(21, 7))
    ax_hidden = fig.add_subplot(131, projection='3d')
    ax_input = fig.add_subplot(132)
    ax_gradient = fig.add_subplot(133)

    # Create animation
    ani = FuncAnimation(fig, partial(update, mlp=mlp, ax_input=ax_input, ax_hidden=ax_hidden, ax_gradient=ax_gradient, X=X, y=y), frames=step_num//10, repeat=False)

    # Save the animation as a GIF
    ani.save(os.path.join(result_dir, "visualize.gif"), writer='pillow', fps=10)
    plt.close()

if __name__ == "__main__":
    activation = "tanh"
    lr = 0.1
    step_num = 1000
    visualize(activation, lr, step_num)
