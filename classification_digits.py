import numpy as np
import matplotlib.pyplot as plt
import struct

# Function to load MNIST images
def load_mnist_images(filename):
    with open(filename, 'rb') as f:
        # Read the metadata
        magic, num_images, rows, cols = struct.unpack(">IIII", f.read(16))
        # Read the image data
        images = np.fromfile(f, dtype=np.uint8).reshape(num_images, rows, cols)
    return images

# Function to load MNIST labels
def load_mnist_labels(filename):
    with open(filename, 'rb') as f:
        # Read the metadata
        magic, num_labels = struct.unpack(">II", f.read(8))
        # Read the label data
        labels = np.fromfile(f, dtype=np.uint8)
    return labels

# Function to prepare images (flatten and normalize)
def prepare_images(images):
    return (images / 255.0).reshape(images.shape[0], -1)

# Function to extract specific digits
def extract_digits(images, labels, digit_pair):
    """
    Extracts images and labels for the specified digit pair.

    Parameters:
    images : numpy.ndarray
        The dataset of images (shape: num_samples x num_features).
    labels : numpy.ndarray
        The dataset of labels (shape: num_samples,).
    digit_pair : tuple
        A tuple specifying the two digits to extract (e.g., (4, 9)).

    Returns:
    filtered_images : numpy.ndarray
        Images corresponding to the specified digits.
    binary_labels : numpy.ndarray
        Labels transformed to binary values: 0 for the first digit, 1 for the second.
    """
    # Find indices of the two digits
    indices = np.isin(labels, digit_pair)
    filtered_images = images[indices]
    filtered_labels = labels[indices]

    # Transform labels to binary: 0 for digit_pair[0], 1 for digit_pair[1]
    binary_labels = (filtered_labels == digit_pair[1]).astype(int)

    return filtered_images, binary_labels

# Linear Classifier Function
def linear_classifier(X, weights):
    """
    Predict labels using a linear model.

    Parameters:
    X : numpy.ndarray
        Input data (m samples x n features)
    weights : numpy.ndarray
        Weight vector including the bias term (n+1 features)
    logits: raw scores from each input sample, the
            further away from 0, the stronger the classification

    Returns:
    predictions : numpy.ndarray
        Predicted labels as 0 or 1
    """
    # Add a bias term to the input data
    bias = np.ones((X.shape[0], 1))  # Shape (m, 1)
    X_bias = np.hstack((bias, X))  # Shape (m, n+1)

    # Linear transformation
    logits = np.dot(X_bias, weights)  # Shape (m,)

    # Convert logits to binary predictions (0 or 1)
    predictions = (logits >= 0).astype(int)

    return predictions

# Xavier Initialization for W and b
def initialize_weights_xavier(input_size, output_size):
    """
    Initializes weights using Xavier initialization.

    Parameters:
    input_size : int
        Number of input features (D).
    output_size : int
        Number of output classes (C).

    Returns:
    W : numpy.ndarray
        Initialized weight matrix (C x D).
    b : numpy.ndarray
        Initialized bias vector (C).
    """
    limit = np.sqrt(2 / (input_size + output_size))  # Xavier scaling factor
    W = np.random.uniform(-limit, limit, size=(output_size, input_size))
    b = np.zeros(output_size)  # Biases initialized to zero
    return W, b

# Softmax function
def softmax(logits):
    """
    Compute softmax probabilities.

    Parameters:
    logits : numpy.ndarray
        Logits from the linear model (shape: m x C, where m is the number of samples,
        and C is the number of classes).

    Returns:
    probabilities : numpy.ndarray
        Softmax probabilities (shape: m x C).
    """
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))  # Stability trick
    probabilities = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
    return probabilities


# Loss function with regularization
def compute_loss(X, y_true, W, b, reg_strength):
    """
    Compute the total loss (cross-entropy + L2 regularization).

    Parameters:
    X : numpy.ndarray
        Input data (m samples x n features).
    y_true : numpy.ndarray
        True labels (m samples, in one-hot encoded form).
    W : numpy.ndarray
        Weight matrix (C x D, where C is the number of classes, D is the number of features).
    b : numpy.ndarray
        Bias vector (C, one per class).
    reg_strength : float
        Regularization strength (lambda).

    Returns:
    loss : float
        The total loss value.
    """
    # Compute logits
    logits = np.dot(X, W.T) + b  # Shape: (m, C)

    # Compute softmax probabilities
    probabilities = softmax(logits)  # Shape: (m, C)

    # Cross-entropy loss (data loss)
    num_samples = X.shape[0]
    data_loss = -np.sum(y_true * np.log(probabilities + 1e-12)) / num_samples  # Add small epsilon to avoid log(0)

    # L2 Regularization loss
    reg_loss = reg_strength * np.sum(W ** 2)

    # Total loss
    total_loss = data_loss + reg_loss
    return total_loss

# Gradient computation
def compute_gradients(X, y_true, W, b, reg_strength):
    """
    Compute gradients of the total loss w.r.t. weights and biases.

    Parameters:
    X : numpy.ndarray - Input data (m samples x n features).
    y_true : numpy.ndarray - One-hot encoded true labels (m samples x C classes).
    W : numpy.ndarray - Weight matrix (C x D features).
    b : numpy.ndarray - Bias vector (C classes).
    reg_strength : float - Regularization strength (lambda).

    Returns:
    grad_W : numpy.ndarray - Gradient w.r.t. weights (C x D).
    grad_b : numpy.ndarray - Gradient w.r.t. biases (C).
    """
    # Compute logits: raw scores for each class
    logits = np.dot(X, W.T) + b

    # Compute softmax probabilities
    probabilities = softmax(logits)

    # Error term: difference between predicted and true labels
    grad_logits = probabilities - y_true

    # Gradients w.r.t. weights and biases
    grad_W = np.dot(grad_logits.T, X) / X.shape[0] + 2 * reg_strength * W  # Add regularization
    grad_b = np.sum(grad_logits, axis=0) / X.shape[0]

    return grad_W, grad_b

# Gradient descent optimization
def optimize_parameters(W, b, grad_W, grad_b, learning_rate):
    updated_W = W - learning_rate * grad_W
    updated_b = b - learning_rate * grad_b
    return updated_W, updated_b

# Function to split data into training and validation sets
def train_validation_split(images, labels, validation_ratio=0.2):
    """
    Splits the dataset into training and validation sets.

    Parameters:
    images : numpy.ndarray
        Input data (num_samples x num_features).
    labels : numpy.ndarray
        Corresponding labels (num_samples,).
    validation_ratio : float
        Proportion of data to use for validation (default: 0.2).

    Returns:
    X_train, X_val, y_train, y_val : tuple
        Training and validation splits for images and labels.
    """
    num_samples = images.shape[0]
    indices = np.arange(num_samples)
    np.random.shuffle(indices)

    split_index = int(num_samples * (1 - validation_ratio))
    train_indices = indices[:split_index]
    val_indices = indices[split_index:]

    X_train, X_val = images[train_indices], images[val_indices]
    y_train, y_val = labels[train_indices], labels[val_indices]

    return X_train, X_val, y_train, y_val

# Function to compute accuracy
def compute_accuracy(y_true, y_pred):
    """
    Compute classification accuracy.

    Parameters:
    y_true : numpy.ndarray
        True labels.
    y_pred : numpy.ndarray
        Predicted labels.

    Returns:
    accuracy : float
        Classification accuracy (percentage).
    """
    return np.mean(y_true == y_pred) * 100


def plot_training_losses(train_losses, learning_rates, regularization_strengths, limit_switch):
    """
    Plots the training losses against epochs for different sets of hyperparameters.

    Parameters:
    - train_losses: Dictionary with keys as (learning_rate, regularization_strength)
                    and values as lists of training losses for each epoch.
    - learning_rates: List of learning rates (floats).
    - regularization_strengths: List of regularization strengths (floats).
    """
    # Create a plot for each combination of learning rates and regularization strengths
    plt.figure(figsize=(12, 8))

    for (lr, reg_strength), losses in train_losses.items():
        # Label for each line in the plot
        label = f"LR={lr}, Reg={reg_strength}"
        epochs = range(1, len(losses) + 1)  # x-axis is the epoch numbers
        plt.plot(epochs, losses, label=label)

    if limit_switch == True: plt.ylim(0, 3)

    # Add plot details
    plt.title("Training Loss vs Epochs for Different Hyperparameters")
    plt.xlabel("Epochs")
    plt.ylabel("Training Loss")
    plt.legend(loc="upper right", bbox_to_anchor=(1.3, 1))  # Adjust legend position
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_accuracies(training_accuracies, validation_accuracies, target_lr, target_reg_strength):
    """
    Plots training accuracy and validation accuracy against epochs.

    Parameters:
    - training_accuracies: List of training accuracy values over epochs.
    - validation_accuracies: List of validation accuracy values over epochs.
    - target_lr: The learning rate used for training (for labeling).
    - target_reg_strength: The regularization strength used for training (for labeling).
    """
    # Create a plot
    epochs = range(1, len(training_accuracies) + 1)  # Epoch numbers (1-based index)
    plt.figure(figsize=(10, 6))

    # Plot training and validation accuracies
    plt.plot(epochs, training_accuracies, label="Training Accuracy")
    plt.plot(epochs, validation_accuracies, label="Validation Accuracy")

    # Add labels, title, and legend
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy (%)")
    plt.title(f"Training and Validation Accuracy vs Epochs\n(LR={target_lr}, Reg={target_reg_strength})")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()

    # Show the plot
    plt.show()

# Paths to your local MNIST files
train_images_path = "data/train-images-idx3-ubyte"
train_labels_path = "data/train-labels-idx1-ubyte"

# Load the dataset
train_images = load_mnist_images(train_images_path)
train_labels = load_mnist_labels(train_labels_path)

# Prepare the images
train_images = prepare_images(train_images)

# Extract images and labels for digits two digits
# Digit pairs to be compared are (4,9), (4,6), (0,1) and (2,7)
digit_pair = (4,6)
images_pair, labels_pair = extract_digits(train_images, train_labels, digit_pair)

# Initialize dimensional parameters
num_classes = 2  # Binary classification for digit pair (e.g., 4 and 9)
num_features = images_pair.shape[1]  # Number of features (784 for MNIST)

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_validation_split(images_pair, labels_pair)

# Convert training labels to one-hot encoding
num_train_samples = y_train.shape[0]
y_train_one_hot = np.zeros((num_train_samples, num_classes))
y_train_one_hot[np.arange(num_train_samples), y_train] = 1

# Hyperparameter grid
learning_rates = [0.01, 0.05, 0.1, 0.5]
regularization_strengths = [0.001, 0.01]

# Variables to store best and worst results
best_accuracy = 0
best_params = {}
best_W, best_b = None, None
worst_accuracy = 100
worst_params = {}

## Variables to store result parameters
# training losses at every epoch and validation accuracies for all hyperparameters
train_losses = {}
val_accuracies = {}
# Initialize lists to store training and validation
# accuracies for a specific hyperparameter combination
target_t_accuracies = []
target_v_accuracies = []
# Known best learning rate and regularization strength based on prior code iterations
target_lr = 0.5
target_reg_strength = 0.001

# Hyperparameter tuning
for lr in learning_rates:
    for reg_strength in regularization_strengths:
        # Initialize weights and biases
        W, b = initialize_weights_xavier(num_features, num_classes)

        # Check if the current combination matches the target hyperparameters
        track_accuracies = (lr == target_lr and reg_strength == target_reg_strength)

        # Check if the key (lr, reg_strength) exists in the dictionary, if not initialize it
        if (lr, reg_strength) not in train_losses:
            train_losses[(lr, reg_strength)] = []

        # Training loop (1000 epochs)
        # epoch = iterations
        for epoch in range(100):
            grad_W, grad_b = compute_gradients(X_train, y_train_one_hot, W, b, reg_strength)
            W, b = optimize_parameters(W, b, grad_W, grad_b, lr)

            # Compute training loss
            train_loss = compute_loss(X_train, y_train_one_hot, W, b, reg_strength)
            train_predictions = np.argmax(np.dot(X_train, W.T) + b, axis=1)
            train_accuracy = compute_accuracy(y_train, train_predictions)

            # Compute validation accuracy
            val_logits = np.dot(X_val, W.T) + b
            val_predictions = np.argmax(val_logits, axis=1)
            val_accuracy = compute_accuracy(y_val, val_predictions)

            # Append training loss for results display
            train_losses[(lr, reg_strength)].append(train_loss)

            # If this is the target combination, store accuracies
            if track_accuracies:
                target_t_accuracies.append(train_accuracy)
                target_v_accuracies.append(val_accuracy)

            print(f"Epoch {epoch+1}: Training Loss = {train_losses[(lr, reg_strength)][-1]:.4f}")

        # Validation predictions
        val_logits = np.dot(X_val, W.T) + b
        val_predictions = np.argmax(val_logits, axis=1)

        # Compute validation accuracy
        val_accuracy = compute_accuracy(y_val, val_predictions)

        # Check if the key (lr, reg_strength) exists in the dictionary, if not initialize it
        if (lr, reg_strength) not in val_accuracies:
            val_accuracies[(lr, reg_strength)] = []

        val_accuracies[(lr, reg_strength)].append(val_accuracy)

        print(f"LR: {lr}, Reg Strength: {reg_strength}, Validation Accuracy: {val_accuracy:.2f}%")

        # Update best parameters if accuracy improves
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_params = {"learning_rate": lr, "regularization_strength": reg_strength}
            best_W, best_b = W, b

        if val_accuracy < worst_accuracy:
            worst_accuracy = val_accuracy
            worst_params = {"learning_rate": lr, "regularization_strength": reg_strength}

# Print the best hyperparameters
print(f"Best Weight and Bias parameters: {best_W} and {best_b}")
print("Best Hyperparameters:")
print(f"Learning Rate: {best_params['learning_rate']}")
print(f"Regularization Strength: {best_params['regularization_strength']}")
print(f"Best Validation Accuracy: {best_accuracy:.2f}%")
print("Worst Hyperparameters:")
print(f"Learning Rate: {worst_params['learning_rate']}")
print(f"Regularization Strength: {worst_params['regularization_strength']}")
print(f"Worst Validation Accuracy: {worst_accuracy:.2f}%")

# Plot training losses against epochs for different hyperparameters
plot_training_losses(train_losses, learning_rates, regularization_strengths, False)
plot_training_losses(train_losses, learning_rates, regularization_strengths, True)
plot_accuracies(target_t_accuracies, target_v_accuracies, target_lr, target_reg_strength)

# Display first 100 images with predicted and true labels using the best parameters
val_logits = np.dot(X_val[:100], best_W.T) + best_b
val_predictions = np.argmax(val_logits, axis=1)

# Dynamic text size: Adjust based on grid size and figure size
grid_size = 10  # 10x10 grid
figsize = 20  # Figure size (20x20)
text_size = figsize / grid_size * 3  # Increase the scaling factor for larger text

print("\nFirst 100 Validation Results:")
plt.figure(figsize=(figsize, figsize))  # Adjust figure size
for i in range(100):
    plt.subplot(grid_size, grid_size, i + 1)  # Arrange in a 10x10 grid
    plt.imshow(X_val[i].reshape(28, 28), cmap="gray")
    # Text position and color
    text_color = 'lime' if val_predictions[i] == y_val[i] else 'red'  # Green for correct, red for incorrect
    plt.text(2, 25, f"P:{val_predictions[i]} T:{y_val[i]}", color=text_color, fontsize=text_size,
             bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))  # Embedded text with black background
    plt.axis("off")
plt.tight_layout()
plt.show()

