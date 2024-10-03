import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


'''def generate_data_linearly_separable(n: int, dim: int, std: float, with_noise = None):
    """
    Generate linearly separable data.

    Parameters:
        n (int): Number of samples.
        dim (int): Dimensionality of the data.
        std (float): Standard deviation for the data distribution.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the generated data,
            corresponding labels, and the separating hyperplane weights.
    """
    np.random.seed(0)

    # w_spartor : (1, dim+1)
    w_spartor = np.random.randn(1, dim + 1)  # Shape (1, dim+1)

    data = np.random.randn(n, dim)    # (n, dim)
    data = std * data
    
    ones = np.ones((n,1))             # (n, 1)

    x = np.concatenate((data, ones), axis=1)            # (n, dim+1)

    # Calculate the scalar product
    # Note: The result will have shape (n, 1) since we are multiplying (1, dim+1) with (n, dim+1)
    scalar_product = np.dot(x, w_spartor.T)  # (n,1)

    # Labeling the values: positive -> 1, negative -> -1
    labels = np.where(scalar_product > 0, 1, -1).astype(int)

    if with_noise is not None:
        num = int(with_noise * n)
        for i in range(num):
            idx = np.random.randint(0, n)
            labels[idx] *= -1

    return data, labels, w_spartor'''



def generate_data_linearly_separable(n: int, dim: int, std: float, with_noise = None, normalize=False):
    """
    Generate linearly separable data with optional normalization.

    Parameters:
        n (int): Number of samples.
        dim (int): Dimensionality of the data.
        std (float): Standard deviation for the data distribution.
        with_noise (float, optional): Percentage of labels to flip as noise.
        normalize (bool, optional): If True, normalize the data across all dimensions.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: A tuple containing the generated data,
            corresponding labels, and the separating hyperplane weights.
    """
    np.random.seed(0)

    # w_spartor : (1, dim+1)
    w_spartor = np.random.randn(1, dim + 1)  # Shape (1, dim+1)

    data = np.random.randn(n, dim)  # (n, dim)
    data = std * data  # Scale by standard deviation

    # Optionally normalize the data across each dimension
    if normalize:
        data -= np.mean(data, axis=0)
        data /= np.std(data, axis=0)

    ones = np.ones((n, 1))  # (n, 1)

    x = np.concatenate((data, ones), axis=1)  # (n, dim+1)

    # Calculate the scalar product
    scalar_product = np.dot(x, w_spartor.T)  # (n,1)

    # Labeling the values: positive -> 1, negative -> -1
    labels = np.where(scalar_product > 0, 1, -1).astype(int)

    # Add noise by flipping labels for a certain percentage of data points
    if with_noise is not None:
        num = int(with_noise * n)
        for i in range(num):
            idx = np.random.randint(0, n)
            labels[idx] *= -1

    return data, labels, w_spartor



def plot_linearly_separable_data(data: np.ndarray, labels: np.ndarray, w_spartor: np.ndarray, title = None):
    """
    Plot linearly separable data along with the separating hyperplane.

    Parameters:
        data (np.ndarray): data of shape (n, 2).
        labels (np.ndarray): Corresponding labels of shape (n,).
        w_spartor (np.ndarray): Weights of the separating hyperplane of shape (1, 3).
    """
    plt.figure(figsize=(8, 6))

    # Plot the data points
    plt.scatter(data[np.where(labels == 1)[0]][:, 0], data[np.where(labels == 1)[0]][:, 1], 
                color='blue', label='Class 1', alpha=0.6)
    plt.scatter(data[np.where(labels == -1)[0]][:, 0], data[np.where(labels == -1)[0]][:, 1], 
                color='red', label='Class -1', alpha=0.6)

    # Calculate the line (y = mx + b) for the hyperplane
    x_values = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 100)
    
    # Extract weights for the hyperplane
    w1, w2, b = w_spartor[0, 0], w_spartor[0, 1], w_spartor[0, 2]
    
    # Calculate corresponding y values for the hyperplane
    y_values = -(w1 * x_values + b) / w2

    # Plot the hyperplane
    plt.plot(x_values, y_values, color='green', label='Separating Hyperplane')

    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    if title is None:
        title = 'Linearly Separable Data with Separating Hyperplane'
    plt.title(title)
    plt.axhline(0, color='grey', lw=0.5, ls='--')
    plt.axvline(0, color='grey', lw=0.5, ls='--')
    plt.legend()
    plt.grid()
    plt.show()


def plot_linearly_separable_data_3d(data: np.ndarray, labels: np.ndarray, w_spartor: np.ndarray):
    """
    Plot linearly separable data along with the separating hyperplane in 3D.

    Parameters:
        data (np.ndarray): Data of shape (n, 3).
        labels (np.ndarray): Corresponding labels of shape (n,).
        w_spartor (np.ndarray): Weights of the separating hyperplane of shape (1, 4).
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot the data points
    ax.scatter(data[np.where(labels == 1)[0]][:, 0], data[np.where(labels == 1)[0]][:, 1], data[np.where(labels == 1)[0]][:, 2], 
               color='blue', label='Class 1', alpha=0.6)
    ax.scatter(data[np.where(labels == -1)[0]][:, 0], data[np.where(labels == -1)[0]][:, 1], data[np.where(labels == -1)[0]][:, 2], 
               color='red', label='Class -1', alpha=0.6)

    # Create a grid for the hyperplane
    x_values = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 10)
    y_values = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 10)
    X, Y = np.meshgrid(x_values, y_values)

    # Calculate corresponding Z values for the hyperplane
    w1, w2, w3, b = w_spartor[0, 0], w_spartor[0, 1], w_spartor[0, 2], w_spartor[0, 3]
    Z = -(w1 * X + w2 * Y + b) / w3

    # Plot the hyperplane
    ax.plot_surface(X, Y, Z, color='green', alpha=0.5, label='Separating Hyperplane')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_zlabel('Feature 3')
    ax.set_title('Linearly Separable Data with Separating Hyperplane in 3D')
    ax.legend()
    plt.show()

