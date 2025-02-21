import os
import random
import numpy as np
import pandas as pd
from classifiers import SVMC

def get_root_directory():
    """Determine the root directory containing the data."""
    return '.' if 'data' in os.listdir('.') else '..'



def load_csv(filepath, sep=',', header=None, index_col=None, as_list=False):
    """Load a CSV file and return its values as a NumPy array or list."""
    data = pd.read_csv(filepath, sep=sep, header=header, index_col=index_col).values
    return data.tolist() if as_list else data
# Load datasets
def load_datasets():
    X_train = [load_csv(f"data/Xtr{i}.csv", as_list=True) for i in range(3)]
    X_train_matrix = [load_csv(f"data/Xtr{i}_mat100.csv", sep=' ') for i in range(3)]
    Y_train = [load_csv(f"data/Ytr{i}.csv", index_col=0) for i in range(3)]
    X_test = [load_csv(f"data/Xte{i}.csv", as_list=True) for i in range(3)]
    X_test_matrix = [load_csv(f"data/Xte{i}_mat100.csv", sep=' ') for i in range(3)]
    # Extract sequences
    X_train = np.array([np.array(x)[:, 1][1:] for x in X_train]).flatten()
    X_test = np.array([np.array(x)[:, 1][1:] for x in X_test]).flatten()
    X_train_matrix = np.array(X_train_matrix).reshape(-1, 100)
    X_test_matrix = np.array(X_test_matrix).reshape(-1, 100)
    Y_train = np.array([y[1:] for y in Y_train]).flatten().astype(int)
    #Y_train[Y_train == 0] = -1
    return X_train, X_train_matrix, Y_train, X_test, X_test_matrix

# One-hot encoding for DNA sequences
def one_hot_encode(letter):
    """Convert a DNA nucleotide (A, C, G, T) into a one-hot vector."""
    mapping = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    return mapping.get(letter, [0, 0, 0, 0])  # Default to all-zero if unknown character

def sequence_to_one_hot(sequence):
    """Convert a DNA sequence into a one-hot vector."""
    return np.array([one_hot_encode(letter) for letter in sequence]).reshape(-1)


def transform_data_to_one_hot(data):
    """Convert a list of DNA sequences into a NumPy array of one-hot vectors."""
    return np.array([sequence_to_one_hot(seq) for seq in data])

# Label encoding for DNA sequences
def sequence_to_label(sequence):
    """Convert a DNA sequence into label encoding."""
    mapping = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    return np.array([mapping[letter] for letter in sequence])

# Patch extraction
def compute_patches(sequence, num_patches):
    """Divide a one-hot encoded sequence into patches."""
    one_hot_vector = sequence_to_one_hot(sequence)
    patch_size = len(one_hot_vector) // num_patches
    remainder = len(one_hot_vector) % num_patches

    if remainder > 0:
        one_hot_vector = np.concatenate((one_hot_vector, np.zeros(num_patches - remainder)))
        patch_size = len(one_hot_vector) // num_patches

    return [one_hot_vector[i * patch_size : (i + 1) * patch_size] for i in range(num_patches)]

# Model selection for SVM
def optimal_svm_params(Gram_train, Gram_val, y_train, y_val, c_values, sv_values):
    """Find the optimal SVM parameters (C, min_sv) by maximizing accuracy."""
    best_c, best_sv, best_score = 1, 1e-4, 0
    for c in c_values:
        for sv in sv_values:
            svm = SVMC(c=c, min_sv=sv)
            svm.fit(Gram_train, y_train)
            y_pred = svm.predict_class(Gram_val).reshape(-1)
            score = accuracy_score(y_val.reshape(-1), y_pred)
            if score > best_score:
                best_c, best_sv, best_score = c, sv, score
    return best_c, best_sv

# Accuracy computation
def accuracy_score(y_true, y_pred):
    """Compute accuracy score for classification."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# Train-test split
def train_test_split(*arrays, test_size=0.5):
    """Split multiple arrays into train and test sets."""
    num_samples = len(arrays[0])
    indices = list(range(num_samples))
    random.shuffle(indices)
    split_point = int(num_samples * (1 - test_size))
    train_indices, test_indices = indices[:split_point], indices[split_point:]

    result = []
    for array in arrays:
        array = np.array(array)
        result.extend([array[train_indices].tolist() if isinstance(array, list) else array[train_indices],
                       array[test_indices].tolist() if isinstance(array, list) else array[test_indices]])
    return result
