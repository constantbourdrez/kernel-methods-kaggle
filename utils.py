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

# Accuracy computation
def accuracy_score(y_true, y_pred):
    """Compute accuracy score for classification."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    correct = np.sum(y_true == y_pred)
    return correct / len(y_true)

# Train-test split
def train_test_split(*arrays, test_size=0.5, not_random=False):
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

def normalize_gram_matrix(gram_matrix):
    """Normalize a non-square Gram matrix."""
    row_norms = np.linalg.norm(gram_matrix, axis=1, keepdims=True)
    col_norms = np.linalg.norm(gram_matrix, axis=0, keepdims=True)
    return gram_matrix / (row_norms @ col_norms)


def cross_validation(X, Y, kernel, classifier, n_folds=5, n_proc = 8):
    """Perform cross-validation for a given classifier using only numpy."""
    num_samples = len(X)
    fold_size = int(0.2 * num_samples)
    indices = np.arange(num_samples)
    np.random.shuffle(indices)


    val_folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(n_folds)]
    train_folds = [np.setdiff1d(indices, val_fold).tolist() for val_fold in val_folds]
    folds = list(zip(train_folds, val_folds))
    scores = []

    for i in range(n_folds):
        print(f"Fold {i}: Train size = {len(folds[i][0])}, Validation size = {len(folds[i][1])}")
        # Split into training and validation sets
        train_indices, val_indices = folds[i]
        X_train, Y_train = X[train_indices], Y[train_indices]
        X_val, Y_val = X[val_indices], Y[val_indices]
        Y_val[Y_val == -1] = 0
        if n_proc != None:
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=n_proc)
            gram_val = kernel.gram_matrix(X_val, X_train, n_proc=n_proc)
        else:
            gram_train = kernel.gram_matrix(X_train, X_train)
            gram_val = kernel.gram_matrix(X_val, X_train)
        classifier.fit(gram_train, Y_train)
        Y_pred = classifier.predict_class(gram_val)
        scores.append(accuracy_score(Y_val, Y_pred))
    return np.mean(scores)
