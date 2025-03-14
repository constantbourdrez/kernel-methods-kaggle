from classifiers import SVMC
from kernels import SpectrumKernel
from utils import *
import argparse
import pickle
import numpy as np

class Ensemble_SVM:
    def __init__(self, classifiers, X_train_list, kernel):
        self.classifiers = classifiers
        self.X_train_list = X_train_list
        self.kernel = kernel
    def predict(self, X):
        for i, classifier in enumerate(self.classifiers):
            gram = self.kernel.gram_matrix(X, self.X_train_list[i], n_proc = 7)
            gram = normalize_gram_matrix(gram)
            predictions = classifier.predict_class(gram)
            predictions[predictions == 0] = -1
            if i == 0:
                ensemble_predictions = predictions
            else:
                ensemble_predictions += predictions
        return (ensemble_predictions > 0).astype(int)


def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.02)
    Y_train[Y_train == 0] = -1
    N_values = [3, 5, 7, 10]
    best_N = N_values[0]
    best_accuracy = 0

    kernel = SpectrumKernel(alphabet='ACGT', n=args.n, use_mismatch=True, m=1)
    svm = SVMC(c=args.c, min_sv=args.min_sv)

    for N in N_values:
        sampled_classifiers = []
        training_samples = []
        print(f"Training Ensemble with N={N}")
        indices = np.random.choice(len(X_train), (N, int(0.9*len(X_train))), replace=True)
        for i in range(N):
            valid_indices = np.setdiff1d(np.arange(len(X_train)), indices[i])
            X_train_sample = X_train[indices[i]]
            Y_train_sample = Y_train[indices[i]]
            X_valid_sample = X_train[valid_indices]
            Y_valid_sample = Y_train[valid_indices]
            gram_train = kernel.gram_matrix(X_train_sample, X_train_sample, n_proc=7)
            gram_valid = kernel.gram_matrix(X_valid_sample, X_train_sample, n_proc=7)
            svm.fit(gram_train, Y_train_sample)
            sampled_classifiers.append(svm)
            training_samples.append(X_train_sample)
            y_hat = svm.predict_class(gram_valid)
            Y_valid_sample[Y_valid_sample == -1] = 0
            accuracy = accuracy_score(Y_valid_sample, y_hat)
            print(f"Trained SVM with Accuracy: {accuracy}")

        ensemble_svm = Ensemble_SVM(sampled_classifiers, training_samples, kernel)
        ensemble_predictions = ensemble_svm.predict(X_valid)
        valid_accuracy = accuracy_score(Y_valid, ensemble_predictions)
        print(f"Ensemble with N={N} Test Accuracy: {valid_accuracy}")

        if valid_accuracy > best_accuracy:
            best_accuracy = valid_accuracy
            best_N = N
            best_ensemble = ensemble_svm

    print(f"Best N found: {best_N} with Accuracy: {best_accuracy}")

    with open("ensemble_model.pkl", "wb") as f:
        pickle.dump(best_ensemble, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Ensemble Training Script with Bagging")
    parser.add_argument('--n', type=int, default=7, help="Length of the substring for spectrum kernel")
    parser.add_argument('--c', type=float, default=0.1, help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, default=0.01, help="Minimum support vector threshold")
    args = parser.parse_args()
    main(args)
