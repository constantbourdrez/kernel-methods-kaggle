from classifiers import SVMC, KernelLogisticRegression, K_means
from kernels import SpectrumKernel, LinearKernel, FisherKernel, RBFKernel, LocalAlignmentKernel, CL_Kernel
from utils import *
import argparse

def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    # Transform data to one-hot encoding
    X_train_one_hot = transform_data_to_one_hot(X_train)
    if args.kernel_type == 'linear':
        kernel = LinearKernel()
        n_proc = None
        X_train = X_train_matrix


    elif args.kernel_type == 'spectrum':
        kernel = SpectrumKernel(alphabet='ACGT', n = 7)
        n_proc = 8

    elif args.kernel_type == 'mismatch':
            kernel = SpectrumKernel(alphabet='ACGT', n=6, use_mismatch=True, m=1)
            n_proc = 7


    elif args.kernel_type == "local_alignment":
        kernel = LocalAlignmentKernel(beta=1)
        n_proc = 8


    elif args.kernel_type == 'rbf':
        kernel = RBFKernel(gamma=args.gamma)
        n_proc = None
        X_train = X_train_matrix

    elif args.kernel_type == 'fischer':
        kernel = FisherKernel(k=3, normalize=True)
        A, pi_0, pi_fin, p= kernel.intialize_parameters(observation_size=64)
        # Train the kernel using the EM_HMM method (or similar training process)
        A, pi_0, pi_fin, p, loss = kernel.EM_HMM(X_train, kernel.k, A, pi_0, pi_fin, p, n_iter=40)
        # Compute the Gram matrix
        n_proc = None

    elif args.kernel_type == 'cl':
        kernel = CL_Kernel(SpectrumKernel(alphabet='ACGT', n=7), SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True))
        n_proc = 8

    else:
        raise ValueError(f"Invalid kernel type: {args.kernel_type}")

    if args.method == 'svm':
        svm = SVMC(c=args.c, min_sv=args.min_sv)
        print('Fitting SVM')
        Y_train[Y_train == 0] = -1
        accuracy = cross_validation(X_train, Y_train, kernel, svm, n_folds= 5, n_proc = n_proc)

    elif args.method == 'logistic':
        logistic_reg = KernelLogisticRegression()
        print('Fitting Logistic Regression')
        accuracy = cross_validation(X_train, Y_train, kernel, logistic_reg, n_folds= 5, n_proc = n_proc)

    elif args.method == 'k_means':
        if args.kernel_type == 'spectrum' or args.kernel_type == 'mismatch' or args.kernel_type == 'cl' or args.kernel_type == 'local_alignment':
            n_proc = 8
        k_means = K_means(k = 70, max_iter=30)
        print('Fitting K-means')
        accuracy = cross_validation(X_train, Y_train, kernel, k_means, n_folds= 5, n_proc = n_proc)
    else:
        raise ValueError(f"Invalid method: {args.method}")

    print(f"Accuracy: {accuracy} for kernel type: {args.kernel_type} and method: {args.method}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--method', type=str, required=True, help="Optimization method (svm, logistic, k-mean)")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (linear, spectrum, string, rbf, local, hmm)")
    parser.add_argument('--c', type=float, default=0.5, help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, default=0.2, help="Minimum support vector threshold")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for RBF kernel")
    args = parser.parse_args()

    main(args)
