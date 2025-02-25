from classifiers import SVMC, KernelLogisticRegression, K_means
from kernels import SpectrumKernel, LinearKernel, FisherKernel, RBFKernel, LocalAlignmentKernel, CL_Kernel
from utils import *
import argparse

def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    X_train, X_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.1)
    # Transform data to one-hot encoding
    X_train_one_hot = transform_data_to_one_hot(X_train)
    X_val_one_hot = transform_data_to_one_hot(X_val)
    if args.kernel_type == 'linear':
        kernel = LinearKernel()
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train_one_hot, X_train_one_hot)
            gram_val = kernel.gram_matrix(X_val_one_hot, X_train_one_hot)

    elif args.kernel_type == 'spectrum':
        kernel = SpectrumKernel(alphabet='ACGT', n = 7)
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
            gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)

    elif args.kernel_type == 'mismatch':
            kernel = SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True)
            if args.method != 'k_means':
                gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
                gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)

    elif args.kernel_type == "local_alignment":
        kernel = LocalAlignmentKernel()
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc = 8)
            gram_val = kernel.gram_matrix(X_val, X_train, n_proc = 8)

    elif args.kernel_type == 'rbf':
        kernel = RBFKernel(gamma=args.gamma)
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train_one_hot, X_train_one_hot)
            gram_val = kernel.gram_matrix(X_val_one_hot, X_train_one_hot)

    elif args.kernel_type == 'fischer':
        kernel = FisherKernel(k=3, normalize=True)
        A, pi_0, pi_fin, p= kernel.intialize_parameters(observation_size=64)
        # Train the kernel using the EM_HMM method (or similar training process)
        A, pi_0, pi_fin, p, loss = kernel.EM_HMM(X_train, kernel.k, A, pi_0, pi_fin, p, n_iter=40)
        # Compute the Gram matrix
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train, X_train, A, pi_0, pi_fin, p)
            epsilon = 1e-5  # Regularization term
            gram_train += epsilon * np.eye(gram_train.shape[0])
            gram_val = kernel.gram_matrix(X_val, X_train, A, pi_0, pi_fin, p)

    elif args.kernel_type == 'cl':
        kernel = CL_Kernel(SpectrumKernel(alphabet='ACGT', n=7), SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True))
        if args.method != 'k_means':
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=8)
            gram_val = kernel.gram_matrix(X_val, X_train, n_proc=8)
    else:
        raise ValueError(f"Invalid kernel type: {args.kernel_type}")

    if args.method == 'svm':
        svm = SVMC(c=args.c, min_sv=args.min_sv)
        print('Fitting SVM')
        y_train[y_train == 0] = -1
        svm.fit(gram_train, y_train)
        print('Predicting SVM')
        y_pred = svm.predict_class(gram_val)

    elif args.method == 'logistic':
        logistic_reg = KernelLogisticRegression()
        print('Fitting Logistic Regression')
        logistic_reg.fit(gram_train, y_train)
        print('Predicting Logistic Regression')
        y_pred = logistic_reg.predict_class(gram_val)
        
    elif args.method == 'k_means':
        if args.kernel_type == 'spectrum' or args.kernel_type == 'mismatch' or args.kernel_type == 'cl' or args.kernel_type == 'local_alignment':
            n_proc = 8
        else:
            n_proc = False
        k_means = K_means(k = 2, max_iter=30, n_proc=n_proc)
        k_means.fit(X_train, y_train, kernel)
        y_pred = k_means.predict(X_train, X_val, kernel)
    else:
        raise ValueError(f"Invalid method: {args.method}")
    print(accuracy_score(y_val, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--method', type=str, required=True, help="Optimization method (svm, logistic, k-mean)")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (linear, spectrum, string, rbf, local, hmm)")
    parser.add_argument('--c', type=float, default=0.5, help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, default=0.1, help="Minimum support vector threshold")
    parser.add_argument("--gamma", type=float, default=1.0, help="Gamma parameter for RBF kernel")
    args = parser.parse_args()

    main(args)
