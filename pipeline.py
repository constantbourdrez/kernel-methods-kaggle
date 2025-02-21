from classifiers import SVMC, KernelLogisticRegression
from kernels import SpectrumKernel, LinearKernel, StringKernel,  HMMFisherKernel, RBFKernel, LocalAlignmentKernel
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
    elif args.kernel_type == 'spectrum':
        kernel = SpectrumKernel(alphabet='ACGT', n = 7)
        gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
        gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)
    elif args.kernel_type == 'mismatch':
            kernel = SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True)
            print(X_train_one_hot.shape, X_train.shape)
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
            gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)
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
    else:
        raise ValueError(f"Invalid method: {args.method}")
    print(accuracy_score(y_val, y_pred))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--method', type=str, required=True, help="Optimization method (svm or logistic)")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (linear, spectrum, string, rbf, local, hmm)")
    parser.add_argument('--c', type=float, default=0.5, help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, default=0.1, help="Minimum support vector threshold")
    args = parser.parse_args()

    main(args)
