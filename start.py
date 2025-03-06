from classifiers import SVMC, KernelLogisticRegression
from kernels import SpectrumKernel, LinearKernel,  RBFKernel, LocalAlignmentKernel, FisherKernel, CL_Kernel
from utils import *
from ensemble_method import Ensemble_SVM
import pickle
import argparse

def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    # Transform data to one-hot encoding
    X_train_one_hot = transform_data_to_one_hot(X_train)
    X_test_one_hot = transform_data_to_one_hot(X_test)

    if args.kernel_type == 'linear':
        kernel = LinearKernel()
    elif args.kernel_type == 'spectrum':
        kernel = SpectrumKernel(alphabet='ACGT', n=7)
        if args.method != 'ensemble':
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
            gram_test = kernel.gram_matrix(X_test, X_train, n_proc=4)
    elif args.kernel_type == 'mismatch':
        kernel = SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True)
        if args.method != 'ensemble':
            gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
            gram_test = kernel.gram_matrix(X_test, X_train, n_proc=4)
    elif args.kernel_type == 'cl':
        kernel = CL_Kernel(SpectrumKernel(alphabet='ACGT', n=7), SpectrumKernel(alphabet='ACGT', n=7, use_mismatch=True))
        gram_train = kernel.gram_matrix(X_train, X_train, n_proc=8)
        gram_test = kernel.gram_matrix(X_test, X_train, n_proc=8)
    else:
        raise ValueError(f"Invalid kernel type: {args.kernel_type}")

    if args.method == 'svm':
        svm = SVMC(c=args.c, min_sv=args.min_sv)
        print('Fitting SVM')
        Y_train[Y_train == 0] = -1
        svm.fit(gram_train, Y_train)
        print('Predicting SVM')
        y_pred = svm.predict_class(gram_test)
    elif args.method == 'logistic':
        logistic_reg = KernelLogisticRegression()
        print('Fitting Logistic Regression')
        logistic_reg.fit(gram_train, Y_train)
        print('Predicting Logistic Regression')
        y_pred = logistic_reg.predict_class(gram_test)
    elif args.method == 'ensemble':
        ensemble_svm = Ensemble_SVM([], [], kernel)
        ensemble_svm = pickle.load(open('ensemble_model.pkl', 'rb'))
        y_pred = ensemble_svm.predict(X_test)
    else:
        raise ValueError(f"Invalid method: {args.method}")



    with open("Yte.csv", 'w') as f:
        f.write('Id,Bound\n')
        for i in range(len(y_pred)):
            f.write(str(i)+','+str(y_pred[i])+'\n')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--method', type=str, required=True, help="Optimization method (svm or logistic)")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (linear, spectrum, string, rbf, local, hmm)")
    parser.add_argument('--c', type=float, default=0.5, help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, default=0.1, help="Minimum support vector threshold")
    args = parser.parse_args()

    main(args)
