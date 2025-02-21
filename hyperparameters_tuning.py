from classifiers import SVMC, KernelLogisticRegression
from kernels import SpectrumKernel, LinearKernel, StringKernel,  HMMFisherKernel, RBFKernel, LocalAlignmentKernel
from utils import *
import argparse
import json

def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    X_train, X_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.1)
    dict_params = {"c": [], "min_sv": [], "n": [], "accuracy": []}
    y_train[y_train == 0] = -1
    for c in args.c:
        for min_sv in args.min_sv:
            for n in args.n:
                if args.kernel_type == 'spectrum':
                    kernel = SpectrumKernel(alphabet='ACGT', n = n)
                    gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
                    gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)
                elif args.kernel_type == 'mismatch':
                    kernel = SpectrumKernel(alphabet='ACGT', n=n, use_mismatch=True)
                    gram_train = kernel.gram_matrix(X_train, X_train, n_proc=4)
                    gram_val = kernel.gram_matrix(X_val, X_train, n_proc=4)
                else:
                    raise ValueError(f"Invalid kernel type: {args.kernel_type}")
                svm = SVMC(c=c, min_sv=min_sv)
                svm.fit(gram_train, y_train)
                y_pred = svm.predict_class(gram_val)
                accuracy = accuracy_score(y_val, y_pred)
                dict_params["c"].append(c)
                dict_params["min_sv"].append(min_sv)
                dict_params["n"].append(n)
                dict_params["accuracy"].append(accuracy)
                print(f"Accuracy: {accuracy} with c={c}, min_sv={min_sv}, n={n}")

    with open(f"grid_search_{args.kernel_type}.json", 'w') as f:
        json.dump(dict_params, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--n', type=int, nargs='+', default=[1], help="Length of the substring for spectrum kernel")
    parser.add_argument('--c', type=float, nargs='+', default=[0.5], help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, nargs='+', default=[0.1], help="Minimum support vector threshold")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (spectrum, mismatch)")
    args = parser.parse_args()

    main(args)
