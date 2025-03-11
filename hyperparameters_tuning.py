from classifiers import SVMC, KernelLogisticRegression
from kernels import SpectrumKernel, LinearKernel, FisherKernel, RBFKernel, LocalAlignmentKernel, CL_Kernel
from utils import *
import argparse
import json


def main(args):
    X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
    if args.kernel_type in ['spectrum', 'mismatch', 'cl']:
        dict_params = {"c": [], "min_sv": [], "n": [], "accuracy": [], "m": []}
    elif args.kernel_type == 'rbf':
        dict_params = {"c": [], "min_sv": [], "gamma": [], "accuracy": []}
    elif args.kernel_type == 'linear':
        dict_params = {"c": [], "min_sv": [], "accuracy": []}
    elif args.kernel_type == 'local_alignment':
        dict_params = {"c": [], "min_sv": [], "beta":[], "accuracy": []}
    elif args.kernel_type == 'fischer':
        dict_params = {"c": [], "min_sv": [], "k":[], "accuracy": []}
    else:
        raise ValueError(f"Invalid kernel type: {args.kernel_type}")
    Y_train[Y_train == 0] = -1
    for c in args.c:
        for min_sv in args.min_sv:
            for n in args.n:
                for gamma in args.gamma:
                    for m in args.m:
                        for beta in args.beta:
                            for k in args.k:
                                if args.kernel_type == 'spectrum':
                                    kernel = SpectrumKernel(alphabet='ACGT', n = n)
                                    n_proc = 8
                                elif args.kernel_type == 'mismatch':
                                    kernel = SpectrumKernel(alphabet='ACGT', n=n, use_mismatch=True, m = m)
                                    n_proc = 8
                                elif args.kernel_type == 'cl':
                                    kernel_1 = SpectrumKernel(alphabet='ACGT', n = n)
                                    kernel_2 = SpectrumKernel(alphabet='ACGT', n=n, use_mismatch=True, m = m)
                                    kernel = CL_Kernel(kernel_1, kernel_2)
                                    n_proc = 8
                                elif args.kernel_type =='rbf':
                                    kernel = RBFKernel(gamma=gamma)
                                    n_proc = None
                                    X_train = X_train_matrix
                                elif args.kernel_type == 'linear':
                                    kernel = LinearKernel()
                                    n_proc = None
                                    X_train = X_train_matrix
                                elif args.kernel_type == "local_alignment":
                                    kernel = LocalAlignmentKernel(beta=args.beta)
                                    n_proc = 4
                                elif args.kernel_type == 'fischer':
                                    kernel = FisherKernel(k = k, normalize=True)
                                    n_proc = 'fischer'
                                else:
                                    raise ValueError(f"Invalid kernel type: {args.kernel_type}")
                                accuracy = cross_validation(X_train, Y_train, kernel, SVMC(c=c, min_sv=min_sv), n_folds= 2, n_proc=n_proc)
                                dict_params["c"].append(c)
                                dict_params["min_sv"].append(min_sv)
                                if args.kernel_type in ['spectrum', 'mismatch', 'cl']:
                                    dict_params["n"].append(n)
                                if args.kernel_type == 'rbf':
                                    dict_params["gamma"].append(gamma)
                                if args.kernel_type in [ 'mismatch', 'cl']:
                                    dict_params["m"].append(m)
                                if args.kernel_type == 'local_alignment':
                                    dict_params["beta"].append(beta)

                                if args.kernel_type == 'fischer':
                                    dict_params["k"].append(k)
                                dict_params["accuracy"].append(accuracy)
                                print(f"Accuracy: {accuracy} with c={c}, n={n}")

    with open(f"grid_search_{args.kernel_type}.json", 'w') as f:
        json.dump(dict_params, f)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Training Script")
    parser.add_argument('--n', type=int, nargs='+', default=[1], help="Length of the substring for spectrum kernel")
    parser.add_argument('--c', type=float, nargs='+', default=[0.5], help="Regularization parameter for SVM")
    parser.add_argument('--min_sv', type=float, nargs='+', default=[0.1], help="Minimum support vector threshold")
    parser.add_argument('--kernel_type', type=str, required=True, help="Kernel type (spectrum, mismatch)")
    parser.add_argument('--gamma', type=float, nargs='+', default=[0.1], help="Gamma parameter for RBF kernel")
    parser.add_argument('--m', type=int, nargs='+', default=[1], help="Number of mismatches for mismatch kernel")
    parser.add_argument('--beta',  type=float, nargs='+', default=[0.5], help="Beta parameter for local alignment kernel")
    parser.add_argument('--k',  type=int, nargs='+', default=[3], help="k parameter for fischer kernel")
    args = parser.parse_args()

    main(args)
