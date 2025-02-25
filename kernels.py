import numpy as np
from multiprocessing import Pool
from itertools import product
from collections import defaultdict
from tqdm import tqdm
from utils import *

class LinearKernel:
    def compute(self, x, y):
        return np.dot(x, y)

    def gram_matrix(self, X, Y=None):
        Y = X if Y is None else Y
        return np.dot(X, Y.T)

class RBFKernel:
    def __init__(self, gamma):
        self.gamma = gamma

    def compute(self, x, y):
        return np.exp(-self.gamma * np.linalg.norm(x - y) ** 2)

    def gram_matrix(self, X, Y=None):
        Y = X if Y is None else Y
        X_norm = np.sum(X ** 2, axis=1).reshape(-1, 1)
        Y_norm = np.sum(Y ** 2, axis=1).reshape(1, -1)
        pairwise_sq_dists = X_norm + Y_norm - 2 * np.dot(X, Y.T)
        return np.exp(-self.gamma * pairwise_sq_dists)

class SpectrumKernel:
    def __init__(self, alphabet, n, use_mismatch=False):
        self.alphabet = alphabet
        self.n = n
        self.use_mismatch = use_mismatch
        self.all_combinations = list(product(alphabet, repeat=n))
        self.ngram_to_index = {ngram: idx for idx, ngram in enumerate(self.all_combinations)}

    def ngrams(self, seq):
        """
        Extract n-grams from the sequence using NumPy for efficiency.
        """
        if not isinstance(seq, str):
            raise ValueError(f"Expected a string sequence, got {type(seq)}: {seq}")

        seq = np.array(list(seq))  # Convert string to character array
        if len(seq) < self.n:
            return []
        return np.lib.stride_tricks.sliding_window_view(seq, self.n).tolist()


    def create_histogram(self, seq):
        """
        Create a histogram of n-grams for a given sequence.
        """

        histogram = np.zeros(len(self.all_combinations), dtype=np.float32)
        for ngram in self.ngrams(seq):
            index = self.ngram_to_index.get(tuple(ngram))
            if index is not None:
                histogram[index] += 1
        return histogram

    def create_histogram_mismatch(self, seq):
        """
        Create a histogram of n-grams considering mismatches.
        """
        histogram = self.create_histogram(seq)
        for ngram in self.ngrams(seq):
            for i, char in enumerate(ngram):
                for letter in self.alphabet:
                    if letter != char:
                        modified_ngram = list(ngram)
                        modified_ngram[i] = letter
                        index = self.ngram_to_index.get(tuple(modified_ngram))
                        if index is not None:
                            histogram[index] += 0.1  # Mismatch penalty
        histogram /= np.linalg.norm(histogram) + 1e-6  # Normalize
        return histogram

    def create_histogram_for_seq(self, seq):
        """
        Select histogram type based on `use_mismatch` flag.
        """
        return self.create_histogram_mismatch(seq) if self.use_mismatch else self.create_histogram(seq)

    @staticmethod
    def compute_idf(histograms):
        """
        Compute inverse document frequency (IDF) for the histograms.
        """
        idf = np.sum(histograms, axis=0)
        return np.maximum(1, np.log10(len(histograms) / (idf + 1e-6)))

    @staticmethod
    def kernel(x1, x2):
        """
        Compute the kernel (similarity) between two histograms.
        """
        return np.dot(x1, x2)

    def compute_self_similarity(self, seq):
        """
        Compute self-similarity for normalization in the Gram matrix.
        """
        hist = self.create_histogram_for_seq(seq)
        return self.kernel(hist, hist) + 1e-6  # Avoid division by zero

    def gram_matrix(self, X, Y=None, n_proc=1, verbose = False):
        """
        Compute the Gram matrix for a set of sequences.
        """
        Y = X if Y is None else Y
        len_X, len_Y = len(X), len(Y)
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        if n_proc > 1:
            with Pool(processes=n_proc) as pool:
                if verbose:
                    print('Computing self-similarity')
                X = X.astype(str)
                sim_X = pool.map(self.compute_self_similarity, X)
                sim_Y = pool.map(self.compute_self_similarity, Y) if X is not Y else sim_X
                if verbose:
                    print('Computing histograms')
                X_hist = np.array([self.create_histogram_for_seq(seq) for seq in X])
                Y_hist = np.array([self.create_histogram_for_seq(seq) for seq in Y])
                gram_matrix = np.dot(X_hist, Y_hist.T) / np.sqrt(np.outer(sim_X, sim_Y))
                if verbose:
                    print('Done')
        else:
            sim_X = [self.compute_self_similarity(seq) for seq in X]
            sim_Y = [self.compute_self_similarity(seq) for seq in Y] if X is not Y else sim_X

            for i in range(len_X):
                for j in range(len_Y):
                    gram_matrix[i, j] = self.kernel(
                        self.create_histogram_for_seq(X[i]),
                        self.create_histogram_for_seq(Y[j])
                    ) / np.sqrt(sim_X[i] * sim_Y[j])

        return gram_matrix

class LocalAlignmentKernel:
    def __init__(self, match_score=1, mismatch_penalty=-1, gap_penalty=-1):
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def _compute_score(self, seq1, seq2):
        """Compute local alignment score using NumPy (vectorized)."""
        len_seq1, len_seq2 = len(seq1), len(seq2)
        score_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1), dtype=np.float32)

        match_matrix = np.array([
            [self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty
             for j in range(1, len_seq2 + 1)]
            for i in range(1, len_seq1 + 1)
        ], dtype=np.float32)

        for i in range(1, len_seq1 + 1):
            for j in range(1, len_seq2 + 1):
                scores = np.array([
                    score_matrix[i-1, j-1] + match_matrix[i-1, j-1],  # Match/Mismatch
                    score_matrix[i-1, j] + self.gap_penalty,  # Gap in seq2
                    score_matrix[i, j-1] + self.gap_penalty,  # Gap in seq1
                    0  # Local alignment (no alignment)
                ])
                score_matrix[i, j] = np.max(scores)

        return np.max(score_matrix)  # Return max local alignment score

    def _compute_score_parallel(self, args):
        """Wrapper for parallel computation."""
        seq1, seq2 = args
        return self._compute_score(seq1, seq2)

    def gram_matrix(self, X, Y=None, n_proc=4):
        """
        Compute the Gram matrix using parallel processing.

        Parameters:
        - X: List of sequences.
        - Y: List of sequences (optional, default is X).
        - n_jobs: Number of processes for parallel execution.

        Returns:
        - gram_matrix: The computed Gram matrix.
        """
        Y = X if Y is None else Y
        len_X, len_Y = len(X), len(Y)
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        # Create list of sequence pairs for parallel computation
        seq_pairs = [(X[i], Y[j]) for i in range(len_X) for j in range(len_Y)]

        # Use multiprocessing Pool
        print('computing results')
        with Pool(n_proc) as pool:
            results = pool.map(self._compute_score_parallel, seq_pairs)
        print('results computed')

        # Reshape results back into matrix form
        gram_matrix[:, :] = np.array(results, dtype=np.float32).reshape(len_X, len_Y)

        return gram_matrix


class CL_Kernel():

    def __init__(self, kernel_1, kernel_2):
        self.kernel_1 = kernel_1
        self.kernel_2 = kernel_2

    def gram_matrix(self, X, Y, n_proc=8):
        gram1 = self.kernel_1.gram_matrix(X, Y, n_proc=n_proc)
        gram2 = self.kernel_2.gram_matrix(X, Y, n_proc=n_proc)
        gram1_norm = normalize_gram_matrix(gram1)
        gram2_norm = normalize_gram_matrix(gram2)
        return 1/2 * (gram1_norm + gram2_norm)


class FisherKernel():
    """ Fisher Kernel class """

    def __init__(self, k, normalize):
        self.k = k
        permutations = self.generate_permutations(k)
        self.index = dict(zip(permutations, np.arange(len(permutations))))
        self.normalize = normalize

    def logplus(self, x, y):
        M = np.maximum(x, y)
        m = np.minimum(x, y)
        return M + np.log(1 + np.exp(m - M))

    def log_pdf(self, x, p):
        return np.log(p[self.index[x]])

    # u is only one sequence
    def alpha_recursion(self, u, A, pi_0, p):
        T = len(u)
        K = len(p)

        A = np.log(A)

        alpha = np.zeros((T, K), dtype=np.float64)

        alpha[0] = np.log(pi_0) + np.array([self.log_pdf(u[0], p[k]) for k in range(K)]).squeeze()

        for t in range(1, T):
            vec = np.array([
                self.log_pdf(u[t], p[k])
                for k in range(K)
            ]).squeeze()

            total = alpha[t - 1, 0] + A[0]
            for k in range(1, K):
                total = self.logplus(total, alpha[t - 1, k] + A[k])

            alpha[t] = vec + total

        return alpha

    def beta_recursion(self, u, A, pi_fin, p):
        T = len(u)
        K = len(p)

        A = np.log(A)

        beta = np.zeros((T, K))

        beta[T - 1] = np.log(pi_fin)

        for t in range(T - 2, -1, -1):
            vec = np.array([
                self.log_pdf(u[t + 1], p[k])
                for k in range(K)
            ]).squeeze()

            total = A[:, 0] + vec[0] + beta[t + 1, 0]
            for k in range(1, K):
                total = self.logplus(total, A[:, k] + vec[k] + beta[t + 1, k])

            beta[t] = total

        return beta

    def proba_hidden(self, t, alpha, beta):
        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return np.exp(prod - total)

    def proba_joint_hidden(self, t, u, alpha, beta, A, p):
        A = np.log(A)

        vec = np.array([
            self.log_pdf(u[t + 1], p[i])
            for i in range(len(p))
        ]).squeeze()
        matrix = alpha[t].reshape(-1, 1) + A + beta[t + 1].reshape(1, -1) + vec.reshape(1, -1)

        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return np.exp(matrix - total)

    def log_likelihood_hmm(self, t, alpha, beta):
        prod = alpha[t] + beta[t]
        total = prod[0]
        for k in range(1, len(prod)):
            total = self.logplus(total, prod[k])

        return total

    def compute_feature_vector(self, X_initial, p, A, pi_0, pi_fin):
        X = self.transform(X_initial, self.k)
        T = len(X[0])
        K = len(p)
        indicator_matrix = self.make_matrix(X)
        alphas = []
        betas = []
        for string in X:
            alphas.append(self.alpha_recursion(string, A, pi_0, p))
            betas.append(self.beta_recursion(string, A, pi_fin, p))

        A2 = np.array([sum(self.proba_joint_hidden(t, string, alpha, beta, A, p) for t in range(T - 1))
                       for string, alpha, beta in zip(X, alphas, betas)])
        features = (A2 / A - A2.sum(axis=2).reshape(len(X), -1, 1)).reshape(len(X), -1)

        p_zt = np.array([[self.proba_hidden(t, alpha, beta) for t in range(T)] for alpha, beta in zip(alphas, betas)])

        p2 = np.zeros((len(X), p.shape[0], p.shape[1]))
        for t in range(len(X)):
            for k in range(K):
                for lettre in range(p.shape[1]):
                    p2[t, k, lettre] = (p_zt[t, :, k] * indicator_matrix[lettre, t, :]).sum()

        features = np.concatenate((features, (p2 / p - p2.sum(axis=2).reshape(len(X), -1, 1)).reshape(len(X), -1)),
                                  axis=1)
        return features

    def gram_matrix(self, X, Y, A, pi_0, pi_fin, p):
        """Compute the Gram matrix using Fisher Kernel."""
        # Compute feature vectors for the entire dataset
        feature_vectors_X = self.compute_feature_vector(X, p, A, pi_0, pi_fin)
        feature_vectors_Y = self.compute_feature_vector(Y, p, A, pi_0, pi_fin)

        # Compute the Gram matrix by taking the dot product between all feature vectors
        G = np.dot(feature_vectors_X, feature_vectors_Y.T)

        # Optionally normalize the Gram matrix if specified
        if self.normalize:
            G /= np.linalg.norm(G, axis=1, keepdims=True)

        return G

    ### Functions to find the likeliest parameters
    def EM_HMM(self, X_initial, K, A, pi_0, pi_fin, p, n_iter=10):
        X = self.transform(X_initial, self.k)

        np.random.seed(10)
        T = len(X[0])
        l = len(X)
        indicator_matrix = self.make_matrix(X)

        loss = []

        for n in tqdm(range(n_iter)):
            alphas = []
            betas = []
            for string in X:
                alphas.append(self.alpha_recursion(string, A, pi_0, p))
                betas.append(self.beta_recursion(string, A, pi_fin, p))

            p_zt = np.array(
                [[self.proba_hidden(t, alpha, beta) for t in range(T)] for alpha, beta in zip(alphas, betas)])
            pi_0 = p_zt.sum(axis=0)[0]
            pi_0 /= pi_0.sum()

            pi_fin = p_zt.sum(axis=0)[-1]
            pi_fin /= pi_fin.sum()

            A = sum(self.proba_joint_hidden(t, string, alpha, beta, A, p) for t in range(T - 1)
                    for string, alpha, beta in
                    zip(X, alphas, betas))
            A /= A.sum(axis=1).reshape(-1, 1)

            for k in range(K):
                for lettre in range(p.shape[1]):
                    p[k, lettre] = (p_zt[:, :, k] * indicator_matrix[lettre, :, :]).sum()
            p /= p.sum(axis=1).reshape(-1, 1)

            loss.append(sum(self.log_likelihood_hmm(0, alpha, beta) for alpha, beta in zip(alphas, betas)))

        return A, pi_0, pi_fin, p, loss

    def make_matrix(self, X):
        X_bis = []
        for string in X:
            X_bis.append(list(string))
        X_bis = np.array(X_bis)
        permutations = self.generate_permutations(self.k)
        return np.array([X_bis == perm for perm in permutations])

    def transform(self, X, k):
        X2 = []
        for x in X:
            temp = []
            for i in range(len(x) - k):
                temp.append(x[i:i + k])
            X2.append(temp)
        return X2

    def generate_permutations(self, k):
        if k == 1:
            return ["A", "C", "G", "T"]
        else:
            l = self.generate_permutations(k - 1)
            l2 = []
            for e in l:
                l2.append(e + "A")
                l2.append(e + "C")
                l2.append(e + "G")
                l2.append(e + "T")

            return l2

    def intialize_parameters(self, observation_size):
        """
        Initialize parameters for the Hidden Markov Model (HMM):
        - Transition matrix A
        - Initial state distribution pi_0
        - Final state distribution pi_fin
        - Observation probability matrix p

        Parameters:
        - K: The number of hidden states
        - observation_size: The size of the observation space (i.e., the alphabet size)
        """
        # Initialize transition matrix A with random values and normalize
        A = np.random.rand(self.k, self.k)
        A /= A.sum(axis=1, keepdims=True)

        # Initialize initial state distribution pi_0 and normalize
        pi_0 = np.random.rand(self.k)
        pi_0 /= pi_0.sum()

        # Initialize final state distribution pi_fin and normalize
        pi_fin = np.random.rand(self.k)
        pi_fin /= pi_fin.sum()

        # Initialize observation probability matrix p with random values
        p = np.random.rand(self.k, observation_size)
        p /= p.sum(axis=1, keepdims=True)

        return A, pi_0, pi_fin, p
