import numpy as np
from multiprocessing import Pool
from itertools import product
from collections import defaultdict

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
        len_X, len_Y = X.shape[0], Y.shape[0]
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        for i in range(len_X):
            for j in range(len_Y):
                gram_matrix[i, j] = self.compute(X[i], Y[j])

        return gram_matrix


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
        """
        Initialize the LocalAlignmentKernel with scoring parameters.

        Parameters:
        - match_score: The score for a match between characters.
        - mismatch_penalty: The penalty for a mismatch.
        - gap_penalty: The penalty for introducing a gap.
        """
        self.match_score = match_score
        self.mismatch_penalty = mismatch_penalty
        self.gap_penalty = gap_penalty

    def _initialize_matrices(self, seq1, seq2):
        """
        Initialize the scoring and traceback matrices for local alignment.

        Parameters:
        - seq1: The first sequence.
        - seq2: The second sequence.

        Returns:
        - score_matrix: The initialized scoring matrix.
        - traceback_matrix: Matrix for traceback to find the optimal alignment.
        """
        len_seq1, len_seq2 = len(seq1), len(seq2)
        score_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1))
        traceback_matrix = np.zeros((len_seq1 + 1, len_seq2 + 1), dtype=int)
        return score_matrix, traceback_matrix

    def _score(self, i, j, seq1, seq2, score_matrix):
        """
        Calculate the score for a specific cell (i, j) based on the dynamic programming recurrence.

        Parameters:
        - i: Row index for seq1.
        - j: Column index for seq2.
        - seq1: The first sequence.
        - seq2: The second sequence.
        - score_matrix: The current scoring matrix.

        Returns:
        - max_score: The maximum score computed for cell (i, j).
        """
        match = self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty

        scores = [
            score_matrix[i-1, j-1] + match,  # match/mismatch
            score_matrix[i-1, j] + self.gap_penalty,  # gap in seq2
            score_matrix[i, j-1] + self.gap_penalty,  # gap in seq1
            0  # local alignment allows zero score (no alignment)
        ]

        return max(scores)

    def compute(self, seq1, seq2):
        """
        Perform local alignment for the given sequences.

        Parameters:
        - seq1: The first sequence.
        - seq2: The second sequence.

        Returns:
        - aligned_seq1: The aligned version of seq1.
        - aligned_seq2: The aligned version of seq2.
        - alignment_score: The local alignment score.
        """
        score_matrix, traceback_matrix = self._initialize_matrices(seq1, seq2)

        # Fill in the score matrix using dynamic programming
        for i in range(1, len(seq1) + 1):
            for j in range(1, len(seq2) + 1):
                score_matrix[i, j] = self._score(i, j, seq1, seq2, score_matrix)

        # Find the maximum score (this is the local alignment score)
        alignment_score = np.max(score_matrix)
        max_i, max_j = np.unravel_index(np.argmax(score_matrix), score_matrix.shape)

        # Traceback to reconstruct the aligned sequences
        aligned_seq1, aligned_seq2 = "", ""
        i, j = max_i, max_j

        while i > 0 and j > 0 and score_matrix[i, j] > 0:
            if score_matrix[i, j] == score_matrix[i-1, j-1] + (self.match_score if seq1[i-1] == seq2[j-1] else self.mismatch_penalty):
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                i -= 1
                j -= 1
            elif score_matrix[i, j] == score_matrix[i-1, j] + self.gap_penalty:
                aligned_seq1 = seq1[i-1] + aligned_seq1
                aligned_seq2 = "-" + aligned_seq2
                i -= 1
            else:
                aligned_seq1 = "-" + aligned_seq1
                aligned_seq2 = seq2[j-1] + aligned_seq2
                j -= 1

        return aligned_seq1, aligned_seq2, alignment_score

    def gram_matrix(self, X, Y=None):
        """
        Compute the Gram matrix for a set of sequences X and Y.

        Parameters:
        - X: A list of sequences.
        - Y: A list of sequences (optional, default is X).

        Returns:
        - gram_matrix: The computed Gram matrix.
        """
        Y = X if Y is None else Y
        len_X, len_Y = len(X), len(Y)
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        for i in range(len_X):
            for j in range(len_Y):
                aligned_seq1, aligned_seq2, score = self.compute(X[i], Y[j])
                gram_matrix[i, j] = score

        return gram_matrix

class StringKernel:
    def __init__(self, alphabet, n):
        """
        Initialize the StringKernel with the given parameters.

        Parameters:
        - alphabet: The alphabet of characters used in the strings.
        - n: The length of n-grams to be used for kernel computation.
        """
        self.alphabet = alphabet
        self.n = n
        self.all_combinations = list(product(alphabet, repeat=n))

    def ngrams(self, seq):
        """
        Generate n-grams from a sequence.

        Parameters:
        - seq: The input sequence.

        Returns:
        - A list of n-grams extracted from the sequence.
        """
        return list(zip(*[seq[i:] for i in range(self.n)]))

    def create_histogram(self, seq):
        """
        Create a histogram for a sequence based on n-grams.

        Parameters:
        - seq: The input sequence.

        Returns:
        - histogram: A histogram vector representing the frequency of each n-gram in the sequence.
        """
        histogram = np.zeros(len(self.all_combinations), dtype=np.float32)
        for ngram in self.ngrams(seq):
            index = self.all_combinations.index(ngram)
            histogram[index] += 1
        return histogram

    def create_histogram_mismatch(self, seq):
        """
        Create a histogram for a sequence with mismatches allowed in n-grams.

        Parameters:
        - seq: The input sequence.

        Returns:
        - histogram: A histogram vector with mismatch handling (allowing small modifications to n-grams).
        """
        histogram = self.create_histogram(seq)
        letters = self.alphabet

        for ngram in self.ngrams(seq):
            index = self.all_combinations.index(ngram)
            for i, char in enumerate(ngram):
                for letter in letters:
                    if letter != char:
                        modified_ngram = list(ngram)
                        modified_ngram[i] = letter
                        mod_index = self.all_combinations.index(tuple(modified_ngram))
                        histogram[mod_index] += 0.1  # Adding a small penalty for mismatches

        return histogram

    @staticmethod
    def compute_idf(histograms):
        """
        Compute the Inverse Document Frequency (IDF) for a set of histograms.

        Parameters:
        - histograms: A list of histograms for each sequence.

        Returns:
        - idf: The IDF vector for the histograms.
        """
        idf = 0.000001 + np.sum(histograms, axis=0)
        return np.maximum(1, np.log10(len(histograms) / idf))

    @staticmethod
    def kernel(x1, x2):
        """
        Compute the dot product between two histograms (kernel computation).

        Parameters:
        - x1: The first histogram.
        - x2: The second histogram.

        Returns:
        - The dot product (kernel value) between the two histograms.
        """
        return np.vdot(x1, x2)

    def gram_matrix(self, X, Y=None, n_proc=1):
        """
        Compute the Gram matrix (similarity matrix) between a set of sequences X and Y.

        Parameters:
        - X: A list of sequences.
        - Y: A list of sequences (optional, default is X).
        - n_proc: The number of processes to use for parallel computation (default is 1).

        Returns:
        - gram_matrix: The computed Gram matrix.
        """
        Y = X if Y is None else Y
        len_X, len_Y = len(X), len(Y)
        gram_matrix = np.zeros((len_X, len_Y), dtype=np.float32)

        if n_proc > 1:
            from multiprocessing import Pool
            with Pool(processes=n_proc) as pool:
                histograms_X = pool.map(self.create_histogram_mismatch, X)
                histograms_Y = pool.map(self.create_histogram_mismatch, Y) if X is not Y else histograms_X

                for i in range(len_X):
                    for j in range(len_Y):
                        gram_matrix[i, j] = self.kernel(histograms_X[i], histograms_Y[j]) / (
                            np.sqrt(np.vdot(histograms_X[i], histograms_X[i])) * np.sqrt(np.vdot(histograms_Y[j], histograms_Y[j]))
                        )
        else:
            histograms_X = [self.create_histogram_mismatch(x) for x in X]
            histograms_Y = [self.create_histogram_mismatch(y) for y in Y] if X is not Y else histograms_X

            for i in range(len_X):
                for j in range(len_Y):
                    gram_matrix[i, j] = self.kernel(histograms_X[i], histograms_Y[j]) / (
                        np.sqrt(np.vdot(histograms_X[i], histograms_X[i])) * np.sqrt(np.vdot(histograms_Y[j], histograms_Y[j]))
                    )

        return gram_matrix

class HMMFisherKernel:
    def __init__(self, n_components, n_features, n_iter=100):
        """
        Initialize the HMM Fisher Kernel with the given parameters.

        Parameters:
        - n_components: The number of hidden states in the HMM.
        - n_features: The number of features in the observation vectors.
        - n_iter: The number of iterations for fitting the HMM (default is 100).
        """
        self.n_components = n_components
        self.n_features = n_features
        self.n_iter = n_iter
        self.startprob_ = np.ones(n_components) / n_components  # Uniform start probabilities
        self.transmat_ = np.ones((n_components, n_components)) / n_components  # Uniform transition probabilities
        self.means_ = np.random.randn(n_components, n_features)  # Random mean vectors for the Gaussian emissions
        self.covars_ = np.array([np.eye(n_features) for _ in range(n_components)])  # Identity covariance matrices

    def fit(self, X):
        """
        Fit the HMM model to a set of sequences using the Expectation-Maximization algorithm.

        Parameters:
        - X: A list of sequences, where each sequence is a 2D array with shape (n_samples, n_features).
        """
        # Perform Expectation-Maximization for a fixed number of iterations
        for _ in range(self.n_iter):
            for seq in X:
                self._expectation_maximization(seq)

    def _expectation_maximization(self, seq):
        """
        Perform one step of the Expectation-Maximization algorithm on a single sequence.

        Parameters:
        - seq: A 2D array (n_samples, n_features) representing a sequence.
        """
        n_samples = seq.shape[0]

        # E-step: Compute the forward-backward probabilities
        alpha, beta, gamma = self._forward_backward(seq)

        # M-step: Update model parameters based on the sufficient statistics
        self._update_parameters(seq, alpha, beta, gamma)

    def _forward_backward(self, seq):
        """
        Perform the forward-backward algorithm to compute the probabilities.

        Parameters:
        - seq: A 2D array (n_samples, n_features) representing a sequence.

        Returns:
        - alpha: The forward probabilities (alpha).
        - beta: The backward probabilities (beta).
        - gamma: The responsibilities (gamma).
        """
        n_samples = seq.shape[0]

        # Initialize the forward (alpha) and backward (beta) variables
        alpha = np.zeros((n_samples, self.n_components))
        beta = np.zeros((n_samples, self.n_components))

        # Forward pass (alpha)
        alpha[0] = self.startprob_ * self._compute_emission_prob(seq[0])
        alpha[0] /= np.sum(alpha[0])  # Normalize

        for t in range(1, n_samples):
            for j in range(self.n_components):
                alpha[t, j] = np.sum(alpha[t-1] * self.transmat_[:, j]) * self._compute_emission_prob(seq[t])[j]
            alpha[t] /= np.sum(alpha[t])  # Normalize

        # Backward pass (beta)
        beta[-1] = 1  # Initialize the backward variable at the last time step

        for t in range(n_samples - 2, -1, -1):
            for i in range(self.n_components):
                beta[t, i] = np.sum(self.transmat_[i] * self._compute_emission_prob(seq[t+1]) * beta[t+1])
            beta[t] /= np.sum(beta[t])  # Normalize

        # Compute the responsibilities (gamma)
        gamma = alpha * beta
        gamma /= np.sum(gamma, axis=1, keepdims=True)

        return alpha, beta, gamma

    def _compute_emission_prob(self, x):
        """
        Compute the emission probabilities (i.e., P(x | state)) for each component.

        Parameters:
        - x: A 1D array representing an observation.

        Returns:
        - emission_probs: A vector of emission probabilities for each state.
        """
        diff = x - self.means_
        prob = np.exp(-0.5 * np.sum(diff ** 2 / self.covars_, axis=1))
        prob /= np.sqrt(np.linalg.det(self.covars_))
        return prob

    def _update_parameters(self, seq, alpha, beta, gamma):
        """
        Update the parameters (start probabilities, transition matrix, means, and covariances) based on the responsibilities.

        Parameters:
        - seq: A 2D array (n_samples, n_features) representing a sequence.
        - alpha: The forward probabilities (alpha).
        - beta: The backward probabilities (beta).
        - gamma: The responsibilities (gamma).
        """
        n_samples = seq.shape[0]

        # Update start probabilities (pi)
        self.startprob_ = gamma[0]
        self.startprob_ /= np.sum(self.startprob_)

        # Update transition matrix (A)
        for t in range(1, n_samples):
            for i in range(self.n_components):
                for j in range(self.n_components):
                    self.transmat_[i, j] += gamma[t-1, i] * gamma[t, j]

        self.transmat_ /= np.sum(self.transmat_, axis=1, keepdims=True)  # Normalize

        # Update means and covariances (B)
        for i in range(self.n_components):
            weighted_sum = np.zeros(self.n_features)
            weighted_cov = np.zeros((self.n_features, self.n_features))
            for t in range(n_samples):
                weighted_sum += gamma[t, i] * seq[t]
                weighted_cov += gamma[t, i] * np.outer(seq[t], seq[t])

            self.means_[i] = weighted_sum / np.sum(gamma[:, i])
            self.covars_[i] = weighted_cov / np.sum(gamma[:, i]) - np.outer(self.means_[i], self.means_[i])

    def compute_fisher_information_matrix(self, X):
        """
        Compute the Fisher Information Matrix for the HMM.

        Parameters:
        - X: A list of sequences, where each sequence is a 2D array with shape (n_samples, n_features).

        Returns:
        - fisher_info: The Fisher Information Matrix for the HMM.
        """
        fisher_info = np.zeros((self.n_components, self.n_components))

        # Compute the Fisher Information Matrix using a simple approximation
        for seq in X:
            alpha, beta, gamma = self._forward_backward(seq)
            fisher_info += np.dot(gamma.T, gamma)

        return fisher_info

    @staticmethod
    def kernel(x1, x2):
        """
        Compute the kernel (similarity) between two Fisher scores using an RBF kernel.

        Parameters:
        - x1: Fisher score vector for the first sequence.
        - x2: Fisher score vector for the second sequence.

        Returns:
        - The kernel value (similarity) between the two sequences.
        """
        gamma = 0.5  # You can adjust this hyperparameter
        return np.exp(-gamma * np.linalg.norm(x1 - x2) ** 2)
