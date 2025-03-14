import numpy as np
import cvxopt


###########"Define frequently used functions
sigmoid = lambda x: 1. / (1 + np.exp(-np.clip(x, -500, 500)))

# Avoid log(0) or log of negative values
epsilon = 1e-10

inverse_sigmoid = lambda x: np.log(1 / (x + epsilon) - 1)

def eta(x, w):
    """Computes eta function used in Kernel Logistic Regression."""
    return sigmoid(np.dot(w.T, x))  # Function eta from the paper

#####################################################################

def compute_m(Kernel_Mat, alpha):
    """Compute m = K * alpha."""
    return Kernel_Mat.dot(alpha)

def compute_P(y, m):
    """Compute the diagonal matrix P for the logistic regression."""
    return np.diag(-sigmoid(-y * m).flatten())

def compute_W(y, m):
    """Compute the diagonal matrix W for the logistic regression."""
    diag_comp = sigmoid(y * m) * (1 - sigmoid(y * m))
    return np.diag(diag_comp.flatten())

def compute_Z(m, y):
    """Compute the vector Z in the IRLS algorithm."""
    return  m + y / (sigmoid(-y * m) + 1e-10)

def compute_alpha(Kernel, W, Z, lambda_reg):
    """Compute alpha using the IRLS algorithm."""
    W_sqrt = np.sqrt(W)
    n = Kernel.shape[0]
    to_inv = W_sqrt.dot(Kernel).dot(W_sqrt) + n * lambda_reg * np.eye(n)
    to_inv = np.linalg.inv(to_inv)
    alpha = W_sqrt.dot(to_inv).dot(W_sqrt).dot(Z)
    return alpha
class KernelLogisticRegression:
    """
    Kernel Logistic Regression binary classifier.

    Attributes:
        alpha_: Learned alpha parameter.
    """

    def __init__(self, init_coef=0):
        """
        Constructor for KernelLogisticRegression.

        Args:
            init_coef: Initial coefficient value. Default is 0.
        """
        self.alpha_ = np.array([init_coef]) if init_coef else 0

    def fit(self, kernel_train, label, alpha=None, tolerance=1e-5, lambda_regularisation=0.5):
        """
        Fits the model using the IRLS algorithm to learn parameters.

        Args:
            kernel_train: Gram matrix for the training set (shape: [n_samples, n_samples]).
            label: True labels of the training set (shape: [n_samples,]).
            alpha: Initial alpha values. Default is random initialization.
            tolerance: Stopping criterion for convergence.
            lambda_regularisation: Regularization parameter.

        Returns:
            None
        """
        if label.ndim == 1:
            label = label.reshape(-1, 1)

        if alpha is None:
            alpha = np.random.rand(kernel_train.shape[0], 1)

        if lambda_regularisation == 0:
            lambda_regularisation = 1e-6
        old_alpha = np.copy(alpha)
        m = compute_m(kernel_train, alpha)
        P = np.nan_to_num(compute_P(label, m))
        W = np.nan_to_num(compute_W(label, m))
        z = compute_Z(m, label)
        alpha = compute_alpha(kernel_train, W, z, lambda_regularisation)
        while np.linalg.norm(alpha - old_alpha) > tolerance:
            print(np.linalg.norm(alpha - old_alpha))
            old_alpha = np.copy(alpha)
            m = compute_m(kernel_train, alpha)
            P = np.nan_to_num(compute_P(label, m))
            W = np.nan_to_num(compute_W(label, m))
            z = compute_Z(m, label)
            alpha = compute_alpha(kernel_train, W, z, lambda_regularisation)
        self.alpha_ = alpha

    def get_coef(self):
        """
        Returns the learned model parameters (alpha).

        Returns:
            List of model parameters (alpha).
        """
        return self.alpha_.flatten().tolist()

    def predict(self, kernel_test):
        """
        Predict the probabilities for the test set.

        Args:
            kernel_test: Gram matrix for the test set (shape: [n_samples_train, n_samples_test]).

        Returns:
            Probabilities of each test sample being class 1.
        """
        prediction = np.dot(self.alpha_.T, kernel_test.T).flatten()
        return sigmoid(prediction)

    def predict_class(self, kernel_test):
        """
        Predict the class labels for the test set.

        Args:
            kernel_test: Gram matrix for the test set (shape: [n_samples_train, n_samples_test]).

        Returns:
            Predicted class labels (0 or 1).
        """
        prediction = (self.predict(kernel_test) >= 0.5).astype(int)
        return prediction


class K_means:
    """
    K-means algorithm for clustering with kernel and class prediction using only numpy.
    """
    def __init__(self, k, max_iter=100):
        self.k = k
        self.max_iter = max_iter

    def fit(self, gram_matrix, y):
        """
        Fit the K-means algorithm using the kernel trick and track class labels.

        Parameters:
        gram_matrix (array-like): Precomputed Gram matrix.
        y (array-like): Class labels (0 or 1 for each data point).
        """
        n_samples = gram_matrix.shape[0]
        centroids_idx = np.random.choice(n_samples, self.k, replace=False)
        self.centroids = gram_matrix[centroids_idx, :]

        prev_centroids = np.zeros_like(self.centroids)
        self.cluster_assignments = np.zeros(n_samples)

        for iteration in range(self.max_iter):
            # Assign points to closest centroid in the kernel space
            for i in range(n_samples):
                distances = np.array([
                    gram_matrix[i, i] - 2 * gram_matrix[i, centroids_idx[c]] + gram_matrix[centroids_idx[c], centroids_idx[c]]
                    for c in range(self.k)
                ])
                self.cluster_assignments[i] = np.argmin(distances)

            # Update centroids
            for k in range(self.k):
                cluster_points = np.where(self.cluster_assignments == k)[0]
                if len(cluster_points) > 0:
                    # Calculate the centroid as the mean of points in the cluster
                    new_centroid = np.mean(gram_matrix[cluster_points][:, cluster_points], axis=0)
                    self.centroids[k, :len(new_centroid)] = new_centroid

            # Check for convergence
            if np.linalg.norm(self.centroids - prev_centroids) < 1e-6:
                break

            prev_centroids = self.centroids.copy()

        # Track the majority class for each cluster
        self.cluster_classes = np.zeros(self.k)
        for k in range(self.k):
            cluster_points = np.where(self.cluster_assignments == k)[0]
            if len(cluster_points) > 0:
                class_counts = np.bincount(y[cluster_points].astype(int))
                self.cluster_classes[k] = np.argmax(class_counts)

        self.centroids_idx = centroids_idx
        self.gram_train = gram_matrix

    def predict_class(self, gram_matrix):
        """
        Predict the class labels for the test set.

        Parameters:
        gram_matrix (array-like): Precomputed Gram matrix for the test set (shape: [n_samples_train, n_samples_test]).

        Returns:
        Predicted class labels (0 or 1).
        """
        n_samples = gram_matrix.shape[0]
        y_pred = np.zeros(n_samples)


        for i in range(n_samples):

            distances = np.array([
                gram_matrix[i, i] - 2 * gram_matrix[i, self.centroids_idx[c]] + self.gram_train[self.centroids_idx[c], self.centroids_idx[c]]
                for c in range(self.k)
            ])
            cluster_assignment = np.argmin(distances)
            y_pred[i] = self.cluster_classes[cluster_assignment]

        return y_pred


class SVMC:
    def __init__(self, c=1, min_sv=1e-4):
        self.alpha_ = None
        self.c = c  # Regularization parameter
        self.min_sv = min_sv
        self.b = 0

    def fit(self, kernel_train, label):
        """
        Solving C-SVM quadratic optimization problem:
            min 1/2 u^T P u + q^T u
            s.t.  Au = b
                  Gu <= h
        """
        n = label.shape[0]
        label = label.reshape(-1, 1)
        diag = np.diag(label.flatten())
        P = diag @ kernel_train @ diag
        Pcvx = cvxopt.matrix(P)
        qcvx = cvxopt.matrix(-np.ones(n))


        if self.c is None:
            G = cvxopt.matrix(-np.eye(n))
            h = cvxopt.matrix(np.zeros(n))
        else:
            G = cvxopt.matrix(np.vstack((-np.eye(n), np.eye(n))))
            h = cvxopt.matrix(np.hstack((np.zeros(n), np.ones(n) * self.c)))

        A = label.T.astype('double')
        Acvx = cvxopt.matrix(A)
        bcvx = cvxopt.matrix(0.0)

        # Solve QP problem
        u = cvxopt.solvers.qp(Pcvx, qcvx, G, h, Acvx, bcvx)

        alpha = np.ravel(u['x'])

        # Select support vectors
        sv = alpha > self.min_sv
        self.alpha_ = alpha[sv]
        self.sv = np.where(sv)[0]  
        self.sv_label = label[sv].flatten()

        print(f"{len(self.alpha_)} support vectors out of {n} points")

        # Compute bias term using a robust method
        self.b = np.mean(
            self.sv_label - np.dot(kernel_train[self.sv][:, self.sv], (self.alpha_ * self.sv_label))
        )

    def get_coef(self):
        """Returns the learned Lagrange multipliers."""
        return list(self.alpha_)

    def predict(self, kernel_test):
        """Predicts the raw decision function values."""
        return np.dot(kernel_test[:, self.sv], (self.alpha_ * self.sv_label)) + self.b

    def predict_class(self, kernel_test):
        """Predicts the class labels (0 or 1)."""
        return (self.predict(kernel_test) >= 0).astype(int)
