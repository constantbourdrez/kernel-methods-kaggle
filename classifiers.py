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



class SVMC:
    def __init__(self, c=1, min_sv=1e-4):
        self.alpha_ = None
        self.c = c  # Regularization parameter (corresponds to 1/2*lambda)
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
        label = label.reshape(-1, 1)  # Ensure label is a column vector
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
        self.sv = np.where(sv)[0]  # Ensure it's a 1D array
        self.sv_label = label[sv].flatten()

        print(f"{len(self.alpha_)} support vectors out of {n} points")

        # Compute bias term
        self.b = np.mean(
            self.sv_label - np.sum(self.alpha_ * self.sv_label * kernel_train[np.ix_(self.sv, self.sv)], axis=1)
        )

    def get_coef(self):
        """Returns the learned Lagrange multipliers."""
        return list(self.alpha_)

    def predict(self, kernel_test):
        """Predicts the raw decision function values."""
        y_predict = np.zeros(kernel_test.shape[1])

        for i in range(kernel_test.shape[1]):
            y_predict[i] = np.sum(self.alpha_ * self.sv_label * kernel_test[self.sv, i])

        return y_predict + self.b

    def predict_class(self, kernel_test):
        """Predicts the class labels (-1 or 1)."""
        return np.where(self.predict(kernel_test) >= 0, 1, -1)

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

    def fit(self, kernel_train, label, alpha=None, tolerance=1, lambda_regularisation=0):
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
            Predicted class labels (-1 or 1).
        """
        prediction = (self.predict(kernel_test) >= 0.5).astype(int)
        return prediction
