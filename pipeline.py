from classifiers import SVMC, KernelLogisticRegression
from kernels import SpectrumKernel, LinearKernel, StringKernel,  HMMFisherKernel, RBFKernel, LocalAlignmentKernel
from utils import *

# Load datasets
X_train, X_train_matrix, Y_train, X_test, X_test_matrix = load_datasets()
X_train, X_val,y_train,y_val = train_test_split(X_train,Y_train,test_size=0.1)
# Transform data to one-hot encoding
X_train_one_hot = transform_data_to_one_hot(X_train)
X_val_one_hot = transform_data_to_one_hot(X_val)
logistic_reg = KernelLogisticRegression()
kernel = LinearKernel()
gram_train = kernel.gram_matrix(X_train_one_hot, X_train_one_hot)
gram_val = kernel.gram_matrix(X_val_one_hot, X_train_one_hot)
logistic_reg.fit(gram_train, y_train)
y_pred = logistic_reg.predict(gram_val)
print(accuracy_score(y_val, y_pred))
