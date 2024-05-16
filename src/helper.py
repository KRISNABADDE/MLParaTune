import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs,make_gaussian_quantiles
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,GradientBoostingClassifier


def load_initial_graph(dataset, dataset_type, ax):
    if dataset_type == 'Well separated classes.':
        X, y = make_blobs(n_samples=150, centers=dataset, random_state=42,cluster_std=3)
    else:
        X, y = make_gaussian_quantiles(n_samples=150, n_features=2, n_classes=dataset, random_state=42)

    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', edgecolor='black', s=50)
    ax.tick_params(axis='both', which='major', labelsize=7)
    return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.05)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.05)

    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T

    return XX, YY, input_array

def logistic_regression_params():
    solver = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
    
    if solver == 'liblinear':
        penalty = st.sidebar.selectbox('Regularization', ('l2', 'l1'))
        l1_ratio = None
    else:
        penalty = st.sidebar.selectbox('Regularization', ('l2', 'none'))
        l1_ratio = None if solver in {'lbfgs', 'newton-cg', 'newton-cholesky', 'sag'} else \
                    int(st.sidebar.number_input('l1 Ratio'))
    
    c_input = float(st.sidebar.number_input('C (Smaller values stronger regularization.)', value=1.0, min_value=0.001 ,step=0.1))
    max_iter = int(st.sidebar.number_input('Max Iterations', value=100))
    if solver == 'liblinear':
        multi_class = st.sidebar.selectbox('Multi Class', ('auto', 'ovr'))
    else:
        multi_class = st.sidebar.selectbox('Multi Class', ('auto', 'ovr', 'multinomial'))

    return solver, penalty, l1_ratio, c_input, max_iter, multi_class

def decision_tree_params():
    criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))
    splitter = st.sidebar.selectbox('Splitter', ('best', 'random'))
    max_depth = st.sidebar.number_input('Max Depth', value=8, min_value=2, step=1)
    min_samples_split = st.sidebar.number_input('Min Samples Split [%]', value=0.01, min_value=0.01, max_value=1.0, step=0.05)
    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf [%]', min_value=0.01, max_value=1.0, value=0.01, step=0.05)
    max_features = st.sidebar.selectbox('Max Features', ('sqrt', 'log2', None))
    ccp_alpha = max(0.0, st.sidebar.number_input('CCP Alpha', value=0.0, min_value=0.0, step=0.01))

    return criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, ccp_alpha

def svm_params():
    kernel = st.sidebar.selectbox('Kernel', ('linear', 'poly', 'rbf', 'sigmoid', 'precomputed', 'callable'))
    degree = int(st.sidebar.number_input('Degree', value=3, min_value=2, step=1))
    c_input = float(st.sidebar.number_input('C (Smaller values stronger regularization.)', value=1.0, min_value=0.001 ,step=0.1))
    gamma = st.sidebar.selectbox('Gamma', ('scale', 'auto'))
    decision_function_shape = st.sidebar.selectbox('Decision Function Shape', ('ovo', 'ovr'))

    return kernel, degree, c_input, gamma, decision_function_shape

def knn_params():
    algorithm = st.sidebar.selectbox('Algorithm', ('ball_tree', 'kd_tree', 'brute'))
    leaf_size = int(st.sidebar.number_input('Leaf Size', value=30, min_value=5, step=1))
    n_neighbors = int(st.sidebar.number_input('Number of Neighbors', value=5, min_value=1, step=1))
    weights = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    p = st.sidebar.selectbox('Power Parameter (Manhattan=1, Euclidean=2)', (1, 2))

    return algorithm, leaf_size, n_neighbors, weights, p

def rf_params():
    n_estimators = int(st.sidebar.number_input('n_estimators', value=100,min_value=50,step=20))
    criterion = st.sidebar.selectbox('Criterion', ('gini', 'entropy', 'log_loss'))
    min_samples_split = st.sidebar.number_input('Min Samples Split [%]', value=0.01, min_value=0.01, max_value=1.0, step=0.05)
    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf [%]', min_value=0.01, max_value=1.0, value=0.01, step=0.05)
    max_features = st.sidebar.selectbox('Max Features', ('sqrt', 'log2', None))
    bootstrap = st.sidebar.selectbox('Bootstrap',(True,False))
    if bootstrap == True:
        max_samples = float(st.sidebar.number_input('max_samples',min_value=0.01, max_value=1.0, value=0.7, step=0.05))
    else:
        max_samples = None
    ccp_alpha = max(0.0, st.sidebar.number_input('CCP Alpha', value=0.0, min_value=0.0, step=0.01))
    return n_estimators, criterion, min_samples_split,min_samples_leaf,max_features, max_samples,ccp_alpha
    
def bagging_params():
    base_classifier = st.sidebar.selectbox(
        'Select base classifier for Bagging',
        ('Logistic Regression', 'Decision Tree', 'SVM', 'KNN', 'Random Forest')
    )

    if base_classifier == 'Logistic Regression':
        base_clf = LogisticRegression()
    elif base_classifier == 'Decision Tree':
        base_clf = DecisionTreeClassifier()
    elif base_classifier == 'SVM':
        base_clf = SVC()
    elif base_classifier == 'KNN':
        base_clf = KNeighborsClassifier()
    elif base_classifier == 'Random Forest':
        base_clf = RandomForestClassifier()

    n_estimators = int(st.sidebar.number_input('n_estimators', value=100, min_value=50, step=20))
    min_samples_split = st.sidebar.number_input('Min Samples Split [%]', value=0.01, min_value=0.01, max_value=1.0, step=0.05)
    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf [%]', min_value=0.01, max_value=1.0, value=0.01, step=0.05)
    max_features = st.sidebar.selectbox('Max Features', ('sqrt', 'log2', None))
    bootstrap = st.sidebar.selectbox('Bootstrap', (True, False))

    if bootstrap == True:
        max_samples = float(st.sidebar.number_input('Max Samples', min_value=0.01, max_value=1.0, value=0.7, step=0.05))
    else:
        max_samples = 1.0

    ccp_alpha = max(0.0, st.sidebar.number_input('CCP Alpha', value=0.0, min_value=0.0, step=0.01))

    return base_clf, n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap, max_samples, ccp_alpha

def gradient_boosting_params():
    learning_rate = st.sidebar.number_input('Learning Rate', value=0.1, min_value=0.001, step=0.1)
    n_estimators = st.sidebar.number_input('Number of Estimators', value=100, min_value=50, step=20)
    max_depth = st.sidebar.number_input('Max Depth', value=3, min_value=1, step=1)
    min_samples_split = st.sidebar.number_input('Min Samples Split', value=2, min_value=2, step=1)
    min_samples_leaf = st.sidebar.number_input('Min Samples Leaf', value=1, min_value=1, step=1)
    max_features = st.sidebar.selectbox('Max Features', ('sqrt', 'log2', None))
    subsample = st.sidebar.number_input('Subsample', value=1.0, min_value=0.01, max_value=1.0, step=0.01)
    ccp_alpha = max(0.0, st.sidebar.number_input('CCP Alpha', value=0.0, min_value=0.0, step=0.01))
    return learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, subsample,ccp_alpha
