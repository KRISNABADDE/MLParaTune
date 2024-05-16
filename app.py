import matplotlib.pyplot as plt
import streamlit as st
from src.helper import *

# Custom styling
custom_style = """
    <style>
        
        .custom-header {
        font-size: 20px;
        padding-bottom: 20px;
        display: flex;
        font-family: emoji;
        }
        .st-emotion-cache-1r4qj8v {
        position: absolute;
        background: rgb(255, 255, 255);
        color: rgb(49, 51, 63);
        inset: 0px;
        color-scheme: light;
        overflow: hidden;}

        .custom-title {
            font-size: 16px;  /* Adjust the font size */
        }
    </style>
"""

st.set_page_config(page_title="Hyperparameter Tuning",page_icon="⚙️")
st.markdown(custom_style, unsafe_allow_html=True)

st.markdown('<h2 class="custom-header">KRSNA </h2>', unsafe_allow_html=True)
st.sidebar.markdown('<h2 class="custom-header">KRSNA </h2>', unsafe_allow_html=True)

st.markdown('<h1 class="custom-title">Machine Learning Algorithms: Hyperparameter Tuning</h1>', unsafe_allow_html=True)



plt.style.use('fivethirtyeight')

st.sidebar.markdown("# Classifier Options")
dataset_type = st.sidebar.selectbox('Datasets',('Well separated classes.','Complex'))
dataset = int(st.sidebar.number_input('Number of classes', value=2))
if dataset < 2:
    st.sidebar.error("Class should be greater than or equal to 2.")
    st.stop()

classifier_choice = st.sidebar.selectbox(
    'Select Classifier',
    ('Logistic Regression', 'Decision Tree', 'SVM', 'KNeighborsClassifier',
     'RandomForestClassifier','Bagging Classifier','GradientBoostingClassifier')
)

default_para = st.sidebar.selectbox('Parameters', ('default', 'Parameters'))

fig, ax = plt.subplots()

X, y = load_initial_graph(dataset,dataset_type, ax)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
orig = st.pyplot(fig)

if classifier_choice == 'Logistic Regression':
    if default_para == 'Parameters':
        solver, penalty, l1_ratio, c_input, max_iter, multi_class = logistic_regression_params()
        clf = LogisticRegression(penalty=penalty, C=c_input, solver=solver, max_iter=max_iter, multi_class=multi_class, l1_ratio=l1_ratio,warm_start=True)
    else:
        clf = LogisticRegression(warm_start=True)
elif classifier_choice == 'Decision Tree':
    if default_para == 'Parameters':
        criterion, splitter, max_depth, min_samples_split, min_samples_leaf, max_features, ccp_alpha = decision_tree_params()
        clf = DecisionTreeClassifier(criterion=criterion, splitter=splitter, max_depth=max_depth,
                                     min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf,
                                     max_features=max_features, ccp_alpha=ccp_alpha)
    else:
        clf = DecisionTreeClassifier()
elif classifier_choice == 'SVM':
    if default_para == 'Parameters':
        kernel, degree, c_input, gamma, decision_function_shape = svm_params()
        clf = SVC(kernel=kernel, degree=degree, C=c_input, gamma=gamma, decision_function_shape=decision_function_shape)
    else:
        clf = SVC()
elif classifier_choice == 'KNeighborsClassifier':
    if default_para == 'Parameters':
        algorithm, leaf_size, n_neighbors, weights, p = knn_params()
        clf = KNeighborsClassifier(algorithm=algorithm, leaf_size=leaf_size, n_neighbors=n_neighbors, weights=weights, p=p,n_jobs=-1)
    else:
        clf = KNeighborsClassifier(n_jobs=-1)
elif classifier_choice == 'RandomForestClassifier':
    if default_para == 'Parameters':
        n_estimators, criterion, min_samples_split,min_samples_leaf,max_features, max_samples,ccp_alpha = rf_params()
        clf = RandomForestClassifier(n_estimators=n_estimators,criterion=criterion,min_samples_split=min_samples_split,
                                    min_samples_leaf=min_samples_leaf, max_features=max_features,max_samples=max_samples,ccp_alpha=ccp_alpha,n_jobs=-1,warm_start=True)
    else:
        clf = RandomForestClassifier(n_jobs=-1,warm_start=True)
elif classifier_choice == 'Bagging Classifier':
    if default_para == 'Parameters':
        base_clf, n_estimators, min_samples_split, min_samples_leaf, max_features, bootstrap, max_samples, ccp_alpha = bagging_params()
        clf = BaggingClassifier(base_clf, n_estimators=n_estimators, bootstrap=bootstrap, max_samples=max_samples,
                            random_state=42,n_jobs=-1,warm_start=True)
    else:
        clf = BaggingClassifier(n_jobs=-1,warm_start=True)
elif classifier_choice == 'GradientBoostingClassifier':
    if default_para == 'Parameters':
        learning_rate, n_estimators, max_depth, min_samples_split, min_samples_leaf, max_features, subsample,ccp_alpha = gradient_boosting_params()
        clf = GradientBoostingClassifier(learning_rate=learning_rate, n_estimators=n_estimators,
                                         max_depth=max_depth, min_samples_split=min_samples_split,
                                         min_samples_leaf=min_samples_leaf, max_features=max_features,
                                         subsample=subsample, random_state=42,ccp_alpha=ccp_alpha,warm_start=True)
    else:
        clf = GradientBoostingClassifier(warm_start=True)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

XX, YY, input_array = draw_meshgrid(X)
labels = clf.predict(input_array)

ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='Set2')
ax.tick_params(axis='both', which='major', labelsize=7)
orig = st.pyplot(fig)
st.subheader(f"Accuracy for {classifier_choice}: {round(accuracy_score(y_test, y_pred),2)}")
