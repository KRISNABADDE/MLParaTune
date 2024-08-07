# Hyperparameter Tuning for Machine Learning Algorithms

This Streamlit application allows users to visualize the decision boundaries of various machine learning classifiers and tune their hyperparameters interactively. The application supports Logistic Regression, Decision Tree, SVM, K-Nearest Neighbors, Random Forest, Bagging Classifier, and Gradient Boosting Classifier.

## Features

- **Interactive Visualization**: Visualize decision boundaries for different classifiers.
- **Hyperparameter Tuning**: Adjust hyperparameters through the sidebar and observe the effect on the model's performance.
- **Dynamic Dataset Generation**: Generate datasets with well-separated classes or complex patterns.

## Setup and Installation

### Prerequisites

- Python 3.7+
- Streamlit
- Scikit-learn
- Matplotlib
- Numpy

### Installation

1. Clone the repository:

```bash
git clone https://github.com/KRISNABADDE/MLParaTune.git
cd MLParaTune
```

2. Create a virtual environment and activate it

```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
3. Install the required dependencies

```bash
pip install -r requirements.txt
```bash

4. Run app

```bash

streamlit run app.py
```

