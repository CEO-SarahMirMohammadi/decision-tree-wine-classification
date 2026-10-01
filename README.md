# Decision Tree Wine Classification

A machine learning project demonstrating **Decision Tree classification** using Scikit-learn's **Wine dataset**.

This project modifies Example 3.11, which originally performs Decision Tree classification on the Iris dataset, and adapts the program to use Scikit-learn's Wine dataset.

The implementation includes:

* Loading the Wine dataset
* Exploring the dataset structure
* Splitting the data into training and testing sets
* Training a Decision Tree classifier
* Predicting classes for the test dataset
* Counting correctly classified samples
* Calculating classification accuracy

## Overview

The original Example 3.11 uses the Iris dataset:

```python
X, y = load_iris(return_X_y=True)
```

This project replaces Iris with:

```python
wine = load_wine()

X = wine.data
y = wine.target
```

The resulting workflow is:

```text
Load Wine Dataset
       ↓
Extract Features and Labels
       ↓
Train/Test Split
       ↓
Train Decision Tree
       ↓
Predict Test Samples
       ↓
Compare Predictions with True Labels
       ↓
Calculate Accuracy
```

## Algorithm

### Decision Tree Classification

A **Decision Tree** is a supervised machine learning algorithm that makes predictions by repeatedly splitting the data according to feature-based conditions.

The tree consists of:

* Root node
* Internal decision nodes
* Branches
* Leaf nodes

Each internal node applies a decision rule such as:

```text
Feature ≤ threshold?
```

The process continues until the observation reaches a leaf containing the predicted class.

Conceptually:

```text
                 Root
                   │
          Feature ≤ Threshold?
              /           \
            Yes            No
            │              │
       Decision          Decision
         Node               Node
        /    \             /    \
       ...    ...         ...    ...
              │
           Leaf Node
              │
          Prediction
```

A decision tree can therefore represent a classification process as a sequence of simple decision rules.

## Dataset

This project uses Scikit-learn's built-in **Wine dataset** through:

```python
from sklearn.datasets import load_wine
```

The dataset contains chemical analysis results of wines belonging to three different cultivars.

The dataset contains:

* **178 samples**
* **13 numerical features**
* **3 classes**

The three target classes are represented by:

```text
0 → Class 0
1 → Class 1
2 → Class 2
```

The dataset features include measurements such as:

* Alcohol
* Malic acid
* Ash
* Alcalinity of ash
* Magnesium
* Total phenols
* Flavanoids
* Nonflavanoid phenols
* Proanthocyanins
* Color intensity
* Hue
* OD280/OD315 of diluted wines
* Proline

## Dataset Loading

The dataset is loaded using:

```python
wine = load_wine()

X = wine.data
y = wine.target
```

Here:

* `X` contains the input features.
* `y` contains the target class labels.

The feature matrix has the shape:

```text
(178, 13)
```

This means that there are 178 observations and 13 features for each observation.

## Train/Test Split

The dataset is divided into training and testing subsets using:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0
)
```

The `test_size=0.5` parameter means that 50% of the dataset is used for testing and the remaining 50% is used for training.

Therefore, approximately:

```text
Training data → 50%
Testing data  → 50%
```

The `random_state=0` parameter makes the split reproducible.

## Model Training

The Decision Tree classifier is created using:

```python
clf = DecisionTreeClassifier(random_state=0)
```

The model is trained using:

```python
clf.fit(X_train, y_train)
```

During training, the decision tree learns rules that separate the three wine classes based on the 13 input features.

## Prediction

After training, the model predicts the classes of the test samples:

```python
y_pred = clf.predict(X_test)
```

The resulting `y_pred` array contains the predicted class for each test observation.

## Evaluation

The original exercise evaluates the model by counting how many test samples were classified correctly.

The number of test samples is calculated using:

```python
N = y_test.shape[0]
```

The number of correctly classified samples is calculated using:

```python
C = (y_test == y_pred).sum()
```

The program then displays:

```text
Total points: N
Correctly labeled points: C
```

## Accuracy

The project also calculates classification accuracy:

```python
accuracy = C / N
```

and displays it as a percentage:

```python
print("Accuracy: %.2f%%" % (accuracy * 100))
```

Accuracy represents the proportion of test observations that were classified correctly.

For example, if:

```text
100 test samples
95 correctly classified
```

then:

```text
Accuracy = 95%
```

## Project Structure

```text
decision-tree-wine-classification/
│
├── decision_tree_wine.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Requirements

| Requirement  | Purpose                                      |
| ------------ | -------------------------------------------- |
| Python 3.x   | Programming language                         |
| Scikit-learn | Dataset, train/test split, and Decision Tree |

## Installation

Clone the repository:

```bash
git clone https://github.com/CEO-SarahMirMohammadi/decision-tree-wine-classification.git
```

Move into the project directory:

```bash
cd decision-tree-wine-classification
```

Create a virtual environment.

### Windows

```powershell
python -m venv .venv
```

Activate the virtual environment:

```powershell
.venv\Scripts\activate
```

Install the dependencies:

```powershell
pip install -r requirements.txt
```

## Running the Program

Run:

```powershell
python decision_tree_wine.py
```

The program will:

1. Load the Wine dataset.
2. Display the dataset dimensions.
3. Display the number and names of classes.
4. Split the dataset into training and testing sets.
5. Train a Decision Tree classifier.
6. Predict the classes of the test samples.
7. Count correctly classified samples.
8. Calculate classification accuracy.

## Example Output

A typical execution will produce output similar to:

```text
Wine dataset shape:
(178, 13)

Number of classes:
3

Class names:
[...]

Training samples:
89
Testing samples:
89

Decision Tree model trained successfully.

Evaluation:
Total points: 89 Correctly labeled points: XX
Accuracy: XX.XX%
```

The exact number of correctly classified samples and the resulting accuracy depend on the trained decision tree and test split.

## Workflow

```text
                 Wine Dataset
                      │
                      ▼
             178 Samples
             13 Features
                      │
                      ▼
              Train/Test Split
                 50% / 50%
                      │
          ┌───────────┴───────────┐
          ▼                       ▼
    Training Data             Test Data
          │                       │
          ▼                       │
   Decision Tree                  │
      Training                   │
          │                       │
          ▼                       │
    Trained Model                 │
          │                       │
          └───────────┬───────────┘
                      ▼
                  Prediction
                      │
                      ▼
             Compare with y_test
                      │
                      ▼
                  Accuracy
```

## Key Concepts

### 1. Supervised Learning

Decision Trees are supervised learning algorithms because the training data contains known target labels.

The model learns:

```text
Features → Target Class
```

### 2. Classification

This project performs multiclass classification.

The model must determine which of the three wine classes each observation belongs to.

### 3. Decision Tree

A Decision Tree creates a sequence of feature-based decisions to reach a final class prediction.

### 4. Training Data

Training data is used to learn the decision rules.

```python
clf.fit(X_train, y_train)
```

### 5. Test Data

Test data is not used during training.

It is used to evaluate how the trained model performs on unseen observations.

### 6. Prediction

Predictions are generated using:

```python
clf.predict(X_test)
```

### 7. Accuracy

Accuracy measures the proportion of correct predictions:

```text
Correct Predictions
────────────────────
Total Predictions
```

## Why Decision Trees?

Decision Trees have several useful characteristics:

* Easy to understand conceptually
* Can model nonlinear relationships
* Can handle numerical features
* Require relatively little preprocessing
* Can be used for both classification and regression
* Produce human-readable decision rules

However, individual decision trees can also overfit their training data, especially when they are allowed to grow without restrictions.

## Important Decision Tree Parameters

The model can be controlled using parameters such as:

### `max_depth`

Controls the maximum depth of the tree.

```python
DecisionTreeClassifier(max_depth=5)
```

### `min_samples_split`

Controls the minimum number of samples required to split an internal node.

```python
DecisionTreeClassifier(min_samples_split=5)
```

### `min_samples_leaf`

Controls the minimum number of samples required in a leaf node.

```python
DecisionTreeClassifier(min_samples_leaf=2)
```

These parameters can be used to control model complexity and reduce overfitting.

## Limitations

A Decision Tree can become overly complex when it learns very detailed patterns from the training data.

This can result in **overfitting**, where the model performs well on training data but performs worse on unseen data.

The current exercise uses the default Decision Tree configuration to remain close to the original Example 3.11 implementation.

## Future Improvements

Possible improvements include:

* Add a confusion matrix.
* Add a classification report.
* Compare training and testing accuracy.
* Tune `max_depth`.
* Experiment with `min_samples_split`.
* Experiment with `min_samples_leaf`.
* Use cross-validation.
* Visualize the trained decision tree.
* Compare Decision Tree with Random Forest.
* Compare Decision Tree with Logistic Regression.
* Compare Decision Tree with Support Vector Machine.
* Analyze feature importance.

## License

This project is licensed under the **MIT License**.

You are free to use, modify, distribute, and build upon this project under the terms of the MIT License.
