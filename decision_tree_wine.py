from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# Load Scikit-learn Wine dataset
wine = load_wine()

X = wine.data
y = wine.target

print("Wine dataset shape:")
print(X.shape)

print("\nNumber of classes:")
print(len(wine.target_names))

print("\nClass names:")
print(wine.target_names)


# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=0
)

print("\nTraining samples:")
print(X_train.shape[0])

print("Testing samples:")
print(X_test.shape[0])


# Create the Decision Tree classifier
clf = DecisionTreeClassifier(random_state=0)

# Train the model
clf.fit(X_train, y_train)

print("\nDecision Tree model trained successfully.")


# Make predictions on the test data
y_pred = clf.predict(X_test)


# Calculate correctly classified samples
N = y_test.shape[0]
C = (y_test == y_pred).sum()

print("\nEvaluation:")
print(
    "Total points: %d Correctly labeled points: %d"
    % (N, C)
)


# Calculate accuracy
accuracy = C / N

print("Accuracy: %.2f%%" % (accuracy * 100))