import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load the Wine Dataset
wine = load_wine()
X, y = wine.data, wine.target

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define initial K values to test 
k_values = [1, 3, 5, 7, 9]

# Train a KNN classifier for each k and record the accuracy scores
accuracies = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    accuracy = accuracy_score(y_test, knn.predict(X_test))
    accuracies.append(accuracy)

# Plot accuracy against K values
plt.scatter(k_values, accuracies)
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("Accuracy vs. K values (Sample Range)")
plt.show()

# Test a wider range of K values
k_values = range(1, 30)
accuracy_list = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy_list.append(accuracy)

# Find and print the best K
best_k = list(k_values)[accuracy_list.index(max(accuracy_list))]
print("Best value of K:", best_k)

# Plot full range
plt.plot(k_values, accuracy_list, marker='o')
plt.xlabel("K")
plt.ylabel("Accuracy")
plt.title("KNN Accuracy for K = 1 to 29")
plt.show()
