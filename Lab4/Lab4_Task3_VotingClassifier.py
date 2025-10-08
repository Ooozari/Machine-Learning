import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

df = pd.read_csv("../Dataset/income_evaluation.csv")
df = df.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
df = pd.get_dummies(df, drop_first=True)

X = df.drop("income_>50K", axis=1)
y = df["income_>50K"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# Initialize classifiers
log_clf = LogisticRegression(random_state=101)
rnd_clf = RandomForestClassifier(random_state=101)
svm_clf = SVC(random_state=101, probability=True)

# Create Voting Classifier (Hard Voting)
voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard'
)

# Fit the ensemble model
voting_clf.fit(X_train, y_train)

# Evaluate each classifier
print("\nModel Accuracies:")
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(f"{clf.__class__.__name__}: {accuracy_score(y_test, y_pred):.4f}")
