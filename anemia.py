import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix

# Load the data
data = pd.read_csv("smallanemia.csv")

# Data cleaning and preprocessing
# (You may need to handle missing values, encode categorical variables, and scale numerical features)

# Split the data into training and testing sets
X = data.drop('Result', axis=1)
y = data['Result']
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train and evaluate models
models = {
    'Random Forest': RandomForestClassifier(),
    'Decision Tree': DecisionTreeClassifier(),
    'Logistic Regression': LogisticRegression(),
    'Naive Bayes': GaussianNB(),
    'Support Vector Machine': SVC(),
    'XGBoost': xgb.XGBClassifier(n_estimators=900, learning_rate=0.1)
}

for name, model in models.items():
    model.fit(train_X, train_y)
    y_pred = model.predict(test_X)
    accuracy = accuracy_score(test_y, y_pred)
    print(f'The accuracy of {name} is: {accuracy}')

    # Cross-validation
    cv_accuracy = cross_val_score(model, X, y, cv=10, scoring='accuracy')
    print(f'The cross-validated accuracy for {name} is: {cv_accuracy.mean()}')

    # Confusion matrix
    cm = confusion_matrix(test_y, y_pred)
    print(f'Confusion matrix for {name}:')
    print(cm)

    # Heatmap for confusion matrix
    sns.heatmap(cm, annot=True, fmt='d')
    plt.title(f'Confusion Matrix for {name}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

# Voting Classifier
voting_clf = VotingClassifier(estimators=[('Random Forest', RandomForestClassifier()),
                                          ('Decision Tree', DecisionTreeClassifier()),
                                          ('Logistic Regression', LogisticRegression()),
                                          ('Naive Bayes', GaussianNB()),
                                          ('Support Vector Machine', SVC()),
                                          ('XGBoost', xgb.XGBClassifier(n_estimators=900, learning_rate=0.1))
                                         ], voting='hard')

voting_clf.fit(train_X, train_y)
y_pred_voting = voting_clf.predict(test_X)
accuracy_voting = accuracy_score(test_y, y_pred_voting)
print(f'The accuracy of Voting Classifier is: {accuracy_voting}')

# Bagging
bagging_clf = BaggingClassifier(DecisionTreeClassifier(),
                                n_estimators=500,
                                bootstrap=True,
                                random_state=42)
bagging_clf.fit(train_X, train_y)
y_pred_bagging = bagging_clf.predict(test_X)
accuracy_bagging = accuracy_score(test_y, y_pred_bagging)
print(f'The accuracy of Bagging Classifier is: {accuracy_bagging}')

# XGBoost
xgboost = xgb.XGBClassifier(n_estimators=900, learning_rate=0.1)
result = cross_val_score(xgboost, X, y, cv=10, scoring='accuracy')
print(f'The cross-validated score for XGBoost is: {result.mean()}')