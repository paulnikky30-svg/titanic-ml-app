import pandas as pd

# Load dataset
df = pd.read_csv("titanic.csv")

# Cleaning
df.drop("Cabin", axis=1, inplace=True)

df["Age"] = df["Age"].fillna(df["Age"].mean())

df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

df["Sex"] = df["Sex"].map({"male":0, "female":1})

df["Embarked"] = df["Embarked"].map({"S":0, "C":1, "Q":2})

df.drop(["Name", "Ticket"], axis=1, inplace=True)

# Input Output
X = df.drop("Survived", axis=1)
y = df["Survived"]

# Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model Training
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=200)

model.fit(X_train, y_train)

print("Model Trained Successfully")

# Prediction
y_pred = model.predict(X_test)

# Print Predictions
print(y_pred)

# Accuracy Check
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

# Confusion_Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred)

print(cm)

# Classification Report

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# Random Forest model

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Create Model
rf_model = RandomForestClassifier(n_estimators=100)

# Train Model
rf_model.fit(X_train, y_train)

# Prediction
rf_pred = rf_model.predict(X_test)

# Accuracy
rf_accuracy = accuracy_score(y_test, rf_pred)

print("Random Forest Accuracy:", rf_accuracy)

# Model Save(pickle)

import pickle

pickle.dump(rf_model, open("titanic_rf_model.pkl", "wb"))

print("Random Forest Model Saved")

# Load saved model + prediction test 

import pickle
import pandas as pd

model = pickle.load(open("titanic_rf_model.pkl", "rb"))

sample = pd.DataFrame([[
    1, 1, 1, 22, 0, 0, 100, 0
]], columns=[
    "PassengerId","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"
])

result = model.predict(sample)

print("Raw Output:", result)

if result[0] == 1:
    print("Prediction: Survived")
else:
    print("Prediction: Not Survived")

