import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 

Dataset = '../Data/Sonar/sonar data.csv'
df = pd.read_csv(Dataset)
X = df.iloc[:, :-1]  # All columns except the last one
y = df.iloc[:, -1]   # Last column
y = np.where(y == 'R', 0, 1)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


perceptron_model = Perceptron(max_iter=100, random_state=42)
perceptron_model.fit(X_train, y_train)


y_pred = perceptron_model.predict(X_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy on the test set: {accuracy * 100:.2f}%")
cm = confusion_matrix(y_test,y_pred)
sns.heatmap(cm,annot=True,linewidths=0.5,fmt='d')