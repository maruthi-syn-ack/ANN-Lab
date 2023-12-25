import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt 
import seaborn as sns 
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input

Dataset = "../Data/Wheat-seed/seeds.csv"

df = pd.read_csv(Dataset)
X = df.iloc[:, 0:7].values
y = df.iloc[:, 7].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train = y_train - 1
y_test = y_test - 1

model = Sequential([
    Input(shape=(X_train.shape[1],)),
    Dense(100, activation='relu'),
    Dense(3, activation='softmax')  
])
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2)


loss, accuracy = model.evaluate(X_test, y_test)
print('Test loss:', loss)
print('Test accuracy:', accuracy)


y_predicted = model.predict(X_test)
y_predicted_labels = [np.argmax(i) for i in y_predicted]


cm = confusion_matrix(y_test, y_predicted_labels)
sns.heatmap(cm, annot=True, fmt="d",  linewidths=.5)
