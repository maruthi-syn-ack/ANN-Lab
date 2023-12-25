import numpy as np 
import pandas as pd 
import seaborn as  sns
import matplotlib.pyplot as plt 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix , accuracy_score , classification_report
from sklearn.model_selection import train_test_split

Datset = '../Data/Pulsar/pulsar_stars.csv'

df = pd.read_csv(Datset)
df.hist(figsize=(12, 10), bins=20)
plt.show()

X = df.drop('target_class', axis=1)
y = df['target_class']

X_train ,X_test ,y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

scaller = StandardScaler()

X_train_scalled  = scaller.fit_transform(X_train)
X_test_scalled =scaller.transform(X_test)

SVC_model = SVC(kernel='linear',C=1.0,random_state=42)
SVC_model.fit(X_train_scalled,y_train)

y_pred = SVC_model.predict(X_test_scalled)

acc = accuracy_score(y_test,y_pred)
cm  = confusion_matrix(y_test,y_pred)
cr = classification_report(y_test,y_pred)

print(f"Accurcy {acc}")
print(f"cr : {cr}")

sns.heatmap(cm,annot=True,fmt='d',linewidths=0.5)