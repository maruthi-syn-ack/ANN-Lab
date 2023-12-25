import numpy as np 
import pandas as pd 
import seaborn as sns 
import matplotlib.pyplot as plt 
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error , r2_score

Datset = '../Data/Time-Series/MLTempDataset1.csv'
df = pd.read_csv(Datset)
# Convert 'Datetime' column to datetime type
df['Datetime'] = pd.to_datetime(df['Datetime'])

df = df.sort_values('Datetime')
# Set 'Datetime' as the index
df.set_index('Datetime', inplace=True)

scaler = MinMaxScaler()
df['Hourly_Temp'] = scaler.fit_transform(df['Hourly_Temp'].values.reshape(-1, 1))

# Create sequences for training
def create_sequences(data, seq_length):
    sequences, targets = [], []
    for i in range(len(data) - seq_length):
        seq = data[i:i+seq_length]
        target = data[i+seq_length]
        sequences.append(seq)
        targets.append(target)
    return np.array(sequences), np.array(targets)
sequence_length = 10
X, y = create_sequences(df['Hourly_Temp'], sequence_length)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

mlp_regressor = MLPRegressor(hidden_layer_sizes=(50,), max_iter=1000)
mlp_regressor.fit(X_train, y_train)

test_predictions = mlp_regressor.predict(X_test)

test_predictions_original = scaler.inverse_transform(test_predictions.reshape(-1, 1))
y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1))

mse = mean_squared_error(y_test_original, test_predictions_original)
r2 =r2_score(y_test_original, test_predictions_original)

print(f'Mean Squared Error on Test Set: {mse}')
print(f'R2 score on Test Set: {r2}')

