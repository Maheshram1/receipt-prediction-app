import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import joblib

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Load and preprocess the data
df = pd.read_csv('./data_daily.csv')
df['# Date'] = pd.to_datetime(df['# Date'])
df.set_index('# Date', inplace=True)

# Extract features
df['day_of_week'] = df.index.dayofweek / 6  # Normalize to [0, 1]
df['day_of_year'] = df.index.dayofyear / 365  # Normalize to [0, 1]

# Normalize the data
scaler = MinMaxScaler()
features = ['Receipt_Count', 'day_of_week', 'day_of_year']
df1 = df
x1 = df1[features].values
y1 = df1['Receipt_Count'].values
df[features] = scaler.fit_transform(df[features])

# Prepare the data for LSTM
X = df[features].values
y = df['Receipt_Count'].values


# Define the sequence length
seq_length = 30

# Create sequences
X_seq, y_seq = [], []
X1_seq, y1_seq = [], []
for i in range(len(X) - seq_length):
    X_seq.append(X[i:i+seq_length])
    y_seq.append(y[i+seq_length])
    X1_seq.append(x1[i:i+seq_length])
    y1_seq.append(y1[i:i+seq_length])

X_seq = np.array(X_seq)
y_seq = np.array(y_seq)
X1_seq = np.array(X1_seq)
y1_seq = np.array(y1_seq)

split_index = 300
# Split the data into training and testing sets
X_train, X_test = X_seq[:split_index], X_seq[split_index:]
y_train, y_test = y_seq[:split_index], y_seq[split_index:]

X1_train, X1_test = X1_seq[:split_index], X1_seq[split_index:]
y1_train, y1_test = y1_seq[:split_index], y1_seq[split_index:]


# Convert to PyTorch tensors
X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).view(-1, 1)
X_test = torch.FloatTensor(X_test)
y_test = torch.FloatTensor(y_test).view(-1, 1)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)

# Define the LSTM model
class ReceiptCountLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(ReceiptCountLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])
        return out

# Instantiate the model
input_size = 3  # Receipt_Count, day_of_week, day_of_year
hidden_size = 128
num_layers = 3
output_size = 1

model = ReceiptCountLSTM(input_size, hidden_size, num_layers, output_size)

# Define loss function and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# Train the model
num_epochs = 200
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for batch_X, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {total_loss/len(train_loader):.4f}')

# Evaluate the model
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f'Test Loss: {test_loss.item():.4f}')

# Function to inverse transform predictions
def inverse_transform_predictions(predictions, scaler):
    # Create a dummy dataframe with the same structure as the original
    dummy_df = pd.DataFrame(columns=features)
    dummy_df['Receipt_Count'] = predictions.flatten()
    dummy_df['day_of_week'] = 0  # These values don't matter for inverse_transform
    dummy_df['day_of_year'] = 0
    
    # Inverse transform
    inverse_transformed = scaler.inverse_transform(dummy_df)
    
    # Return only the Receipt_Count column
    return inverse_transformed[:, 0]

# Inverse transform test predictions and actual values
test_predictions_original = inverse_transform_predictions(test_predictions.numpy(), scaler)
# y_test_original = inverse_transform_predictions(y_test.numpy(), scaler)
y_test_original = y1_test

# Function to predict receipts for 2022
def predict_2022_receipts(model, scaler, last_sequence):
    predictions = []
    current_sequence = last_sequence.copy()
    
    start_date = datetime(2022, 1, 1)
    end_date = datetime(2022, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        # Prepare input
        X = torch.FloatTensor(current_sequence).unsqueeze(0)
        
        # Make prediction
        model.eval()
        with torch.no_grad():
            prediction = model(X)
        
        # Inverse transform the prediction
        prediction_original = inverse_transform_predictions(prediction.numpy(), scaler)[0]
        
        predictions.append((current_date, prediction_original))
        
        # Update sequence
        new_day = np.array([[prediction.item(), current_date.weekday() / 6, 1+current_date.timetuple().tm_yday / 365]])
        current_sequence = np.vstack((current_sequence[1:], new_day))
        
        current_date += timedelta(days=1)
    
    return predictions

# Get the last sequence from 2021 for prediction
last_date = df.index.max()
last_sequence = df.loc[last_date - timedelta(days=seq_length-1):last_date, features].values

# Predict receipts for 2022
predictions_2022 = predict_2022_receipts(model, scaler, last_sequence)

# Convert predictions to DataFrame
df_predictions = pd.DataFrame(predictions_2022, columns=['Date', 'Predicted_Receipts'])
df_predictions.set_index('Date', inplace=True)

# Group by month and calculate the sum
monthly_predictions = df_predictions.resample('M').sum()

print("Monthly Receipt Predictions for 2022:")
print(monthly_predictions)

# Visualization function
def visualize_predictions(df, predictions_2022, test_predictions, y_test):
    plt.figure(figsize=(15, 10))

    # Plot historical data
    historical_data = scaler.inverse_transform(df[features])[:, 0]
    plt.plot(df.index, historical_data, label='Historical Data', color='blue', alpha=0.7)

    # Plot test predictions
    test_dates = df.index[-len(y1_test):]
    plt.scatter(test_dates, y1[-len(y1_test):], color='green', alpha=0.5, label='Actual Test Values')
    plt.scatter(test_dates, test_predictions, color='orange', alpha=0.5, label='Test Predictions')

    # Plot 2022 predictions
    dates_2022 = [p[0] for p in predictions_2022]
    values_2022 = [p[1] for p in predictions_2022]
    plt.plot(dates_2022, values_2022, label='2022 Predictions', color='red', alpha=0.7)

    plt.title('Receipt Count: Historical Data, Test Predictions, and 2022 Forecast')
    plt.xlabel('Date')
    plt.ylabel('Receipt Count')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('receipt_predictions_with_test.png')
    plt.close()

# Visualize the results
visualize_predictions(df, predictions_2022, test_predictions_original, y_test_original)

# Save the model
torch.save(model.state_dict(), 'receipt_prediction_model.pth')

# Save the scaler
joblib.dump(scaler, 'receipt_prediction_scaler.joblib')

print("Model and scaler saved successfully.")
