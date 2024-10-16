# app.py
import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime, timedelta
import streamlit as st
import plotly.graph_objs as go

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

# Load the trained model and scaler
model = ReceiptCountLSTM(input_size=3, hidden_size=128, num_layers=3, output_size=1)
model.load_state_dict(torch.load('receipt_prediction_model.pth', map_location=torch.device('cpu')))
model.eval()

scaler = joblib.load('receipt_prediction_scaler.joblib')

# Function to predict receipts for a given year
def predict_yearly_receipts(year, last_sequence):
    predictions = []
    current_sequence = last_sequence.copy()
    
    start_date = datetime(year, 1, 1)
    end_date = datetime(year, 12, 31)
    current_date = start_date
    
    while current_date <= end_date:
        # Prepare input
        X = torch.FloatTensor(current_sequence).unsqueeze(0)
        
        # Make prediction
        with torch.no_grad():
            prediction = model(X)
        
        # Inverse transform the prediction
        prediction_original = scaler.inverse_transform(
            np.hstack([prediction.numpy(), np.zeros((1, 2))])
        )[0][0]
        
        predictions.append((current_date, prediction_original))
        
        # Update sequence
        new_day = np.array([[prediction.item(), current_date.weekday() / 6, year - 2021 + current_date.timetuple().tm_yday / 365]])
        current_sequence = np.vstack((current_sequence[1:], scaler.transform(new_day)))
        
        current_date += timedelta(days=1)
    
    return predictions

# Streamlit app
st.title('Receipt Count Prediction App')

# User input for the year
year = st.number_input('Enter the year for prediction:', min_value=2022, max_value=2030, value=2022)

if st.button('Predict'):
    # Load the last known sequence (you may need to adjust this based on your data)
    df = pd.read_csv('data_daily.csv')
    df['# Date'] = pd.to_datetime(df['# Date'])
    df.set_index('# Date', inplace=True)
    df['day_of_week'] = df.index.dayofweek / 6
    df['day_of_year'] = df.index.dayofyear / 365
    features = ['Receipt_Count', 'day_of_week', 'day_of_year']
    last_sequence = scaler.transform(df[features].values[-30:])

    # Make predictions
    predictions = predict_yearly_receipts(year, last_sequence)

    # Create DataFrame and resample to monthly
    df_predictions = pd.DataFrame(predictions, columns=['Date', 'Predicted_Receipts'])
    df_predictions.set_index('Date', inplace=True)
    monthly_predictions = df_predictions.resample('M').sum()

    # Display monthly predictions
    st.write(f"Monthly Receipt Predictions for {year}:")
    st.write(monthly_predictions)

    # Visualize predictions
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df_predictions.index, y=df_predictions['Predicted_Receipts'],
                             mode='lines', name='Daily Predictions'))
    fig.add_trace(go.Scatter(x=monthly_predictions.index, y=monthly_predictions['Predicted_Receipts'],
                             mode='markers', name='Monthly Sum'))

    fig.update_layout(title=f'Receipt Count Predictions for {year}',
                      xaxis_title='Date',
                      yaxis_title='Receipt Count')

    st.plotly_chart(fig)

# Instructions
st.markdown("""
## How to use this app:
1. Enter the year for which you want to predict receipt counts.
2. Click the 'Predict' button.
3. View the monthly predictions and the visualization.

Note: This model has been trained on historical data up to 2021. Predictions for years beyond 2022 may be less accurate due to potential changes in trends and patterns.
""")
