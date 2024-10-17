# Receipt Prediction App

This application predicts receipt counts for a given year using an LSTM neural network model. It includes a Streamlit web interface for easy interaction.

## Features

- LSTM-based prediction model
- Interactive Streamlit web app
- Docker containerization for easy deployment

## Prerequisites

- Python 3.9+
- Docker (optional)

## Installation

1. Clone the repository:
git clone (https://github.com/Maheshram1/receipt-prediction-app.git)
cd receipt-prediction-app

2. Create a virtual environment and activate it:
python3 -m venv venv
source venv/bin/activate

3. Install the required packages:
pip install -r requirements.txt

## Usage

### Running with Python

1. Train the model:
python fetch.py

2. Start the Streamlit app:
streamlit run app.py

3. Open a web browser and go to `http://localhost:8501`

### Running with Docker

1. Build the Docker image:
docker build -t receipt-prediction-app .

2. Run the Docker container:
docker run -p 8501:8501 receipt-prediction-app

3. Open a web browser and go to `http://localhost:8501`

## Model Details

The prediction model uses an LSTM neural network trained on historical receipt data. It takes into account the day of the week and day of the year as features.
