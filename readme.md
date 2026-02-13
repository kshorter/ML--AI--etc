# EV Stock Prediction Game 🎮📈

An interactive gamified Streamlit application that challenges you to predict Tesla stock movements against an AI-powered Random Forest model.

## Features

- 🎯 Test your trading skills against machine learning predictions
- 🔥 Track your score, streak, and achievements
- 💡 Get helpful hints (RSI, Momentum, AI confidence)
- 📊 Visualize stock data and technical indicators
- ⚙️ Multiple difficulty levels (Easy, Medium, Hard)

## Prerequisites

- Python 3.7 or higher
- pip package manager

## Installation

1. **Install required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   The main dependencies include:
   - streamlit
   - pandas
   - numpy
   - plotly
   - scikit-learn

## Running the Streamlit App

To start the Streamlit server and launch the application:

```bash
streamlit run streamlit_app.py
```

The app will automatically open in your default web browser at `http://localhost:8501`

### Alternative Options

- **Specify a custom port:**
  ```bash
  streamlit run streamlit_app.py --server.port 8080
  ```

- **Run without auto-opening browser:**
  ```bash
  streamlit run streamlit_app.py --server.headless true
  ```

## How to Play

1. Click **"New Challenge"** to start
2. Analyze the stock information and chart
3. Decide if the stock will gain >2% in the next 5 trading days
4. Make your prediction: BUY or HOLD
5. Compare your prediction with the AI model
6. Earn points and unlock achievements!

## Data Requirements

The app requires Tesla stock data located at:
- `sql/stock_market_data.csv`

Ensure this file exists before running the application.

## Troubleshooting

- **Import errors:** Make sure all dependencies are installed with `pip install -r requirements.txt`
- **Data file not found:** Verify the `sql/stock_market_data.csv` file exists
- **Port already in use:** Use a different port with `--server.port` option

---

**Enjoy the game and good luck beating the AI! 🚀**
