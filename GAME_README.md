# 🎮 EV Stock Prediction Game

An interactive, gamified Streamlit application that challenges you to predict Tesla stock movements using the same features as a trained ML model!

## 🌟 Features

### Game Mechanics
- **Challenge Mode**: Predict whether Tesla stock will gain >2% in the next 5 trading days
- **Difficulty Levels**: 
  - 🟢 Easy: 3 hints, +1 point per correct guess
  - 🟡 Medium: 2 hints, +2 points per correct guess
  - 🔴 Hard: 1 hint, +3 points per correct guess
- **Smart Hints**: Progressive hints reveal RSI, Momentum, and AI Confidence
- **Real-time Feedback**: See how you compare to the AI model's predictions

### Scoring System
- Track your **Score**, **Accuracy**, and **Streak**
- Build winning streaks to maximize points
- Reset stats anytime to start fresh

### 🏆 Achievements
Unlock achievements as you play:
- 🎯 **First Win** - Score your first point
- 🔥 **Hot Streak** - Get 3 correct predictions in a row
- ⭐ **Perfect 5** - Achieve a 5-prediction winning streak
- 👑 **Master Trader** - Reach 20 points
- 🔮 **Fortune Teller** - Make 10 total predictions

### Visualizations
- 📈 **Price Charts**: View 30-day historical price trends
- 📊 **Feature Importance**: See what the AI considers most important
- 🎯 **Real-time Metrics**: Stock prices, RSI, momentum, and more

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package installer)

### Step 1: Install Required Packages

Open a terminal/command prompt in this directory and run:

```bash
pip install -r requirements.txt
```

Or install packages individually:

```bash
pip install streamlit pandas numpy plotly scikit-learn matplotlib seaborn
```

### Step 2: Verify Installation

Check if Streamlit is installed:

```bash
streamlit --version
```

## 🎯 How to Run

### Method 1: Using Command Line

Navigate to this directory and run:

```bash
streamlit run streamlit_app.py
```

### Method 2: Using Python

```bash
python -m streamlit run streamlit_app.py
```

The game will automatically open in your default web browser at `http://localhost:8501`

## 🎮 How to Play

1. **Start a Challenge**: Click the "🎲 New Challenge" button
2. **Analyze the Data**: Review the stock price chart and metrics
3. **Use Hints (Optional)**: Click "🔍 Show Hint" to reveal technical indicators
4. **Make Your Prediction**: Choose BUY (Good Pick) or HOLD (Not a Good Pick)
5. **Submit**: Click "✅ Submit Guess" to see if you match the AI's prediction
6. **Track Progress**: Watch your score, streak, and unlock achievements!

## 📚 Understanding the Game

### What is a "Good Pick"?
A stock is considered a "Good Pick" if it will gain **more than 2%** in the next **5 trading days**.

### Technical Indicators Explained

- **RSI (Relative Strength Index)**: 
  - Below 30 = Oversold (potential buy signal)
  - Above 70 = Overbought (potential sell signal)
  
- **Momentum**: Rate of price change over 5 days
  - Positive momentum suggests upward trend
  - Negative momentum suggests downward trend

- **AI Confidence**: The model's certainty in its prediction (0-100%)

### The AI Model

The game uses a **Random Forest Classifier** trained on Tesla stock data with these features:
- **SMA (Simple Moving Average)**: Trend indicators
- **EMA (Exponential Moving Average)**: Recent price focus
- **RSI**: Overbought/oversold signals
- **MACD**: Momentum indicator
- **Volatility**: Price stability measure
- **Volume Change**: Trading activity
- **HL Range**: Daily price range

## 🎯 Tips for Success

1. **Use hints strategically** - They're limited based on difficulty!
2. **Watch for RSI extremes** - <30 or >70 can signal reversals
3. **Check momentum** - Positive momentum often continues short-term
4. **Build streaks** - Consecutive wins maximize your score
5. **Learn from mistakes** - The AI shows why it made each prediction

## 📁 File Structure

```
.
├── streamlit_app.py           # Main game application
├── requirements.txt           # Python dependencies
├── GAME_README.md            # This file
├── sql/
│   └── stock_market_data.csv # Stock data (required)
├── scripts/
│   └── ev_stock_predictor.py # Original ML model
└── notebooks/
    └── ev_stock_prediction.ipynb # Jupyter notebook
```

## ⚠️ Important Notes

- **Data Required**: The game needs `sql/stock_market_data.csv` to function
- **For Educational Purposes**: This is a learning tool, not investment advice
- **Historical Data**: Predictions are based on historical data patterns
- **Always DYOR**: Do Your Own Research before making real investment decisions

## 🐛 Troubleshooting

### Streamlit Not Found
```bash
pip install streamlit --upgrade
```

### Missing Dependencies
```bash
pip install -r requirements.txt --upgrade
```

### Data File Not Found
Ensure `sql/stock_market_data.csv` exists in the correct location.

### Port Already in Use
Run Streamlit on a different port:
```bash
streamlit run streamlit_app.py --server.port 8502
```

## 🎉 Enjoy the Game!

Challenge yourself, compete with friends, and see if you can outsmart the AI! 

**Can you achieve a perfect streak? Can you unlock all achievements?**

Good luck, trader! 📈🚀
