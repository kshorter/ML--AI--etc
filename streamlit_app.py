"""
EV Stock Prediction Game - Gamified Streamlit Application
Challenge yourself against the ML model!
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from datetime import datetime
import random

# Page configuration
st.set_page_config(
    page_title="📈 EV Stock Prediction Game",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for gamification
st.markdown("""
    <style>
    .big-font {
        font-size:50px !important;
        font-weight: bold;
        text-align: center;
    }
    .score-box {
        background-color: #1E1E1E;
        padding: 20px;
        border-radius: 10px;
        border: 2px solid #4CAF50;
        text-align: center;
    }
    .correct {
        background-color: #4CAF50;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .incorrect {
        background-color: #f44336;
        color: white;
        padding: 10px;
        border-radius: 5px;
        font-weight: bold;
    }
    .hint-box {
        background-color: #FFF3CD;
        padding: 15px;
        border-radius: 5px;
        border-left: 5px solid #FFC107;
        color: #856404;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'score' not in st.session_state:
    st.session_state.score = 0
if 'total_guesses' not in st.session_state:
    st.session_state.total_guesses = 0
if 'streak' not in st.session_state:
    st.session_state.streak = 0
if 'best_streak' not in st.session_state:
    st.session_state.best_streak = 0
if 'hints_used' not in st.session_state:
    st.session_state.hints_used = 0
if 'current_index' not in st.session_state:
    st.session_state.current_index = None
if 'game_started' not in st.session_state:
    st.session_state.game_started = False
if 'achievements' not in st.session_state:
    st.session_state.achievements = []
if 'difficulty' not in st.session_state:
    st.session_state.difficulty = 'Medium'

@st.cache_data
def load_and_prepare_data():
    """Load and prepare the stock data with features"""
    # Load data
    data_path = 'sql/stock_market_data.csv'
    df = pd.read_csv(data_path)
    
    # Skip first two rows (Ticker row and Date row) and extract TSLA data
    # TSLA is in columns with .2 suffix: Close.2, High.2, Low.2, Open.2, Volume.2
    df = df.iloc[2:].reset_index(drop=True)  # Skip first 2 rows
    
    tsla_close = pd.to_numeric(df['Close.2'], errors='coerce')
    tsla_high = pd.to_numeric(df['High.2'], errors='coerce')
    tsla_low = pd.to_numeric(df['Low.2'], errors='coerce')
    tsla_open = pd.to_numeric(df['Open.2'], errors='coerce')
    tsla_volume = pd.to_numeric(df['Volume.2'], errors='coerce')
    
    tsla_data = pd.DataFrame({
        'Close': tsla_close,
        'High': tsla_high,
        'Low': tsla_low,
        'Open': tsla_open,
        'Volume': tsla_volume
    })
    
    tsla_data = tsla_data.dropna()
    
    # Feature Engineering
    tsla_data['SMA_5'] = tsla_data['Close'].rolling(window=5).mean()
    tsla_data['SMA_20'] = tsla_data['Close'].rolling(window=20).mean()
    tsla_data['EMA_12'] = tsla_data['Close'].ewm(span=12).mean()
    tsla_data['Momentum'] = tsla_data['Close'].pct_change(5)
    tsla_data['Volatility'] = tsla_data['Close'].pct_change().rolling(window=5).std()
    
    delta = tsla_data['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
    rs = gain / loss
    tsla_data['RSI'] = 100 - (100 / (1 + rs))
    
    ema_12 = tsla_data['Close'].ewm(span=12).mean()
    ema_26 = tsla_data['Close'].ewm(span=26).mean()
    tsla_data['MACD'] = ema_12 - ema_26
    tsla_data['HL_Range'] = (tsla_data['High'] - tsla_data['Low']) / tsla_data['Close']
    tsla_data['Volume_Change'] = tsla_data['Volume'].pct_change()
    
    future_return = tsla_data['Close'].shift(-5) / tsla_data['Close'] - 1
    tsla_data['Target'] = (future_return > 0.02).astype(int)
    tsla_data['Future_Return'] = future_return * 100
    
    tsla_data = tsla_data.dropna()
    
    return tsla_data

@st.cache_resource
def train_model(tsla_data):
    """Train the Random Forest model"""
    feature_columns = ['SMA_5', 'SMA_20', 'EMA_12', 'Momentum', 'Volatility', 
                       'RSI', 'MACD', 'HL_Range', 'Volume_Change']
    
    X = tsla_data[feature_columns]
    y = tsla_data['Target']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    rf_model = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    rf_model.fit(X_scaled, y)
    
    return rf_model, scaler, feature_columns

def get_achievement_emoji(name):
    """Get emoji for achievement"""
    emojis = {
        'First Win': '🎯',
        'Hot Streak': '🔥',
        'Perfect 5': '⭐',
        'Master Trader': '👑',
        'Fortune Teller': '🔮',
        'Risk Taker': '🎲',
        'Analyzer': '🔍',
        'Consistent': '💎'
    }
    return emojis.get(name, '🏆')

def check_achievements():
    """Check and award achievements"""
    new_achievements = []
    
    if st.session_state.score >= 1 and 'First Win' not in st.session_state.achievements:
        new_achievements.append('First Win')
        st.session_state.achievements.append('First Win')
    
    if st.session_state.streak >= 3 and 'Hot Streak' not in st.session_state.achievements:
        new_achievements.append('Hot Streak')
        st.session_state.achievements.append('Hot Streak')
        
    if st.session_state.streak >= 5 and 'Perfect 5' not in st.session_state.achievements:
        new_achievements.append('Perfect 5')
        st.session_state.achievements.append('Perfect 5')
        
    if st.session_state.score >= 20 and 'Master Trader' not in st.session_state.achievements:
        new_achievements.append('Master Trader')
        st.session_state.achievements.append('Master Trader')
        
    if st.session_state.total_guesses >= 10 and 'Fortune Teller' not in st.session_state.achievements:
        new_achievements.append('Fortune Teller')
        st.session_state.achievements.append('Fortune Teller')
    
    return new_achievements

# Load data and train model
tsla_data = load_and_prepare_data()
model, scaler, feature_columns = train_model(tsla_data)

# Calculate predictions for all data
X_all = tsla_data[feature_columns]
X_all_scaled = scaler.transform(X_all)
predictions = model.predict(X_all_scaled)
probabilities = model.predict_proba(X_all_scaled)[:, 1]

# Header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🎮 EV Stock Prediction Game 📈</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center;'>Challenge the AI and Predict Tesla Stock Movements!</h3>", unsafe_allow_html=True)

# Sidebar - Game Stats and Controls
with st.sidebar:
    st.header("🎯 Your Stats")
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Score", st.session_state.score)
        st.metric("Streak", f"{st.session_state.streak} 🔥")
    with col2:
        accuracy = (st.session_state.score / st.session_state.total_guesses * 100) if st.session_state.total_guesses > 0 else 0
        st.metric("Accuracy", f"{accuracy:.1f}%")
        st.metric("Best Streak", st.session_state.best_streak)
    
    st.divider()
    
    # Difficulty Selection
    st.header("⚙️ Settings")
    difficulty = st.selectbox(
        "Difficulty Level",
        ['Easy', 'Medium', 'Hard'],
        index=1,
        help="Easy: More hints available\nMedium: Balanced gameplay\nHard: Limited hints, higher stakes"
    )
    st.session_state.difficulty = difficulty
    
    # Difficulty info
    if difficulty == 'Easy':
        st.info("🟢 Easy Mode: 3 hints available, +1 point per correct guess")
        max_hints = 3
        points_per_correct = 1
    elif difficulty == 'Medium':
        st.info("🟡 Medium Mode: 2 hints available, +2 points per correct guess")
        max_hints = 2
        points_per_correct = 2
    else:
        st.info("🔴 Hard Mode: 1 hint available, +3 points per correct guess")
        max_hints = 1
        points_per_correct = 3
    
    st.divider()
    
    # Achievements
    st.header("🏆 Achievements")
    if st.session_state.achievements:
        for achievement in st.session_state.achievements:
            st.success(f"{get_achievement_emoji(achievement)} {achievement}")
    else:
        st.info("Play to unlock achievements!")
    
    st.divider()
    
    # Reset button
    if st.button("🔄 Reset Game", use_container_width=True):
        st.session_state.score = 0
        st.session_state.total_guesses = 0
        st.session_state.streak = 0
        st.session_state.best_streak = 0
        st.session_state.hints_used = 0
        st.session_state.current_index = None
        st.session_state.game_started = False
        st.session_state.achievements = []
        st.rerun()

# Main Game Area
if not st.session_state.game_started or st.session_state.current_index is None:
    st.info("👆 Click 'New Challenge' to start playing!")
    
    if st.button("🎲 New Challenge", use_container_width=True, type="primary"):
        # Select random index from data
        st.session_state.current_index = random.randint(50, len(tsla_data) - 10)
        st.session_state.game_started = True
        st.session_state.hints_used = 0
        st.rerun()
else:
    idx = st.session_state.current_index
    
    # Get current data point
    current_data = tsla_data.iloc[idx]
    current_features = X_all_scaled[idx].reshape(1, -1)
    model_prediction = predictions[idx]
    model_confidence = probabilities[idx]
    
    # Display challenge
    col1, col2, col3 = st.columns([2, 2, 1])
    
    with col1:
        st.subheader("📊 Stock Information")
        metric_col1, metric_col2, metric_col3 = st.columns(3)
        with metric_col1:
            st.metric("Close Price", f"${current_data['Close']:.2f}")
        with metric_col2:
            st.metric("High", f"${current_data['High']:.2f}")
        with metric_col3:
            st.metric("Low", f"${current_data['Low']:.2f}")
        
        # Price chart (last 30 days)
        chart_data = tsla_data.iloc[max(0, idx-30):idx+1]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=chart_data['Close'],
            mode='lines',
            name='Close Price',
            line=dict(color='#4CAF50', width=2)
        ))
        fig.update_layout(
            title="📈 Recent Price Movement (30 Days)",
            xaxis_title="Days Ago",
            yaxis_title="Price ($)",
            height=300,
            showlegend=False
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Your Challenge")
        st.markdown("""
        <div class='hint-box'>
        <h4>Will this be a GOOD PICK?</h4>
        <p>A "Good Pick" means the stock will gain <b>>2%</b> in the next 5 trading days.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### Make Your Prediction:")
        
        user_guess = st.radio(
            "What's your call?",
            ["🚀 BUY (Good Pick)", "⏸️ HOLD (Not a Good Pick)"],
            key=f"guess_{idx}",
            label_visibility="collapsed"
        )
        
        # Confidence slider
        user_confidence = st.slider(
            "How confident are you?",
            0, 100, 50,
            help="Slide to indicate your confidence level"
        )
        
        # Submit button
        submit_col1, submit_col2 = st.columns(2)
        with submit_col1:
            submit_guess = st.button("✅ Submit Guess", use_container_width=True, type="primary")
        with submit_col2:
            skip_challenge = st.button("⏭️ Skip", use_container_width=True)
    
    with col3:
        st.subheader("💡 Hints")
        max_hints = {'Easy': 3, 'Medium': 2, 'Hard': 1}[st.session_state.difficulty]
        hints_remaining = max_hints - st.session_state.hints_used
        
        st.info(f"Hints: {hints_remaining}/{max_hints}")
        
        if hints_remaining > 0:
            if st.button("🔍 Show Hint", use_container_width=True):
                st.session_state.hints_used += 1
                
            if st.session_state.hints_used >= 1:
                st.success(f"RSI: {current_data['RSI']:.1f}")
                if current_data['RSI'] > 70:
                    st.caption("⚠️ Overbought")
                elif current_data['RSI'] < 30:
                    st.caption("📈 Oversold")
                    
            if st.session_state.hints_used >= 2:
                momentum = current_data['Momentum'] * 100
                st.success(f"Momentum: {momentum:+.2f}%")
                
            if st.session_state.hints_used >= 3:
                st.success(f"AI Confidence: {model_confidence:.1%}")
    
    # Process guess
    if submit_guess:
        user_prediction = 1 if "BUY" in user_guess else 0
        correct = (user_prediction == model_prediction)
        
        st.session_state.total_guesses += 1
        
        if correct:
            points_earned = points_per_correct
            st.session_state.score += points_earned
            st.session_state.streak += 1
            if st.session_state.streak > st.session_state.best_streak:
                st.session_state.best_streak = st.session_state.streak
            
            st.success(f"""
            ### 🎉 CORRECT! 
            **You earned {points_earned} points!**
            
            - Your Prediction: {'BUY' if user_prediction == 1 else 'HOLD'}
            - AI Prediction: {'BUY' if model_prediction == 1 else 'HOLD'}
            - AI Confidence: {model_confidence:.1%}
            - Actual Return: {current_data['Future_Return']:+.2f}%
            - Current Streak: {st.session_state.streak} 🔥
            """)
        else:
            st.session_state.streak = 0
            st.error(f"""
            ### ❌ INCORRECT
            **Better luck next time!**
            
            - Your Prediction: {'BUY' if user_prediction == 1 else 'HOLD'}
            - AI Prediction: {'BUY' if model_prediction == 1 else 'HOLD'}
            - AI Confidence: {model_confidence:.1%}
            - Actual Return: {current_data['Future_Return']:+.2f}%
            - Streak Reset to 0
            """)
        
        # Check for new achievements
        new_achievements = check_achievements()
        if new_achievements:
            for achievement in new_achievements:
                st.balloons()
                st.success(f"🏆 Achievement Unlocked: {get_achievement_emoji(achievement)} {achievement}")
        
        # Next challenge button
        if st.button("➡️ Next Challenge", use_container_width=True, type="primary"):
            st.session_state.current_index = random.randint(50, len(tsla_data) - 10)
            st.session_state.hints_used = 0
            st.rerun()
    
    if skip_challenge:
        st.session_state.current_index = random.randint(50, len(tsla_data) - 10)
        st.session_state.hints_used = 0
        st.rerun()

# Bottom section - Model insights
st.divider()
st.header("🤖 About the AI Model")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Feature Importance")
    feature_importance = pd.DataFrame({
        'Feature': feature_columns,
        'Importance': model.feature_importances_
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        feature_importance,
        x='Importance',
        y='Feature',
        orientation='h',
        title='What the AI Considers Most Important',
        color='Importance',
        color_continuous_scale='Greens'
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("How It Works")
    st.markdown("""
    The AI uses a **Random Forest Classifier** trained on Tesla stock data with these features:
    
    - **SMA (Simple Moving Average)**: Trend indicators
    - **EMA (Exponential Moving Average)**: Recent price focus
    - **RSI (Relative Strength Index)**: Overbought/oversold signals
    - **MACD**: Momentum indicator
    - **Momentum**: Rate of price change
    - **Volatility**: Price stability measure
    - **Volume Change**: Trading activity
    - **HL Range**: Daily price range
    
    The model predicts if the stock will gain **>2%** in the next **5 trading days**.
    """)

st.info("💡 **Pro Tip**: Use the hints wisely! RSI below 30 often indicates oversold (potential buy), while above 70 indicates overbought (potential sell).")
