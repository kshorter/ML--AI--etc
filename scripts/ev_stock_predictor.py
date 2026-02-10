"""
EV Stock Prediction Model
Machine Learning for Predicting Good EV Stock Picks
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

print("=" * 80)
print("EV STOCK PREDICTION MODEL - MACHINE LEARNING ANALYSIS")
print("=" * 80)
print()

# ==================== 1. Load Data ====================
print("Step 1: Loading stock market data...")
data_path = 'sql/stock_market_data.csv'
df = pd.read_csv(data_path, skiprows=[0, 2])
print(f"✓ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
print()

# ==================== 2. Data Preprocessing ====================
print("Step 2: Preprocessing Tesla (TSLA) EV stock data...")

# Extract TSLA data
tsla_close = pd.to_numeric(df[('Close', 'TSLA')], errors='coerce')
tsla_high = pd.to_numeric(df[('High', 'TSLA')], errors='coerce')
tsla_low = pd.to_numeric(df[('Low', 'TSLA')], errors='coerce')
tsla_open = pd.to_numeric(df[('Open', 'TSLA')], errors='coerce')
tsla_volume = pd.to_numeric(df[('Volume', 'TSLA')], errors='coerce')

tsla_data = pd.DataFrame({
    'Close': tsla_close,
    'High': tsla_high,
    'Low': tsla_low,
    'Open': tsla_open,
    'Volume': tsla_volume
})

tsla_data = tsla_data.dropna()
print(f"✓ TSLA data cleaned: {len(tsla_data)} valid trading days")
print()

# ==================== 3. Feature Engineering ====================
print("Step 3: Creating technical indicators and features...")

# Simple Moving Averages
tsla_data['SMA_5'] = tsla_data['Close'].rolling(window=5).mean()
tsla_data['SMA_20'] = tsla_data['Close'].rolling(window=20).mean()

# Exponential Moving Average
tsla_data['EMA_12'] = tsla_data['Close'].ewm(span=12).mean()

# Momentum (Price change over 5 days)
tsla_data['Momentum'] = tsla_data['Close'].pct_change(5)

# Volatility (Standard deviation of returns)
tsla_data['Volatility'] = tsla_data['Close'].pct_change().rolling(window=5).std()

# RSI (Relative Strength Index)
delta = tsla_data['Close'].diff()
gain = (delta.where(delta > 0, 0)).rolling(window=5).mean()
loss = (-delta.where(delta < 0, 0)).rolling(window=5).mean()
rs = gain / loss
tsla_data['RSI'] = 100 - (100 / (1 + rs))

# MACD (Moving Average Convergence Divergence)
ema_12 = tsla_data['Close'].ewm(span=12).mean()
ema_26 = tsla_data['Close'].ewm(span=26).mean()
tsla_data['MACD'] = ema_12 - ema_26

# High-Low Range
tsla_data['HL_Range'] = (tsla_data['High'] - tsla_data['Low']) / tsla_data['Close']

# Volume change
tsla_data['Volume_Change'] = tsla_data['Volume'].pct_change()

# Target: Good pick if 5-day future return > 2%
future_return = tsla_data['Close'].shift(-5) / tsla_data['Close'] - 1
tsla_data['Target'] = (future_return > 0.02).astype(int)

tsla_data = tsla_data.dropna()

print(f"✓ Features engineered: {len(tsla_data.columns)} total columns")
print(f"  Technical indicators: SMA_5, SMA_20, EMA_12, Momentum, Volatility, RSI, MACD, HL_Range, Volume_Change")
print()

# ==================== 4. Prepare Training Data ====================
print("Step 4: Preparing training and test datasets...")

feature_columns = ['SMA_5', 'SMA_20', 'EMA_12', 'Momentum', 'Volatility', 
                   'RSI', 'MACD', 'HL_Range', 'Volume_Change']

X = tsla_data[feature_columns]
y = tsla_data['Target']

# Normalize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"✓ Training set: {X_train.shape[0]} samples")
print(f"✓ Testing set: {X_test.shape[0]} samples")
print(f"  Good picks in training: {y_train.sum()} ({y_train.mean()*100:.1f}%)")
print(f"  Good picks in testing: {y_test.sum()} ({y_test.mean()*100:.1f}%)")
print()

# ==================== 5. Train Model ====================
print("Step 5: Training Random Forest Classifier...")

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train, y_train)
print("✓ Model training complete!")
print()

# ==================== 6. Evaluate Model ====================
print("Step 6: Evaluating model performance...")
print()

y_train_pred = rf_model.predict(X_train)
y_test_pred = rf_model.predict(X_test)
y_test_proba = rf_model.predict_proba(X_test)[:, 1]

# Metrics
accuracy = accuracy_score(y_test, y_test_pred)
precision = precision_score(y_test, y_test_pred)
recall = recall_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred)

print("=" * 60)
print("MODEL PERFORMANCE METRICS")
print("=" * 60)
print(f"Accuracy:  {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"Precision: {precision:.4f} ({precision*100:.2f}%)")
print(f"Recall:    {recall:.4f} ({recall*100:.2f}%)")
print(f"F1-Score:  {f1:.4f}")
print()

cm = confusion_matrix(y_test, y_test_pred)
print("Confusion Matrix:")
print(f"  True Negatives:  {cm[0,0]:4d}  |  False Positives: {cm[0,1]:4d}")
print(f"  False Negatives: {cm[1,0]:4d}  |  True Positives:  {cm[1,1]:4d}")
print()

fpr, tpr, _ = roc_curve(y_test, y_test_proba)
roc_auc = auc(fpr, tpr)
print(f"ROC AUC Score: {roc_auc:.4f}")
print()

# ==================== 7. Feature Importance ====================
print("=" * 60)
print("FEATURE IMPORTANCE RANKINGS")
print("=" * 60)
feature_importance = pd.DataFrame({
    'Feature': feature_columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

for idx, row in feature_importance.iterrows():
    bar = '█' * int(row['Importance'] * 100)
    print(f"{row['Feature']:15s} | {bar} {row['Importance']:.4f}")
print()

# ==================== 8. Recent Predictions ====================
print("=" * 60)
print("RECENT EV STOCK PREDICTIONS (LAST 20 TRADING DAYS)")
print("=" * 60)
print("Prediction: 1 = Good Pick (Expected >2% return), 0 = Avoid")
print()

recent_data = X_scaled[-20:]
recent_predictions = rf_model.predict(recent_data)
recent_probabilities = rf_model.predict_proba(recent_data)[:, 1]

results_df = pd.DataFrame({
    'Day': range(len(recent_predictions), 0, -1),
    'Prediction': ['BUY ✓' if p == 1 else 'HOLD ✗' for p in recent_predictions],
    'Confidence': recent_probabilities,
    'Rating': ['★★★★★' if p >= 0.8 else '★★★★' if p >= 0.6 else '★★★' if p >= 0.5 else '★★' if p >= 0.4 else '★' for p in recent_probabilities]
})

print(results_df.to_string(index=False))
print()

good_picks = (recent_predictions == 1).sum()
avg_confidence = recent_probabilities[recent_predictions == 1].mean() if good_picks > 0 else 0

print("=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"✓ Good Picks Identified: {good_picks}/{len(recent_predictions)} days ({good_picks/len(recent_predictions)*100:.1f}%)")
print(f"✓ Average Confidence (Good Picks): {avg_confidence:.2%}")
print(f"✓ Model Accuracy: {accuracy:.2%}")
print(f"✓ Most Important Feature: {feature_importance.iloc[0]['Feature']}")
print()
print("Note: This model predicts short-term EV stock movements based on technical")
print("indicators. Always do your own research before making investment decisions!")
print("=" * 60)
