import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import warnings

warnings.simplefilter(action='ignore')

print("Step 2: Training the Hybrid Mismatch Model...")

df = pd.read_csv('ufc-master.csv')
df['date'] = pd.to_datetime(df['date'])

# Target Variable
df['Is_Finish'] = df['finish'].apply(lambda x: 1 if 'KO/TKO' in str(x).upper() or 'SUB' in str(x).upper() else 0)

# Feature Engineering
df['title_bout'] = df['title_bout'].astype(int)
df['Is_Womens_Bout'] = df['weight_class'].apply(lambda x: 1 if isinstance(x, str) and 'Women' in x else 0)
df['Is_Opposite_Stance'] = (df['R_Stance'].fillna('O') != df['B_Stance'].fillna('O')).astype(int)

# Lethality Core
r_rate = (df['R_win_by_KO/TKO'] + df['R_win_by_Submission']) / df['R_wins'].replace(0, 1)
b_rate = (df['B_win_by_KO/TKO'] + df['B_win_by_Submission']) / df['B_wins'].replace(0, 1)
df['Mean_Of_Finishing_Rate'] = (r_rate.fillna(0) + b_rate.fillna(0)) / 2

# Absolute Differences (Mismatch Indicators)
df['Abs_Age_Dif'] = np.abs(df['age_dif'])
df['Abs_Reach_Dif'] = np.abs(df['reach_dif'])
df['Abs_Win_Streak_Dif'] = np.abs(df['win_streak_dif'])
df['Abs_Sig_Str_Dif'] = np.abs(df['sig_str_dif'])
df['Abs_TD_Dif'] = np.abs(df['R_avg_TD_landed'] - df['B_avg_TD_landed'])
df['Abs_Sub_Att_Dif'] = np.abs(df['R_avg_SUB_ATT'] - df['B_avg_SUB_ATT'])

features = ['Abs_Age_Dif', 'Abs_Reach_Dif', 'Abs_Win_Streak_Dif', 'title_bout', 
            'Is_Womens_Bout', 'Is_Opposite_Stance', 'Mean_Of_Finishing_Rate', 
            'Abs_Sig_Str_Dif', 'Abs_TD_Dif', 'Abs_Sub_Att_Dif']

# Data Split (Using pre-2026 data for training to avoid leakage)
train_data = df[df['date'].dt.year < 2026].dropna(subset=features + ['Is_Finish'])
test_data = df[(df['date'] >= '2026-01-01') & (df['date'] <= '2026-05-09')].dropna(subset=features + ['Is_Finish'])

# Model Training (Optimized with n_estimators=1000)
rf = RandomForestClassifier(n_estimators=1000, random_state=18)
rf.fit(train_data[features], train_data['Is_Finish'])

# Evaluation
predictions = rf.predict(test_data[features])
accuracy = accuracy_score(test_data['Is_Finish'], predictions)

print(f"Evaluated Matches (2026 Season): {len(test_data)}")
print(f"Model Out-of-Sample Accuracy: {accuracy * 100:.2f}%")

# Export Model
joblib.dump(rf, 'ufc_model.pkl')
print("Model saved successfully as 'ufc_model.pkl'.")
