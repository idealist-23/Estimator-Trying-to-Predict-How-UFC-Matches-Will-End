import pandas as pd
import numpy as np

print("Step 1: Building Fighter Profiles (Fighter_Profiles.csv)...")

df = pd.read_csv('ufc-master.csv')
df['date'] = pd.to_datetime(df['date'])

# Freeze historical data prior to 2026 to prevent target leakage
df_history = df[df['date'].dt.year < 2026].copy()
df_history = df_history.iloc[::-1].reset_index(drop=True)

all_names = pd.concat([df_history['R_fighter'], df_history['B_fighter']]).dropna().unique()
all_profiles = pd.DataFrame(all_names, columns=['Fighter'])

metrics = ['Height_cms', 'age', 'Stance', 'Reach_cms', 'current_win_streak', 
           'avg_SIG_STR_landed', 'avg_SUB_ATT', 'avg_TD_landed', 
           'win_by_KO/TKO', 'win_by_Submission', 'wins']

for m in metrics:
    r_dict = df_history.set_index('R_fighter')[f'R_{m}'].dropna().to_dict()
    b_dict = df_history.set_index('B_fighter')[f'B_{m}'].dropna().to_dict()
    cum_dict = {**b_dict, **r_dict}
    all_profiles[m] = all_profiles['Fighter'].map(cum_dict)

# Calculate Lethality Engine (Finish Rate)
all_profiles['finish_rate'] = (all_profiles['win_by_KO/TKO'] + all_profiles['win_by_Submission']) / all_profiles['wins']
all_profiles['finish_rate'] = all_profiles['finish_rate'].replace([np.inf, -np.inf], 0).fillna(0)

all_profiles.to_csv('Fighter_Profiles.csv', index=False)
print(f"Process Complete. {len(all_profiles)} fighter profiles saved.")
