UFC Matchmaker Prototype: Predicting Finishes vs. Decisions
The Vision
Most machine learning projects in MMA focus on predicting who will win. I wanted to build something different: a model that predicts how a fight will end. After the Paramount broadcasting deal, the UFC's matchmaking philosophy has clearly shifted toward high-tempo, action-packed bouts that result in finishes. I built this prototype to see if an algorithm could identify these explosive matchups the same way human matchmakers do.

The Approach: "Hybrid Mismatch"
Instead of simply comparing raw stats, I engineered features based on the concept of a "mismatch." During the development process, I realized a few harsh truths about fight data:

Age gaps are tricky: A 2-year age difference means nothing in the cage. Age only becomes a fatal variable when the gap hits a critical threshold (e.g., 6-7 years).

Wrestling cancels out: Two elite wrestlers often respect each other's grappling so much that the fight turns into a 15-minute kickboxing match.

The Lethality Core: To capture the true "kill or be killed" aura of a fight, I created a custom Mean_Lethality_Score that averages the historical finish rates of both fighters, acting as the primary engine for the model.

The 2026 Reality Check (Backtest)
I trained a Random Forest model on nearly a decade of UFC fights (2015–2024). To see how it handles the unpredictable nature of the sport, I blind-tested it on 116 fights from early 2026 (leading up to the Adesanya vs. Pyfer card).

The overall accuracy landed at 58%, which is a solid edge over a coin flip in a highly volatile sport. However, the real story is in the distribution:

🎯 Fight Night Main Card: 73% Correct (27/37)

🎯 Fight Night Prelims: 51% Correct (22/43)

🎯 PPV Main Card: 53% Correct (8/15)

🎯 PPV Prelims: 52% Correct (11/21)

The Insight: The model excels (73%) when dealing with established veterans on Fight Night main cards where there is a wealth of historical data. However, it struggles in PPV title fights (where elite fighters fight far more cautiously) and completely hits a wall during the Prelims.

Future Roadmap & V2.0
This is an early prototype, and analyzing its blind spots gave me a clear roadmap for the next iteration:

Solving the "Cold Start" Problem: The low accuracy in the prelims is heavily tied to debutants having zero UFC data. For V2.0, I plan to integrate data from regional promotions (via Tapology/Sherdog) to accurately assess incoming fighters.

Time Decay & Strength of Schedule: Currently, a knockout from 2018 holds the same weight as a knockout from last month. The model also overestimates the finish probability of fighters who crushed unranked opponents but struggle to finish top-tier contenders (e.g., Umar Nurmagomedov). Future models will heavily weigh a fighter's last 3–5 fights to capture their current "Prime."

Switching to XGBoost: Random Forest proved to be too "safe" and conservative. Because certain parameters (like age_dif) only impact the fight after passing specific thresholds, XGBoost will be much more effective at capturing these non-linear, extreme edge cases.

Thanks to MDABBERT for providing the foundational Ultimate UFC Dataset on Kaggle. Here is the link: "https://www.kaggle.com/datasets/mdabbert/ultimate-ufc-dataset"
