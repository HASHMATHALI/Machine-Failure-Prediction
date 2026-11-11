# Machine Failure Prediction ⚙️

Predicting machine failure from sensor readings, so maintenance can be scheduled before a breakdown
instead of after one.

## Dataset

`data.csv` — **944 readings**, 9 sensor features and a binary target.

| Feature | Meaning (as provided) |
| :-- | :-- |
| footfall | Number of people or objects passing the machine |
| tempMode | Temperature mode / setting |
| AQ | Air quality index near the machine |
| USS | Ultrasonic sensor reading |
| CS | Current sensor reading |
| VOC | Volatile organic compound level |
| RP | Rotational position / RPM |
| IP | Input pressure |
| Temperature | Operating temperature |
| **fail** | 1 = failure, 0 = no failure |

Class balance: 551 no-failure, 393 failure (42% failures). No missing values.

## Method

1. Exploratory analysis: correlations, class balance, feature distributions, feature-vs-failure plots.
2. **Hold out 20% of the real readings as a test set first.**
3. Balance the training set only with SMOTE, and fit the scaler on training data only.
4. Compare six models: Logistic Regression, Random Forest, Gradient Boosting, SVM, KNN, XGBoost.
5. Tune Random Forest with 5-fold grid search on F1.
6. Inspect feature importance, confusion matrix and ROC curve.

> **Why the order matters.** An earlier version oversampled with SMOTE *before* splitting. That puts
> synthetic rows built from test machines into the training set and lets the test set contain
> synthetic rows, which inflates every score. The notebook now splits first.

## Results

On 189 held-out real readings (110 no-failure, 79 failure):

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
| :-- | --: | --: | --: | --: | --: |
| Logistic Regression | 0.905 | 0.859 | 0.924 | 0.890 | 0.964 |
| Random Forest | 0.910 | 0.861 | 0.937 | 0.897 | 0.973 |
| Gradient Boosting | 0.910 | 0.878 | 0.911 | 0.894 | 0.968 |
| **SVM** | **0.926** | **0.911** | 0.911 | **0.911** | **0.977** |
| K-Nearest Neighbors | 0.884 | 0.828 | 0.911 | 0.868 | 0.946 |
| XGBoost | 0.900 | 0.866 | 0.899 | 0.882 | 0.961 |
| Tuned Random Forest | 0.905 | 0.859 | 0.924 | 0.890 | 0.975 |

A 10-fold cross-validated Random Forest (SMOTE inside each fold) gives ROC-AUC **0.963 ± 0.021**
and F1 **0.879 ± 0.038**, so the single split above is representative.

Most important features (tuned Random Forest): **VOC** (43%), AQ (17%), USS (13%), CS (6%),
Temperature (5%).

Takeaways:

- All six model families land within a few points of each other; the signal is strong and mostly
  carried by VOC and air-quality readings.
- Tuning the forest did not beat the default on this split. SVM is the best single model here.
- Recall on failures is about 0.91–0.94: roughly 1 failure in 12 would still be missed, which is the
  number a maintenance team should weigh against false alarms.

## Run it

```bash
pip install pandas numpy scikit-learn imbalanced-learn xgboost matplotlib seaborn jupyter
jupyter notebook machine_failure_prediction.ipynb
```

## Future work

- Time-ordered evaluation if timestamps become available (train on the past, test on the future).
- Cost-sensitive threshold choice: pick the operating point from the cost of a missed failure
  versus an unnecessary inspection.
- SHAP explanations per prediction for maintenance engineers.

---
Author: Mohammed Hashmath Ali
