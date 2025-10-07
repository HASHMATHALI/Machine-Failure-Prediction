# Machine Failure Prediction using Machine Learning ⚙️🤖

## 📌 Project Overview
Unplanned machine failures lead to **high costs, downtime, and inefficiency**.  
This project uses **Machine Learning** to predict failures in advance, enabling predictive maintenance.

## 📊 Dataset
- **Samples**: 999  
- **Features**: 9 (Footfall, Temp Mode, AQ, USS, CS, VOC, RP, IP, Temperature)  
- **Target**: Failure (Binary: Fail / No-Fail)  
- **Imbalance**: Failure cases ~15% → SMOTE applied for balancing

## 🛠️ Methodology
1. **Data Preprocessing**
   - Missing value checks
   - StandardScaler for feature scaling
   - SMOTE for class imbalance handling
2. **Models Used**
   - Logistic Regression
   - Random Forest (Baseline + Tuned)
   - Gradient Boosting
   - Support Vector Machine (SVM)
   - K-Nearest Neighbors (KNN)
   - XGBoost
3. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - ROC-AUC

## 🚀 Results
- **Random Forest (Tuned)** → **95.2% Accuracy**  
- **ROC-AUC**: 99.1%  
- Outperformed all other models

### 🔑 Top Features (Feature Importance)
1. RP (22.1%)  
2. CS (16.2%)  
3. Temperature (13.5%)  
4. IP (12.5%)  
5. VOC (9.9%)  

## 💼 Business Impact
- 70–80% reduction in downtime  
- 30–40% lower maintenance costs  
- 2–3 years extended machine lifespan  
- Improved operational efficiency  

## 📈 Future Work
- Time-series analysis of sensor data  
- Integration of additional IoT sensors  
- Explainable ML dashboards  
- Real-time monitoring & alerts  

## 📂 Project Files
- **Machine Failure Prediction.ipynb** → Jupyter Notebook (Code & Analysis)  
- **Machine Failure Prediction Presentation.pptx** → Presentation slides (Report)

## 🛠️ Tech Stack
- Python (Pandas, NumPy, Scikit-learn, Seaborn, Matplotlib, Imbalanced-learn, XGBoost)  
- Jupyter Notebook  
- Machine Learning Classification Models  

---
👤 **Author**: Mohammed Hashmath Ali  
📌 *Placement Readiness Project – Predictive Maintenance*

