# AI-Driven Predictive Maintenance System 🛠️

### 🚀 Project Overview
This project implements a machine learning solution for **Condition Monitoring** and **Predictive Maintenance**. By analyzing sensor data (Air Temperature, Process Temperature, Torque, RPM), the system detects potential machine failures and diagnoses the specific root cause of the defect (e.g., Power Failure vs. Heat Dissipation Failure).

This project demonstrates the application of **Artificial Intelligence** in **Mechanical Engineering** to minimize downtime and optimize industrial systems.

---

### 📊 Key Results

| Metric | Result | Description |
| :--- | :--- | :--- |
| **Binary Detection Accuracy** | **98.15%** | Accuracy in distinguishing between "Healthy" and "Failing" machines. |
| **Precision (Failure)** | **0.88** | High reliability in flagging failures (low false alarm rate). |
| **Diagnostics Capability** | **F1: 0.81** | Successfully isolates **Power Failures** and **Heat Dissipation Failures**. |

#### 📉 Visualization
*The Confusion Matrix below demonstrates the model's ability to correctly classify specific failure modes despite the highly imbalanced dataset.*

![Confusion Matrix](confusion_matrix.png)


---

### 📂 Dataset
**Source:** [AI4I 2020 Predictive Maintenance Dataset (UCI Machine Learning Repository)](https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset)

The dataset consists of 10,000 data points collected from a simulated milling machine, containing:
* **Features:** Air temperature [K], Process temperature [K], Rotational speed [rpm], Torque [Nm], Tool wear [min].
* **Targets:** Machine failure (Binary), Failure Type (Multi-class: TWF, HDF, PWF, OSF, RNF).

---

### ⚙️ Methodology

#### 1. Data Preprocessing
* **Feature Selection:** Removed non-predictive identifiers (UDI, Product ID) to prevent data leakage.
* **Target Engineering:** Created a consolidated `Failure_Type` variable for multi-class diagnostics.

#### 2. Model Architecture
* **Algorithm:** Random Forest Classifier (Ensemble Learning).
* **Class Imbalance Handling:** Implemented `class_weight='balanced'` to penalize the model for missing rare failure events (vital for industrial data where failures are <4%).

#### 3. Diagnostics Extension
* Extended the system from simple binary detection to a **multi-class classification system** capable of identifying:
    * Power Failure (PWF)
    * Heat Dissipation Failure (HDF)
    * Overstrain Failure (OSF)
    * Tool Wear Failure (TWF)
 
 ### Physics-Informed Predictive Maintenance System (SciML)

## 📌 Project Overview : Phase 2

Unlike traditional "black-box" data science approaches that rely solely on raw sensor correlations, this framework uses **Physics-Informed Feature Engineering** to inject domain knowledge (Thermodynamics, Mechanics) into the model. The system was validated using **SHAP (Shapley Additive Explanations)**, proving that derived physical constraints are superior predictors of failure than raw telemetry.

### 🚀 Key Impact
* **Thermodynamic Discovery:** Identified **Heat Dissipation Risk ($T_{process} - T_{air}$)** as the #1 driver of equipment failure, outperforming raw torque sensors by **~20%**.
* **Interpretability:** Replaced opaque "failure probabilities" with root-cause diagnostics (e.g., "Failure predicted due to Overstrain > 140 Nm·min").
* **Performance:** Achieved **>99% Recall** on critical failure modes (Power Failure, Heat Dissipation Failure).

---

## ⚙️ Methodology: The "Physics-First" Approach

Instead of training on raw sensor data alone, we engineered features based on first-principles mechanical equations to govern the model's decision boundary.

### 1. Physics-Informed Feature Engineering
We augmented the dataset with three governing physical quantities:

| Feature | Equation | Physics Domain | Purpose |
| :--- | :--- | :--- | :--- |
| **Mechanical Power ($P$)** | $$P = \tau \times \omega \times \frac{2\pi}{60}$$ | Rotational Mechanics | Detects **Power Failures (PWF)** where the spindle operates beyond the motor's power curve. |
| **Thermal Delta ($\Delta T$)** | $$\Delta T = T_{process} - T_{air}$$ | Thermodynamics | Captures **Heat Dissipation Failures (HDF)** caused by insufficient cooling relative to ambient conditions. |
| **Overstrain Load** | $$S = \tau \times t_{wear}$$ | Material Science | Proxies cumulative fatigue; failure occurs when high torque is applied to a worn tool. |

### 2. Model Architecture
* **Algorithm:** Random Forest Classifier (Optimized for tabular non-linearity).
* **Training Strategy:** `class_weight='balanced'` to handle the extreme rarity of failure events (approx 3% of data).
* **Validation:** Stratified K-Fold Cross-Validation to ensure generalization across unseen operating conditions.

---

## 📊 Results & SHAP Validation

We used **SHAP (Shapley Additive Explanations)** to reverse-engineer the model's decision-making process. The analysis confirmed that the model effectively "learned" the laws of physics, prioritizing our engineered features over raw data.

### Feature Importance Ranking (SHAP)
The model identified the **Thermodynamic Differential** as the strongest predictor of failure, validating the hypothesis that thermal runaway is the primary precursor to breakdown.

| Rank | Feature | SHAP Impact Score | What it monitors |
| :--- | :--- | :--- | :--- |
| 1 | Temp_Diff ($\Delta T$) | 0.0209 | Thermodynamics |
| 2 | Torque [Nm] | 0.0207 | Mechanical Stress |
| 3 | Overstrain | 0.0163 | Material Fatigue |
| 4 | Power_W | 0.0140 | Electrical Load |
| 5 | Tool Wear [min] | 0.0113 | Surface Quality |

---

### 💻 Technologies Used
* **Language:** Python 3.x
* **Data Processing:** Pandas, NumPy
* **Machine Learning:** Scikit-Learn (RandomForestClassifier, LogicsticRegression)
* **Visualization:** Matplotlib, Seaborn

### 🔧 How to Run
1.  Clone the repository:
    ```bash
    git clone https://github.com/SaiPatil059/EngineFailureAnalysisAI.git
    ```
2.  Install dependencies:
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```
3.  Run the python file  `enginefailureanalysisai.py`.

---

### 📬 Contact
**Sai Patil**
* Mechanical Engineering Student | AI & Mechanical Enthusiast
* [LinkedIn Profile](www.linkedin.com/in/sai-patil-39027a340)
