# 7072CEM – Machine Learning Assignment  
## Bank Marketing Prediction using Multiple ML Algorithms (bank-full Dataset)

### 🔍 Project Overview
This project analyzes the **bank-full dataset** from a Portuguese bank marketing campaign to predict whether a customer will **subscribe to a term deposit**.  
The dataset is widely used for supervised machine learning tasks and contains both **numerical** and **categorical** variables describing client attributes, campaign interactions, and socio-economic indicators.

The main objective is to build and compare multiple machine learning models to determine which algorithm performs best for binary classification.

---

## 📂 Dataset Description – *bank-full*
The **bank-full.csv** dataset contains:

- **45,212 rows**  
- **17 input features** (age, job, education, marital, default, balance, contact types, durations, etc.)  
- **1 target variable**:  
  - `y` → *yes/no* (whether the client subscribed to a deposit)

The dataset contains:
- Numeric features  
- Categorical features with high cardinality  
- Imbalanced target variable (majority: “no”)  

This makes it ideal for comparing different machine learning algorithms.

---

# 🧠 Machine Learning Models Used
The project uses **four algorithms**, selected for interpretability, performance, and coverage of different ML families.

### 1️⃣ **SVM – RBF Kernel**
- Non-linear & strong performer  
- Works very well with complex decision boundaries  
- Requires feature scaling  
- Generalizes well by maximizing margins  

### 2️⃣ **Random Forest Classifier**
- Handles high-dimensional categorical data  
- Naturally robust to imbalance  
- Provides feature importance  
- Marker-friendly explainability  

### 3️⃣ **KNN Classifier**
- Simple & intuitive  
- Non-parametric  
- Requires scaling  
- Used for contrast against more complex models  

### 4️⃣ **Decision Tree**
- Easy to visualize  
- Helps demonstrate tree-based model logic  
- Baseline for Random Forest comparison  

---

# 🛠️ Project Workflow
1. Import and clean dataset  
2. Handle categorical variables (label encoding / one-hot)  
3. Split dataset into train/test  
4. Apply feature scaling where needed  
5. Train four ML algorithms  
6. Evaluate using:  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-Score  
   - Confusion Matrix  
7. Compare performance  
8. Present key findings  

---

# 📈 Visualizations
Recommended images to include in the `images/` folder:

- Correlation heatmap  
- Decision Tree plot  
- Confusion matrices for each model  
- Feature importance bar chart  
- Performance comparison graph  
```md
![Confusion Matrix](images/confusion_matrix.png)
