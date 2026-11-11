# Titanic Survival Prediction

A machine learning classification project based on the **Titanic passenger dataset**.  
The notebook covers data loading, data cleaning, exploratory data analysis (EDA), feature encoding, correlation analysis, model training, evaluation, and hyperparameter tuning.

## 📌 Project Overview

The objective of this project is to predict whether a Titanic passenger survived (`1`) or did not survive (`0`) using passenger-related features.

### Target Variable

- **`Survived`**
  - `0` → Did not survive
  - `1` → Survived

## 📂 Dataset

The notebook uses the `train.csv` Titanic dataset.

Original dataset structure:

| Column | Description |
|---|---|
| `PassengerId` | Unique passenger identifier |
| `Survived` | Survival status / target variable |
| `Pclass` | Passenger class |
| `Name` | Passenger name |
| `Sex` | Passenger gender |
| `Age` | Passenger age |
| `SibSp` | Number of siblings/spouses aboard |
| `Parch` | Number of parents/children aboard |
| `Ticket` | Ticket number |
| `Fare` | Passenger fare |
| `Cabin` | Cabin information |
| `Embarked` | Port of embarkation |

The dataset contains **891 rows and 12 columns**.

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib / Seaborn
- Scikit-learn
- Jupyter Notebook

## 🔄 Project Workflow

```text
Load Dataset
     ↓
Initial Data Inspection
     ↓
Missing Value Analysis
     ↓
Data Cleaning
     ↓
Feature Selection
     ↓
Categorical Encoding
     ↓
Correlation Analysis
     ↓
Exploratory Data Analysis
     ↓
Train-Test Split
     ↓
Feature Scaling
     ↓
Model Training
     ↓
Model Evaluation
     ↓
Hyperparameter Tuning
     ↓
Model Comparison
```

## 1. Data Loading and Initial Inspection

The dataset is loaded using Pandas:

```python
df = pd.read_csv("train.csv")
cdf = df.copy()
```

Initial inspection includes:

- Dataset shape
- First few rows
- Random sample
- Data types
- Descriptive statistics
- Missing-value counts
- Duplicate-value checks

Initial dataset shape:

```text
(891, 12)
```

## 2. Missing Value Analysis

Missing values were checked using:

```python
cdf.isna().sum()
```

The main missing-value issues were found in:

- `Age`
- `Cabin`
- `Embarked`

### Age

Missing `Age` values were replaced using the mean age:

```python
cdf['Age'] = cdf['Age'].fillna(cdf.Age.mean())
```

### Embarked

Missing `Embarked` values were handled using the mode.

### Cabin

`Cabin` contains a large amount of missing information, so it was removed from the modeling dataset.

## 3. Data Cleaning and Feature Selection

The following columns were removed:

```text
Cabin
Ticket
PassengerId
Name
```

Reason:

- `Cabin` → substantial missing information
- `Ticket` → not directly used in the current modeling approach
- `PassengerId` → identifier rather than a predictive feature
- `Name` → raw textual information not used in this version

The modeling dataset therefore focuses on useful structured passenger features.

## 4. Categorical Encoding

### Embarked

`Embarked` was converted into numerical features using `OneHotEncoder`.

```python
OneHotEncoder(sparse_output=False)
```

This creates separate binary columns for the different embarkation ports.

### Sex

`Sex` was converted into numerical values using `LabelEncoder`.

## 5. Correlation Analysis

A correlation matrix was generated using:

```python
correlation = cdf.corr(numeric_only=True)
```

This was used to understand relationships between the numerical features and the target variable.

## 6. Exploratory Data Analysis

The notebook includes visual analysis of:

- Survival class distribution
- Survival rate by passenger class and sex
- Age distribution
- Age vs survival
- Fare distribution

Examples of questions explored:

- How is survival distributed?
- How does passenger class relate to survival?
- How does gender relate to survival?
- What is the distribution of passenger ages?
- What is the distribution of fares?

## 7. Model Training

The target and features were separated:

```python
X = cdf.drop(['Survived'], axis=1)
Y = cdf.Survived
```

The dataset was divided into training and testing sets using an **80:20 split** with stratification.

```python
train_test_split(
    X,
    Y,
    test_size=0.20,
    stratify=Y,
    random_state=42
)
```

The test set contains **179 observations**.

## 🤖 Machine Learning Models

The notebook evaluates the following classification algorithms:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Decision Tree
4. Random Forest

The models were evaluated using:

- Accuracy
- F1 Score
- Precision
- Recall
- Classification Report
- Confusion Matrix

## 📊 Model Results

### Test-set performance

| Model | Accuracy | F1 Score |
|---|---:|---:|
| Logistic Regression | 80.45% | 0.7244 |
| KNN | 81.01% | 0.7213 |
| Decision Tree | 77.65% | 0.6923 |
| Random Forest | **82.12%** | **0.7333** |
| Tuned Random Forest | 79.89% | 0.7049 |

Based on the recorded test-set results in the notebook, the **Random Forest model before GridSearchCV achieved the highest test accuracy (82.12%) and highest test F1 score (0.7333)** among the evaluated test predictions.

## 🔧 Hyperparameter Tuning

`GridSearchCV` with 5-fold cross-validation was used for model tuning.

### KNN

The notebook searched for:

```text
n_neighbors = 1 to 11
```

Best parameter:

```text
n_neighbors = 3
```

Best cross-validation F1 score:

```text
0.7423
```

### Decision Tree

The notebook searched:

```text
max_depth = 1 to 14
```

Best parameter:

```text
max_depth = 9
```

Best cross-validation F1 score:

```text
0.7392
```

### Random Forest

The notebook searched:

```text
max_depth = 1 to 14
n_estimators = 100, 150, None
```

Best parameters:

```text
max_depth = 7
n_estimators = 150
```

Best cross-validation F1 score:

```text
0.7529
```

## 📈 Evaluation Metrics

### Accuracy

Measures the overall percentage of correct predictions.

```text
Accuracy = Correct Predictions / Total Predictions
```

### Precision

Measures how many predicted positive cases were actually positive.

### Recall

Measures how many actual positive cases were correctly identified.

### F1 Score

The harmonic mean of precision and recall:

```text
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```

For this project, F1 score is useful because it considers both precision and recall rather than relying only on accuracy.

## 🧠 Key Learning Outcomes

This project demonstrates practical understanding of:

- Pandas data loading and inspection
- Missing-value handling
- Feature selection
- Categorical encoding
- Correlation analysis
- Exploratory Data Analysis
- Train-test splitting
- Stratified sampling
- Feature scaling
- Classification algorithms
- Confusion matrix
- Precision, recall and F1 score
- Cross-validation
- GridSearchCV
- Hyperparameter tuning
- Model comparison

## 📁 Repository Structure

Recommended GitHub structure:

```text
Titanic-Survival-Prediction/
│
├── Titanic_survival.ipynb
├── train.csv
├── README.md
└── requirements.txt
```

## ▶️ How to Run

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd Titanic-Survival-Prediction
```

### 2. Install dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

### 3. Start Jupyter Notebook

```bash
jupyter notebook
```

### 4. Open

```text
Titanic_survival.ipynb
```

Make sure `train.csv` is available in the same directory as the notebook.

## 🚀 Future Improvements

Possible improvements for a future version:

- Better feature engineering from `Name` and `Ticket`
- Extract cabin/deck information instead of completely dropping `Cabin`
- Compare additional algorithms such as SVM, Gradient Boosting and XGBoost
- Perform more extensive hyperparameter tuning
- Handle class imbalance if required
- Build a reusable prediction pipeline
- Deploy the trained model using FastAPI or Flask
- Create a simple React-based prediction interface

## 👨‍💻 Author

**Pavan Koli**

Computer Engineering | Data Analytics & Machine Learning

---

⭐ If you find this project useful, consider giving the repository a star.
