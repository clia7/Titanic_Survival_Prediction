# Titanic - Machine Learning from Disaster 🚢

This project is focusing on predicting passenger survival using machine learning.
The entire pipeline was built, optimized, and submitted to [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic).

Using this clean, end-to-end engineered data pipeline and a classification model, I achieved an **Kaggle Accuracy Score of 77.51%**.

## 📊 Project Overview & Results
* **Local Validation Score:** ~84.36% Accuracy (evaluated via `train_test_split` with `random_state=42`)
* **Official Kaggle Test Score:** **77.51%** (Solid placement in the top 50% for a first honest submission)
* **Model Used:** Logistic Regression (`LogisticRegression` with `max_iter=1000`)
* **Feature Set:** 27 optimized features, including engineered variables for titles, family sizes, and deck locations.

## 🛠️ Data Pipeline & Key Technical Learnings

### 1. Robust Data Preprocessing (Anti-Data-Leakage)
To ensure true model generalization, missing data imputation for numerical values (`Age`, `Fare`) and categorical values (`Embarked`) was performed strictly within separate contexts for the training and test sets. Calculating medians independently prevents *data leakage*, making the pipeline production-ready and reliable on unseen test data.

### 2. Feature Engineering & Selection
The features were meticulously prepared to capture social status and family dynamics on the ship:
* **Title Extraction:** Extracted titles from names (e.g., Mr, Mrs, Miss, Master) to capture social standing.
* **Family & Group Metrics:** Built `FamilySize`, `TicketGroupSize`, and `isAlone` tracking to identify groups traveling together.
* **Deck Mapping:** Extracted `Deck` letters from the cabin data to capture structural location features on the Titanic.

### 3. Structural Alignment & Matrix Geometry
Because machine learning models process tabular data strictly as mathematical matrices (relying on indexing rather than reading text-based column names), preserving column geometry is critical. The pipeline ensures that the test dataset is explicitly filtered and re-aligned (`X_test_final = test_df[model_features]`) to exactly match the 27-feature structure of the training matrix, preventing positional errors during inference.

## 📂 Repository File Structure
* `Titanic_Survival_Prediction.ipynb`: The complete pipeline notebook featuring Exploratory Data Analysis (EDA), engineering, strict column alignment, and full-dataset training.
* `submission.csv`: The clean, final prediction matrix successfully mapped to `PassengerId` and exported for Kaggle evaluation.

## 🚀 Environment & Libraries
* **Python 3**
* **Pandas & NumPy** (Data manipulation, missing data handling, and matrix transformations)
* **Matplotlib & Seaborn** (Exploratory Data Analysis and distribution tracking)
* **Scikit-Learn** (Train-test splits, model fitting, and metrics evaluation)
