# ElevateLabs-AI-ML
Internship Task Submissions for Elevate Labs

## 🧰 Tools & Libraries Used
- Python 3
- Pandas
- NumPy
- Matplotlib
- Seaborn

---

## 📂 Task: Titanic Dataset Cleaning
### Objective:
To clean and preprocess the Titanic dataset in preparation for a machine learning pipeline.

### Steps Performed:
1. Loaded and explored the dataset.
2. Handled missing values using median (for Age) and mode (for Embarked).
3. Dropped sparse columns like 'Cabin'.
4. Applied one-hot encoding on categorical features (`Sex`, `Embarked`).
5. Standardized numerical features (`Age`, `Fare`).
6. Visualized and reviewed outliers using boxplots.

### Files Included:
- `titanic_cleaning.py`: Python script for data cleaning
- `Titanic-Dataset.csv`: Original dataset
-  Screenshots: Boxplots for `Age` and `Fare`
- `README.md`: Task explanation
