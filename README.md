# ML-6-MBA-Admission-2025
This project analyzes and predicts MBA admissions using machine learning techniques. The dataset used contains various attributes related to applicants, including test scores, GPA, and other factors influencing admission decisions.

# MBA Admission Class 2025 Prediction

## Overview

![image](https://github.com/user-attachments/assets/e1b62894-5428-4104-b98b-61053abb5446)

## Dataset
The dataset used is `MBA.csv`, which includes:
- GMAT Scores
- GPA Scores
- Work Experience
- Extracurricular Activities
- Admission Decision (Target Variable)

## Steps to Run the Project

### 1. Install Dependencies
Ensure you have Python installed along with the required libraries.

```bash
pip install pandas numpy matplotlib seaborn scikit-learn missingno
```

### 2. Load the Data

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import missingno as msno

warnings.filterwarnings('ignore')
sns.set_style('whitegrid')

# Load dataset
df = pd.read_csv('MBA.csv')
# Display first few rows
df.head()
```

## Exploratory Data Analysis (EDA)

### Data Summary
The dataset consists of several key features that impact MBA admissions. The summary statistics provide insights into the distribution of numerical features such as GMAT scores, GPA, and work experience. Categorical features like extracurricular activities and admission decisions help in understanding patterns in the data.

### Data Cleaning Insights
- **Duplicates**: The dataset was checked for duplicate records to ensure data integrity.
- **Missing Values**: Some missing values were identified, and visualization using a missing value matrix helped in understanding the extent of the issue.
- **Outliers**: Key numerical features were analyzed for outliers, as extreme values in GMAT or GPA could impact predictions.

### Data Distribution and Trends
- **GPA Scores**: The GPA distribution showed that most applicants had a GPA between 3.0 and 4.0, with a slight skew towards higher GPAs.
- **GMAT Scores**: The GMAT scores were normally distributed, with most scores falling between 600 and 750.
- **Work Experience**: The dataset revealed that a significant portion of applicants had prior work experience, which could influence admissions positively.

### Correlation Analysis
- A **correlation heatmap** showed a strong relationship between GMAT scores, GPA, and the likelihood of admission.
- Work experience had a moderate correlation with admission, indicating that higher experience levels slightly improve acceptance chances.
- Some features exhibited weak correlations, suggesting they might not contribute significantly to the model’s predictive power.

## Model Training

### Decision Tree Model
A **Decision Tree Classifier** was chosen for prediction due to its interpretability and efficiency. The tree was trained with a depth of 5 and a minimum of 5 samples per leaf to avoid overfitting.

```python
from sklearn.tree import DecisionTreeClassifier
dtree = DecisionTreeClassifier(max_depth=5, min_samples_leaf=5, random_state=100)
dtree.fit(X_train, y_train)
```

### Decision Tree Visualization
The trained decision tree was visualized to understand how the model makes decisions based on feature splits.

```python
from sklearn.tree import plot_tree
plt.figure(figsize=(15,9))
tree = plot_tree(dtree, filled=True, feature_names=X.columns, class_names=['Rejected', 'Accepted'], fontsize=8)
plt.show()
```
![image](https://github.com/user-attachments/assets/bad7e40a-7927-4a57-8349-d52b16cadc3f)

## Insights
- **GMAT and GPA scores** are the most important predictors of admission decisions.
- **Work experience** has a moderate impact but is not the strongest factor.
- **The Decision Tree model provides an interpretable structure** to understand the decision-making process.
- **Visualizing the tree allows easy identification of key factors influencing admissions.**

## Future Improvements
- Tune hyperparameters to improve model accuracy.
- Use more advanced models like Random Forest or XGBoost for better generalization.
- Incorporate additional features such as interview scores to enhance predictions.

## Repository Structure
```
|-- main.ipynb  # Jupyter Notebook with analysis
|-- MBA.csv     # Dataset file
|-- README.md   # Project documentation
```

## Contributing
Feel free to open issues or submit pull requests to improve this project.

## License
This project is licensed under the MIT License.
