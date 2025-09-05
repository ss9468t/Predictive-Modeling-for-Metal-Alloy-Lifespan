
This repository contains the implementation of multiple **machine learning techniques** applied to a provided dataset for predictive modeling, classification, clustering, and image recognition tasks. The work is structured around the following key stages:

---

## 📂 Project Structure

1. **Data Loading & Exploration**
   - Imported and explored the dataset using **Pandas**, **Matplotlib**, and **Seaborn**.
   - Performed summary statistics, correlation analysis, and missing value checks.
   - Conducted initial visualization with histograms, boxplots, scatterplots, and pairplots.

2. **Regression Analysis**
   - Applied multiple regression models to predict **Lifespan** of parts:
     - **Linear Regression**
     - **Random Forest Regressor** (with GridSearchCV for hyperparameter tuning)
     - **Decision Tree Regressor**
   - Achieved high accuracy with Random Forest (**R² ≈ 0.98**).
   - Residual analysis and actual vs. predicted plots included.

3. **Binary Classification**
   - Converted `Lifespan` into a binary target (`is_defective`).
   - Applied classifiers with **SMOTE** resampling to address imbalance:
     - Logistic Regression
     - Support Vector Machine (SVM)
     - Random Forest Classifier
   - Evaluation with:
     - **Confusion Matrix**
     - **Classification Report**
     - **ROC Curve & AUC**

4. **Convolutional Neural Network (CNN)**
   - Implemented a deep learning pipeline using **TensorFlow/Keras** for image classification.
   - Dataset preprocessing included:
     - Organizing images by defect type
     - Splitting into training/validation sets
     - Applying augmentation (rotation, zoom, shift, flip)
   - CNN architecture:
     - Multiple Conv2D + MaxPooling layers
     - Dense hidden layer + Dropout
     - Softmax output for 4 classes
   - Achieved up to **95% validation accuracy**.

5. **Clustering**
   - Applied **K-Means clustering** to group parts based on numerical features (`Lifespan`, `coolingRate`, `quenchTime`, `forgeTime`).
   - Determined optimal clusters using:
     - **Elbow Method**
     - **Silhouette Score**
   - Visualized clusters with scatter plots.

---

## 🛠️ Technologies Used
- **Python 3.x**
- **Pandas, NumPy** – Data manipulation
- **Matplotlib, Seaborn** – Visualization
- **Scikit-learn** – Regression, Classification, Clustering
- **Imbalanced-learn** – SMOTE oversampling
- **TensorFlow/Keras** – Deep learning (CNN)
- **Jupyter Notebook / Google Colab** – Development environment

---

## 📊 Key Results
- **Regression**: Random Forest model explained **98% variance** in lifespan prediction.
- **Classification**: Random Forest achieved the highest accuracy (**94%**) in defect classification.
- **CNN**: Model reached **~95% validation accuracy** on defect image dataset.
- **Clustering**: Optimal cluster size determined via silhouette score.

---

## 🚀 How to Run
1. Clone the repository:
   ```bash
   git clone https://github.com/ss9468t/Predictive-Modeling-for-Metal-Alloy-Lifespan.git
   cd Predictive-Modeling-for-Metal-Alloy-Lifespan
