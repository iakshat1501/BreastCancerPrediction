# 🧠 Breast Cancer Prediction Using Machine Learning

This project focuses on predicting whether a tumor is benign or malignant using supervised machine learning models trained on real breast cancer diagnostic data.

---

## 📁 Project Structure
BreastCancerPrediction/
├── Cancer_Data.csv # Breast cancer dataset
├── DATAMINING_PROJECT.ipynb # Jupyter notebook with full ML pipeline
├── model.pkl # Trained ML model
├── welcome.py # Python script for model inference
├── image.jpg # Visualization image or UI element
├── Project_Report.docx # Detailed report of the project


---

## 🎯 Objective

To build a robust and accurate classification model that:
- Identifies breast cancer diagnoses (benign or malignant)
- Uses key features from a clinical dataset
- Supports decision-making in healthcare diagnostics

---

## 📊 Dataset: `Cancer_Data.csv`

The dataset contains attributes derived from digitized images of fine needle aspirates (FNA) of breast masses.

### Key Features:
- `radius_mean`, `texture_mean`, `perimeter_mean`, etc.
- `concavity`, `symmetry`, `fractal_dimension`
- `diagnosis` (target): `M` (malignant) or `B` (benign)

Dataset Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))

---

## 🔬 Models Used

- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest Classifier
- Decision Tree Classifier
- k-Nearest Neighbors (KNN)

---

## 📈 Workflow Summary

1. **Data Preprocessing**
   - Handling missing values
   - Feature scaling (StandardScaler)
   - Encoding target labels

2. **Model Training**
   - Trained multiple models with train-test split
   - Used cross-validation for performance metrics

3. **Model Evaluation**
   - Accuracy
   - Confusion Matrix
   - ROC Curve / AUC

4. **Model Deployment**
   - Exported model with `joblib`/`pickle` (`model.pkl`)
   - `welcome.py` script used for loading model and predictions

---


## 📌 Results
Achieved over 95% accuracy with Random Forest and SVM

Effective feature selection improved model performance

ROC-AUC: ~0.98 on best models

---

## 📄 Project Report
A detailed explanation of:

Data cleaning & transformation

Model comparisons

Results interpretation

Check the file: Project_Report.docx

---

## 🔮 Future Improvements
Integration with Flask or Streamlit for web-based inference

Real-time clinical application with user upload interface

Hyperparameter optimization with GridSearchCV

---


## 👤 Author
Akshat Garg

```bash
pip install pandas numpy scikit-learn matplotlib seaborn

