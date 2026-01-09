# Loan Approval Prediction

This project implements a **machine learning pipeline** to predict whether a loan
application will be approved or rejected based on applicant features.

The purpose of this project is educational: to demonstrate **data preprocessing,
model training, and evaluation** using classical supervised learning algorithms in Python.

---

## 📌 Problem Description

Loan approval is a **binary classification problem** where the goal is to predict
whether an applicant qualifies for a loan based on historical data.

Given a dataset containing applicant attributes, the system learns patterns that
help estimate approval decisions.

---

## 🧠 Machine Learning Approach

- **Problem Type:** Binary Classification  
- **Algorithms Used:**
  - K-Nearest Neighbors (KNN)
  - Decision Tree Classifier
- **Workflow:**
  1. Load dataset
  2. Preprocess and clean data
  3. Train models
  4. Evaluate performance using metrics

---

## 📂 Project Structure

### File Description

- **main.py**  
  Entry point of the application.  
  Loads the dataset, calls preprocessing functions, trains the models, and evaluates results.

- **processing.py**  
  Handles data preprocessing tasks such as feature preparation and dataset handling.

- **models.py**  
  Contains the implementation and training logic for:
  - K-Nearest Neighbors (KNN)
  - Decision Tree models

- **metrics.py**  
  Computes and displays evaluation metrics for the trained models.

- **loan_data.csv**  
  Dataset used for training and testing the models.

- **requirements.txt**  
  Lists all required Python dependencies.

---

## 📦 Requirements

Before running the project, install the required dependencies:

```bash
pip install -r requirements.txt

