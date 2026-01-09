from processing import process_data
from models import train_knn, train_decision_tree
from metrics import evaluate_model
import pandas as pd

filename = 'loan_data.csv'
x_train, x_test, y_train, y_test = process_data(filename)


knn = train_knn(x_train, y_train)
print("--- KNN ---")
evaluate_model(knn, x_test, y_test)


dt = train_decision_tree(x_train, y_train)
print("\n--- Decision Tree ---")
evaluate_model(dt, x_test, y_test)

