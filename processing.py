import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def process_data(filename):
    df = pd.read_csv(filename)
    print(df.head())

    le = LabelEncoder()
    df['employment_status'] = le.fit_transform(df['employment_status'])

    x = df.drop('loan_approved', axis=1)
    y = df['loan_approved']

    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)

    x_train, x_test, y_train, y_test = train_test_split(
        x_scaled, y, test_size=0.2, random_state=42, stratify=y
    )

    return x_train, x_test, y_train, y_test

