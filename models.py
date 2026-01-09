from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def train_knn(X_train, y_train):
    knn = KNeighborsClassifier(n_neighbors=3)
    knn.fit(X_train, y_train)
    return knn


def train_decision_tree(X_train, y_train):
    dt = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_split=10)
    dt.fit(X_train, y_train)
    return dt
