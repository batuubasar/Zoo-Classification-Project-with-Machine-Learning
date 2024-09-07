from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
from ucimlrepo import fetch_ucirepo

# fetch dataset
zoo = fetch_ucirepo(id=111)

# data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets
y = y.squeeze()

# Veriyi eğitim ve test setlerine ayırma
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# MLPClassifier (Yapay Sinir Ağı) ile sınıflandırma
mlp_clf = MLPClassifier()
mlp_clf.fit(X_train, y_train)
mlp_predictions = mlp_clf.predict(X_test)

# k-NN (k-Nearest Neighbors) ile sınıflandırma
knn_clf = KNeighborsClassifier()
knn_clf.fit(X_train, y_train)
knn_predictions = knn_clf.predict(X_test)

# 10-fold cross validation için MLPClassifier performansı
mlp_scores = cross_val_score(mlp_clf, X, y, cv=10, scoring='accuracy')
mlp_precision = cross_val_score(mlp_clf, X, y, cv=10, scoring='precision_macro')
mlp_recall = cross_val_score(mlp_clf, X, y, cv=10, scoring='recall_macro')

# 10-fold cross validation için k-NN performansı
knn_scores = cross_val_score(knn_clf, X, y, cv=10, scoring='accuracy')
knn_precision = cross_val_score(knn_clf, X, y, cv=10, scoring='precision_macro')
knn_recall = cross_val_score(knn_clf, X, y, cv=10, scoring='recall_macro')

# Sonuçları yazdırma
print("MLP Classifier (Yapay Sinir Ağı) Performansı:")
print("Accuracy:", mlp_scores)
print("Precision:", mlp_precision)
print("Recall:", mlp_recall)

print("\nk-NN (k-Nearest Neighbors) Performansı:")
print("Accuracy:", knn_scores)
print("Precision:", knn_precision)
print("Recall:", knn_recall)
