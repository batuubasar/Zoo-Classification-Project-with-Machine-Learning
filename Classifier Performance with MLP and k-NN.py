from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
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
# Hata (confusion) matrislerini hesaplama
mlp_confusion_matrix = confusion_matrix(y_test, mlp_predictions)
knn_confusion_matrix = confusion_matrix(y_test, knn_predictions)
# Hata (confusion) matrislerini yazdırma
print("MLP Classifier (Yapay Sinir Ağı) Hata Matrisi:")
print(mlp_confusion_matrix)
print("")
print("k-NN (k-Nearest Neighbors) Hata Matrisi:")
print(knn_confusion_matrix)

