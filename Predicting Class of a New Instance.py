from sklearn.model_selection import train_test_split
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

# Yeni örnek özniteliklerini bir veri noktası olarak oluştur
new_instance = [[0,0,1,0,0,1,1,0,0,0,0,0,6,0,0,0]]

# MLPClassifier ile tahmin yap
mlp_prediction = mlp_clf.predict(new_instance)
print("MLP Classifier (Yapay Sinir Ağı) ile tahmin edilen sınıf:", mlp_prediction)

# k-NN ile tahmin yap
knn_prediction = knn_clf.predict(new_instance)
print("k-NN (k-Nearest Neighbors) ile tahmin edilen sınıf:", knn_prediction)
