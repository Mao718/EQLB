from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def evaluate(train_representation, test_representation, train_label, test_label, neighbors=5):
    classifier = KNeighborsClassifier(n_neighbors=neighbors)
    classifier.fit(train_representation, train_label)
    y_pred = classifier.predict(test_representation)
    print('accuracy =', accuracy_score(test_label, y_pred))
