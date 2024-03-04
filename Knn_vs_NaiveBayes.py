import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Calea către directorul cu imagini
dataset_path = './Dataset'

# Funcție pentru încărcarea și preprocesarea imaginilor
def load_images(directory):
    images = []
    labels = []
    label_names = ['botine', 'pantofi', 'sandale']
    for label in label_names:
        path = os.path.join(directory, label)
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, (64, 64))
            images.append(img.flatten())
            labels.append(label)
    return np.array(images), np.array(labels), label_names

# Încărcarea datelor
images, labels, label_names = load_images(dataset_path)

# Împărțirea datelor în set de antrenament și set de test
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# Antrenarea modelelor KNN și Naive Bayes
# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)
nb_predictions = nb.predict(X_test)

# Evaluarea modelelor
print("KNN Performance:")
print(classification_report(y_test, knn_predictions, target_names=label_names))
print("Accuracy:", accuracy_score(y_test, knn_predictions))

print("\nNaive Bayes Performance:")
print(classification_report(y_test, nb_predictions, target_names=label_names))
print("Accuracy:", accuracy_score(y_test, nb_predictions))

# Funcție pentru afișarea matricei de confuzie
def plot_confusion_matrix(cm, classes, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Adevărat')
    plt.xlabel('Prezis')
    plt.show()

# Calcularea și afișarea matricei de confuzie pentru KNN
knn_cm = confusion_matrix(y_test, knn_predictions)
plot_confusion_matrix(knn_cm, label_names, "KNN Confusion Matrix")

# Calcularea și afișarea matricei de confuzie pentru Naive Bayes
nb_cm = confusion_matrix(y_test, nb_predictions)
plot_confusion_matrix(nb_cm, label_names, "Naive Bayes Confusion Matrix")
