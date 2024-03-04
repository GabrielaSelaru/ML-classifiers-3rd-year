from sklearn.ensemble import RandomForestClassifier
import cv2
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
dataset_path = './Dataset'

# Funcție pentru încărcarea și preprocesarea imaginilor
def load_images(directory):
    images = []
    labels = []
    label_names = ['botine', 'pantofi','sandale']
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

# Inițializează și antrenează un clasificator Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=50)
rf_classifier.fit(X_train,y_train)

# Realizează predicții pe setul de testare
y_pred = rf_classifier.predict(X_test)

# Definește un dicționar de mapare a claselor numerice la etichete
class_mapping = {0: 'sanda', 1: 'botină', 2: 'pantof'}

# Afișează câteva exemple de imagini din setul de testare și predicțiile asociate
num_images_to_display = 5
selected_indices = np.random.choice(len(X_test), num_images_to_display, replace=False)

for i, idx in enumerate(selected_indices):
    plt.subplot(1, num_images_to_display, i + 1)
    plt.imshow(X_test[idx].reshape(64,64,1),cmap='gray')
    plt.title(f' {y_pred[idx]}')
    plt.axis('off')

plt.show()

report = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(report)

def plot_confusion_matrix(cm, classes, title):
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('Adevărat')
    plt.xlabel('Prezis')
    plt.show()

cm_test= confusion_matrix(y_test, y_pred)
y_train_pred=rf_classifier.predict(X_train)
cm_train= confusion_matrix(y_train, y_train_pred)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
sns.heatmap(cm_train, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - train")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.subplot(1, 2, 2)
sns.heatmap(cm_test, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - test")
plt.ylabel('True Label')
plt.xlabel('Predicted Label')

plt.tight_layout()
plt.show()

# Calculul acuratetei pe setul de antrenare
accuracy_train = rf_classifier.score(X_train, y_train)

# Calculul acuratetei pe setul de testare
accuracy_test = rf_classifier.score(X_test, y_test)

# Prezentarea rezultatelor
print("Acuratetea pe setul de antrenare:", accuracy_train)
print("Acuratetea pe setul de testare:", accuracy_test)
