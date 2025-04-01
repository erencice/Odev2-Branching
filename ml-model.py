import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Veriyi yükleme
X_train = pd.read_csv('X_train.csv')
X_test = pd.read_csv('X_test.csv')
y_train = pd.read_csv('y_train.csv')
y_test = pd.read_csv('y_test.csv')

# Model oluşturma
model = LogisticRegression()

# Modeli eğitim verisi ile eğitme
model.fit(X_train, y_train)

# Test verisi ile tahmin yapma
y_pred = model.predict(X_test)

# Sonuçları değerlendirme
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)

# Sonuçları yazdırma
print("Model Accuracy: ", accuracy)
print("Confusion Matrix: \n", conf_matrix)
