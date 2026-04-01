import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

iris_df = pd.read_csv('iris.csv')

X = iris_df.drop('species', axis=1)
y = iris_df['species']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)

y_pred = knn.predict(X_test)
print("--- IRIS Dataset Results ---")
print(f"Accuracy Score: {accuracy_score(y_test, y_pred)}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))




diabetes_df = pd.read_csv('diabetes.csv')

cols_fix = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
for col in cols_fix:
    diabetes_df[col] = diabetes_df[col].replace(0, diabetes_df[col].median())

X_d = diabetes_df.drop('Outcome', axis=1)
y_d = diabetes_df['Outcome']

X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_d, y_d, test_size=0.2, random_state=42)

scaler_d = StandardScaler()
X_train_d = scaler_d.fit_transform(X_train_d)
X_test_d = scaler_d.transform(X_test_d)

knn_d = KNeighborsClassifier(n_neighbors=11) #
knn_d.fit(X_train_d, y_train_d)

y_pred_d = knn_d.predict(X_test_d)
print("\n--- Diabetes Dataset Results ---")
print(f"Accuracy Score: {accuracy_score(y_test_d, y_pred_d)}")
print("Confusion Matrix:\n", confusion_matrix(y_test_d, y_pred_d))




import matplotlib.pyplot as plt
import seaborn as sns

heart_df = pd.read_csv('heart.csv')
X_h = heart_df.drop('target', axis=1)
y_h = heart_df['target']

X_train_h, X_test_h, y_train_h, y_test_h = train_test_split(X_h, y_h, test_size=0.2, random_state=42)

scaler_h = StandardScaler()
X_train_h = scaler_h.fit_transform(X_train_h)
X_test_h = scaler_h.transform(X_test_h)

scores = []
for k in range(1, 21):
    knn_h = KNeighborsClassifier(n_neighbors=k)
    knn_h.fit(X_train_h, y_train_h)
    scores.append(knn_h.score(X_test_h, y_test_h))

best_k = scores.index(max(scores)) + 1
print(f"\nBest K value found: {best_k} with score: {max(scores)}")

final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train_h, y_train_h)
y_pred_h = final_knn.predict(X_test_h)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix(y_test_h, y_pred_h), annot=True, fmt='d', cmap='Blues')
plt.title(f'Confusion Matrix (K={best_k})')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

report = classification_report(y_test_h, y_pred_h, output_dict=True)
sns.heatmap(pd.DataFrame(report).iloc[:-1, :].T, annot=True, cmap='YlGnBu')
plt.title('Classification Report')
plt.show()



