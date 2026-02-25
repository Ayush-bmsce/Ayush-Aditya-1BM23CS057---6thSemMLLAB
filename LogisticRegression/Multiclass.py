import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Load the dataset and Preprocessing
# Note: Ensure the csv files are in your working directory
try:
    zoo_df = pd.read_csv('zoo-data.csv')
    
    # Preprocessing: 
    # Usually, 'animal_name' is a unique identifier and should be dropped 
    # as it provides no predictive value.
    if 'animal_name' in zoo_df.columns:
        zoo_df = zoo_df.drop('animal_name', axis=1)

    # Separate features (X) and target (y)
    X = zoo_df.drop('class_type', axis=1)
    y = zoo_df['class_type']

    # Split into Training (80%) and Testing (20%) sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 2. Build Logistic Regression Model
    # We use multi_class='multinomial' for targets with more than two categories
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # 3. Measure the accuracy
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    # 4. Plot the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix - Zoo Animal Classification')
    plt.show()

except FileNotFoundError:
    print("Error: Please ensure 'zoo-data.csv' is in the same folder as this script.")
