import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

%matplotlib inline

print("Starting Salary Prediction Script...")

# Step 1: CSV file path
file_path = r"C:\Users\BMSCECSE-SH\Desktop\ML Lab 3\salary.csv"

# Step 2: Load CSV if exists, else use dummy data
if os.path.exists(file_path):
    df = pd.read_csv(file_path)
    print("CSV loaded successfully!")
else:
    print("CSV file not found. Using dummy dataset.")
    data = {
        "YearsExperience": [1,2,3,4,5,6,7,8,9,10],
        "Salary": [45000,50000,60000,65000,70000,75000,80000,85000,90000,95000]
    }
    df = pd.DataFrame(data)

# Step 3: Check for missing values
print("\nMissing values in each column:")
print(df.isnull().sum())

# Step 4: Handle missing values by filling with mean
df['YearsExperience'] = df['YearsExperience'].fillna(df['YearsExperience'].mean())
df['Salary'] = df['Salary'].fillna(df['Salary'].mean())

# Step 5: Clean column names
df.columns = df.columns.str.strip()

# Step 6: Prepare data for regression
X = df[['YearsExperience']]  # Independent variable
y = df['Salary']             # Dependent variable

# Step 7: Build Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Step 8: Predict salary for 12 years of experience
years = 12
predicted_salary = model.predict([[years]])[0]
print(f"\nPredicted salary for {years} years of experience: ${predicted_salary:.2f}")

# Step 9: Plot data and regression line
plt.figure(figsize=(10,6))
plt.scatter(X, y, color='blue', label='Actual Salary')
plt.plot(X, model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.scatter([years], [predicted_salary], color='green', s=100, label=f'Prediction ({years} yrs)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.title('Salary Prediction based on Experience')
plt.legend()
plt.grid(True)
plt.show()
