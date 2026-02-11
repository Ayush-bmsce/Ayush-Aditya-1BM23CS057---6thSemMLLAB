import pandas as pd
from sklearn.linear_model import LinearRegression
import numpy as np

# Mapping words to numbers
word_to_num = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12
}

# Load the hiring data
hiring_file = r"C:\Users\BMSCECSE-SH\Desktop\ML Lab 3\hiring.csv"
df = pd.read_csv(hiring_file)

# Convert experience words to numbers
df['experience'] = df['experience'].str.lower().map(word_to_num)

# Fill missing values with mean
df['experience'].fillna(df['experience'].mean(), inplace=True)
df['test_score(out of 10)'].fillna(df['test_score(out of 10)'].mean(), inplace=True)
df['interview_score(out of 10)'].fillna(df['interview_score(out of 10)'].mean(), inplace=True)

# Rename columns for convenience
df.columns = ['Experience', 'TestScore', 'InterviewScore', 'Salary']

print("Hiring Data after preprocessing:")
print(df)

# Prepare data for model
X = df[['Experience', 'TestScore', 'InterviewScore']]
y = df['Salary']

# Build Multiple Linear Regression model
model = LinearRegression()
model.fit(X, y)

# Predict salaries for new candidates
candidates = np.array([[2, 9, 6],
                       [12, 10, 10]])
predictions = model.predict(candidates)

for i, salary in enumerate(predictions):
    print(f"\nPredicted salary for candidate {i+1}: ${salary:.2f}")
