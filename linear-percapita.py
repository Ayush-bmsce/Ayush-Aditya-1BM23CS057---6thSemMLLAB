import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Step 1: Use your data directly (no CSV)


df = pd.read_csv("C:\\Users\\BMSCECSE-SH\\Desktop\\ML Lab 3\\canada_per_capita_income.csv")

# Step 2: Prepare data
X = df[['year']]
y = df['per capita income (US$)']

# Step 3: Train model
model = LinearRegression()
model.fit(X, y)

# Step 4: Predict 2020
pred_2020 = model.predict([[2020]])[0]
print("Predicted per capita income for 2020:", round(pred_2020, 2))

# Step 5: Plot data and regression line
plt.figure(figsize=(12,6))
plt.scatter(df['year'], df['per capita income (US$)'], color='blue', label='Actual Data')
plt.plot(df['year'], model.predict(X), color='red', linewidth=2, label='Regression Line')
plt.scatter(2020, pred_2020, color='green', s=100, label=f'Prediction 2020: {round(pred_2020,2)}')
plt.xlabel('Year')
plt.ylabel('Per Capita Income (US$)')
plt.title('Canada Per Capita Income Prediction')
plt.legend()
plt.grid(True)
plt.show()
