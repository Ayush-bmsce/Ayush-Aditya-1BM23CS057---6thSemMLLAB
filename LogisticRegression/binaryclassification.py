import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("C:\\Users\\BMSCECSE-SH\\Desktop\\Logistic Regression\\HR_comma_sep.csv")

# Identify variables with impact
# Grouping by 'left' to see mean values of numerical features
print(df.groupby('left').mean(numeric_only=True))







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("C:\\Users\\BMSCECSE-SH\\Desktop\\Logistic Regression\\HR_comma_sep.csv")

# Identify variables with impact
# Grouping by 'left' to see mean values of numerical features
pd.crosstab(df.salary, df.left).plot(kind='bar')
plt.title('Retention by Salary Level')
plt.ylabel('Number of Employees')
plt.show()








import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("C:\\Users\\BMSCECSE-SH\\Desktop\\Logistic Regression\\HR_comma_sep.csv")
pd.crosstab(df.Department, df.left).plot(kind='bar', figsize=(10,6))
plt.title('Retention by Department')
plt.ylabel('Number of Employees')
plt.show()







import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("C:\\Users\\BMSCECSE-SH\\Desktop\\Logistic Regression\\HR_comma_sep.csv")



# Narrowing down features
sub_df = df[['satisfaction_level', 'average_montly_hours', 'promotion_last_5years', 'salary']]

# Handling categorical data (Salary)
salary_dummies = pd.get_dummies(sub_df.salary, prefix="salary")
df_with_dummies = pd.concat([sub_df, salary_dummies], axis='columns')
X = df_with_dummies.drop('salary', axis='columns')

y = df.left

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Build Model
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

accuracy = model.score(X_test, y_test)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Quick prediction example
# Predict for a person with [0.5 sat, 200 hrs, 0 promo, 0 low, 1 medium, 0 high salary]
# print(model.predict([[0.5, 200, 0, 0, 1, 0]]))





