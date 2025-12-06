import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("imdb-movies-dataset.csv")
df.head()

df.shape

df.isnull().sum()
df = df.dropna() 
df.columns = df.columns.str.strip()

#show how many duplicts data
duplicate_count = df.duplicated(subset='Title').sum()
duplicate_count

##remove duplicts
df = df.drop_duplicates(subset='Title', keep='first')
duplicate_count = df.duplicated(subset='Title').sum()
duplicate_count


df['VotesCount'] = df['VotesCount'].str.replace(',', '', regex=False).astype(int)
df['ReviewCount'] = df['ReviewCount'].str.replace(',', '', regex=False).astype(int)
df['Year'] = df['Year'].astype(int)
df['Rating'] = df['Rating'].astype(float)
df['Metascore'] = df['Metascore'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Title'] = df['Title'].astype(str)

sns.pairplot(df)
plt.show()

numeric_df=df.select_dtypes(include=['float64', 'int64'])
numeric_df.corr()
sns.heatmap(numeric_df.corr(),annot=True,linewidths=1)
plt.show()

##from above heatmap it show that i will use duration and metascore and votescount

df['Rating'].plot.density()
plt.show()


plt.hist(df['Rating'])
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

#display relation between rating and year
plt.scatter(df['Year'], df['Rating'])
plt.xlabel("Year")
plt.ylabel("Rating")
plt.show()

#display relation between rating and Votes
plt.scatter(df['VotesCount'], df['Rating'])
plt.xscale("log")
plt.xlabel("Votes (log scale)")
plt.ylabel("Rating")
plt.show()

#display rating intervals with seaborn 
import seaborn as sns
sns.histplot(df['Rating'], bins=10, kde=True, color='skyblue')
plt.xlabel("Rating")     
plt.ylabel("Count")      
plt.title("Distribution of Movie Ratings") 
plt.show()

#display relation between rating and ReviewCount
plt.scatter(df['ReviewCount'], df['Rating'])
plt.xscale("log")
plt.xlabel("Votes (log scale)")
plt.ylabel("Rating")
plt.show()

import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['VotesCount'],y=df['Rating'])
plt.show()

sns.boxplot(x=df['Duration'],y=df['Rating'])
plt.show()

sns.boxplot(x=df['Metascore'],y=df['Rating'])
plt.show()

chossen_cols = ['VotesCount', 'Metascore', 'Duration']
df_clean = df.copy()

for col in chossen_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # This line automatically checks ALL rows for this column
    df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]


sns.boxplot(x=df_clean['Metascore'], y=df_clean['Rating'])
plt.show()

sns.boxplot(x=df_clean['VotesCount'], y=df_clean['Rating'])
plt.show()

sns.boxplot(x=df_clean['Duration'], y=df_clean['Rating'])
plt.show()

df_clean.shape

X = df_clean[['Metascore','VotesCount','Duration']].copy()
y = df_clean['Rating']

X['VotesCount'] = np.log1p(X['VotesCount'])
X['Duration'] = np.log1p(X['Duration'])
X['Metascore'] = np.log1p(X['Metascore'])

from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

##train model
model = LinearRegression()
model.fit(X_train, y_train)

from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("MSE:", mse)
rmse = np.sqrt(0.3119)
print("RMSE:",rmse)

y_pred = model.predict(X_train) 
from sklearn import metrics

r2 = metrics.r2_score(y_train, y_pred)
print("R-squared value of this fit:", round(r2, 3))

from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

# MSE & RMSE
mse = metrics.mean_squared_error(y_test, y_pred_test)
rmse = np.sqrt(mse)

# R² score
r2 = metrics.r2_score(y_test, y_pred_test)

print(f"MSE: {mse:.3f}")
print(f"RMSE: {rmse:.3f}")
print(f"R²: {r2:.3f}")

plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Random Forest Predictions vs Actual Ratings")
plt.plot([0,10], [0,10], color='red', linestyle='--')  # perfect prediction line
plt.show()

