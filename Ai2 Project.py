#**Load the Dataset**
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("imdb-movies-dataset.csv")
df.head()
df.shape
#Data Preprocessing
df.isnull().sum()
df = df.dropna() 
df.columns = df.columns.str.strip()
#remove dublicts

duplicate_count = df.duplicated(subset='Title').sum()
duplicate_count
df = df.drop_duplicates(subset='Title', keep='first')
duplicate_count = df.duplicated(subset='Title').sum()
print(duplicate_count)
df.shape
#Convert data types
df['VotesCount'] = df['VotesCount'].str.replace(',', '', regex=False).astype(int)
df['ReviewCount'] = df['ReviewCount'].str.replace(',', '', regex=False).astype(int)
df['Year'] = df['Year'].astype(int)
df['Rating'] = df['Rating'].astype(float)
df['Metascore'] = df['Metascore'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Title'] = df['Title'].astype(str)
df['Certificate'] = pd.to_numeric(df['Certificate'].astype(str).str.replace(',', ''), errors='coerce').fillna(0).astype(int)

#**Data Visualization**

#**Pairplots using seaborn**
sns.pairplot(df)
plt.show()
numeric_df=df.select_dtypes(include=['float64', 'int64'])
numeric_df.corr()
sns.heatmap(numeric_df.corr(),annot=True,linewidths=1)
plt.show()
df['Rating'].plot.density()
plt.show()


plt.hist(df['Rating'],)
plt.xlabel("Rating")
plt.ylabel("Count")
plt.show()

df['Rating'].plot.hist(bins=20,figsize=(5,5),edgecolor='k',)
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
#handle outliers
import seaborn as sns
import matplotlib.pyplot as plt

sns.boxplot(x=df['VotesCount'],y=df['Rating'])
plt.show()

sns.boxplot(x=df['Duration'],y=df['Rating'])
plt.show()
sns.boxplot(x=df['Metascore'],y=df['Rating'])
plt.show()
#**Remove Outliers**
# chossen_cols = ['VotesCount', 'Metascore', 'Duration']
# df_clean = df.copy()

# for col in chossen_cols:
#     Q1 = df_clean[col].quantile(0.25)
#     Q3 = df_clean[col].quantile(0.75)
#     IQR = Q3 - Q1
#     lower_bound = Q1 - 1.5 * IQR
#     upper_bound = Q3 + 1.5 * IQR
    
#     # This line automatically checks ALL rows for this column
#     df_clean = df_clean[(df_clean[col] >= lower_bound) & (df_clean[col] <= upper_bound)]
df_clean=df
#after remove outliers
sns.boxplot(x=df_clean['Metascore'], y=df_clean['Rating'])
plt.show()
sns.boxplot(x=df_clean.sample(2000)['VotesCount'], y=df_clean.sample(2000)['Rating'])
plt.show()
sns.boxplot(x=df_clean['Duration'], y=df_clean['Rating'])
plt.show()
df_clean.shape
### Feature Selection
#X = df_clean[['Metascore','VotesCount','Duration','Year','ReviewCount','Certificate']].copy()
X = df_clean[['Metascore','VotesCount','Duration','Year','ReviewCount']].copy()
y = df_clean['Rating']
X.shape
X.sample(10,ignore_index=False)
#**Handle skewed numeric columns (log-transform Votes & ReviewCount)**
X['VotesCount'] = np.log1p(X['VotesCount'])
X['ReviewCount'] = np.log1p(X['ReviewCount'])


### Model fit and training using LinearRegression
#**Import linear regression model estimator from scikit-learn and instantiate**
from sklearn.linear_model import LinearRegression
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
##train model
model = LinearRegression()
model.fit(X_train, y_train)
#**Evaluate**
from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)
mse = mean_squared_error(y_test, pred)
print("MSE:", mse)
rmse = np.sqrt(mse)
print("RMSE:",rmse)
r2 = metrics.r2_score(y_test, pred)
print("R-squared value of this fit:", round(r2, 3))
### Model fit and training using Random Forest
from sklearn.ensemble import RandomForestRegressor
rf = RandomForestRegressor(
     random_state=42,n_estimators=300,max_depth=50,min_samples_split=2,min_samples_leaf=11)
rf.fit(X_train, y_train)
y_pred_train = rf.predict(X_train)
y_pred_test = rf.predict(X_test)

def mse():
    mse_training = metrics.mean_squared_error(y_train, y_pred_train)
    rmse_training = np.sqrt(mse_training)
    r2_training = metrics.r2_score(y_train, y_pred_train)

    print("Training Results:")
    print(f"MSE (train): {mse_training:.3f}")
    print(f"RMSE (train): {rmse_training:.3f}")
    print(f"R² (train): {r2_training:.3f}")


    mse_test = metrics.mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = metrics.r2_score(y_test, y_pred_test)

    print("\nTesting Results:")
    print(f"MSE (test): {mse_test:.3f}")
    print(f"RMSE (test): {rmse_test:.3f}")
    print(f"R² (test): {r2_test:.3f}")
mse()    
plt.figure(figsize=(8,6))
sns.scatterplot(x=y_test, y=y_pred_test)
plt.xlabel("Actual Ratings")
plt.ylabel("Predicted Ratings")
plt.title("Random Forest Predictions vs Actual Ratings")
plt.plot([0,10], [0,10], color='red', linestyle='--')  
plt.show()
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, X, y, cv=5, scoring='r2')
print("",scores.mean())
#Feature Importance
importances = rf.feature_importances_
feature_names = X.columns

sorted_idx = importances.argsort()[::-1]

plt.figure(figsize=(10,8))
sns.barplot(x=importances[sorted_idx], y=feature_names[sorted_idx])
plt.title("Feature Importances")
plt.xlabel("Importance")
plt.ylabel("Features")
plt.show()
#**KNN Algorithm**
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)


best_r2 = -1
best_k = None

for k in range(1, 200):
    knn = KNeighborsRegressor(n_neighbors=k)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    print(f"K={k}, R²={r2:.3f}")
    
    if r2 > best_r2:
        best_r2 = r2
        best_k = k

print("\nBest K:", best_k)
print("Best R²:", best_r2)
