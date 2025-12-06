#columns work on Review_Count Votes Metascore Rating Duration Year Title
import pandas as pd


df = pd.read_csv("imdb-movies-dataset.csv")
df = df.dropna() 
# to remove any spaces from column name
df.columns = df.columns.str.strip()

duplicate_count = df.duplicated(subset='Title').sum()
print(f"Number of duplicate movies: {duplicate_count}")
##remove dublicts
df = df.drop_duplicates(subset='Title', keep='first')
duplicate_count = df.duplicated(subset='Title').sum()
print(f"Number of duplicate movies: {duplicate_count}")

df['Votes'] = df['Votes'].str.replace(',', '', regex=False).astype(int)
df['ReviewCount'] = df['ReviewCount'].str.replace(',', '', regex=False).astype(int)
df['Year'] = df['Year'].astype(int)
df['Rating'] = df['Rating'].astype(float)
df['Metascore'] = df['Metascore'].astype(int)
df['Duration'] = df['Duration'].astype(int)
df['Title'] = df['Title'].astype(str)

